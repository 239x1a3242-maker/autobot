"""
Autobot Thinking Response Generator - Synchronous Generation Module
Pure generation logic: takes model + prompt → returns dict with thinking/answer

Input/Output Contract:
- Input: model, tokenizer, system_message, user_prompt, device, max_context_length, max_tokens, 
         max_tokens_hard_limit, temperature, tools_json
- Output: Dict with keys {think, raw_think, text, raw_text, template_token_count, formatted_prompt, 
         input_length, generated_tokens, elapsed, tokens_per_sec, success, error}
"""

import json
import time
import torch
import traceback
from typing import Optional, Dict, Any
from transformers import TextIteratorStreamer
from threading import Thread


def strip_special_tokens(text: str) -> str:
    """
    Remove special tokens from text.
    
    Args:
        text: Text potentially containing special tokens
        
    Returns:
        Cleaned text with special tokens removed
    """
    special_tokens = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "<|startoftext|>",
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<|tool_list_start|>",
        "<|tool_list_end|>",
        "<think>",
        "</think>",
    ]
    cleaned = text
    for tok in special_tokens:
        if tok in cleaned:
            cleaned = cleaned.replace(tok, "")
    return cleaned


def generate_autobot_thinking(
    model: Any,
    tokenizer: Any,
    system_message: str,
    user_prompt: str,
    device: str = "cuda",
    max_context_length: int = 32768,
    max_tokens: int = 4500,
    max_tokens_hard_limit: int = 24000,
    temperature: float = 0.7,
    tools_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate response from Autobot Thinking model (synchronous).
    
    **INPUTS:**
    - model: Loaded transformer model instance
    - tokenizer: Loaded tokenizer instance
    - system_message: System prompt/instructions for the model
    - user_prompt: User query/message
    - device: "cuda" or "cpu"
    - max_context_length: Maximum context window (default 32768)
    - max_tokens: Target generation length (default 4500)
    - max_tokens_hard_limit: Hard limit on generation (default 24000)
    - temperature: Sampling temperature 0.0-1.0 (default 0.7)
    - tools_json: Optional JSON string with available tools
    
    **OUTPUTS:**
    Returns dict with:
    - think: str - Thinking content (cleaned, without tags)
    - raw_think: str - Raw thinking (with special tokens)
    - text: str - Final answer (cleaned, without special tokens)
    - raw_text: str - Raw answer (with special tokens)
    - template_token_count: int - Tokens after template applied
    - formatted_prompt: str - Full formatted prompt to model
    - input_length: int - Input token count
    - generated_tokens: int - Tokens generated
    - elapsed: float - Generation time in seconds
    - tokens_per_sec: float - Generation speed
    - success: bool - Whether generation succeeded
    - error: str - Error message if success=False (empty string if success=True)
    """
    
    print(f"\n🎯 GENERATE_AUTOBOT_THINKING: '{user_prompt[:60]}...'")
    print(f"📊 Config: max_tokens={max_tokens}, temp={temperature}, device={device}")
    
    start_time = time.time()
    
    # ===== VALIDATION =====
    if not model or not tokenizer:
        error_msg = "Model or tokenizer not available."
        print(f"❌ {error_msg}")
        return {
            "think": "",
            "raw_think": "",
            "text": "",
            "raw_text": "",
            "template_token_count": 0,
            "formatted_prompt": "",
            "input_length": 0,
            "generated_tokens": 0,
            "elapsed": time.time() - start_time,
            "tokens_per_sec": 0.0,
            "success": False,
            "error": error_msg
        }
    
    try:
        # ===== BUILD MESSAGES =====
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"  📝 Messages prepared")
        
        # ===== APPLY CHAT TEMPLATE =====
        try:
            if tools_json:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=json.loads(tools_json) if isinstance(tools_json, str) else tools_json,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except TypeError:
            # Fallback if tools parameter not supported
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        template_token_count = len(formatted_prompt)
        print(f"  ✅ Chat template applied ({template_token_count} chars)")
        
        # ===== TOKENIZE =====
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_context_length - max_tokens,
        ).to(device)
        
        input_length = inputs['input_ids'].shape[1]
        print(f"  ✅ Tokenized ({input_length} tokens)")
        
        # ===== CREATE STREAMER =====
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            timeout=300.0
        )
        
        # ===== GENERATION CONFIG =====
        generation_config = {
            **inputs,
            "max_new_tokens": min(max_tokens, max_tokens_hard_limit),
            "do_sample": temperature > 0,
            "top_p": 0.1,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
            "use_cache": True,
        }
        
        if temperature <= 0.1:
            generation_config["do_sample"] = False
            generation_config.pop("top_p", None)
            generation_config.pop("top_k", None)
        else:
            generation_config["temperature"] = max(0.1, min(temperature, 1.0))
        
        # ===== START GENERATION THREAD =====
        thread = Thread(target=model.generate, kwargs=generation_config)
        thread.start()
        print(f"  🔄 Generation started")
        
        # ===== STREAM TOKENS =====
        raw_think = ""
        raw_text = ""
        in_thinking_block = False
        token_count = 0
        
        for new_text in streamer:
            if not new_text:
                continue
            
            token_count += 1
            
            # Track thinking blocks
            if "<think>" in new_text:
                in_thinking_block = True
                new_text = new_text.replace("<think>", "")
            
            if "</think>" in new_text:
                in_thinking_block = False
                new_text = new_text.replace("</think>", "")
            
            # Accumulate raw content
            if in_thinking_block:
                raw_think += new_text
            else:
                raw_text += new_text
        
        thread.join(timeout=2.0)
        
        # ===== CLEAN TOKENS =====
        think = strip_special_tokens(raw_think)
        text = strip_special_tokens(raw_text)
        
        elapsed = time.time() - start_time
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
        
        print(f"  ✅ Generation complete: {token_count} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} t/s)")
        
        # ===== RETURN RESULT =====
        return {
            "think": think,
            "raw_think": raw_think,
            "text": text,
            "raw_text": raw_text,
            "template_token_count": template_token_count,
            "formatted_prompt": formatted_prompt,
            "input_length": input_length,
            "generated_tokens": token_count,
            "elapsed": round(elapsed, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "success": True,
            "error": ""
        }
    
    except torch.cuda.OutOfMemoryError:
        error_msg = "CUDA out of memory. Try reducing max_tokens or using CPU."
        print(f"  ❌ {error_msg}")
        elapsed = time.time() - start_time
        return {
            "think": "",
            "raw_think": "",
            "text": "",
            "raw_text": "",
            "template_token_count": 0,
            "formatted_prompt": "",
            "input_length": 0,
            "generated_tokens": 0,
            "elapsed": round(elapsed, 3),
            "tokens_per_sec": 0.0,
            "success": False,
            "error": error_msg
        }
    
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(f"  ❌ {error_msg}")
        traceback.print_exc()
        elapsed = time.time() - start_time
        return {
            "think": "",
            "raw_think": "",
            "text": "",
            "raw_text": "",
            "template_token_count": 0,
            "formatted_prompt": "",
            "input_length": 0,
            "generated_tokens": 0,
            "elapsed": round(elapsed, 3),
            "tokens_per_sec": 0.0,
            "success": False,
            "error": error_msg
        }
