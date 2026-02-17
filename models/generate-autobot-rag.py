"""
Generation helper for autobot-rag.
"""

import time
from typing import Any, Dict

import torch

SPECIAL_TOKENS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|startoftext|>"]


def generate_autobot_response(
    model: Any,
    tokenizer: Any,
    system_message: str,
    user_prompt: str,
    device: str,
    max_tokens: int,
    temperature: float,
    max_context_length: int,
    max_tokens_hard_limit: int,
    model_label: str = "autobot-rag",
) -> Dict[str, Any]:
    """
    Generate a full non-streaming response payload.
    """
    print("[generate-autobot-rag] =======================================")
    print("[generate-autobot-rag] generate_autobot_response called")
    print(f"[generate-autobot-rag] model={model_label}")
    print(f"[generate-autobot-rag] device={device}")
    print(f"[generate-autobot-rag] max_tokens={max_tokens}, temperature={temperature}")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    print("[generate-autobot-rag] Applying chat template...")
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(
        f"[generate-autobot-rag] Formatted prompt length={len(formatted_prompt)} chars"
    )

    template_token_count = len(tokenizer.encode(formatted_prompt, add_special_tokens=False))
    print(f"[generate-autobot-rag] Template token count={template_token_count}")

    print("[generate-autobot-rag] Tokenizing prompt...")
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_context_length - max_tokens,
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    print(f"[generate-autobot-rag] Tokenized input length={input_len} tokens")

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
        "use_cache": True,
    }

    if temperature <= 0.1:
        generation_config["do_sample"] = False
        generation_config.pop("top_p", None)
        generation_config.pop("top_k", None)
        print("[generate-autobot-rag] Deterministic mode enabled")
    else:
        generation_config["temperature"] = max(0.1, min(temperature, 1.0))

    print("[generate-autobot-rag] Running model.generate...")
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**generation_config)
    elapsed = time.time() - start_time

    generated_ids = output_ids[0][input_len:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    cleaned_text = raw_text
    for token in SPECIAL_TOKENS:
        cleaned_text = cleaned_text.replace(token, "")
    cleaned_text = cleaned_text.strip()

    generated_tokens = int(generated_ids.shape[0])
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[generate-autobot-rag] Generation complete: tokens={generated_tokens}, elapsed={elapsed:.2f}s, tps={tokens_per_sec:.1f}"
    )
    print("[GEN] Returning generated response payload")
    print("[generate-autobot-rag] =======================================")

    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "template_token_count": template_token_count,
        "formatted_prompt": formatted_prompt,
        "input_length": input_len,
        "generated_tokens": int(generated_ids.shape[0]),
    }
