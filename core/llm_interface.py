"""
LLM Interface for AutoBot
Abstraction layer for local language model interactions.
Uses LFM2.5-1.2B model (AshokGakr/model-tiny) for all tasks.

Model: LFM2.5-1.2B-Instruct (AshokGakr/model-tiny)
- Architecture: 16 layers, 1.2B parameters
- Chat Template: ChatML-like format with special tokens
- Generation Parameters: temperature=0.1, top_k=50, top_p=0.1, repetition_penalty=1.05
"""

import os, json, re, traceback, asyncio, time, sys
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import logging
from typing import Dict, Any, Optional
import torch
LFM_MODEL_NAME = "./models"
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_TOKENS_DEFAULT = 4500
MAX_TOKENS_HARD_LIMIT = 24000  # Conservative limit for stability
DEFAULT_TEMPERATURE = 0.7  # LFM2.5 default is 0.1, but keep adjustable
STREAM_DELAY = 0.01
MAX_CONTEXT_LENGTH = 32768  # Safe context length for LFM2.5 (max 32768)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Global model variables
lfm_tokenizer = None
lfm_model = None

class LLMInterface:
    """Interface to local LFM2.5 model for all NLP tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.model_available = False

    async def initialize(self):
        """Load and initialize all ML models with optimized settings for LFM2.5"""
        global lfm_tokenizer, lfm_model
    
        print("\n" + "="*50)
        print("🔄 LOADING AI MODELS - LFM2.5-1.2B-INSTRUCT")
        print("="*50)
    
        # ==== OPTIMIZED LFM2.5 LOADING ====
        lfm_success = False
        try:
            print(f"📥 Loading GAKR tiny model model")
            print(f"model loading from {LFM_MODEL_NAME}")
            # Load tokenizer with correct settings
            lfm_tokenizer = AutoTokenizer.from_pretrained(
                LFM_MODEL_NAME,
                padding_side="left",  # Essential for streaming
                trust_remote_code=False,
                use_fast=True
            )
        
            # IMPORTANT: LFM2.5 uses special tokens <|im_start|> and <|im_end|>
            # No need to set pad token separately as the tokenizer handles it
        
            print(f"   Tokenizer loaded with vocab size: {lfm_tokenizer.vocab_size}")
            print(f"   Chat template available: {lfm_tokenizer.chat_template is not None}")
        
            # Configure model loading based on available hardware
            load_kwargs = {
                "torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                "trust_remote_code": False,
                "device_map": "auto" if DEVICE == "cuda" else None,
            }
        
            lfm_model = AutoModelForCausalLM.from_pretrained(
                LFM_MODEL_NAME,
                **load_kwargs
            )
        
            # If device_map not used, manually move to device
            if DEVICE != "cuda" or load_kwargs.get("device_map") is None:
                lfm_model = lfm_model.to(DEVICE)

            lfm_model.eval()
        
            # Enable eval mode for inference optimizations
            lfm_model.config.use_cache = True  # Enable KV cache for faster generation
        
            print(f"✅ GAKR tiny model loaded successfully")
            print(f"   Model device: {lfm_model.device}")
            print(f"   Model dtype: {lfm_model.dtype}")
            print(f"   Model parameters: ~1.2B")
            print(f"   Context window: 32,768 tokens")
            lfm_success = True
        
        except Exception as e:
            print(f"❌ Failed to load LFM2.5 model: {e}")
            traceback.print_exc()
        
            # Fallback: Try with simpler settings
            try:
                print("🔄 Trying simpler loading method...")
                lfm_tokenizer = AutoTokenizer.from_pretrained(LFM_MODEL_NAME)
            
                lfm_model = AutoModelForCausalLM.from_pretrained(
                    LFM_MODEL_NAME,
                    torch_dtype=torch.float32
                )
                lfm_model.to(DEVICE).eval()
                print(f"✅ LFM2.5 model loaded (fallback method)")
                lfm_success = True
            except Exception as e2:
                print(f"❌ Fallback loading failed: {e2}")
                lfm_tokenizer = None
                lfm_model = None
    
        print("="*50)
        print(f"📊 MODEL LOADING SUMMARY:")
        print(f"  • GAKR tiny Model: {'✅' if lfm_success else '❌'}")
        print(f"  • Device: {DEVICE}")
        print(f"  • Max Context: {MAX_CONTEXT_LENGTH} tokens")
        print("="*50 + "\n")
    
        # Update instance variables and model availability flag
        if lfm_success:
            self.model = lfm_model
            self.tokenizer = lfm_tokenizer
            self.model_available = True
    
        return lfm_success


    async def generate_with_intent_model(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using the model with recommended parameters."""
        if not self.model_available or self.model is None:
            raise RuntimeError("Model not available - was not initialized")

        try:
            # Use provided system prompt or default
            if system_prompt is None:
                system_prompt = """You are AutoBot, a helpful AI assistant powered by advanced AI language models.

**Available Capabilities:**
1. **Web Search**: Research and gather information from the internet
   - Optimized for information retrieval
   - Returns: structured search results with summaries
   
2. **General Conversation**: Answer questions and have discussions

Be concise, accurate, and always identify which capability is needed for the user's request."""

            # Format as chat message for LFM2.5 ChatML template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template to format the messages
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,  
                tokenize=False,
                add_generation_prompt=True,
            )
            
            print(f"🔍 Formatted prompt length: {len(formatted_prompt)} characters")
            print(f"🔍 Sample formatted prompt: {formatted_prompt[:200]}...")

            
            # Tokenize the formatted prompt
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT_LENGTH ,
            ).to(DEVICE)
            
            input_length = inputs['input_ids'].shape[1]
            print(f"🔍 Tokenized input: {input_length} tokens")

            temperature=0.7  # Use default temperature for intent classification, can be adjusted if needed
            # Generate response with LFM2.5 recommended parameters
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS_HARD_LIMIT,
                    do_sample=temperature > 0,
                    temperature=max(0.1, min(temperature, 1.0)),
                    top_p=0.1,  # LFM2.5 default
                    top_k=50,   # LFM2.5 default
                    repetition_penalty=1.05,  # LFM2.5 default
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache
                )
            
            # Decode only the generated tokens (skip the input)
            prompt_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                output_ids[0][prompt_len:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"✗ Model generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def generate_with_reasoning_model(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using reasoning - uses same model with same parameters."""
        if not self.model_available or self.model is None:
            raise RuntimeError("Model not available - was not initialized")

        try:
            # Use provided system prompt or default
            if system_prompt is None:
                system_prompt = """You are AutoBot, a helpful AI assistant powered by advanced AI language models.

**Available Capabilities:**
1. **Web Search**: Research and gather information from the internet
2. **General Conversation**: Answer questions and have discussions

Provide detailed, thoughtful, accurate and helpful responses."""

            # Format as chat message for LFM2.5 ChatML template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template to format the messages
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,  
                tokenize=False,
                add_generation_prompt=True,
            )
            
            print(f"🔍 Formatted prompt length: {len(formatted_prompt)} characters")
            print(f"🔍 Sample formatted prompt: {formatted_prompt[:200]}...")

            
            # Tokenize the formatted prompt
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT_LENGTH ,
            ).to(DEVICE)
            
            input_length = inputs['input_ids'].shape[1]
            print(f"🔍 Tokenized input: {input_length} tokens")

            temperature=0.7  # Use default temperature for intent classification, can be adjusted if needed
            # Generate response with LFM2.5 recommended parameters
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS_HARD_LIMIT,
                    do_sample=temperature > 0,
                    temperature=max(0.1, min(temperature, 1.0)),
                    top_p=0.1,  # LFM2.5 default
                    top_k=50,   # LFM2.5 default
                    repetition_penalty=1.05,  # LFM2.5 default
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache
                )
            
            # Decode only the generated tokens (skip the input)
            prompt_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                output_ids[0][prompt_len:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"✗ Model generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def generate_response(self, prompt: str, context: Optional[Dict] = None, use_reasoning: bool = True) -> str:
        """Generate a response using the LLM."""
        formatted_prompt = self._format_prompt(prompt, context)
        if use_reasoning:
            return await self.generate_with_reasoning_model(formatted_prompt)
        else:
            return await self.generate_with_intent_model(formatted_prompt)

    def _format_prompt(self, user_input: str, context: Optional[Dict]) -> str:
        """Format the prompt for the LLM."""
        # For LFM2.5, we use the chat template which handles system prompt via apply_chat_template
        # Here we just return the user input as-is
        return user_input
