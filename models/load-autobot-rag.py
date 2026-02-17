"""
Model loader for autobot-rag.
"""

from typing import Any, Optional, Tuple
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_autobot_rag(model_path: str, device: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load tokenizer and model from local autobot-rag directory.
    Returns (tokenizer, model). If loading fails, returns (None, None).
    """
    print("[load-autobot-rag] ==========================================")
    print("[load-autobot-rag] Starting autobot-rag load")
    print(f"[load-autobot-rag] model_path={model_path}")
    print(f"[load-autobot-rag] target_device={device}")

    tokenizer = None
    model = None

    try:
        print("[load-autobot-rag] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=False,
            use_fast=True,
        )
        print(f"[load-autobot-rag] Tokenizer ready (vocab={tokenizer.vocab_size})")
        print(
            f"[load-autobot-rag] Chat template available={tokenizer.chat_template is not None}"
        )

        print("[load-autobot-rag] Loading model weights...")
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "trust_remote_code": False,
            "device_map": "auto" if device == "cuda" else None,
        }

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if device != "cuda" or load_kwargs.get("device_map") is None:
            print(f"[load-autobot-rag] Moving model to {device}...")
            model = model.to(device)

        model.eval()
        model.config.use_cache = True

        print("[load-autobot-rag] Model load complete")
        print(f"[load-autobot-rag] model_device={model.device}")
        print(f"[load-autobot-rag] model_dtype={model.dtype}")
        print("[load-autobot-rag] ==========================================")
        return tokenizer, model

    except Exception as primary_error:
        print(f"[load-autobot-rag] Primary loading failed: {primary_error}")
        traceback.print_exc()

        # Fallback load path
        try:
            print("[load-autobot-rag] Trying fallback loading mode...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
            model = model.to(device)
            model.eval()
            model.config.use_cache = True
            print("[load-autobot-rag] Fallback load succeeded")
            print("[load-autobot-rag] ==========================================")
            return tokenizer, model
        except Exception as fallback_error:
            print(f"[load-autobot-rag] Fallback loading failed: {fallback_error}")
            traceback.print_exc()
            print("[load-autobot-rag] Returning empty model state")
            print("[load-autobot-rag] ==========================================")
            return None, None

