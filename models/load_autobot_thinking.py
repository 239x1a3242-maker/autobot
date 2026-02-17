"""
Autobot Thinking Model Loader
Handles loading and initialization of the Autobot thinking model with optimized settings
"""

import os
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_NAME = "./models/autobot-thinking"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_LENGTH = 32768

# Global model variables
autobot_tokenizer = None
autobot_model = None


def get_device_info() -> dict:
    """Get device and environment information"""
    return {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
    }


async def load_autobot_thinking_model():
    """
    Load and initialize the Autobot Thinking model with optimized settings
    
    Returns:
        tuple: (tokenizer, model, success_bool)
    """
    global autobot_tokenizer, autobot_model
    
    print("\n" + "="*60)
    print("🚀 AUTOBOT THINKING MODEL LOADER")
    print("="*60)
    
    # Print device info
    device_info = get_device_info()
    print(f"\n📊 DEVICE INFORMATION:")
    print(f"  • Device: {device_info['device'].upper()}")
    print(f"  • CUDA Available: {device_info['cuda_available']}")
    print(f"  • PyTorch Version: {device_info['torch_version']}")
    if device_info['cuda_version'] != "N/A":
        print(f"  • CUDA Version: {device_info['cuda_version']}")
    
    autobot_success = False
    
    try:
        print(f"\n📥 LOADING MODEL:")
        print(f"  • Model Name: Autobot Thinking")
        print(f"  • Model Path: {MODEL_NAME}")
        print(f"  • Context Length: {MAX_CONTEXT_LENGTH} tokens")
        
        # ====== LOAD TOKENIZER ======
        print(f"\n🔹 Step 1: Loading Tokenizer...")
        autobot_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side="left",  # Essential for streaming
            trust_remote_code=False,
            use_fast=True
        )
        print(f"  ✅ Tokenizer loaded successfully")
        print(f"     • Vocab Size: {autobot_tokenizer.vocab_size}")
        print(f"     • Chat Template Available: {autobot_tokenizer.chat_template is not None}")
        print(f"     • Padding Side: left")
        
        # ====== LOAD MODEL ======
        print(f"\n🔹 Step 2: Loading Model Weights...")
        
        # Configure model loading based on available hardware
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            "trust_remote_code": False,
            "device_map": "auto" if DEVICE == "cuda" else None,
        }
        
        print(f"     • Data Type: {load_kwargs['torch_dtype']}")
        print(f"     • Device Map: {load_kwargs['device_map']}")
        
        autobot_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **load_kwargs
        )
        print(f"  ✅ Model weights loaded successfully")
        
        # ====== CONFIGURE MODEL ======
        print(f"\n🔹 Step 3: Configuring Model...")
        
        # If device_map not used, manually move to device
        if DEVICE != "cuda" or load_kwargs.get("device_map") is None:
            print(f"     • Moving model to {DEVICE}...")
            autobot_model = autobot_model.to(DEVICE)
        
        # Set to evaluation mode
        autobot_model.eval()
        print(f"  ✅ Model set to evaluation mode")
        
        # Enable inference optimizations
        autobot_model.config.use_cache = True
        print(f"  ✅ KV cache enabled for faster generation")
        
        # ====== MODEL SUMMARY ======
        print(f"\n📋 MODEL SUMMARY:")
        print(f"  • Model Name: Autobot Thinking")
        print(f"  • Model Size: ~1.2B parameters")
        print(f"  • Device: {autobot_model.device}")
        print(f"  • Data Type: {autobot_model.dtype}")
        print(f"  • Context Window: {MAX_CONTEXT_LENGTH} tokens")
        print(f"  • Inference Optimization: KV Cache enabled")
        
        autobot_success = True
        print(f"\n✅ MODEL LOADED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n❌ FAILED TO LOAD MODEL: {e}")
        print(f"\n📋 ERROR DETAILS:")
        traceback.print_exc()
        
        # ====== FALLBACK ATTEMPT ======
        try:
            print(f"\n🔄 ATTEMPTING FALLBACK LOAD METHOD...")
            print(f"  • Using simpler configuration...")
            
            autobot_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=False
            )
            print(f"  ✅ Tokenizer loaded (fallback)")
            
            autobot_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32
            )
            autobot_model.to(DEVICE).eval()
            print(f"  ✅ Model loaded (fallback method)")
            print(f"  ✅ FALLBACK LOAD SUCCESSFUL")
            
            autobot_success = True
            
        except Exception as e2:
            print(f"\n❌ FALLBACK LOAD FAILED: {e2}")
            print(f"\n📋 FALLBACK ERROR DETAILS:")
            traceback.print_exc()
            autobot_tokenizer = None
            autobot_model = None
            autobot_success = False
    
    # ====== FINAL STATUS ======
    print(f"\n" + "="*60)
    print(f"📊 LOADING STATUS:")
    print(f"  • Tokenizer Loaded: {'✅ YES' if autobot_tokenizer else '❌ NO'}")
    print(f"  • Model Loaded: {'✅ YES' if autobot_model else '❌ NO'}")
    print(f"  • Success: {'✅ YES' if autobot_success else '❌ NO'}")
    print("="*60 + "\n")
    
    return autobot_tokenizer, autobot_model, autobot_success


def get_model():
    """Get loaded model"""
    return autobot_model


def get_tokenizer():
    """Get loaded tokenizer"""
    return autobot_tokenizer


def is_model_loaded():
    """Check if model is loaded"""
    return autobot_model is not None and autobot_tokenizer is not None
