# load-autobot-rag.py Documentation

## Overview
This module provides functionality to load the autobot-rag language model and its tokenizer from a local directory. It implements a robust loading mechanism with fallback handling to ensure reliable model initialization across different hardware configurations.

## Function: `load_autobot_rag()`

### Purpose
Loads both the tokenizer and model from a specified local autobot-rag directory path. If loading fails with the primary method, it attempts a fallback loading approach before returning None values.

### Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | The file system path to the autobot-rag model directory (e.g., `./autobot-rag/`) |
| `device` | `str` | Target device for model inference: `"cpu"` or `"cuda"` |

### Output

Returns a tuple: `Tuple[Optional[Any], Optional[Any]]`

| Return Value | Type | Description |
|--------------|------|-------------|
| `tokenizer` | `Optional[Any]` | AutoTokenizer instance loaded from model_path, or `None` if loading fails |
| `model` | `Optional[Any]` | AutoModelForCausalLM instance, or `None` if loading fails |

**Success case:** `(AutoTokenizer, AutoModelForCausalLM)`  
**Failure case:** `(None, None)`

### How It Works

#### Phase 1: Primary Loading
1. **Tokenizer Loading**
   - Loads AutoTokenizer from the model_path
   - Configuration: `padding_side="left"`, `trust_remote_code=False`, `use_fast=True`
   - Logs vocabulary size and chat template availability

2. **Model Loading**
   - For CUDA devices: Uses `torch.bfloat16` precision with automatic device mapping
   - For CPU devices: Uses `torch.float32` precision
   - Sets model to evaluation mode (`model.eval()`)
   - Enables KV cache for faster inference (`model.config.use_cache = True`)

#### Phase 2: Device Placement
- If CUDA with device_map: Model automatically distributed across GPUs
- If CPU or CUDA without device_map: Model explicitly moved to target device using `.to(device)`

#### Phase 3: Fallback Loading (on primary failure)
- Simplified loading with default parameters
- Uses `torch.float32` for all devices
- Manual device placement using `.to(device)`
- Falls back to returning `(None, None)` if this also fails

### Logging Output

The function provides detailed console logging with `[load-autobot-rag]` prefix:

- Model path and target device information
- Tokenizer loading status with vocabulary size
- Model weight loading progress
- Device placement confirmation
- Data type (dtype) confirmation
- Error details if loading fails

### Error Handling

1. **Primary Load Failure:**
   - Catches exception and logs error details
   - Prints full traceback
   - Attempts Phase 3 fallback loading

2. **Fallback Load Failure:**
   - Catches secondary exception
   - Prints traceback
   - Returns `(None, None) `

### Dependencies

- `torch`: PyTorch library for tensor operations
- `transformers.AutoTokenizer`: Automatic tokenizer loading
- `transformers.AutoModelForCausalLM`: Automatic model loading for causal language modeling

### Example Usage

```python
from load_autobot_rag import load_autobot_rag

# Load model for GPU processing
tokenizer, model = load_autobot_rag(
    model_path="./autobot-rag/",
    device="cuda"
)

if tokenizer is None or model is None:
    print("Failed to load model")
else:
    print(f"Model loaded successfully: {model.dtype}")

# Load model for CPU processing
tokenizer_cpu, model_cpu = load_autobot_rag(
    model_path="./autobot-rag/",
    device="cpu"
)
```

### Key Configuration Details

| Setting | Primary Mode | Fallback Mode |
|---------|--------------|---------------|
| Precision (CUDA) | `torch.bfloat16` | `torch.float32` |
| Precision (CPU) | `torch.float32` | `torch.float32` |
| Device Mapping | Auto (CUDA) | Manual |
| Trust Remote Code | False | False |
| Use Fast Tokenizer | True | Default |
| Evaluation Mode | Yes | Yes |
| KV Cache Enabled | Yes | Yes |
| Padding Side | Left | Default |

### Important Notes

- The function returns `(None, None)` on any loading failure, allowing graceful degradation in applications
- CUDA device mapping is automatic, enabling better optimization for multi-GPU setups
- CPU loading is slower due to full precision (float32) usage
- The tokenizer's chat template availability is logged for debugging purposes
- All special tokens are properly configured for the model
