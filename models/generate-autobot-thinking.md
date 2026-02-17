# Generate Autobot Thinking - Documentation

## Overview

The `generate_autobot_thinking.py` module is responsible for **generating text responses from the Autobot Thinking model** given a prompt. It handles the complete generation pipeline including formatting, tokenization, generation config, thinking/answer separation, and token streaming.

---

## Purpose

- **Pure generation:** Convert system message + user prompt → thinking text + answer text
- **Synchronous:** Non-async, returns complete dict
- **Thinking separation:** Separates model's internal thinking from final answer
- **Tool awareness:** Supports tools in generation for orchestration scenarios
- **Error handling:** Comprehensive error reporting with detailed logging

---

## File Location

### Directory Structure
```
models/
├── autobot-thinking/              # Model weights directory
│   ├── model.safetensors
│   ├── config.json
│   └── (other model files)
└── generate_autobot_thinking.py   # ← This file
```

### This Module Only
**This documentation covers ONLY `generate_autobot_thinking.py`**
- Standalone module for response generation
- Takes loaded model + tokenizer → returns response dict
- Independent pure generation logic

---

## API Reference

### 1. `generate_autobot_thinking(model, tokenizer, ...) → Dict[str, Any]`

**Purpose:** Generate a complete response from the model (synchronous)

**Function Signature:**

```python
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
```

**Input Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | Any | Required | Loaded Autobot Thinking model instance |
| `tokenizer` | Any | Required | Loaded tokenizer instance |
| `system_message` | str | Required | System prompt (instructions) |
| `user_prompt` | str | Required | User query/message |
| `device` | str | "cuda" | "cuda" for GPU or "cpu" for CPU |
| `max_context_length` | int | 32768 | Max context window (don't change) |
| `max_tokens` | int | 4500 | Target generation length |
| `max_tokens_hard_limit` | int | 24000 | Hard limit (don't exceed) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-1.0) |
| `tools_json` | str | None | JSON string of available tools |

**Output Dictionary:**

```python
{
    "think": str,                           # Cleaning thinking (no special tokens)
    "raw_think": str,                       # Raw thinking (with <think> tags)
    "text": str,                            # Final answer (cleaned)
    "raw_text": str,                        # Raw answer (with special tokens)
    "template_token_count": int,            # Characters in formatted prompt
    "formatted_prompt": str,                # Full prompt sent to model
    "input_length": int,                    # Input token count
    "generated_tokens": int,                # Tokens generated
    "elapsed": float,                       # Generation time in seconds
    "tokens_per_sec": float,                # Generation speed
    "success": bool,                        # True if generation succeeded
    "error": str                            # Error message if success=False
}
```

---

## Complete Usage Example

### Basic Usage

```python
from models.load_autobot_thinking import load_autobot_thinking_model
from models.generate_autobot_thinking import generate_autobot_thinking
import torch

# Step 1: Load models
tokenizer, model, success = load_autobot_thinking_model()
if not success:
    print("Failed to load models")
    exit()

# Step 2: Prepare prompts
system_message = "You are a helpful AI assistant."
user_prompt = "What is quantum computing?"

# Step 3: Generate response
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message=system_message,
    user_prompt=user_prompt,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_tokens=2000,
    temperature=0.7
)

# Step 4: Check result
if result["success"]:
    print("🎯 Thinking Process:")
    print(result["think"])
    print("\n📝 Final Answer:")
    print(result["text"])
    print(f"\n⏱️ Generated {result['generated_tokens']} tokens in {result['elapsed']}s")
else:
    print(f"❌ Error: {result['error']}")
```

---

### Advanced Usage with Tools

```python
import json
from models.generate_autobot_thinking import generate_autobot_thinking

# Define available tools
tools = [
    {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
]

# Generate with tools
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="You are an AI with access to web search.",
    user_prompt="What are the latest AI trends?",
    device="cuda",
    max_tokens=3000,
    temperature=0.5,
    tools_json=json.dumps(tools)  # Pass tools
)

# Check if thinking mentions tools
if "web_search" in result["raw_text"]:
    print("Model decided to use web_search tool")
```

---

### Standalone Script Example

```python
#!/usr/bin/env python3
# models/test_generator.py

import torch
from load_autobot_thinking import load_autobot_thinking_model
from generate_autobot_thinking import generate_autobot_thinking

# Step 1: Load models
print("🔄 Loading models...")
tokenizer, model, success = load_autobot_thinking_model()

if not success:
    print("❌ Failed to load models")
    exit(1)

print("✅ Models loaded!")

# Step 2: Generate response
print("\n🎯 Generating response...")
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="You are a helpful AI assistant.",
    user_prompt="Explain quantum computing in 100 words.",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_tokens=2000,
    temperature=0.7
)

# Step 3: Display results
if result["success"]:
    print("\n✅ Generation successful!\n")
    print("💭 Thinking:")
    print("-" * 50)
    print(result["think"])
    print("\n📝 Answer:")
    print("-" * 50)
    print(result["text"])
    print("\n⏱️ Metrics:")
    print(f"   Tokens: {result['generated_tokens']}")
    print(f"   Time: {result['elapsed']}s")
    print(f"   Speed: {result['tokens_per_sec']} t/s")
else:
    print(f"\n❌ Generation failed: {result['error']}")
```

### Direct Function Call

```python
# Minimal direct usage
from load_autobot_thinking import load_autobot_thinking_model
from generate_autobot_thinking import generate_autobot_thinking
import torch

tokenizer, model, _ = load_autobot_thinking_model()

result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="You are helpful.",
    user_prompt="Say hello?",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print(result["text"] if result["success"] else result["error"])
```

---

## Internal Working

### Step 1: Validation

```python
if not model or not tokenizer:
    return error_dict("Model or tokenizer not available.")
```

- Checks if models are loaded
- Returns error dict if validation fails

### Step 2: Build Messages

```python
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]
```

- Creates ChatML format messages
- Follows Autobot Thinking chat template

### Step 3: Apply Chat Template

```python
if tools_json:
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tools=json.loads(tools_json),
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
```

- Converts messages to model's expected format
- Adds tool information if provided
- Adds generation prompt special token

**Example output:**
```
<|im_start|>system
You are a helpful AI assistant.
<|im_end|>
<|im_start|>user
What is quantum computing?
<|im_end|>
<|im_start|>assistant

```

### Step 4: Tokenization

```python
inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=max_context_length - max_tokens,  # Reserve space for generation
).to(device)
```

- Converts text to token IDs
- Truncates if too long
- Moves to device (GPU/CPU)
- Reserves space for generation

### Step 5: Generation Configuration

```python
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

# Adjust for temperature
if temperature <= 0.1:
    generation_config["do_sample"] = False
    generation_config.pop("top_p", None)
    generation_config.pop("top_k", None)
else:
    generation_config["temperature"] = max(0.1, min(temperature, 1.0))
```

**Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_new_tokens` | 4500 | Tokens to generate |
| `do_sample` | True | Use sampling vs greedy |
| `top_p` | 0.1 | Nucleus sampling cutoff |
| `top_k` | 50 | Top-k sampling |
| `repetition_penalty` | 1.05 | Penalize repeated tokens |
| `use_cache` | True | Use KV cache for speed |

### Step 6: Generation Thread

```python
thread = Thread(target=model.generate, kwargs=generation_config)
thread.start()
```

- Starts generation in background thread
- Allows non-blocking token streaming

### Step 7: Token Streaming

```python
raw_think = ""
raw_text = ""
in_thinking_block = False

for new_text in streamer:
    if "<think>" in new_text:
        in_thinking_block = True
        new_text = new_text.replace("<think>", "")
    
    if "</think>" in new_text:
        in_thinking_block = False
        new_text = new_text.replace("</think>", "")
    
    if in_thinking_block:
        raw_think += new_text
    else:
        raw_text += new_text
```

**Process:**
1. Collects tokens from streamer
2. Tracks if inside `<think>` block
3. Separates thinking from final answer
4. Accumulates raw text

**Example thinking block:**
```
<think>
Let me think about quantum computing...
It uses qubits instead of bits...
Qubits can be both 0 and 1...
</think>

Quantum computing is...
```

### Step 8: Token Cleaning

```python
think = strip_special_tokens(raw_think)
text = strip_special_tokens(raw_text)
```

Removes special tokens:
- `<|im_start|>`, `<|im_end|>`
- `<|tool_call_start|>`, `<|tool_call_end|>`
- `<think>`, `</think>`

### Step 9: Return Result

```python
return {
    "think": think,
    "raw_think": raw_think,
    "text": text,
    "raw_text": raw_text,
    "template_token_count": template_token_count,
    "formatted_prompt": formatted_prompt,
    "input_length": input_length,
    "generated_tokens": token_count,
    "elapsed": elapsed,
    "tokens_per_sec": tokens_per_sec,
    "success": True,
    "error": ""
}
```

---

## Helper Functions

### 2. `strip_special_tokens(text: str) → str`

**Purpose:** Remove special tokens from text

**Special Tokens Removed:**
- `<|im_end|>`, `<|im_start|>`
- `<|endoftext|>`, `<|startoftext|>`
- `<|tool_call_start|>`, `<|tool_call_end|>`
- `<|tool_list_start|>`, `<|tool_list_end|>`
- `<think>`, `</think>`

**Usage Example:**

```python
from models.generate_autobot_thinking import strip_special_tokens

raw_text = "<|im_start|>This is <think>a test</think><|im_end|>"
cleaned = strip_special_tokens(raw_text)
print(cleaned)  # Output: This is a test
```

---

## Error Handling

### CUDA Out of Memory

```python
except torch.cuda.OutOfMemoryError:
    return {
        "success": False,
        "error": "CUDA out of memory. Try reducing max_tokens or using CPU.",
        # ... other fields empty
    }
```

**Solutions:**
- Reduce `max_tokens` (try 2000 instead of 4500)
- Use CPU: pass `device="cpu"`
- Clear GPU cache: `torch.cuda.empty_cache()`

### General Exceptions

```python
except Exception as e:
    return {
        "success": False,
        "error": f"Generation error: {str(e)}",
        # ... other fields empty
    }
```

---

## Generation Parameters Guide

### Temperature

**What it does:** Controls randomness of generation

```
temperature = 0.0   → Greedy (deterministic, repetitive)
temperature = 0.5   → Balanced (creative but coherent)
temperature = 0.7   → Default (creative and natural)
temperature = 1.0   → Maximum randomness
```

**Recommended values:**
- Q&A: `0.3 - 0.5`
- General: `0.6 - 0.8`
- Creative: `0.8 - 1.0`

### max_tokens

**What it does:** Target length of generation

```
max_tokens = 500     → Short responses
max_tokens = 2000    → Medium responses
max_tokens = 4500    → Long responses (default)
max_tokens = 24000   → Very long (max limit)
```

**Memory impact:** Higher max_tokens = more GPU memory needed

---

## Performance Metrics

### Generation Speed

| Device | max_tokens | Speed | Time |
|--------|-----------|-------|------|
| CUDA (H100) | 2000 | ~70 t/s | 28s |
| CUDA (A100) | 2000 | ~40 t/s | 50s |
| CUDA (RTX 4090) | 2000 | ~30 t/s | 65s |
| CPU | 500 | ~3 t/s | 165s |

### Memory Usage

| Device | max_tokens | Memory |
|--------|-----------|--------|
| CUDA | 2000 | 3.5 GB |
| CUDA | 4500 | 4.2 GB |
| CPU | 1000 | 6 GB RAM |

---

## Output Examples

### Example 1: Successful Generation

```python
{
    "think": "The user is asking about quantum computing. Let me explain the key concepts...",
    "raw_think": "<think>The user is asking about quantum computing...</think>",
    "text": "Quantum computing is a revolutionary computing paradigm that uses quantum bits (qubits)...",
    "raw_text": "Quantum computing is a revolutionary computing paradigm that uses quantum bits (qubits)...",
    "template_token_count": 285,
    "formatted_prompt": "<|im_start|>system\nYou are helpful...",
    "input_length": 45,
    "generated_tokens": 187,
    "elapsed": 5.32,
    "tokens_per_sec": 35.15,
    "success": True,
    "error": ""
}
```

### Example 2: CUDA Out of Memory

```python
{
    "think": "",
    "raw_think": "",
    "text": "",
    "raw_text": "",
    "template_token_count": 0,
    "formatted_prompt": "",
    "input_length": 0,
    "generated_tokens": 0,
    "elapsed": 0.15,
    "tokens_per_sec": 0.0,
    "success": False,
    "error": "CUDA out of memory. Try reducing max_tokens or using CPU."
}
```

### Example 3: Model Not Loaded

```python
{
    "think": "",
    "raw_think": "",
    "text": "",
    "raw_text": "",
    "template_token_count": 0,
    "formatted_prompt": "",
    "input_length": 0,
    "generated_tokens": 0,
    "elapsed": 0.01,
    "tokens_per_sec": 0.0,
    "success": False,
    "error": "Model or tokenizer not available."
}
```

---

## Best Practices

### ✅ DO

1. **Check success flag**
   ```python
   result = generate_autobot_thinking(...)
   if result["success"]:
       use_result(result)
   else:
       handle_error(result["error"])
   ```

2. **Use appropriate max_tokens**
   ```python
   # For short questions
   max_tokens = 1000
   
   # For complex analysis
   max_tokens = 3000
   ```

3. **Handle errors gracefully**
   ```python
   try:
       result = generate_autobot_thinking(...)
   except Exception as e:
       log_error(e)
   ```

4. **Monitor generation speed**
   ```python
   speed = result["tokens_per_sec"]
   if speed < 5:
       log_warning("Slow generation detected")
   ```

### ❌ DON'T

1. **Don't ignore success flag**
   ```python
   # ❌ Bad
   result = generate_autobot_thinking(...)
   print(result["text"])  # Might be empty!
   ```

2. **Don't exceed hard limits**
   ```python
   # ❌ Bad
   max_tokens = 50000  # Will cause OOM
   
   # ✅ Good
   max_tokens = min(user_input, 24000)
   ```

3. **Don't regenerate for same prompt**
   ```python
   # ❌ Bad
   for i in range(3):
       result = generate_autobot_thinking(prompt)
   
   # ✅ Good
   result = generate_autobot_thinking(prompt)
   cache_result(prompt, result)
   ```

---

## Summary

| Aspect | Details |
|--------|-------------|
| **Module Type** | Standalone generator |
| **Location** | `models/generate_autobot_thinking.py` |
| **Main Function** | `generate_autobot_thinking(...)` |
| **Return Type** | `Dict[str, Any]` |
| **Processing** | Synchronous (blocking) |
| **Input** | model, tokenizer, system_message, user_prompt, device, max_tokens, temperature |
| **Output** | think, text, raw_think, raw_text, metrics, success, error |
| **Thinking Support** | ✅ Yes (separated from answer) |
| **Tool Support** | ✅ Yes (via tools_json parameter) |
| **Generation Time** | 20-60 seconds avg |
| **Dependencies** | torch, transformers |
| **Error Handling** | Comprehensive error dict |
| **Memory Required** | 3.5-4.2 GB GPU |

