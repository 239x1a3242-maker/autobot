# Autobot Thinking - Complete Usage Guide

## Overview

**Autobot Thinking** is a lightweight, high-performance reasoning model. This model excels at on-device deployment with advanced reasoning capabilities while maintaining an extremely small footprint (~1.2B parameters).

The local implementation uses two core modules:
- **`load_autobot_thinking.py`** - Handles model & tokenizer loading
- **`generate_autobot_thinking.py`** - Generates responses with thinking separation

---

## Model Specifications

### Key Details

| Property | Value |
|----------|-------|
| **Model Name** | Autobot Thinking |
| **Parameters** | 1.17 Billion |
| **Architecture** | 16 layers (10 double-gated LIV conv blocks + 6 GQA blocks) |
| **Context Length** | 32,768 tokens |
| **Vocabulary** | 65,536 tokens |
| **Training Data** | 28 Trillion tokens |

### Why This Model?

✅ **Best-in-class performance** - Rivals much larger models (1.2B rivaling 7B+)  
✅ **Edge deployment** - Runs under 1GB of memory, perfect for local deployment  
✅ **Fast inference** - 239 tok/s on AMD CPU, 82 tok/s on mobile NPU  
✅ **Extended training** - 28T tokens pre-training + reinforcement learning  
✅ **Reasoning capability** - Separates thinking from final answer  
✅ **Tool use support** - Built-in function calling capability  

### Architecture Advantages

- **Hybrid approach** - Combines Autobot Thinking architecture with extended pre-training
- **Efficient design** - Double-gated LIV convolution blocks for better performance
- **Group Query Attention** - Reduces memory footprint while maintaining quality
- **Long context** - 32K token window supports complex reasoning tasks

---

## Performance Benchmarks

### Model Comparison

| Benchmark | Autobot Thinking | Qwen3-1.7B (thinking) | Autobot Thinking-Instruct | Llama 3.2 1B |
|-----------|-------|-------|-------|-------|
| **GPQA** | 37.86 | 36.93 | 38.89 | 16.57 |
| **MMLU-Pro** | 49.65 | 56.68 | 44.35 | 20.80 |
| **IFBench** | 88.42 | 71.65 | 86.23 | 52.37 |
| **AIME25** | 44.85 | 25.88 | 47.33 | 15.93 |

**Winner:** Autobot Thinking outperforms comparable models in most reasoning benchmarks.

### Inference Speed

| Device | Framework | Throughput | Memory | Context |
|--------|-----------|-----------|--------|---------|
| **AMD Ryzen AI 395+** | FastFlowLM (NPU) | 60 tok/s | 1600MB | Full 32K |
| **AMD Ryzen AI 9 HX 370** | llama.cpp (Q4_0) | 116 tok/s | 856MB | Full 32K |
| **Qualcomm Snapdragon X Elite** | NexaML (NPU) | 63 tok/s | 0.9GB | Full context |
| **Mobile (Galaxy S25 Ultra)** | llama.cpp (Q4_0) | 70 tok/s | 719MB | Full context |

**Key Insight:** Sustains ~52 tok/s at 16K context and ~46 tok/s even at full 32K context.

---

## Getting Started

### Prerequisites

```bash
pip install torch transformers safetensors
```

### Directory Structure

```
models/
├── autobot-thinking/              # Model weights directory
│   ├── model.safetensors          # Model weights
│   ├── config.json                # Model configuration
│   ├── generation_config.json     # Generation settings
│   ├── tokenizer.json             # Tokenizer vocabulary
│   ├── tokenizer_config.json      # Tokenizer configuration
│   ├── special_tokens_map.json    # Special tokens mapping
│   └── chat_template.jinja        # Chat template for formatting
├── load_autobot_thinking.py       # Model loader module
├── generate_autobot_thinking.py   # Generator module
└── autobot-thinking-use.md        # This file
```

---

## Step 1: Loading the Model

### Using the Loader Module

The `load_autobot_thinking.py` module handles model initialization with automatic device detection:

```python
from load_autobot_thinking import load_autobot_thinking_model

# Load the model and tokenizer
tokenizer, model, success = load_autobot_thinking_model()

if success:
    print("✅ Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
else:
    print("❌ Failed to load model")
```

### What the Loader Does

1. **Detects device** - Checks for CUDA GPU availability
2. **Sets precision** - Uses `bfloat16` for GPU (faster, better precision)
3. **Loads weights** - Loads `model.safetensors` from disk
4. **Loads tokenizer** - Initializes tokenizer with chat template
5. **Validates loading** - Returns success status

### Device Configuration

| Device | Data Type | Memory | Speed |
|--------|-----------|--------|-------|
| **NVIDIA GPU** | bfloat16 | ~2.4GB | Fast |
| **CPU Only** | float32 | ~4.8GB | Slower |

### Advanced: Get Device Info

```python
from load_autobot_thinking import get_device_info, is_model_loaded

# Get detailed device information
info = get_device_info()
print(f"Device: {info['device']}")
print(f"GPU Name: {info.get('device_name', 'N/A')}")
print(f"CUDA Available: {info['cuda_available']}")
print(f"Data Type: {info['using_dtype']}")
print(f"Loaded: {is_model_loaded()}")
```

---

## Step 2: Generating Responses

### Basic Generation

The `generate_autobot_thinking.py` module generates responses with thinking separation:

```python
from load_autobot_thinking import load_autobot_thinking_model
from generate_autobot_thinking import generate_autobot_thinking
import json

# Step 1: Load model
tokenizer, model, success = load_autobot_thinking_model()
if not success:
    print("Failed to load model")
    exit(1)

# Step 2: Prepare prompts
system_message = "You are a helpful AI assistant trained in reasoning and problem-solving."
user_prompt = "What is photosynthesis and why is it important?"

# Step 3: Generate response
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message=system_message,
    user_prompt=user_prompt,
    device="cuda",
    temperature=0.7,
    max_tokens=4500
)

# Step 4: Display results
if result['success']:
    print("\n=== THINKING ===")
    print(result['think'])
    print("\n=== ANSWER ===")
    print(result['text'])
    print(f"\n⏱️ Generated in {result['elapsed']:.2f}s")
    print(f"📊 {result['tokens_per_sec']:.1f} tokens/sec")
else:
    print(f"❌ Generation failed: {result['error']}")
```

### Output Structure

The generator returns a comprehensive dictionary:

```python
{
    # Main outputs
    "think": str,                           # Thinking without special tokens
    "text": str,                            # Final answer (cleaned)
    
    # Raw outputs (with special tokens)
    "raw_think": str,                       # Raw thinking with <think> tags
    "raw_text": str,                        # Raw answer with special tokens
    
    # Metadata
    "template_token_count": int,            # Characters in formatted prompt
    "formatted_prompt": str,                # Complete prompt sent to model
    "input_length": int,                    # Input token count
    "generated_tokens": int,                # Tokens generated
    "elapsed": float,                       # Generation time in seconds
    "tokens_per_sec": float,                # Generation speed
    
    # Status
    "success": bool,                        # True if generation succeeded
    "error": str                            # Error message if failed
}
```

### Generation Parameters

```python
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="You are a helpful assistant.",
    user_prompt="Your question here",
    
    # Device configuration
    device="cuda",  # or "cpu"
    
    # Context settings
    max_context_length=32768,     # Model's total context window
    
    # Generation limits
    max_tokens=4500,              # Target generation length
    max_tokens_hard_limit=24000,  # Never exceed this
    
    # Sampling
    temperature=0.7,              # 0.1=deterministic, 1.0=random
    
    # Tool integration (optional)
    tools_json=None  # JSON string of available tools
)
```

### Temperature Guide

| Temperature | Behavior | Use Case |
|---|---|---|
| **0.1** | Deterministic, focused | Accurate answers, code |
| **0.3-0.5** | Focused with variety | Balanced responses |
| **0.7** | Balanced (default) | General purpose |
| **0.9-1.0** | Creative, unpredictable | Creative writing |

---

## Complete Example: Q&A System

```python
#!/usr/bin/env python3
"""
Complete example: Q&A system with thinking and reasoning
"""

from load_autobot_thinking import load_autobot_thinking_model
from generate_autobot_thinking import generate_autobot_thinking
import json

def run_qa_system():
    print("🚀 Loading Autobot Thinking model...")
    tokenizer, model, success = load_autobot_thinking_model()
    
    if not success:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    
    # System personality
    system_message = """You are Autobot Thinking, an advanced AI assistant specializing in:
- Complex reasoning and problem-solving
- Data analysis and synthesis
- Technical explanations
- Step-by-step thinking

Always show your reasoning process clearly."""
    
    questions = [
        "Explain quantum computing and its applications",
        "What are the key differences between machine learning and deep learning?",
        "How does blockchain technology work?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)
        
        result = generate_autobot_thinking(
            model=model,
            tokenizer=tokenizer,
            system_message=system_message,
            user_prompt=question,
            device="cuda",
            temperature=0.7,
            max_tokens=4500
        )
        
        if result['success']:
            print(f"\n📍 THINKING PROCESS:")
            print("-" * 40)
            print(result['think'][:500] + "..." if len(result['think']) > 500 else result['think'])
            
            print(f"\n✨ FINAL ANSWER:")
            print("-" * 40)
            print(result['text'])
            
            print(f"\n📊 STATISTICS:")
            print(f"   ⏱️  Time: {result['elapsed']:.2f}s")
            print(f"   🔤 Speed: {result['tokens_per_sec']:.1f} tok/s")
            print(f"   📈 Tokens: {result['generated_tokens']}")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    run_qa_system()
```

---

## Tool Use & Function Calling

### What is Tool Use?

Tool use (function calling) allows the AI to call external functions to answer questions:

```
User: "What is the status of candidate ID 12345?"
Model: Thinks → Calls get_candidate_status(candidate_id="12345")
System: Returns candidate info
Model: Interprets result → Provides final answer
```

### Defining Tools

```python
tools_json = json.dumps([
    {
        "name": "web_search",
        "description": "Search the web for current information",
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
    },
    {
        "name": "get_candidate_status",
        "description": "Get recruitment status for a candidate",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "Unique candidate identifier"
                }
            },
            "required": ["candidate_id"]
        }
    }
])
```

### Using Tools in Generation

```python
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="You are a helpful assistant with access to tools.",
    user_prompt="What is the status of candidate ID 12345?",
    device="cuda",
    tools_json=tools_json  # Pass the tools
)

# Model response will include tool calls in special tokens
print(result['raw_text'])  # Contains <|tool_call_start|> ... <|tool_call_end|>
```

### Tool Response Format

The model outputs tool calls like:

```
Checking the current status of candidate ID 12345.
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>
```

After tool execution, provide result as:

```python
tool_result = {"candidate_id": "12345", "status": "Interview Scheduled", "date": "2026-02-20"}

# Continue with tool result
user_prompt_with_result = f"""
Tool result: {json.dumps(tool_result)}

What is the current status based on this result?
"""
```

---

## Advanced: Tool Use Workflow

```python
import json
import re

def extract_tool_calls(response_text):
    """Extract tool calls from model response"""
    pattern = r'<\|tool_call_start\|>\[(.*?)\]<\|tool_call_end\|>'
    matches = re.findall(pattern, response_text)
    return matches

def process_with_tools(model, tokenizer, user_prompt, tools_json):
    """Full tool use workflow"""
    
    system_message = "You are an AI assistant with access to tools. Use tools when needed."
    
    # Initial generation
    result = generate_autobot_thinking(
        model=model,
        tokenizer=tokenizer,
        system_message=system_message,
        user_prompt=user_prompt,
        device="cuda",
        tools_json=tools_json
    )
    
    if not result['success']:
        return result
    
    # Extract tool calls
    tool_calls = extract_tool_calls(result['raw_text'])
    
    if tool_calls:
        print(f"🔧 Found {len(tool_calls)} tool calls")
        
        # In production: execute tools and collect results
        # For this example, we just show the calls
        for call in tool_calls:
            print(f"   → {call[:100]}...")
    
    return result
```

---

## Real-World Examples

### Example 1: Customer Support Analysis

```python
system = """You are a customer support AI. Analyze customer issues and provide solutions.
Show your reasoning process and think through problems carefully."""

query = """
A customer reports: "My app crashes when uploading files larger than 100MB"
What are the most likely causes and solutions?
"""

result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message=system,
    user_prompt=query,
    temperature=0.6  # More focused
)

print("THINKING:", result['think'][:300])
print("SOLUTION:", result['text'])
```

### Example 2: Code Review

```python
system = """You are an expert code reviewer. Analyze code for:
- Performance issues
- Security vulnerabilities
- Best practices
- Maintainability

Think through each aspect carefully."""

code = '''
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
'''

query = f"Review this Python code:\n\n{code}"

result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message=system,
    user_prompt=query,
    temperature=0.5
)

print(result['text'])
```

### Example 3: Data Analysis

```python
system = """You are a data analyst. Interpret data and provide insights.
Consider:
- Trends and patterns
- Anomalies
- Relationships between variables
- Recommendations"""

data = """
Monthly Sales Data:
Jan: $45,000
Feb: $42,000
Mar: $48,000
Apr: $51,000
May: $49,000
Jun: $55,000
"""

query = f"Analyze this data and provide insights:\n\n{data}"

result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message=system,
    user_prompt=query,
    temperature=0.7
)

print(result['text'])
```

---

## Best Practices

### Do's ✅

- ✅ **Load model once** - Reuse model instance across requests
- ✅ **Use appropriate temperature** - Lower for factual, higher for creative
- ✅ **Batch requests** - Process multiple prompts efficiently
- ✅ **Monitor tokens** - Keep track of generated tokens
- ✅ **Clear system messages** - Be specific about expected behavior
- ✅ **Handle errors gracefully** - Check `success` status always
- ✅ **Use tools strategically** - Only when external data is needed

### Don'ts ❌

- ❌ **Don't reload model repeatedly** - Creates unnecessary overhead
- ❌ **Don't set max_tokens > max_tokens_hard_limit** - Will cause errors
- ❌ **Don't use extreme temperatures** - 0.0 or 1.5+ cause issues
- ❌ **Don't ignore error messages** - They indicate real problems
- ❌ **Don't send extremely long prompts** - May hit memory limits
- ❌ **Don't use GPU memory foolishly** - Monitor VRAM usage

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory (OOM)

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Reduce `max_tokens` parameter
- Use `device="cpu"` for CPU inference
- Clear GPU memory before loading: `torch.cuda.empty_cache()`
- Quantize model to reduce memory

```python
# CPU fallback
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="...",
    user_prompt="...",
    device="cpu",  # Use CPU
    max_tokens=2000  # Reduce tokens
)
```

### Issue 2: Slow Generation

**Symptom:** Generation takes > 30 seconds

**Solutions:**
- Ensure `device="cuda"` (not CPU)
- Reduce `max_tokens`
- Use lower context length if possible
- Check system load

```python
# Monitor speed
print(f"Speed: {result['tokens_per_sec']:.1f} tok/s")
# CPU: 5-10 tok/s
# GPU: 50-150 tok/s
```

### Issue 3: Repeated or Nonsensical Output

**Symptom:** Model generates garbage or repeats text

**Solutions:**
- Increase `temperature` (too low = repetitive)
- Provide clearer system message
- Reduce `max_tokens`
- Try different random seed

```python
# Better prompting
system = "Be clear, concise, and accurate in your response."
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message=system,
    user_prompt=query,
    temperature=0.8  # Increase if too low
)
```

---

## Integration Example: FastAPI Web Server

```python
from fastapi import FastAPI
from pydantic import BaseModel
from load_autobot_thinking import load_autobot_thinking_model
from generate_autobot_thinking import generate_autobot_thinking

app = FastAPI()

# Load model at startup
tokenizer, model, _ = load_autobot_thinking_model()

class QueryRequest(BaseModel):
    prompt: str
    system: str = "You are a helpful assistant"
    temperature: float = 0.7

class QueryResponse(BaseModel):
    thinking: str
    answer: str
    tokens: int
    speed: float

@app.post("/api/analyze")
async def analyze(request: QueryRequest):
    result = generate_autobot_thinking(
        model=model,
        tokenizer=tokenizer,
        system_message=request.system,
        user_prompt=request.prompt,
        device="cuda",
        temperature=request.temperature
    )
    
    if result['success']:
        return QueryResponse(
            thinking=result['think'][:200] + "..." if len(result['think']) > 200 else result['think'],
            answer=result['text'],
            tokens=result['generated_tokens'],
            speed=result['tokens_per_sec']
        )
    else:
        raise HTTPException(status_code=500, detail=result['error'])
```

---

## Performance Optimization

### Memory Usage

| Configuration | Memory | Speed |
|---|---|---|
| **bfloat16 + CUDA** | ~2.4GB | Very fast (100+ tok/s) |
| **float32 + CUDA** | ~4.8GB | Fast |
| **float32 + CPU** | ~4.8GB | Slow (5-10 tok/s) |
| **Quantized (Q4)** | ~700MB | Medium (40-60 tok/s) |

### Optimization Tips

1. **Use bfloat16** - Balances speed and memory
2. **Batch inference** - Process multiple requests together
3. **Cache embeddings** - For repeated prompts
4. **Use quantization** - For resource-constrained devices
5. **Monitor GPU memory** - Use `nvidia-smi`

---

## Deployment Scenarios

### Scenario 1: Local Desktop Application
- Device: Laptop with NVIDIA GPU
- Config: bfloat16, full model, batch size 1
- Speed: ~80-120 tok/s

### Scenario 2: Mobile/Edge Device
- Device: Qualcomm Snapdragon, NPU
- Config: Quantized (Q4), optimized for edge
- Speed: ~60-80 tok/s

### Scenario 3: Server Deployment
- Device: NVIDIA H100 GPU
- Config: bfloat16, batched requests
- Speed: ~500+ tok/s (with batching)

### Scenario 4: CPU-Only Deployment
- Device: AMD/Intel CPU
- Config: float32, optimized with llama.cpp
- Speed: ~50-100 tok/s depending on CPU

---

## Citation & References

### How to Cite

```bibtex
@article{autobot2026thinking,
  author = {Autobot},
  title = {Autobot Thinking: On-Device Reasoning Under 1GB},
  journal = {Autobot Blog},
  year = {2026},
  note = {www.autobot.ai/blog/autobot-thinking-on-device-reasoning-under-1gb}
}

@article{autobot2025thinking,
  title={Autobot Thinking Technical Report},
  author={Autobot},
  journal={arXiv preprint arXiv:2511.23404},
  year={2025}
}
```

### Additional Resources

- **HuggingFace Model Card**: https://huggingface.co/Autobot/Autobot-Thinking
- **Documentation**: https://docs.autobot.ai/thinking
- **Autobot AI Blog**: https://www.autobot.ai/blog
- **Technical Paper**: https://arxiv.org/abs/2511.23404

---

## Module Reference

### load_autobot_thinking.py

**Purpose:** Load and initialize the model with device detection

**Key Functions:**
```python
# Load the model
tokenizer, model, success = load_autobot_thinking_model()

# Get current model
model = get_model()

# Get current tokenizer
tokenizer = get_tokenizer()

# Check if loaded
is_loaded = is_model_loaded()

# Get device info
info = get_device_info()
```

### generate_autobot_thinking.py

**Purpose:** Generate responses with thinking separation

**Key Function:**
```python
result = generate_autobot_thinking(
    model=model,
    tokenizer=tokenizer,
    system_message="...",
    user_prompt="...",
    temperature=0.7,
    max_tokens=4500
)
```

---

## Quick Start Checklist

- [ ] Install dependencies: `pip install torch transformers safetensors`
- [ ] Place model in `models/autobot-thinking/` directory
- [ ] Copy `load_autobot_thinking.py` to `models/`
- [ ] Copy `generate_autobot_thinking.py` to `models/`
- [ ] Load model: `tokenizer, model, success = load_autobot_thinking_model()`
- [ ] Generate response: `result = generate_autobot_thinking(...)`
- [ ] Monitor performance: Check `result['tokens_per_sec']`
- [ ] Deploy and scale as needed

---

## Support & Questions

For issues or questions:
- Check HuggingFace Model Card: https://huggingface.co/Autobot/Autobot-Thinking
- Visit Autobot: https://autobot.ai/
- Enterprise support: sales@autobot.ai

---

**Last Updated:** February 16, 2026  
**Model Version:** Autobot Thinking  
**Documentation Version:** 1.0
