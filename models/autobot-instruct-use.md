# autobot-instruct-use.md — Use Cases & Guide

## Overview

This guide shows **how to use the autobot-instruct instruction model** with:
- `load-autobot-instruct.py` — Loading the model
- `generate-autobot-instruct.py` — Generating responses
- Real **input/output examples** for each use case

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [Module 1: load-autobot-instruct.py](#module-1-load-autobot-instructpy)
3. [Module 2: generate-autobot-instruct.py](#module-2-generate-autobot-instructpy)
4. [Use Cases & Examples](#use-cases--examples)
5. [Input/Output Formats](#inputoutput-formats)
6. [Common Patterns](#common-patterns)
7. [Error Handling & Troubleshooting](#error-handling--troubleshooting)

---

## Module Overview

```
load-autobot-instruct.py
  └─ load_autobot_instruct(base_dir, device) → (tokenizer, model, model_dir)
  
generate-autobot-instruct.py
  └─ generate_autobot_instruct(model, tokenizer, ...) → dict with text, metadata
```

**Both modules work together:**
1. **load** → Get model + tokenizer ready
2. **generate** → Create responses from prompts

---

## Module 1: load-autobot-instruct.py

### Function Signature

```python
def load_autobot_instruct(
    base_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """Load tokenizer + model and return (tokenizer, model, resolved_model_dir)."""
```

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dir` | str or None | None | Root path to search for model. If None, uses repo root |
| `device` | str or None | None | Device: `'cuda'` or `'cpu'`. If None, auto-detects CUDA |

### Return Values

```python
(tokenizer, model, model_dir)

tokenizer: AutoTokenizer
  - vocab_size: 32000
  - padding_side: 'left'
  - has chat_template: True
  
model: AutoModelForCausalLM
  - device: cuda:0 or cpu
  - dtype: torch.bfloat16 (CUDA) or torch.float32 (CPU)
  - eval mode: True
  - use_cache: True
  
model_dir: str
  - Absolute path to /path/to/models/autobot-instruct
```

### Model Discovery Process

The loader searches for `config.json` in these locations (in order):
1. `{base_dir}/models/autobot-instruct/config.json`
2. `{base_dir}/models/config.json`

If found → Uses that directory
If not found → Raises `FileNotFoundError`

### Use Cases

#### **Use Case 1: Load with Auto Device Detection**

```python
from models.load_autobot_instruct import load_autobot_instruct
import torch

# Load model (auto-detects CUDA if available)
tokenizer, model, model_dir = load_autobot_instruct()

print(f"Loaded from: {model_dir}")
print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
```

**Output:**
```
Loaded from: C:\Users\gajja\Documents\cars\optimas-prime-instruct\models\autobot-instruct
Device: cuda:0
Dtype: torch.bfloat16
```

#### **Use Case 2: Force CPU Load (Testing)**

```python
# Force CPU for development/testing
tokenizer, model, model_dir = load_autobot_instruct(device="cpu")

print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
```

**Output:**
```
Device: cpu
Dtype: torch.float32
```

#### **Use Case 3: Force CUDA Load**

```python
# Ensure CUDA usage (fail if GPU not available)
tokenizer, model, model_dir = load_autobot_instruct(device="cuda")

print(f"Device: {model.device}")
print(f"Model ready for generation")
```

**Output:**
```
Device: cuda:0
Model ready for generation
```

#### **Use Case 4: Custom Base Directory**

```python
# If model is in different location
tokenizer, model, model_dir = load_autobot_instruct(
    base_dir="/custom/path/to/project",
    device="cuda"
)

print(f"Loaded from: {model_dir}")
```

**Output:**
```
Loaded from: /custom/path/to/project/models/autobot-instruct
```

### Load Errors & Solutions

```python
# ERROR 1: FileNotFoundError - Model directory not found
# SOLUTION: Ensure ./models/autobot-instruct/ exists

# ERROR 2: RuntimeError - Both primary and fallback load failed
# SOLUTION: Check model files are valid (config.json, model.safetensors)

# ERROR 3: CUDA not available but requested
# SOLUTION: Use device="cpu" instead

# ERROR 4: Device mismatch
# SOLUTION: Check torch.cuda.is_available() first
```

---

## Module 2: generate-autobot-instruct.py

### Function Signature

```python
def generate_autobot_instruct(
    model,
    tokenizer,
    system_message: str,
    user_prompt: str,
    device: str,
    max_context_length: int,
    max_tokens: int,
    max_tokens_hard_limit: int,
    temperature: float,
    tools_json: Optional[List[Dict[str, Any]]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate one response and return generated text payload."""
```

### Input Parameters (Required)

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `model` | Model | `model` | Loaded LFM2 model from load_autobot_instruct() |
| `tokenizer` | Tokenizer | `tokenizer` | Loaded tokenizer from load_autobot_instruct() |
| `system_message` | str | `"You are helpful"` | System role/instruction (ignored if `messages` provided) |
| `user_prompt` | str | `"Explain AI"` | User's question/prompt (ignored if `messages` provided) |
| `device` | str | `'cuda'` or `'cpu'` | Where to run generation |
| `max_context_length` | int | `4096` | Total context window limit (max: 128000) |
| `max_tokens` | int | `512` | Target tokens to generate |
| `max_tokens_hard_limit` | int | `1024` | Safety hard limit |
| `temperature` | float | `0.7` | Sampling temperature (0.1-2.0) |

### Input Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools_json` | list | None | List of tool definitions for tool calling |
| `messages` | list | None | Message history (if provided, ignores system_message + user_prompt) |

### Input Format Examples

#### **Format 1: Simple System + User Prompt**

```python
result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You are a programming expert.",
    user_prompt="Write a Python function to reverse a string.",
    device='cuda',
    max_context_length=2048,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
)
```

#### **Format 2: With Conversation History (messages)**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process..."},
    {"role": "user", "content": "Tell me more about chlorophyll."},
]

result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    messages=messages,  # Use messages instead of system_message + user_prompt
    device='cuda',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
)
```

#### **Format 3: With Tools**

```python
tools_json = [
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    }
]

result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You have access to web_search. Use it when needed.",
    user_prompt="What are the latest AI developments in 2026?",
    device='cuda',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
    tools_json=tools_json,  # Pass tools
)
```

### Return Value Structure

```python
{
    'text': str,                    # Clean response (special tokens removed)
    'raw_text': str,                # Full response with special tokens
    'template_token_count': int,    # Tokens after chat template applied
    'formatted_prompt': str,        # Full prompt sent to model
    'input_length': int,            # Number of input tokens processed
    'generated_tokens': int,        # Number of tokens generated
}
```

### Return Value Examples

#### **Example 1: Simple Response**

```python
result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You are helpful.",
    user_prompt="What is 2+2?",
    device='cuda',
    max_context_length=2048,
    max_tokens=32,
    max_tokens_hard_limit=64,
    temperature=0.1,
)

print("Result type:", type(result))  # <class 'dict'>
print("Text:", result['text'])
print("Generated tokens:", result['generated_tokens'])
```

**Output:**
```
Result type: <class 'dict'>
Text: 2 + 2 equals 4.
Generated tokens: 8
```

#### **Example 2: With Metadata**

```python
result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="Explain in detail.",
    user_prompt="What is photosynthesis?",
    device='cuda',
    max_context_length=4096,
    max_tokens=512,
    max_tokens_hard_limit=1024,
    temperature=0.7,
)

print("Full result dictionary:")
print(f"  text: {result['text'][:100]}...")
print(f"  raw_text: {result['raw_text'][:100]}...")
print(f"  template_token_count: {result['template_token_count']}")
print(f"  input_length: {result['input_length']}")
print(f"  generated_tokens: {result['generated_tokens']}")
```

**Output:**
```
Full result dictionary:
  text: Photosynthesis is the process by which plants convert light energy...
  raw_text: <|im_start|>Photosynthesis is the process by which plants convert light...
  template_token_count: 23
  input_length: 42
  generated_tokens: 156
```

---

## Use Cases & Examples

### Use Case 1: Simple Question & Answer

```python
from models.load_autobot_instruct import load_autobot_instruct
from models.generate_autobot_instruct import generate_autobot_instruct
import torch

# STEP 1: Load
tokenizer, model, model_dir = load_autobot_instruct()

# STEP 2: Generate
result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You are a helpful assistant.",
    user_prompt="What is artificial intelligence?",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
)

# STEP 3: Use result
print("ANSWER:")
print(result['text'])
print(f"\nTokens generated: {result['generated_tokens']}")
print(f"Input tokens used: {result['input_length']}")
```

**Input:**
```
system_message: "You are a helpful assistant."
user_prompt: "What is artificial intelligence?"
temperature: 0.7 (balanced)
```

**Output:**
```
ANSWER:
Artificial intelligence (AI) refers to computer systems designed to perform tasks that 
typically require human intelligence. This includes learning from experience, recognizing 
patterns, understanding language, and making decisions. Modern AI powers applications 
including chatbots, recommendation systems, autonomous vehicles, and medical diagnostics.

Tokens generated: 89
Input tokens used: 42
```

### Use Case 2: Code Generation (Low Temperature)

```python
result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You are an expert Python developer. Write clean, well-commented code.",
    user_prompt="Write a function to find all prime numbers up to n.",
    device='cuda',
    max_context_length=4096,
    max_tokens=512,
    max_tokens_hard_limit=1024,
    temperature=0.2,  # Low = more deterministic code
)

print(result['text'])
```

**Input:**
```
system_message: "You are an expert Python developer. Write clean, well-commented code."
user_prompt: "Write a function to find all prime numbers up to n."
temperature: 0.2 (deterministic)
```

**Output:**
```
def find_primes(n: int) -> list:
    """
    Find all prime numbers up to n using Sieve of Eratosthenes.
    
    Args:
        n: Upper limit (inclusive)
    
    Returns:
        List of prime numbers up to n
    """
    if n < 2:
        return []
    
    # Create boolean array and mark all subsequent numbers as prime
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [num for num in range(2, n + 1) if is_prime[num]]

# Example usage
primes = find_primes(30)
print(primes)  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

### Use Case 3: Long Document Summarization

```python
document = """
The Industrial Revolution began in Britain in the late 18th century with the invention 
of the steam engine. This led to mechanization of agriculture and the rise of factory 
systems. Workers migrated to cities, creating both opportunities and social problems. 
The textile industry was transformed first, followed by iron, coal, and transportation. 
This period fundamentally changed society: urbanization increased, the middle class grew, 
and technology became central to economic life. Environmental pollution became visible 
for the first time. Eventually, higher productivity led to increased living standards 
and better access to education, though working conditions initially deteriorated.
"""

result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You are a history teacher. Summarize accurately and concisely.",
    user_prompt=f"Summarize this in 2-3 sentences:\n\n{document}",
    device='cuda',
    max_context_length=2048,
    max_tokens=128,
    max_tokens_hard_limit=256,
    temperature=0.5,
)

print(result['text'])
```

**Input:**
```
system_message: "You are a history teacher. Summarize accurately and concisely."
user_prompt: [document text] + "Summarize this in 2-3 sentences:"
temperature: 0.5 (balanced)
```

**Output:**
```
The Industrial Revolution transformed Britain in the late 18th century through innovations 
like the steam engine, which mechanized production and sparked factory systems. This shift 
caused massive urbanization and social upheaval, though eventually led to higher productivity 
and improved living standards. Environmental pollution and exploitative working conditions 
were negative consequences that emerged during this transformative period.
```

### Use Case 4: Multi-Turn Conversation

```python
# Initialize conversation
messages = [
    {"role": "system", "content": "You are a friendly biology tutor."}
]

# Turn 1
messages.append({
    "role": "user",
    "content": "What is photosynthesis?"
})

result1 = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    device='cuda',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
)

print(f"Q: What is photosynthesis?")
print(f"A: {result1['text']}\n")
messages.append({"role": "assistant", "content": result1['text']})

# Turn 2 - Follow-up question
messages.append({
    "role": "user",
    "content": "Can you explain the light-dependent reactions?"
})

result2 = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    device='cuda',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
)

print(f"Q: Can you explain the light-dependent reactions?")
print(f"A: {result2['text']}\n")
messages.append({"role": "assistant", "content": result2['text']})

# Turn 3 - Another question
messages.append({
    "role": "user",
    "content": "What is the Calvin cycle?"
})

result3 = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    messages=messages,
    device='cuda',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
)

print(f"Q: What is the Calvin cycle?")
print(f"A: {result3['text']}")
```

**Input (messages format):**
```
messages = [
    {"role": "system", "content": "You are a friendly biology tutor."},
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "[previous response]"},
    {"role": "user", "content": "Can you explain the light-dependent reactions?"},
]
```

**Output:**
```
Q: What is photosynthesis?
A: Photosynthesis is the process by which plants, algae, and some bacteria convert 
light energy from the sun into chemical energy stored in glucose. This process occurs 
primarily in the leaves and requires three main ingredients: light, water, and carbon 
dioxide. The process produces glucose (food for the plant) and oxygen (released as 
a byproduct).

Q: Can you explain the light-dependent reactions?
A: The light-dependent reactions occur in the thylakoid membranes of the chloroplasts. 
These reactions require direct sunlight and involve splitting water molecules, which 
releases oxygen. The energy from light is used to create ATP and NADPH, which are 
energy-carrying molecules that power the next stage.

Q: What is the Calvin cycle?
A: The Calvin cycle, also called the light-independent reactions, occurs in the stroma 
of chloroplasts. This cycle uses the ATP and NADPH produced by the light reactions to 
fix carbon dioxide from the air into glucose. The cycle doesn't require light directly 
but depends on the products from the light reactions.
```

### Use Case 5: Tool Calling (Web Search)

```python
tools_json = [
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for"
                }
            },
            "required": ["query"],
        },
    }
]

result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="""You have access to web search. Use <|tool_call_start|>web_search(query="...")<|tool_call_end|> 
when asked about current events or recent information.""",
    user_prompt="What are the latest AI breakthroughs in 2026?",
    device='cuda',
    max_context_length=4096,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.7,
    tools_json=tools_json,
)

print("Raw response (may contain tool call):")
print(result['raw_text'])
print("\nCleaned response:")
print(result['text'])
```

**Input:**
```
system_message: [with tool calling instructions]
user_prompt: "What are the latest AI breakthroughs in 2026?"
tools_json: [web_search tool definition]
```

**Output:**
```
Raw response (may contain tool call):
<|tool_call_start|>web_search(query="latest AI breakthroughs 2026")<|tool_call_end|>

Cleaned response:
[Tool call detected - search would be executed]
```

**Then detect and handle:**
```python
from tool_detector import detect_tool_call

detector = detect_tool_call(result)

if detector.get("type") == "tool":
    tool_name = detector.get("args", {}).get("tool_name")  # "web_search"
    tool_args = detector.get("args", {}).get("args", {})   # {"query": "..."}
    print(f"Execute: {tool_name}({tool_args})")
else:
    print(f"Regular response: {result['text']}")
```

---

## Input/Output Formats

### Format 1: Simple Question & Answer

**Input:**
```python
system_message: str = "You are a helpful assistant."
user_prompt: str = "What is 2 + 2?"
temperature: float = 0.1
max_tokens: int = 32
```

**Output (result dict):**
```python
{
    'text': '2 + 2 equals 4.',
    'raw_text': '<|im_start|>2 + 2 equals 4.<|im_end|>',
    'template_token_count': 15,
    'formatted_prompt': '...[full formatted prompt]...',
    'input_length': 20,
    'generated_tokens': 8,
}
```

### Format 2: Multi-turn Conversation

**Input:**
```python
messages = [
    {"role": "system", "content": "You are a teacher."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is artificial intelligence..."},
    {"role": "user", "content": "Explain machine learning."},
]
temperature: float = 0.7
max_tokens: int = 256
```

**Output (result dict):**
```python
{
    'text': 'Machine learning is a subset of AI where...',
    'raw_text': '<|im_start|>Machine learning is a subset...<|im_end|>',
    'template_token_count': 42,
    'formatted_prompt': '...[full conversation history formatted]...',
    'input_length': 156,
    'generated_tokens': 145,
}
```

### Format 3: With Tools

**Input:**
```python
tools_json = [
    {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        }
    }
]
system_message: str = "You have web search tool. Use it for current info."
user_prompt: str = "Current AI breakthroughs?"
temperature: float = 0.7
```

**Output (when tool is called):**
```python
{
    'text': '',  # Empty after tool stripping
    'raw_text': '<|tool_call_start|>web_search(query="AI breakthroughs 2026")<|tool_call_end|>',
    'template_token_count': 48,
    'formatted_prompt': '...[prompt with tool definitions]...',
    'input_length': 182,
    'generated_tokens': 18,
}
```

Then detect with:
```python
from tool_detector import detect_tool_call

detector = detect_tool_call(result)
# Returns: {"type": "tool", "args": {"tool_name": "web_search", "args": {"query": "..."}}}
```

### Format 4: Code Generation

**Input:**
```python
system_message: str = "You are a Python expert. Write clean code."
user_prompt: str = "Write a function to reverse a list."
temperature: float = 0.2  # Low temp for code
max_tokens: int = 256
```

**Output (result dict):**
```python
{
    'text': '''def reverse_list(lst):
    """Reverse a list."""
    return lst[::-1]

# Example
print(reverse_list([1, 2, 3]))  # [3, 2, 1]''',
    'raw_text': '<|im_start|>def reverse_list...<|im_end|>',
    'template_token_count': 20,
    'formatted_prompt': '...[formatted prompt]...',
    'input_length': 45,
    'generated_tokens': 89,
}
```

### Accessing Result Fields

```python
result = generate_autobot_instruct(...)

# Get the clean response text (most common use)
response_text = result['text']
print(response_text)

# Get response with special tokens visible
raw_response = result['raw_text']
print(raw_response)

# Get generation statistics
tokens_used = result['generated_tokens']
input_tokens = result['input_length']
total_tokens = tokens_used + input_tokens
print(f"Total tokens: {total_tokens}")

# Get the actual formatted prompt sent to model (for debugging)
formatted = result['formatted_prompt']

# Get template statistics
template_count = result['template_token_count']
```

---

## Common Patterns

### Pattern 1: Batch Processing

```python
from models.load_autobot_instruct import load_autobot_instruct
from models.generate_autobot_instruct import generate_autobot_instruct
import torch

tokenizer, model, _ = load_autobot_instruct()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "How does blockchain work?"
]

results = []
for i, prompt in enumerate(prompts, 1):
    print(f"Processing {i}/{len(prompts)}...")
    result = generate_autobot_instruct(
        model=model,
        tokenizer=tokenizer,
        system_message="You are an expert. Explain clearly.",
        user_prompt=prompt,
        device=device,
        max_context_length=2048,
        max_tokens=256,
        max_tokens_hard_limit=512,
        temperature=0.6,
    )
    results.append({'prompt': prompt, 'answer': result['text']})

# Use results
for item in results:
    print(f"Q: {item['prompt']}")
    print(f"A: {item['answer']}\n")
```

### Pattern 2: Validation Before Generation

```python
def validate_generation_inputs(
    prompt: str, 
    tokenizer, 
    max_context_length: int = 4096
) -> dict:
    """Validate inputs before generation"""
    
    # Check prompt length
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)
    
    validation = {
        'valid': True,
        'token_count': token_count,
        'warnings': [],
        'recommended_max_tokens': max_context_length - token_count - 100,
    }
    
    if token_count > max_context_length * 0.8:
        validation['valid'] = False
        validation['warnings'].append(
            f"Prompt too long: {token_count} tokens "
            f"(limit: {max_context_length})"
        )
    
    if token_count > max_context_length - 256:
        validation['warnings'].append(
            "Very little room for output. Consider shortening prompt."
        )
    
    return validation

# Usage
validation = validate_generation_inputs("Your long prompt...", tokenizer)
if not validation['valid']:
    print("Cannot generate:", validation['warnings'])
else:
    print(f"Token count: {validation['token_count']}")
    print(f"Recommended max_tokens: {validation['recommended_max_tokens']}")
```

### Pattern 3: Handling Tool Calls in Loop

```python
from tool_detector import detect_tool_call

def generate_with_tools(
    model, 
    tokenizer, 
    prompt: str, 
    device: str,
    max_tool_calls: int = 5
):
    """Generate with tool calling support"""
    
    messages = [
        {"role": "system", "content": "You can use web_search tool."},
        {"role": "user", "content": prompt},
    ]
    
    tools_json = [
        {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        }
    ]
    
    tool_calls = 0
    
    while tool_calls < max_tool_calls:
        result = generate_autobot_instruct(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_context_length=4096,
            max_tokens=256,
            max_tokens_hard_limit=512,
            temperature=0.7,
            tools_json=tools_json,
        )
        
        detector = detect_tool_call(result)
        
        if detector.get("type") == "tool":
            tool_calls += 1
            tool_info = detector.get("args", {})
            tool_name = tool_info.get("tool_name")
            tool_args = tool_info.get("args", {})
            
            print(f"[Tool Call {tool_calls}] {tool_name}({tool_args})")
            
            # Add to conversation
            messages.append({"role": "assistant", "content": result['raw_text']})
            
            # Execute tool (mock)
            tool_result = f"Results from {tool_name}"
            messages.append({"role": "tool", "content": tool_result})
        else:
            # Final response
            return result['text'], tool_calls
    
    raise Exception(f"Max tool calls ({max_tool_calls}) exceeded")

# Usage
response, calls_made = generate_with_tools(model, tokenizer, 
                                          "What's new in AI?", device)
print(f"Response: {response}")
print(f"Tool calls made: {calls_made}")
```

### Pattern 4: Memory-Efficient Streaming

```python
import asyncio

async def stream_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    chunk_size: int = 50
):
    """Stream response in chunks"""
    
    result = generate_autobot_instruct(
        model=model,
        tokenizer=tokenizer,
        system_message="You are helpful.",
        user_prompt=prompt,
        device=device,
        max_context_length=4096,
        max_tokens=512,
        max_tokens_hard_limit=1024,
        temperature=0.7,
    )
    
    text = result['text']
    
    # Stream in chunks
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        print(chunk, end='', flush=True)
        await asyncio.sleep(0.01)  # Simulate network delay
    
    print()  # Newline
    return result['generated_tokens']

# Usage
# tokens = asyncio.run(stream_response(model, tokenizer, "...", device))
```

### Pattern 5: Error Recovery

```python
def generate_with_fallback(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_retries: int = 2
) -> str:
    """Generate with automatic retry on failure"""
    
    import torch
    
    for attempt in range(max_retries):
        try:
            result = generate_autobot_instruct(
                model=model,
                tokenizer=tokenizer,
                system_message="You are helpful.",
                user_prompt=prompt,
                device=device,
                max_context_length=4096,
                max_tokens=256,
                max_tokens_hard_limit=512,
                temperature=0.7,
            )
            return result['text']
        
        except torch.cuda.OutOfMemoryError:
            print(f"Retry {attempt+1}: OOM - clearing cache...")
            torch.cuda.empty_cache()
            
            if attempt == max_retries - 1:
                # Last attempt - use smaller context
                result = generate_autobot_instruct(
                    model=model,
                    tokenizer=tokenizer,
                    system_message="Be brief.",
                    user_prompt=prompt,
                    device=device,
                    max_context_length=1024,  # Smaller
                    max_tokens=128,  # Smaller
                    max_tokens_hard_limit=256,
                    temperature=0.7,
                )
                return result['text']
        
        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
    
    return None

# Usage
try:
    response = generate_with_fallback(model, tokenizer, "Your prompt", device)
    print(response)
except Exception as e:
    print(f"Failed: {e}")
```

---

## Error Handling & Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Model files missing | Check `./models/autobot-instruct/config.json` exists |
| `CUDA OutOfMemory` | GPU memory full | Reduce `max_tokens`, use `device='cpu'` |
| `RuntimeError` (model load) | Invalid model files | Verify model weights and config are valid |
| `TypeError` (chat template) | Tokenizer version mismatch | Update transformers library |

### Quick Fixes

```python
# Fix 1: CUDA out of memory
torch.cuda.empty_cache()
result = generate_autobot_instruct(..., max_tokens=128)  # Reduce

# Fix 2: Model not found
import os
print(os.path.exists("./models/autobot-instruct/config.json"))

# Fix 3: Device issues
device = 'cpu'  # Fall back to CPU
tokenizer, model, _ = load_autobot_instruct(device=device)

# Fix 4: Very long prompts
prompt = prompt[:2000]  # Limit prompt length
result = generate_autobot_instruct(..., user_prompt=prompt)
```

---

## Summary

| Component | Function | Input | Output |
|-----------|----------|-------|--------|
| **load_autobot_instruct()** | Load model | base_dir, device | tokenizer, model, model_dir |
| **generate_autobot_instruct()** | Generate text | model, tokenizer, prompt, params | dict with text, metadata |
| **detect_tool_call()** | Detect tools | result dict | tool info or regular response |

> **Start here:** Load → Generate → Extract text from result['text']

---

## See Also

- [load-autobot-instruct.py](load-autobot-instruct.py) — Implementation details
- [generate-autobot-instruct.py](generate-autobot-instruct.py) — Generation implementation
- [app.py](../app.py) — Backend integration example
- [tool-detector.py](../tool-detector.py) — Tool call detection
