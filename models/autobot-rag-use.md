# Autobot-RAG Complete Usage Guide

## Overview
This guide demonstrates how to use the autobot-rag model with the complete pipeline: loading the model, retrieving context from RAG (Retrieval-Augmented Generation), and generating intelligent responses. This is based on the production implementation in `app.py`.

---

## Table of Contents
1. [Step 1: Loading the Autobot-RAG Model](#step-1-loading-the-autobot-rag-model)
2. [Step 2: Initializing RAG Pipeline](#step-2-initializing-rag-pipeline)
3. [Step 3: Retrieving Context from RAG](#step-3-retrieving-context-from-rag)
4. [Step 4: Generating Responses](#step-4-generating-responses)
5. [Complete Workflow Example](#complete-workflow-example)
6. [API Integration Example](#api-integration-example)

---

## Step 1: Loading the Autobot-RAG Model

### Purpose
Load the tokenizer and model from the local autobot-rag directory before any generation can occur.

### Input Requirements
- **Model Path**: `./models/autobot-rag/` (directory containing model files)
- **Device**: `"cpu"` or `"cuda"` (CUDA recommended for performance)

### Code Implementation

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from load-autobot-rag import load_autobot_rag

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./models/autobot-rag"

tokenizer, model = load_autobot_rag(
    model_path=model_path,
    device=device
)

# Validation
if tokenizer is None or model is None:
    print("❌ Failed to load model")
    sys.exit(1)
else:
    print("✅ Model loaded successfully")
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"   Model device: {model.device}")
    print(f"   Model dtype: {model.dtype}")
```

### Output Details

| Item | Type | Description |
|------|------|-------------|
| `tokenizer` | `AutoTokenizer` | Configured tokenizer with chat template support |
| `model` | `AutoModelForCausalLM` | Language model ready for generation |
| `tokenizer.vocab_size` | `int` | Vocabulary size (typically ~100K+ tokens) |
| `model.device` | `torch.device` | Device allocation (cuda:0, cpu, etc.) |
| `model.dtype` | `torch.dtype` | Precision (bfloat16 for CUDA, float32 for CPU) |

### Return Values

**Success:**
```python
(AutoTokenizer, AutoModelForCausalLM)
```

**Failure:**
```python
(None, None)
```

---

## Step 2: Initializing RAG Pipeline

### Purpose
Set up the RAG (Retrieval-Augmented Generation) system to retrieve relevant context from a vector store before generating responses.

### Input Requirements
- **Vector Store Path**: Path to stored document embeddings (e.g., `./vector_store`)
- **Must have**: Pre-indexed documents in vector store (from ingestion pipeline)

### Code Implementation

```python
from rag_pipeline import RAGPipeline

# Determine vector store location
def get_rag_store_path() -> str:
    """Find the vector store path"""
    env_path = os.getenv("RAG_VECTOR_STORE_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Check common locations
    for candidate in ["./vector_store", "/data/vector_store", "/data"]:
        if os.path.exists(candidate):
            return candidate
    
    return "./vector_store"

# Initialize RAG pipeline
try:
    rag_store_path = get_rag_store_path()
    rag_pipeline = RAGPipeline(store_path=rag_store_path)
    print(f"✅ RAG pipeline ready (store: {rag_store_path})")
except Exception as e:
    print(f"❌ RAG pipeline init failed: {e}")
    rag_pipeline = None
```

### Output Details

| Item | Type | Description |
|------|------|-------------|
| `rag_pipeline` | `RAGPipeline` | Initialized pipeline instance |
| Status | `str` | "ready" if successful, None if failed |
| Store Path | `str` | Location of vector store |

---

## Step 3: Retrieving Context from RAG

### Purpose
Query the vector store to retrieve relevant documents/chunks related to the user's query.

### Input Parameters

```python
context, results = rag_pipeline.retrieve_context(
    query="What is Python?",           # User's question
    top_k=3,                           # Return top 3 most relevant chunks
    max_context_chunks=3,              # Maximum chunks to include
    filters=None                       # Optional metadata filters
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | User's search query |
| `top_k` | `int` | 3 | Number of top results to retrieve |
| `max_context_chunks` | `int` | 3 | Maximum chunks to combine into context |
| `filters` | `Dict` | None | Metadata filters for retrieval |

### Return Values

```python
# context: Object with .context_text and .sources attributes
context.context_text    # Combined text from retrieved chunks
context.sources         # List of source documents

# results: List of individual chunks retrieved
len(results)   # Number of chunks (typically 0-3)
```

### Code Implementation

```python
# Retrieve context from RAG
rag_context_text = ""
rag_sources = []

if rag_pipeline:
    try:
        ctx, results = rag_pipeline.retrieve_context(
            query=user_prompt,
            top_k=3,
            max_context_chunks=3,
            filters=None,
        )
        if ctx:
            rag_context_text = ctx.context_text
            rag_sources = ctx.sources
            print(f"✅ RAG retrieved {len(results)} chunks")
            print(f"   Sources: {rag_sources}")
            print(f"   Context length: {len(rag_context_text)} chars")
    except Exception as e:
        print(f"❌ RAG retrieval error: {e}")
```

### Output Example

```python
# Retrieved context structure
context.context_text:
"""
Python Documentation (Library Reference):
The Python Standard Library contains several built-in modules...

[Source 1]
Python is an interpreted, high-level language...

[Source 2]
Used for: web development, data science, automation...
"""

context.sources:
["python-docs-library.md", "python-overview.md"]

results:
[
    {"text": "Python is an interpreted...", "similarity": 0.92},
    {"text": "Python was created by Guido van Rossum...", "similarity": 0.89},
    {"text": "Popular Python frameworks...", "similarity": 0.87}
]
```

---

## Step 4: Generating Responses

### Purpose
Generate a complete, context-aware response using the RAG-retrieved context combined with the user's query.

### Input Parameters

```python
from generate_autobot_rag import generate_autobot_response

response_payload = generate_autobot_response(
    model=model,                          # Loaded model from Step 1
    tokenizer=tokenizer,                  # Loaded tokenizer from Step 1
    system_message="""You are GAKR AI, a helpful assistant.
    
Rules:
- Begin with a straightforward response
- Follow the format: Descriptive, Diagnostic, Predictive, Prescriptive
- Keep answers clear and practical
- Use bullet points for clarity
""",
    user_prompt="""Use the following context to answer questions:

RAG CONTEXT:
{retrieved_context_from_step_3}

USER QUERY: {user_question}
""",
    device=device,                        # "cpu" or "cuda"
    max_tokens=256,                       # Maximum tokens to generate
    temperature=0.7,                      # Sampling temperature (0=deterministic, 1=creative)
    max_context_length=32768,             # Model's max context window
    max_tokens_hard_limit=512,            # Hard cap on generation
    model_label="autobot-rag"             # For logging
)
```

| Parameter | Type | Recommended | Description |
|-----------|------|-------------|-------------|
| `model` | `AutoModelForCausalLM` | — | Loaded model |
| `tokenizer` | `AutoTokenizer` | — | Loaded tokenizer |
| `system_message` | `str` | — | System context/instructions |
| `user_prompt` | `str` | — | User query + RAG context |
| `device` | `str` | "cuda" | Device for inference |
| `max_tokens` | `int` | 256-512 | Max generation length |
| `temperature` | `float` | 0.1-0.7 | 0=deterministic, 1=max randomness |
| `max_context_length` | `int` | 32768 | Model's context window |
| `max_tokens_hard_limit` | `int` | 512-1024 | Safety cap |
| `model_label` | `str` | "autobot-rag" | Debug label |

### Return Value

```python
response_payload = {
    "text": "Python is a high-level programming language...",
    "raw_text": "<|im_end|>Python is a high-level...<|endoftext|>",
    "template_token_count": 42,
    "formatted_prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>assistant\n",
    "input_length": 42,
    "generated_tokens": 87
}
```

| Key | Type | Description |
|-----|------|-------------|
| `text` | `str` | Cleaned response (special tokens removed) |
| `raw_text` | `str` | Raw generation with special tokens |
| `template_token_count` | `int` | Tokens in formatted prompt |
| `formatted_prompt` | `str` | Full chat-templated prompt |
| `input_length` | `int` | Tokens in user input |
| `generated_tokens` | `int` | Tokens generated by model |

### Code Implementation

```python
from generate_autobot_rag import generate_autobot_response

try:
    # Build the full context message
    context_parts = []
    
    if rag_context_text:
        context_parts.append(f"RAG CONTEXT:\n{rag_context_text}")
    
    if file_text:  # Files uploaded by user
        context_parts.append(f"FILE CONTENT:\n{file_text}")
    
    context = "\n\n".join(context_parts) if context_parts else "No additional context."
    
    # Build full user message
    full_user_message = f"""Use the following context to answer questions:
{context}

USER QUERY: {user_prompt}
"""
    
    # Generate response
    generation_result = generate_autobot_response(
        model=model,
        tokenizer=tokenizer,
        system_message=system_message,
        user_prompt=full_user_message,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_tokens_hard_limit=MAX_TOKENS_HARD_LIMIT,
        model_label="autobot-rag",
    )
    
    # Extract results
    response_text = generation_result["text"]
    input_tokens = generation_result["input_length"]
    output_tokens = generation_result["generated_tokens"]
    
    print(f"✅ Generation complete:")
    print(f"   Response: {response_text[:100]}...")
    print(f"   Input tokens: {input_tokens}")
    print(f"   Output tokens: {output_tokens}")
    
except Exception as e:
    print(f"❌ Generation error: {e}")
    traceback.print_exc()
```

---

## Complete Workflow Example

### Full End-to-End Pipeline

```python
import torch
import traceback
from load_autobot_rag import load_autobot_rag
from generate_autobot_rag import generate_autobot_response
from rag_pipeline import RAGPipeline

# ========== INITIALIZATION ==========

# 1. Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {DEVICE}")

# 2. Load model (Step 1)
print("📦 Loading model...")
tokenizer, model = load_autobot_rag(
    model_path="./models/autobot-rag",
    device=DEVICE
)

if tokenizer is None or model is None:
    print("❌ Model loading failed")
    exit(1)

print("✅ Model loaded successfully")

# 3. Initialize RAG (Step 2)
print("🔍 Initializing RAG...")
try:
    rag_pipeline = RAGPipeline(store_path="./vector_store")
    print("✅ RAG pipeline ready")
except Exception as e:
    print(f"❌ RAG init failed: {e}")
    rag_pipeline = None

# ========== QUERY PROCESSING ==========

# User input
user_query = "What are the main features of Python?"

# 4. Retrieve context (Step 3)
print(f"\n🔎 Searching RAG for: '{user_query}'")

rag_context = ""
rag_sources = []

if rag_pipeline:
    try:
        ctx, results = rag_pipeline.retrieve_context(
            query=user_query,
            top_k=3,
            max_context_chunks=3
        )
        if ctx:
            rag_context = ctx.context_text
            rag_sources = ctx.sources
            print(f"✅ Retrieved {len(results)} chunks")
            print(f"   Sources: {rag_sources}")
    except Exception as e:
        print(f"❌ RAG error: {e}")

# 5. Generate response (Step 4)
print("\n🤖 Generating response...")

system_message = """You are GAKR AI Assistant.

Rules:
- Answer questions clearly and accurately
- Use RAG context when available
- Format: Descriptive, Diagnostic, Predictive, Prescriptive
- Be concise but informative"""

full_prompt = f"""Context to use for answering:
{rag_context if rag_context else "General knowledge"}

User Query: {user_query}"""

try:
    result = generate_autobot_response(
        model=model,
        tokenizer=tokenizer,
        system_message=system_message,
        user_prompt=full_prompt,
        device=DEVICE,
        max_tokens=256,
        temperature=0.7,
        max_context_length=32768,
        max_tokens_hard_limit=512,
        model_label="autobot-rag"
    )
    
    # Extract and display results
    response_text = result["text"]
    input_len = result["input_length"]
    output_len = result["generated_tokens"]
    
    print(f"\n{'='*60}")
    print("RESPONSE:")
    print(f"{'='*60}")
    print(response_text)
    print(f"{'='*60}")
    print(f"Tokens: Input={input_len}, Output={output_len}")
    if rag_sources:
        print(f"Sources: {', '.join(rag_sources)}")
    print(f"{'='*60}\n")
    
except Exception as e:
    print(f"❌ Generation failed: {e}")
    traceback.print_exc()
```

### Expected Output

```
🔧 Device: cuda
📦 Loading model...
[load-autobot-rag] ==========================================
[load-autobot-rag] Starting autobot-rag load
[load-autobot-rag] Tokenizer ready (vocab=100352)
[load-autobot-rag] Model load complete
[load-autobot-rag] model_device=cuda:0
[load-autobot-rag] model_dtype=torch.bfloat16
[load-autobot-rag] ==========================================
✅ Model loaded successfully

🔍 Initializing RAG...
✅ RAG pipeline ready

🔎 Searching RAG for: 'What are the main features of Python?'
✅ Retrieved 3 chunks
   Sources: ['python-overview.md', 'python-features.md']

🤖 Generating response...
[generate-autobot-rag] =======================================
[generate-autobot-rag] generate_autobot_response called
[generate-autobot-rag] Formatted prompt length=487 chars
[generate-autobot-rag] Template token count=42
[generate-autobot-rag] Tokenized input length=42 tokens
[generate-autobot-rag] Generation complete: tokens=87, elapsed=1.24s, tps=70.2
[generate-autobot-rag] =======================================

============================================================
RESPONSE:
============================================================
Python's main features include:

**Descriptive** - Python is a high-level, interpreted language known for simplicity and readability. It supports multiple programming paradigms (object-oriented, functional, procedural).

**Diagnostic** - Its popularity stems from:
- Easy-to-learn syntax
- Extensive standard library
- Rich ecosystem of third-party packages
- Strong community support

**Predictive** - Python will continue dominating in data science, AI/ML, and automation.

**Prescriptive** - To get started: install Python, learn basic syntax, explore popular frameworks.
============================================================
Tokens: Input=42, Output=87
Sources: python-overview.md, python-features.md
============================================================
```

---

## API Integration Example

### FastAPI Endpoint Implementation

```python
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List

@app.post("/api/analyze")
async def analyze(
    prompt: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    max_tokens: int = Form(256),
    temperature: float = Form(0.7),
):
    """
    Main analysis endpoint with RAG and file support
    
    Request:
        prompt: User question
        files: Optional uploaded files
        max_tokens: Max generation length (default 256)
        temperature: Sampling temperature (default 0.7)
    
    Response:
        {
            "response": "Generated answer text",
            "meta": {
                "input_tokens": 42,
                "output_tokens": 87,
                "elapsed": 1.24,
                "rag_sources": ["doc1.md", "doc2.md"]
            }
        }
    """
    
    # Validation
    if not prompt and not files:
        raise HTTPException(400, "Empty prompt and no files")
    
    if not lfm_model:
        raise HTTPException(503, "Model not loaded")
    
    # Enforce limits
    max_tokens = min(max_tokens, MAX_TOKENS_HARD_LIMIT)
    temperature = max(0.0, min(temperature, 2.0))
    
    print(f"Request: '{prompt[:50]}...' | tokens={max_tokens} | temp={temperature}")
    
    try:
        # Process uploaded files
        file_text = ""
        if files:
            for f in files:
                try:
                    content = await f.read(min(MAX_FILE_SIZE, 1024*1024))
                    file_text += f"\n--- File: {f.filename} ---\n"
                    file_text += content.decode("utf-8", errors="ignore")
                except Exception as e:
                    print(f"File error {f.filename}: {e}")
        
        # Retrieve RAG context
        rag_context_text = ""
        rag_sources = []
        
        if rag_pipeline_instance:
            try:
                ctx, results = rag_pipeline_instance.retrieve_context(
                    query=prompt,
                    top_k=3,
                    max_context_chunks=3
                )
                if ctx:
                    rag_context_text = ctx.context_text
                    rag_sources = ctx.sources
            except Exception as e:
                print(f"RAG retrieval error: {e}")
        
        # Build context
        context_parts = []
        if rag_context_text:
            context_parts.append(f"RAG CONTEXT:\n{rag_context_text}")
        if file_text:
            context_parts.append(f"FILE CONTENT:\n{file_text}")
        
        context = "\n\n".join(context_parts) if context_parts else "No context"
        
        full_user_message = f"""Use context to answer:
{context}

QUERY: {prompt}"""
        
        # Generate response
        result = generate_autobot_response(
            model=lfm_model,
            tokenizer=lfm_tokenizer,
            system_message=SYSTEM_MESSAGE,
            user_prompt=full_user_message,
            device=DEVICE,
            max_tokens=max_tokens,
            temperature=temperature,
            max_context_length=MAX_CONTEXT_LENGTH,
            max_tokens_hard_limit=MAX_TOKENS_HARD_LIMIT,
        )
        
        return JSONResponse(content={
            "response": result["text"],
            "meta": {
                "model": "autobot-rag",
                "input_tokens": result["input_length"],
                "output_tokens": result["generated_tokens"],
                "rag_sources": rag_sources,
            }
        })
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=What is Python?" \
  -F "max_tokens=256" \
  -F "temperature=0.7"
```

### Example Response

```json
{
  "response": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence...",
  "meta": {
    "model": "autobot-rag",
    "input_tokens": 42,
    "output_tokens": 87,
    "rag_sources": ["python-docs.md", "python-features.md"]
  }
}
```

---

## Key Parameters Reference

### Generation Configuration

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | 0.0-2.0 | 0.7 | 0=greedy, 0.7=balanced, 1.0=creative |
| `max_tokens` | 1-512 | 256 | Maximum generation length |
| `top_p` | 0.0-1.0 | 0.1 | Nucleus sampling threshold |
| `top_k` | 1-100 | 50 | Top-k candidates |
| `repetition_penalty` | 1.0-2.0 | 1.05 | Penalty for repeating tokens |

### Context Settings

| Setting | Recommended | Purpose |
|---------|-------------|---------|
| `max_context_length` | 32768 | Model's token window limit |
| `max_tokens_hard_limit` | 512-1024 | Safety cap on generation |
| `top_k` (RAG) | 3-5 | Number of chunks to retrieve |
| `max_context_chunks` | 3 | Max chunks in combined context |

---

## Error Handling

```python
import torch
import traceback

try:
    # Generate response
    result = generate_autobot_response(...)
    
except torch.cuda.OutOfMemoryError:
    print("💾 CUDA out of memory - reduce max_tokens or use CPU")
    
except RuntimeError as e:
    print(f"❌ Generation failed: {e}")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    traceback.print_exc()
```

---

## Best Practices

1. **Always check model loading status** before generation
2. **Validate prompt length** to prevent truncation
3. **Use RAG context** when available (significantly improves quality)
4. **Set appropriate temperature** (0.1-0.7 for factual, 0.7-1.0 for creative)
5. **Enforce token limits** to prevent out-of-memory errors
6. **Log generation metrics** for monitoring performance
7. **Handle file uploads safely** (validate size and format)
8. **Use CUDA when available** for 10-20x speedup

---

## Summary

```
┌─────────────────────────────────────────────────────┐
│         AUTOBOT-RAG COMPLETE PIPELINE              │
├─────────────────────────────────────────────────────┤
│ 1. Load Model                                       │
│    Input: model_path, device                        │
│    Output: tokenizer, model                         │
│                                                     │
│ 2. Initialize RAG Pipeline                         │
│    Input: vector_store_path                         │
│    Output: rag_pipeline instance                    │
│                                                     │
│ 3. Retrieve Context                                 │
│    Input: user_query, top_k                         │
│    Output: context_text, sources, results           │
│                                                     │
│ 4. Generate Response                                │
│    Input: model, tokenizer, prompt, params          │
│    Output: text, tokens, metadata                   │
│                                                     │
│ 5. Return to User                                   │
│    Format: JSON with response + metadata            │
└─────────────────────────────────────────────────────┘
```
