
# generate-autobot-instruct.py — Documentation

## Purpose
Generation utilities for the Autobot Instruct model: format chat prompts with the tokenizer's chat template, run `model.generate()`, decode the generated tokens, and return a simple generation payload.

## Public API
- `generate_autobot_instruct(model, tokenizer, system_message: str, user_prompt: str, device: str, max_context_length: int, max_tokens: int, max_tokens_hard_limit: int, temperature: float, tools_json: Optional[List[Dict[str, Any]]] = None, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]`

## Description
Prepares chat messages (either via `messages` or from `system_message` + `user_prompt`), applies the tokenizer's `apply_chat_template` to build the formatted prompt and measure template tokens, tokenizes and truncates inputs to fit `max_context_length`, runs `model.generate()` with configured generation hyperparameters, decodes the generated token ids back to text, strips module-defined special tokens from the visible text, and returns a concise payload containing the generated text and metadata.

## Inputs
- `model` — a loaded Hugging Face causal LM (e.g., returned by `load_autobot_instruct`).
- `tokenizer` — corresponding tokenizer; expected to support `apply_chat_template`.
- `system_message` (str) — assistant/system-level instruction.
- `user_prompt` (str) — user's prompt string (used when `messages` is not supplied).
- `device` (str) — device string (e.g., `'cuda'` or `'cpu'`) used to place inputs and generation.
- `max_context_length` (int) — total context length allowed by the model; generator truncates input to `max_context_length - max_tokens`.
- `max_tokens` (int) — requested new tokens to generate.
- `max_tokens_hard_limit` (int) — absolute cap on `max_new_tokens`.
- `temperature` (float) — controls sampling; if <= 0.1 deterministic decoding is used.
- `tools_json` (Optional[List[Dict[str,Any]]]) — optional list of tool metadata passed to the chat template (the generator will include it in template formatting but does not itself parse tool calls).
- `messages` (Optional[List[Dict[str,str]]]) — optional pre-composed chat message list (each item has keys like `role` and `content`).

## Outputs / Return value
Returns a `dict` with the generated text payload and metadata:

- `text` (str): cleaned string with module `SPECIAL_TOKENS` removed.
- `raw_text` (str): full decoded generated text (may include special tokens).
- `template_token_count` (int): token count of the templated prompt as measured by the tokenizer.
- `formatted_prompt` (str): the prompt string after `apply_chat_template` formatting.
- `input_length` (int): input token count fed to the model.
- `generated_tokens` (int): number of tokens produced by the model.

## Generation internals and tuning
- The function calls `tokenizer.apply_chat_template` twice (once with `tokenize=True` to measure tokens, and once with `tokenize=False` to produce the prompt), with fallbacks when the tokenizer doesn't accept a `tools` kwarg.
- `generation_config` includes the following defaults and constraints:
  - `max_new_tokens`: `min(max_tokens, max_tokens_hard_limit)`
  - `do_sample`: `temperature > 0`
  - `temperature`: clipped to `[0.1, 1.0]` when sampling
  - `top_p`: `0.1`
  - `top_k`: `50`
  - `repetition_penalty`: `1.05`
  - `no_repeat_ngram_size`: `3`
  - `pad_token_id`, `eos_token_id`, `use_cache`
- When `temperature <= 0.1`, sampling is disabled (`do_sample=False`) and temperature/top_p/top_k are removed to enable deterministic decoding.
- The function runs `model.generate(**generation_config)` inside `torch.no_grad()` and decodes only the generated suffix tokens (input prefix removed) before returning the payload.

## Special tokens
The module defines `SPECIAL_TOKENS` (e.g., `<|im_end|>`, `<|tool_list_start|>`, etc.). These are stripped from the returned `text` but retained in `raw_text`.

## Examples
```python
from models.generate_autobot_instruct import generate_autobot_instruct

# assume `model` and `tokenizer` were loaded via `load_autobot_instruct`
result = generate_autobot_instruct(
    model=model,
    tokenizer=tokenizer,
    system_message="You are an assistant.",
    user_prompt="Explain the difference between A and B.",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    max_context_length=2048,
    max_tokens=256,
    max_tokens_hard_limit=512,
    temperature=0.2,
)

print(result['text'])
```

## Notes and caveats
- This module is generation-only: it does not perform tool-call detection or parsing. If you need tool-call semantics, use a separate parser or an earlier version that implemented parsing heuristics.
- The generator expects the tokenizer to support `apply_chat_template`. If that API changes, the function has fallbacks for omission of the `tools` kwarg when calling the template.

