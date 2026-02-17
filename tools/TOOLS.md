# Tool Schemas

This file documents the JSON tool schema used by AutoBot's tool-use agent.

## web_search

```json
{
  "name": "web_search",
  "description": "Search the web for current information, facts, or recent updates.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "max_results": {"type": "integer", "description": "Maximum number of results", "default": 4},
      "workers": {"type": "integer", "description": "Parallel worker count", "default": 6}
    },
    "required": ["query"]
  }
}
```

## send_email

```json
{
  "name": "send_email",
  "description": "Send an email to recipients with subject and body.",
  "parameters": {
    "type": "object",
    "properties": {
      "to": {"type": "string", "description": "Recipient(s), comma separated"},
      "subject": {"type": "string", "description": "Email subject"},
      "body": {"type": "string", "description": "Plain text body"},
      "body_html": {"type": "string", "description": "Optional HTML body"},
      "attachments": {"type": "array", "items": {"type": "string"}, "description": "Optional attachment paths"},
      "cc": {"type": "string", "description": "Optional CC recipient(s)"},
      "bcc": {"type": "string", "description": "Optional BCC recipient(s)"}
    },
    "required": ["to", "subject", "body"]
  }
}
```

## Structured Output Notes

- `web_search` returns structured JSON with keys such as:
  - `status`
  - `query`
  - `results_count`
  - `stats`
  - `results`
- `send_email` returns structured JSON status from the send-email service.
