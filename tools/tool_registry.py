"""
Tool Registry for AutoBot.
Manages loading and execution of local tools from the tools/ folder only.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


tools_json = [
    {
        "name": "web_search",
        "description": "Search the web for current information, facts, or recent updates.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 4,
                },
                "workers": {
                    "type": "integer",
                    "description": "Number of worker threads for scraping",
                    "default": 6,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to recipients with subject and body.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address(es), comma separated for multiple recipients.",
                },
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Plain text email body."},
                "body_html": {
                    "type": "string",
                    "description": "Optional HTML body.",
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional attachment paths.",
                },
                "cc": {
                    "type": "string",
                    "description": "Optional CC recipient(s), comma separated.",
                },
                "bcc": {
                    "type": "string",
                    "description": "Optional BCC recipient(s), comma separated.",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
]


class ToolRegistry:
    """Registry for supported local tools."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tools_dir = Path(__file__).resolve().parent

    async def initialize(self):
        """Load enabled local tools from tools/ folder."""
        enabled_tools = self.config.get("tools", {}).get("enabled", ["web_search", "send_email"])
        enabled_normalized = {self._normalize_tool_name(name) for name in enabled_tools}

        if "web_search" in enabled_normalized:
            self._load_web_search()

        if "send_email" in enabled_normalized:
            self._load_send_email()

        self.logger.info("Tool registry initialized with tools: %s", sorted(self.tools.keys()))

    def _load_web_search(self):
        web_search_dir = self.tools_dir / "web-search"
        if str(web_search_dir) not in sys.path:
            sys.path.insert(0, str(web_search_dir))

        try:
            from search import run_search

            self.tools["web_search"] = {
                "type": "web_search",
                "runner": run_search,
                "initialized": True,
            }
            self.logger.info("Loaded tool: web_search")
        except Exception as exc:
            self.logger.exception("Failed to load web_search: %s", exc)

    def _load_send_email(self):
        module_path = self.tools_dir / "send-email" / "send-email.py"
        if not module_path.exists():
            self.logger.warning("send_email tool file not found: %s", module_path)
            return

        try:
            spec = importlib.util.spec_from_file_location("autobot_send_email_tool", str(module_path))
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to create module spec for send_email tool")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            send_email = getattr(module, "send_email", None)
            if not callable(send_email):
                raise RuntimeError("send_email function not found in send-email.py")

            self.tools["send_email"] = {
                "type": "send_email",
                "runner": send_email,
                "initialized": True,
            }
            self.logger.info("Loaded tool: send_email")
        except Exception as exc:
            self.logger.exception("Failed to load send_email: %s", exc)

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters."""
        normalized = self._normalize_tool_name(tool_name)
        if normalized not in self.tools:
            raise ValueError(f"Tool {normalized} not found or not initialized")

        tool_config = self.tools[normalized]
        tool_type = tool_config.get("type")

        if tool_type == "web_search":
            return await self._execute_web_search(tool_config, params)
        if tool_type == "send_email":
            return await self._execute_send_email(tool_config, params)

        raise ValueError(f"Unknown tool type: {tool_type}")

    async def _execute_web_search(self, tool_config: Dict[str, Any], params: Dict[str, Any]) -> str:
        query = str(params.get("query", "")).strip()
        if not query:
            return json.dumps({"status": "error", "error": "Missing query parameter"}, indent=2)

        max_results = int(params.get("max_results", 4))
        workers = int(params.get("workers", 6))

        loop = asyncio.get_event_loop()
        try:
            results, stats = await loop.run_in_executor(
                None,
                tool_config["runner"],
                query,
                max_results,
                workers,
            )
            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "results_count": len(results),
                    "stats": stats,
                    "results": results,
                },
                indent=2,
                default=str,
            )
        except Exception as exc:
            self.logger.exception("Web search execution failed: %s", exc)
            return json.dumps({"status": "error", "error": str(exc)}, indent=2)

    async def _execute_send_email(self, tool_config: Dict[str, Any], params: Dict[str, Any]) -> str:
        to = params.get("to")
        subject = params.get("subject")
        body = params.get("body")

        if not to or not subject or not body:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Missing required parameters: to, subject, body",
                },
                indent=2,
            )

        body_html = params.get("body_html")
        attachments = params.get("attachments") or []
        cc = params.get("cc")
        bcc = params.get("bcc")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: tool_config["runner"](
                    to=to,
                    subject=subject,
                    body=body,
                    body_html=body_html,
                    attachments=attachments,
                    cc=cc,
                    bcc=bcc,
                ),
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            self.logger.exception("send_email execution failed: %s", exc)
            return json.dumps({"status": "error", "error": str(exc)}, indent=2)

    async def shutdown(self):
        self.logger.info("Shutting down tool registry")
        self.tools.clear()

    def _normalize_tool_name(self, tool_name: str) -> str:
        normalized = str(tool_name).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "email":
            return "send_email"
        if normalized == "websearch":
            return "web_search"
        return normalized
