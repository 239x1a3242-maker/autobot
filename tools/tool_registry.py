"""
Tool Registry for AutoBot
Manages and executes tool integrations.
"""

import asyncio
import logging
import json
import sys
from typing import Dict, Any
from pathlib import Path

tools_json = [
    {
        "name": "web_search",
        "description": "Search the web for current information, facts, or recent updates.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to one or more recipients. Supports plain text, HTML, attachments, CC, and BCC.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address(es). For multiple recipients, separate with commas (e.g., 'alice@example.com, bob@example.com')."
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line."
                },
                "body": {
                    "type": "string",
                    "description": "Plain text body of the email."
                },
                "body_html": {
                    "type": "string",
                    "description": "Optional HTML version of the email body."
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to attach to the email (optional)."
                },
                "cc": {
                    "type": "string",
                    "description": "Carbon copy recipient(s). Multiple emails can be comma-separated (optional)."
                },
                "bcc": {
                    "type": "string",
                    "description": "Blind carbon copy recipient(s). Multiple emails can be comma-separated (optional)."
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
]


# Add tools directory to path for dynamic imports
tools_dir = Path(__file__).parent
web_search_dir = tools_dir / "web-search"

sys.path.insert(0, str(web_search_dir))

class ToolRegistry:
    """Registry for currently supported tools."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tools: Dict[str, Any] = {}

    async def initialize(self):
        """Load enabled tools."""
        enabled_tools = self.config.get('tools', {}).get('enabled', ['web_search'])
        
        try:
            # Load web_search tool
            if 'web_search' in enabled_tools or 'web-search' in enabled_tools:
                try:
                    from search import run_search
                    
                    self.tools['web_search'] = {
                        'type': 'web_search',
                        'runner': run_search,
                        'initialized': True
                    }
                    self.logger.info("Loaded tool: web_search")
                except Exception as e:
                    self.logger.error(f"Failed to load web_search: {e}")
                    self.tools['web_search'] = None

        except Exception as e:
            self.logger.error(f"Error initializing tools: {e}")

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters."""
        tool_name_normalized = tool_name.replace('-', '_')
        
        if tool_name_normalized not in self.tools or self.tools[tool_name_normalized] is None:
            raise ValueError(f"Tool {tool_name_normalized} not found or not initialized")

        tool_config = self.tools[tool_name_normalized]
        tool_type = tool_config.get('type')

        try:
            if tool_type == 'web_search':
                return await self._execute_web_search(tool_config, params)
            else:
                raise ValueError(f"Unknown tool type: {tool_type}")

        except Exception as e:
            self.logger.error(f"Tool execution failed ({tool_name}): {e}")
            return json.dumps({'error': str(e)}, indent=2)

    async def _execute_web_search(self, tool_config: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Execute web search using search engine."""
        try:
            query = params.get('query', '')
            max_results = params.get('max_results', 2)
            workers = params.get('workers', 6)
            
            if not query:
                return json.dumps({'error': 'Missing query parameter'}, indent=2)
            
            # Run search in thread pool to avoid blocking (search is synchronous)
            loop = asyncio.get_event_loop()
            results, stats = await loop.run_in_executor(
                None, 
                tool_config['runner'], 
                query, 
                max_results, 
                workers
            )
            
            return json.dumps({
                'status': 'success',
                'query': query,
                'results_count': len(results),
                'stats': stats,
                'results': results
            }, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Web search execution error: {e}")
            return json.dumps({'error': str(e)}, indent=2)

    async def shutdown(self):
        """Shutdown all tools."""
        self.logger.info("Shutting down tool registry")
        self.tools.clear()
