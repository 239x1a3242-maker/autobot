"""
Planner Module for AutoBot
Creates execution plans for complex user requests using Phi-4-Reasoning.
"""

import json
import logging
from typing import Dict, Any

class Planner:
    """Plans multi-step actions based on user intent using LLM."""

    def __init__(self, config: Dict[str, Any], llm_interface):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = llm_interface

    async def create_plan(self, user_goal: str, intent_json: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an execution plan for the given goal using Phi-4."""
        
        # Select system prompt based on detected intent
        intent = intent_json.get('intent', 'general_conversation')
        
        if intent == 'web_search':
            system_prompt = """You are AutoBot's planning agent for web search tasks.

**Web Search Tool Output Format:**
The web_search tool returns:
{
  "query": "optimized search query",
  "parameters": {"max_results": 2, "workers": 6},
  "stats": {"search_engine": {...}, "cleaner": {...}},
  "structured_results": [
    {
      "url": "string",
      "title": "string", 
      "content": "summarized content",
      "extraction_status": "success|failed",
      "relevance_score": 0.0-1.0
    }
  ]
}

**Your task:** Create a plan that:
1. Generates an optimized search query for the web_search tool
2. Specifies parameters (max_results, workers count)
3. Defines how to process and present the search results

Return JSON plan with steps that use the web_search tool."""

        else:  # general_conversation
            system_prompt = """You are AutoBot's planning agent for general conversation and analysis tasks.

**Your task:** Create a plan that:
1. Identifies what information or analysis is needed
2. Breaks down complex requests into manageable steps
3. Uses LLM reasoning for analysis and responses
4. Optionally combines LLM responses with tool usage if needed

Return JSON plan with steps that use the LLM model for reasoning and conversation."""

        prompt = f"""Create an execution plan for: {user_goal}
Intent: {intent}
Intent info: {json.dumps(intent_json)}

Return JSON plan: {{"plan_id": "string", "steps": [{{"id": "string", "purpose": "string", "inputs": [], "tool_or_model": "web_search|llm", "success_criteria": "string", "timeout_sec": 30}}]}}

Ensure each step has the correct tool_or_model specified based on the intent."""

        response = await self.llm.generate_with_reasoning_model(prompt, system_prompt)
        
        try:
            # Parse JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                plan = json.loads(json_str)
                # Validate structure
                if 'plan_id' not in plan:
                    plan['plan_id'] = f"plan_{hash(user_goal) % 10000}"
                if 'steps' not in plan:
                    plan['steps'] = []
                for step in plan['steps']:
                    if 'id' not in step:
                        step['id'] = f"step_{plan['steps'].index(step)}"
                    if 'purpose' not in step:
                        step['purpose'] = ""
                    if 'inputs' not in step:
                        step['inputs'] = []
                    if 'tool_or_model' not in step:
                        step['tool_or_model'] = "llm"
                    elif step['tool_or_model'] not in ["web_search", "llm"]:
                        step['tool_or_model'] = "llm"
                    if 'success_criteria' not in step:
                        step['success_criteria'] = "completed successfully"
                    if 'timeout_sec' not in step:
                        step['timeout_sec'] = 30
                return plan
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse plan JSON: {response}")
        
        # Fallback to simple plan
        return self._create_simple_plan(user_goal, intent_json.get('intent', 'general_conversation'))

    def _create_simple_plan(self, user_goal: str, intent: str) -> Dict[str, Any]:
        """Create a simple fallback plan based on intent with tool-specific details."""
        plan = {
            "plan_id": f"simple_{hash(user_goal) % 10000}",
            "steps": []
        }

        if intent == "web_search":
            plan["steps"].append({
                "id": "web_search_task",
                "purpose": "Search and gather information using web search tool",
                "inputs": [user_goal],
                "tool_or_model": "web_search",
                "success_criteria": "Search results retrieved and cleaned",
                "timeout_sec": 30
            })
            plan["tool_output_format"] = "JSON with structured_results containing URLs, titles, and content"
        else:
            # Default to LLM for general conversation and unknown intents
            plan["steps"].append({
                "id": "general_task",
                "purpose": "Process user request using language model",
                "inputs": [user_goal],
                "tool_or_model": "llm",
                "success_criteria": "Task completed with a helpful response",
                "timeout_sec": 30
            })

        return plan
