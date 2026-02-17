"""
Intent Classifier for AutoBot
Classifies user input into predefined intent categories using Phi-3.5 model.
"""

import json
import logging
from typing import Dict, Any

class IntentClassifier:
    """Classifies user intents for routing to appropriate modules using LLM."""

    INTENT_CATEGORIES = [
        "web_search",          # For information requests, analysis, queries, etc.
        "general_conversation" # For non-actionable intents
    ]

    def __init__(self, config: Dict[str, Any], llm_interface):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = llm_interface

    async def classify(self, user_input: str) -> Dict[str, Any]:
        """Classify the intent of the user input and return structured JSON."""
        system_prompt = """You are AutoBot's intent classifier. Analyze user input and classify it into one of these categories:

**Intent Categories:**
1. **web_search** - For information gathering and research
   - Keywords: what, who, when, where, how, explain, search, find, analyze, check, research
   - Tools used: Web search engine with content cleaning
   - Expected output: Optimized search query and structured results

2. **general_conversation** - For greetings, opinions, and general chat
   - Keywords: hello, hi, thank you, how are you, tell me, discuss
   - No external tools needed
   - Expected output: Direct conversation response

**Output Format:**
Return valid JSON: {"intent": "string", "confidence": 0.0-1.0, "entities": [], "required_tools": [], "safety_flags": []}"""

        prompt = f"""Classify intent for user input: "{user_input}"

Return JSON: {{"intent": "string", "confidence": float, "entities": [], "required_tools": [], "safety_flags": [], "reasoning": "brief explanation"}}"""

        response = await self.llm.generate_with_intent_model(prompt, system_prompt)
        
        try:
            # Try to parse JSON from response
            # Look for JSON block in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                # Validate required fields
                if 'intent' not in result:
                    result['intent'] = 'general_conversation'
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                if 'entities' not in result:
                    result['entities'] = []
                if 'required_tools' not in result:
                    result['required_tools'] = []
                if 'safety_flags' not in result:
                    result['safety_flags'] = []
                return result
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON from LLM response: {response}")
        
        # Fallback to rule-based classification
        intent = self._rule_based_classify(user_input)
        return {
            "intent": intent,
            "confidence": 0.7,
            "entities": [],
            "required_tools": [],
            "safety_flags": []
        }

    def _rule_based_classify(self, user_input: str) -> str:
        """Fallback rule-based classification."""
        input_lower = user_input.lower()

        # Web search intents: information requests, analysis, monitoring, help, research
        if any(word in input_lower for word in ["what", "who", "when", "where", "how", "tell me", "explain", "show me",
                                                  "search", "find", "look for", "analyze", "check", "monitor", "watch", "status", 
                                                  "performance", "research", "help", "assist", "guide", "tutorial", "how to"]):
            return "web_search"

        # Greetings and general conversation
        if any(word in input_lower for word in ["hello", "hi", "hey", "good morning", "good evening", "nice to meet", "thank", "bye", "goodbye"]):
            return "general_conversation"

        # Default to web_search for unknown inputs
        return "web_search"
