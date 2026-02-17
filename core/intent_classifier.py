"""
Intent classifier utilities for AutoBot.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional


class IntentClassifier:
    """Classifies user intents for high-level routing metadata."""

    INTENT_CATEGORIES = [
        "latest_information",
        "knowledge_query",
        "send_email",
        "general_conversation",
    ]

    def __init__(self, config: Dict[str, Any], llm_interface):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = llm_interface

    async def classify(self, user_input: str) -> Dict[str, Any]:
        """Classify user input and return normalized structured JSON."""
        system_prompt = (
            "You are AutoBot intent+goal classifier. "
            "Return valid JSON only with keys: "
            "intent, goal, confidence, entities, required_tools, safety_flags. "
            "intent must be one of: latest_information, knowledge_query, send_email, general_conversation."
        )
        prompt = (
            f'Classify intent for user input: "{user_input}"\n\n'
            "Rules:\n"
            "- Use latest_information for current/live/latest facts, prices, news.\n"
            "- Use send_email only if user asks to send email.\n"
            "- Use knowledge_query for explanation/summarization from existing knowledge/docs (non-latest).\n"
            "- Use general_conversation for greetings/chit-chat/simple talk.\n\n"
            'Return JSON: {"intent":"latest_information|knowledge_query|send_email|general_conversation",'
            '"goal":"one short sentence","confidence":0.0,'
            '"entities":[],"required_tools":[],"safety_flags":[]}'
        )

        response = await self.llm.generate_with_intent_model(prompt, system_prompt)
        parsed = self._extract_json_object(response)
        if isinstance(parsed, dict):
            return self._normalize(parsed)

        relaxed = self._extract_relaxed_fields(response)
        if isinstance(relaxed, dict):
            return self._normalize(relaxed)

        self.logger.warning(
            "Failed to parse JSON from LLM response (ascii-safe): %s",
            self._safe_log_text(response),
        )
        return self._default_result()

    def _normalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        intent = str(result.get("intent", "general_conversation")).strip().lower()
        aliases = {
            "web_search": "latest_information",
            "web": "latest_information",
            "search": "latest_information",
            "rag": "knowledge_query",
            "local_docs": "knowledge_query",
            "local_doc": "knowledge_query",
            "email": "send_email",
            "general_converse": "general_conversation",
            "general conversation": "general_conversation",
            "conversation": "general_conversation",
        }
        intent = aliases.get(intent, intent)
        if intent not in self.INTENT_CATEGORIES:
            intent = "knowledge_query"

        goal = str(result.get("goal", "")).strip()
        if not goal:
            goal = "Understand the user request and respond accurately."

        confidence = result.get("confidence", 0.5)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.5
        confidence_value = max(0.0, min(1.0, confidence_value))

        entities = result.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        required_tools = result.get("required_tools", [])
        if not isinstance(required_tools, list):
            required_tools = []

        safety_flags = result.get("safety_flags", [])
        if not isinstance(safety_flags, list):
            safety_flags = []

        return {
            "intent": intent,
            "goal": goal,
            "confidence": confidence_value,
            "entities": entities,
            "required_tools": required_tools,
            "safety_flags": safety_flags,
        }

    def _extract_json_object(self, text: Any) -> Optional[Dict[str, Any]]:
        if text is None:
            return None
        raw = str(text).strip()
        if not raw:
            return None

        if raw.startswith("{") and raw.endswith("}"):
            try:
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(raw):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(raw[idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    def _safe_log_text(self, text: Any, limit: int = 600) -> str:
        raw = str(text or "")
        safe = raw.encode("ascii", errors="backslashreplace").decode("ascii", errors="ignore")
        if len(safe) <= limit:
            return safe
        return safe[:limit] + "... [truncated]"

    def _extract_relaxed_fields(self, text: Any) -> Optional[Dict[str, Any]]:
        raw = str(text or "")
        if not raw:
            return None

        intent_match = re.search(r'["\']?intent["\']?\s*:\s*["\']([^"\'}\n]+)', raw, flags=re.IGNORECASE)
        goal_match = re.search(r'["\']?goal["\']?\s*:\s*["\']([^"\'}\n]+)', raw, flags=re.IGNORECASE)
        confidence_match = re.search(r'["\']?confidence["\']?\s*:\s*([0-9]*\.?[0-9]+)', raw, flags=re.IGNORECASE)

        if not intent_match and not goal_match and not confidence_match:
            return None

        payload: Dict[str, Any] = {}
        if intent_match:
            payload["intent"] = intent_match.group(1).strip()
        if goal_match:
            payload["goal"] = goal_match.group(1).strip()
        if confidence_match:
            payload["confidence"] = confidence_match.group(1).strip()
        payload.setdefault("entities", [])
        payload.setdefault("required_tools", [])
        payload.setdefault("safety_flags", [])
        return payload

    def _default_result(self) -> Dict[str, Any]:
        return {
            "intent": "knowledge_query",
            "goal": "Understand the request and provide the most relevant response.",
            "confidence": 0.3,
            "entities": [],
            "required_tools": [],
            "safety_flags": [],
        }
