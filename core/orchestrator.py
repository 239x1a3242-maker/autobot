"""
AutoBot single-agent orchestrator.

This orchestrator intentionally supports only one architecture:
- tool_use

Removed flows:
- react
- planning
- multi_agent
- pev
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.intent_classifier import IntentClassifier
from core.llm_interface import LLMInterface
from interfaces.text_interface import TextInterface
from interfaces.voice_interface import VoiceInterface
from memory.memory_manager import MemoryManager
from tools.tool_registry import ToolRegistry, tools_json as TOOL_SCHEMA

try:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except Exception:
    PydanticOutputParser = None
    PromptTemplate = None
    RunnableLambda = None
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

try:
    from langchain_community.document_loaders import (
        BSHTMLLoader,
        CSVLoader,
        PyPDFLoader,
        TextLoader,
    )
except Exception:
    BSHTMLLoader = None
    CSVLoader = None
    PyPDFLoader = None
    TextLoader = None


class ToolDecision(BaseModel):
    action: str = Field(default="final", description="tool or final")
    thought: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""


@dataclass
class ToolUseResult:
    response: str
    attempts: int
    tool_steps: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    """Single tool-use orchestrator for AutoBot."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.memory = MemoryManager(config)
        self.llm = LLMInterface(config)
        self.intent_classifier = IntentClassifier(config, self.llm)
        self.tool_registry = ToolRegistry(config)

        self.text_interface = None
        if config.get("interfaces", {}).get("text", {}).get("enabled", False):
            try:
                self.text_interface = TextInterface(config)
            except Exception as exc:
                self.logger.warning("Text interface disabled due to initialization error: %s", exc)

        self.voice_interface = None
        if config.get("interfaces", {}).get("voice", {}).get("enabled", False):
            try:
                self.voice_interface = VoiceInterface(config)
            except Exception as exc:
                self.logger.warning("Voice interface disabled due to initialization error: %s", exc)

        self.max_retries = 3
        self.max_agent_steps = int(
            config.get("agentic", {}).get(
                "tool_use_max_steps",
                config.get("agentic", {}).get("react_max_iterations", 4),
            )
        )
        self.debug_enabled = bool(config.get("debug", {}).get("enabled", True))

        self.execution_trace: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, str]] = []

        self.project_root = Path(__file__).resolve().parents[1]
        self.log_file = self._resolve_log_file()

        self.langchain_enabled = bool(config.get("agentic", {}).get("langchain", {}).get("enabled", True))
        lc_cfg = config.get("agentic", {}).get("langchain", {})
        self.langchain_chunk_size = int(lc_cfg.get("chunk_size", 900))
        self.langchain_chunk_overlap = int(lc_cfg.get("chunk_overlap", 120))
        self.langchain_use_local_doc_loader = bool(lc_cfg.get("use_local_doc_loader", True))
        self.local_doc_max_files = int(lc_cfg.get("local_doc_max_files", 10))
        self.local_doc_max_chunks = int(lc_cfg.get("local_doc_max_chunks", 8))
        self.local_doc_max_chars = int(lc_cfg.get("local_doc_max_chars", 7000))
        self.local_doc_default_paths = lc_cfg.get("local_documents", [])
        self.langchain_ready = False
        self.decision_parser = None
        self.decision_chain = None
        self.decision_prompt = None
        self.synthesis_chain = None
        self.synthesis_prompt = None

    async def run_with_interface(self, interface):
        """Run AutoBot with the selected user interface."""
        self.logger.info("AutoBot orchestrator starting (single tool_use flow)")

        await self.memory.initialize()
        await self.tool_registry.initialize()
        await self.llm.initialize()
        self._build_langchain_components()

        await interface.run(self.handle_input)
        self.logger.info("AutoBot orchestrator shutting down")

    async def handle_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Main request pipeline using one tool-use agent only."""
        context = context or {}
        start_ts = time.time()
        trace_id = f"tool_use_{int(start_ts * 1000)}"

        try:
            await self.memory.add_short_term(user_input)
            intent = await self._safe_classify_intent(user_input)
            local_knowledge = await self._prepare_local_knowledge_context(user_input, context, intent)

            result = await self._run_tool_use_with_retries(
                user_input=user_input,
                context=context,
                intent=intent,
                local_knowledge=local_knowledge,
            )

            response = self._clean_text(result.response)
            if not response:
                response = "I could not generate a response."

            await self.memory.add_interaction(user_input, response, intent.get("intent", "unknown"))
            self.chat_history.append({"user": user_input, "assistant": response})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

            self.execution_trace.append(
                {
                    "trace_id": trace_id,
                    "flow": "tool_use",
                    "attempts": result.attempts,
                    "tool_steps": result.tool_steps,
                    "failures": result.failures,
                    "local_docs": local_knowledge.get("sources", []),
                    "duration_sec": round(time.time() - start_ts, 3),
                    "timestamp": time.time(),
                }
            )
            if len(self.execution_trace) > 200:
                self.execution_trace = self.execution_trace[-200:]

            return response

        except Exception as exc:
            self.logger.exception("handle_input failed: %s", exc)
            return "I encountered an error while processing your request."

    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible method retained for callers expecting this API."""
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        return {
            "status": "single_tool_use_mode",
            "message": "Planning flow is disabled. Using tool_use flow only.",
            "steps": steps,
        }

    async def generate_response(
        self,
        user_input: str,
        execution_results: Dict[str, Any],
        intent_json: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> str:
        """Backward-compatible method retained for callers expecting this API."""
        del execution_results
        del intent_json
        del plan
        return await self.handle_input(user_input)

    async def get_debug_status(self, limit: int = 20) -> Dict[str, Any]:
        """Expose minimal debugging information."""
        return {
            "flow": "tool_use",
            "langchain_ready": self.langchain_ready,
            "loaded_models": self.llm.get_loaded_models(),
            "tools": sorted(self.tool_registry.tools.keys()),
            "recent_trace": self.execution_trace[-max(1, limit):],
        }

    def _build_langchain_components(self):
        """Build LangChain parser/prompt/runnable chain for tool-use decisions."""
        if not (self.langchain_enabled and LANGCHAIN_AVAILABLE):
            self.langchain_ready = False
            return

        try:
            self.decision_parser = PydanticOutputParser(pydantic_object=ToolDecision)
            tool_manifest = self._tool_manifest_text()
            decision_tool_schema = list(TOOL_SCHEMA) + [
                {
                    "name": "local_docs_rag",
                    "description": "Use local documents for static/non-latest knowledge and summarize with RAG.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Knowledge query for local docs"}
                        },
                        "required": ["query"],
                    },
                }
            ]
            tool_schema_json = json.dumps(decision_tool_schema, indent=2, default=str)
            self.decision_prompt = PromptTemplate.from_template(
                """
You are AutoBot's STRICT tool-use controller.
Choose exactly one next action: "tool" or "final".

Allowed tools:
{tool_manifest}

Tool JSON schema:
{tool_schema_json}
Reference docs: tools/TOOLS.md and tools/send-email/SEND_EMAIL_TOOL.md

Hard Rules:
1) Use only tools listed above.
2) Use first-turn intent+goal to guide decision.
3) If request is greeting or general conversation, do NOT call tools. Use action="final".
4) For latest/current/live data use action="tool" with tool_name="web_search".
5) For email sending use action="tool" with tool_name="send_email".
6) For non-latest knowledge/information from local documents use action="tool" with tool_name="local_docs_rag".
7) If scratchpad already contains enough data to answer, use action="final".
8) For tool action, include structured tool_input JSON only; no prose.
9) send_email tool_input requires: to, subject, body.
10) web_search tool_input requires: query.
11) local_docs_rag tool_input requires: query.
12) Return valid JSON only.

Intent and Goal (first turn):
{intent_json}

Local Knowledge Context (from local docs):
{local_knowledge_context}

User Input:
{user_input}

Conversation Context:
{conversation_context}

Scratchpad (previous tool observations):
{scratchpad}

Failure Context from Logs (if any):
{failure_context}

{format_instructions}
                """.strip(),
                partial_variables={
                    "format_instructions": self.decision_parser.get_format_instructions(),
                    "tool_manifest": tool_manifest,
                    "tool_schema_json": tool_schema_json,
                },
            )

            self.decision_chain = (
                self.decision_prompt
                | RunnableLambda(self._decision_model_invoke)
                | RunnableLambda(self._parse_tool_decision)
            ).with_retry(stop_after_attempt=3)

            self.synthesis_prompt = PromptTemplate.from_template(
                """
You are AutoBot final-response synthesizer.
Use tool observations to produce a structured, query-focused answer.

Rules:
1) If tool output contains errors, clearly state what failed.
2) Use only information present in observations, local knowledge, and user query.
3) Response format:
- Summary
- Structured Findings
- Predictions / Recommendations (only if applicable)
4) Keep response concise and useful.

User Query:
{user_input}

Tool Observations (JSON):
{scratchpad_json}

Local Knowledge Context:
{local_knowledge_context}
                """.strip()
            )
            self.synthesis_chain = (
                self.synthesis_prompt
                | RunnableLambda(self._synthesis_model_invoke)
                | RunnableLambda(self._clean_text)
            ).with_retry(stop_after_attempt=3)

            self.langchain_ready = True
            self.logger.info("LangChain tool-use decision chain initialized")
        except Exception as exc:
            self.langchain_ready = False
            self.logger.exception("Failed to initialize LangChain decision chain: %s", exc)

    async def _safe_classify_intent(self, user_input: str) -> Dict[str, Any]:
        try:
            return await self.intent_classifier.classify(user_input)
        except Exception as exc:
            self.logger.warning("Intent classification failed: %s", exc)
            return {
                "intent": "knowledge_query",
                "goal": "Understand the request and provide a useful response.",
                "confidence": 0.0,
                "entities": [],
                "required_tools": [],
                "safety_flags": [],
            }

    async def _run_tool_use_with_retries(
        self,
        user_input: str,
        context: Dict[str, Any],
        intent: Dict[str, Any],
        local_knowledge: Dict[str, Any],
    ) -> ToolUseResult:
        failures: List[Dict[str, Any]] = []
        failure_context = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                response, tool_steps = await self._run_tool_use_once(
                    user_input=user_input,
                    context=context,
                    intent=intent,
                    failure_context=failure_context,
                    local_knowledge=local_knowledge,
                )
                return ToolUseResult(
                    response=response,
                    attempts=attempt,
                    tool_steps=tool_steps,
                    failures=failures,
                )
            except Exception as exc:
                failure_context = self._read_recent_error_logs(limit=60)
                failure_record = {
                    "attempt": attempt,
                    "error": str(exc),
                    "log_excerpt": failure_context,
                    "timestamp": time.time(),
                }
                failures.append(failure_record)
                self.logger.exception("tool_use attempt %s failed: %s", attempt, exc)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.25 * attempt)

        latest_log_excerpt = ""
        if failures:
            latest_log_excerpt = self._truncate_text(failures[-1].get("log_excerpt", ""), 1200)

        fallback_message = (
            "I could not complete the request after 3 retries. "
            "I attempted replanning using failure logs each retry.\n"
            "Please check logs/autobot.log for detailed traces."
        )
        if latest_log_excerpt:
            fallback_message += f"\n\nFailure log excerpt:\n{latest_log_excerpt}"
        return ToolUseResult(
            response=fallback_message,
            attempts=self.max_retries,
            tool_steps=[],
            failures=failures,
        )

    async def _run_tool_use_once(
        self,
        user_input: str,
        context: Dict[str, Any],
        intent: Dict[str, Any],
        failure_context: str,
        local_knowledge: Dict[str, Any],
    ) -> tuple[str, List[Dict[str, Any]]]:
        del context

        scratchpad: List[Dict[str, Any]] = []

        for step_idx in range(1, self.max_agent_steps + 1):
            decision = await self._decide_next_action(
                user_input=user_input,
                scratchpad=scratchpad,
                failure_context=failure_context,
                local_knowledge=local_knowledge,
                intent=intent,
            )

            action = (decision.action or "").strip().lower()
            if action == "final":
                final_answer = self._clean_text(decision.final_answer)
                if final_answer:
                    return final_answer, scratchpad
                break

            if action != "tool":
                raise RuntimeError(f"Unsupported action from controller: {decision.action}")

            normalized_tool_name = self._normalize_tool_name(decision.tool_name)
            normalized_input = self._coerce_tool_input(
                tool_name=normalized_tool_name,
                tool_input=decision.tool_input,
                user_input=user_input,
            )
            if normalized_tool_name == "local_docs_rag":
                local_query = self._clean_text(normalized_input.get("query")) or user_input
                local_text = self._truncate_text(local_knowledge.get("text", ""), self.local_doc_max_chars)
                if not local_text:
                    scratchpad.append(
                        {
                            "step": step_idx,
                            "tool": normalized_tool_name,
                            "input": normalized_input,
                            "output": "No relevant local document context available.",
                            "status": "error",
                        }
                    )
                    continue

                rag_summary = await self._summarize_with_rag(
                    query=local_query,
                    tool_name="local_docs_rag",
                    raw_observation=local_text,
                )
                scratchpad.append(
                    {
                        "step": step_idx,
                        "tool": normalized_tool_name,
                        "input": normalized_input,
                        "output": self._truncate_text(rag_summary, 4000),
                        "status": "ok",
                    }
                )
                continue

            if normalized_tool_name not in self.tool_registry.tools:
                scratchpad.append(
                    {
                        "step": step_idx,
                        "tool": normalized_tool_name,
                        "input": normalized_input,
                        "output": f"Tool '{normalized_tool_name}' is unavailable.",
                        "status": "error",
                    }
                )
                continue

            tool_output = await self.tool_registry.execute_tool(
                normalized_tool_name,
                normalized_input,
            )
            if normalized_tool_name == "web_search":
                tool_output = await self._summarize_with_rag(
                    query=user_input,
                    tool_name="web_search",
                    raw_observation=tool_output,
                )
            scratchpad.append(
                {
                    "step": step_idx,
                    "tool": normalized_tool_name,
                    "input": normalized_input,
                    "output": self._truncate_text(tool_output, 4000),
                    "status": "ok",
                }
            )

        synthesized = await self._synthesize_response(
            user_input=user_input,
            scratchpad=scratchpad,
            local_knowledge=local_knowledge,
        )
        return synthesized, scratchpad

    async def _decide_next_action(
        self,
        user_input: str,
        scratchpad: List[Dict[str, Any]],
        failure_context: str,
        local_knowledge: Dict[str, Any],
        intent: Dict[str, Any],
    ) -> ToolDecision:
        local_knowledge_context = self._truncate_text(local_knowledge.get("text", ""), self.local_doc_max_chars)
        intent_name = self._clean_text((intent or {}).get("intent", "")).lower()

        if not scratchpad:
            if intent_name == "general_conversation":
                return ToolDecision(action="final", thought="Intent route: general conversation", final_answer="")
            if intent_name == "send_email":
                return ToolDecision(
                    action="tool",
                    thought="Intent route: send email",
                    tool_name="send_email",
                    tool_input=self._coerce_tool_input("send_email", {}, user_input),
                    final_answer="",
                )
            if intent_name == "latest_information":
                return ToolDecision(
                    action="tool",
                    thought="Intent route: latest information",
                    tool_name="web_search",
                    tool_input=self._coerce_tool_input("web_search", {}, user_input),
                    final_answer="",
                )
            if intent_name == "knowledge_query" and local_knowledge_context:
                return ToolDecision(
                    action="tool",
                    thought="Intent route: local knowledge",
                    tool_name="local_docs_rag",
                    tool_input=self._coerce_tool_input("local_docs_rag", {}, user_input),
                    final_answer="",
                )

        if not self.langchain_ready or self.decision_chain is None:
            return await self._fallback_decision(
                user_input=user_input,
                scratchpad=scratchpad,
                local_knowledge_context=local_knowledge_context,
                intent=intent,
            )

        conversation_context = self._conversation_context_text()
        payload = {
            "user_input": user_input,
            "intent_json": json.dumps(intent or {}, indent=2, default=str),
            "conversation_context": conversation_context,
            "scratchpad": json.dumps(scratchpad, indent=2, default=str) if scratchpad else "(empty)",
            "failure_context": failure_context or "(none)",
            "local_knowledge_context": local_knowledge_context or "(none)",
        }
        try:
            decision = await self.decision_chain.ainvoke(payload)
            if isinstance(decision, ToolDecision):
                normalized = self._normalize_decision(decision, user_input)
                return self._apply_decision_guards(normalized, user_input, intent, local_knowledge_context)
            if isinstance(decision, dict):
                normalized = self._normalize_decision(ToolDecision(**decision), user_input)
                return self._apply_decision_guards(normalized, user_input, intent, local_knowledge_context)
        except Exception as exc:
            self.logger.warning("LangChain decision failed, using model fallback decision: %s", exc)
        fallback = await self._fallback_decision(
            user_input=user_input,
            scratchpad=scratchpad,
            local_knowledge_context=local_knowledge_context,
            intent=intent,
        )
        return self._apply_decision_guards(fallback, user_input, intent, local_knowledge_context)

    async def _decision_model_invoke(self, prompt_value: Any) -> str:
        prompt_text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return await self.llm.generate_with_model(
            "autobot_instruct",
            prompt_text,
            system_prompt=(
                "You are a strict tool-routing policy engine. "
                "Return only JSON that follows the schema."
            ),
            temperature=0.0,
            max_tokens=320,
        )

    async def _synthesis_model_invoke(self, prompt_value: Any) -> str:
        prompt_text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return await self.llm.generate_with_model(
            "autobot_instruct",
            prompt_text,
            system_prompt=(
                "You are AutoBot final answer model. Use summarized tool observations and answer clearly. "
                "Do not reveal chain-of-thought."
            ),
            temperature=0.1,
            max_tokens=900,
        )

    def _parse_tool_decision(self, model_output: Any) -> ToolDecision:
        if self.decision_parser is None:
            raise RuntimeError("Decision parser is not initialized")

        text = str(model_output)
        try:
            parsed = self.decision_parser.parse(text)
            if isinstance(parsed, ToolDecision):
                return parsed
            if hasattr(parsed, "model_dump"):
                return ToolDecision(**parsed.model_dump())
            if isinstance(parsed, dict):
                return ToolDecision(**parsed)
        except Exception:
            pass

        extracted = self._extract_json_object(text)
        if isinstance(extracted, dict):
            return ToolDecision(**extracted)
        raise ValueError("Unable to parse tool decision")

    def _normalize_decision(self, decision: ToolDecision, user_input: str) -> ToolDecision:
        action = self._clean_text(decision.action).lower()
        tool_name = self._normalize_tool_name(decision.tool_name)
        final_answer = self._clean_text(decision.final_answer)

        if action in {"final", "final_answer", "answer", "response", "respond", "no_tool"}:
            return ToolDecision(
                action="final",
                thought=decision.thought,
                tool_name="",
                tool_input={},
                final_answer=final_answer,
            )

        if action in {"tool", "tool_call", "call_tool", "use_tool", "tool_use"}:
            if not tool_name:
                tool_name = "web_search"
            return ToolDecision(
                action="tool",
                thought=decision.thought,
                tool_name=tool_name,
                tool_input=decision.tool_input or {},
                final_answer="",
            )

        action_as_tool = self._normalize_tool_name(action)
        if action_as_tool in {"web_search", "send_email", "local_docs_rag"}:
            return ToolDecision(
                action="tool",
                thought=decision.thought,
                tool_name=action_as_tool,
                tool_input=decision.tool_input or {},
                final_answer="",
            )

        if tool_name in {"web_search", "send_email", "local_docs_rag"}:
            return ToolDecision(
                action="tool",
                thought=decision.thought,
                tool_name=tool_name,
                tool_input=decision.tool_input or {},
                final_answer="",
            )

        if final_answer:
            return ToolDecision(
                action="final",
                thought=decision.thought,
                tool_name="",
                tool_input={},
                final_answer=final_answer,
            )

        return ToolDecision(action="final", thought="Normalized to final", final_answer="")

    def _apply_decision_guards(
        self,
        decision: ToolDecision,
        user_input: str,
        intent: Dict[str, Any],
        local_knowledge_context: str,
    ) -> ToolDecision:
        intent_name = self._clean_text((intent or {}).get("intent", "")).lower()
        lowered = user_input.lower()

        if decision.action == "tool":
            tool_name = self._normalize_tool_name(decision.tool_name)

            if intent_name == "general_conversation":
                return ToolDecision(action="final", thought="General conversation guard", final_answer="")

            if tool_name == "send_email":
                asked_email = ("send email" in lowered) or ("email to" in lowered) or bool(
                    re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_input)
                )
                if not asked_email:
                    return ToolDecision(action="final", thought="Email tool guard", final_answer="")

            if tool_name == "local_docs_rag" and not local_knowledge_context:
                return ToolDecision(action="final", thought="No local docs context available", final_answer="")

            if tool_name == "web_search" and intent_name == "knowledge_query" and local_knowledge_context:
                return ToolDecision(
                    action="tool",
                    thought="Use local docs for non-latest knowledge",
                    tool_name="local_docs_rag",
                    tool_input={"query": user_input},
                    final_answer="",
                )

        return decision

    async def _summarize_with_rag(self, query: str, tool_name: str, raw_observation: Any) -> str:
        observation_text = self._truncate_text(raw_observation, 12000)
        prompt = (
            f"User query:\n{query}\n\n"
            f"Tool used: {tool_name}\n\n"
            f"Raw tool output:\n{observation_text}\n\n"
            "Summarize into structured findings with only relevant facts for final answer."
        )
        return await self.llm.generate_with_rag_model(
            prompt,
            system_prompt=(
                "You are AutoBot RAG summarizer. Summarize tool observations accurately. "
                "Do not add facts not present in tool output."
            ),
        )

    async def _fallback_decision(
        self,
        user_input: str,
        scratchpad: List[Dict[str, Any]],
        local_knowledge_context: str,
        intent: Dict[str, Any],
    ) -> ToolDecision:
        prompt = (
            "Return one JSON object with fields: action, thought, tool_name, tool_input, final_answer.\n"
            "action must be exactly 'tool' or 'final'.\n"
            f"Intent JSON: {json.dumps(intent or {}, default=str)}\n"
            f"User input: {user_input}\n"
            f"Scratchpad: {json.dumps(scratchpad, default=str)}\n"
            f"Local knowledge context: {self._truncate_text(local_knowledge_context, 1500)}\n"
        )
        try:
            raw = await self.llm.generate_with_model(
                "autobot_instruct",
                prompt,
                system_prompt="You are a strict decision model. Return valid JSON only.",
                temperature=0.0,
                max_tokens=220,
            )
            extracted = self._extract_json_object(raw)
            if isinstance(extracted, dict):
                return self._normalize_decision(ToolDecision(**extracted), user_input)
        except Exception as exc:
            self.logger.warning("Model fallback decision failed: %s", exc)
        return ToolDecision(action="final", thought="Fallback default", final_answer="")

    async def _synthesize_response(
        self,
        user_input: str,
        scratchpad: List[Dict[str, Any]],
        local_knowledge: Dict[str, Any],
    ) -> str:
        local_knowledge_context = self._truncate_text(local_knowledge.get("text", ""), self.local_doc_max_chars)

        if not scratchpad:
            prompt = f"User query:\n{user_input}\n\n"
            if local_knowledge_context:
                prompt += f"Local knowledge context (if relevant):\n{local_knowledge_context}\n\n"
            prompt += "Provide a concise final response."
            return await self.llm.generate_with_intent_model(
                prompt,
                system_prompt=(
                    "You are AutoBot final response model. "
                    "Answer clearly and concisely. Use local context only when relevant."
                ),
            )

        if self.langchain_ready and self.synthesis_chain is not None:
            return await self.synthesis_chain.ainvoke(
                {
                    "user_input": user_input,
                    "scratchpad_json": json.dumps(scratchpad, indent=2, default=str),
                    "local_knowledge_context": local_knowledge_context or "(none)",
                }
            )

        synthesis_prompt = (
            "User query:\n"
            f"{user_input}\n\n"
            "Summarized tool observations (JSON):\n"
            f"{json.dumps(scratchpad, indent=2, default=str)}\n\n"
            "Local knowledge context:\n"
            f"{local_knowledge_context or '(none)'}\n\n"
            "Generate the final answer for the user."
        )
        return await self.llm.generate_with_model(
            "autobot_instruct",
            synthesis_prompt,
            system_prompt="You are AutoBot final answer model. Use summarized observations and respond clearly.",
            temperature=0.1,
            max_tokens=900,
        )

    async def _prepare_local_knowledge_context(
        self,
        user_input: str,
        context: Dict[str, Any],
        intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.langchain_use_local_doc_loader:
            return {"text": "", "sources": [], "errors": []}

        explicit_local_docs = bool(context.get("local_documents"))
        intent_name = self._clean_text((intent or {}).get("intent", "")).lower()
        if not explicit_local_docs and intent_name != "knowledge_query":
            return {"text": "", "sources": [], "errors": []}

        paths = self._collect_local_document_paths(context)
        if not paths:
            return {"text": "", "sources": [], "errors": []}

        docs, errors = await asyncio.to_thread(self._load_local_documents_sync, paths)
        if not docs:
            return {"text": "", "sources": [], "errors": errors}

        chunks = self._split_local_documents(docs)
        selected = self._select_relevant_chunks(user_input, chunks, self.local_doc_max_chunks)
        if not selected and not explicit_local_docs:
            return {"text": "", "sources": [], "errors": errors}

        text = self._format_local_knowledge(selected, self.local_doc_max_chars)
        if text and not self._local_context_is_relevant(user_input, text) and not explicit_local_docs:
            return {"text": "", "sources": [], "errors": errors}

        sources = sorted({item.get("source", "") for item in selected if item.get("source")})
        return {"text": text, "sources": sources, "errors": errors}

    def _collect_local_document_paths(self, context: Dict[str, Any]) -> List[Path]:
        candidates: List[Any] = []
        from_context = context.get("local_documents", [])
        if isinstance(from_context, (str, Path)):
            candidates.append(from_context)
        elif isinstance(from_context, list):
            candidates.extend(from_context)

        from_config = self.local_doc_default_paths
        if isinstance(from_config, (str, Path)):
            candidates.append(from_config)
        elif isinstance(from_config, list):
            candidates.extend(from_config)

        supported_suffixes = {
            ".txt",
            ".md",
            ".rst",
            ".log",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".html",
            ".htm",
            ".pdf",
            ".ipynb",
            ".py",
        }
        results: List[Path] = []
        seen = set()
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(str(candidate))
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            else:
                path = path.resolve()

            if not path.exists():
                continue
            if path.is_dir():
                for sub in path.rglob("*"):
                    if not sub.is_file():
                        continue
                    if sub.suffix.lower() not in supported_suffixes:
                        continue
                    key = str(sub)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(sub)
                    if len(results) >= self.local_doc_max_files:
                        return results
                continue

            if path.suffix.lower() in supported_suffixes:
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    results.append(path)
                    if len(results) >= self.local_doc_max_files:
                        return results
        return results

    def _load_local_documents_sync(self, paths: List[Path]) -> tuple[List[Dict[str, str]], List[str]]:
        loaded_docs: List[Dict[str, str]] = []
        errors: List[str] = []

        for path in paths[: self.local_doc_max_files]:
            try:
                docs = self._load_single_local_document(path)
                for item in docs:
                    content = self._clean_text(item.get("content"))
                    if not content:
                        continue
                    loaded_docs.append({"source": str(path), "content": content})
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        return loaded_docs, errors

    def _load_single_local_document(self, path: Path) -> List[Dict[str, str]]:
        suffix = path.suffix.lower()

        if suffix == ".ipynb":
            raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            cells = raw.get("cells", []) if isinstance(raw, dict) else []
            docs: List[Dict[str, str]] = []
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                source = cell.get("source", [])
                if isinstance(source, list):
                    content = "".join(source)
                else:
                    content = str(source or "")
                content = content.strip()
                if not content:
                    continue
                cell_type = str(cell.get("cell_type", "cell"))
                docs.append({"content": f"[{cell_type}]\n{content}"})
            return docs if docs else [{"content": ""}]

        if suffix == ".pdf" and PyPDFLoader is not None:
            docs = PyPDFLoader(str(path)).load()
            return [{"content": str(getattr(d, "page_content", ""))} for d in docs]

        if suffix == ".csv" and CSVLoader is not None:
            docs = CSVLoader(str(path)).load()
            return [{"content": str(getattr(d, "page_content", ""))} for d in docs]

        if suffix in {".html", ".htm"} and BSHTMLLoader is not None:
            docs = BSHTMLLoader(str(path)).load()
            return [{"content": str(getattr(d, "page_content", ""))} for d in docs]

        if TextLoader is not None:
            docs = TextLoader(str(path), encoding="utf-8").load()
            return [{"content": str(getattr(d, "page_content", ""))} for d in docs]

        text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"content": text}]

    def _split_local_documents(self, docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        chunks: List[Dict[str, str]] = []
        splitter = None
        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.langchain_chunk_size,
                chunk_overlap=self.langchain_chunk_overlap,
            )

        for item in docs:
            source = item.get("source", "")
            content = item.get("content", "")
            if not content:
                continue

            if splitter is None:
                chunks.append({"source": source, "content": content[: self.langchain_chunk_size]})
                continue

            for part in splitter.split_text(content):
                if part.strip():
                    chunks.append({"source": source, "content": part.strip()})
        return chunks

    def _select_relevant_chunks(
        self,
        user_input: str,
        chunks: List[Dict[str, str]],
        max_chunks: int,
    ) -> List[Dict[str, str]]:
        if not chunks:
            return []

        stopwords = {
            "the",
            "this",
            "that",
            "with",
            "from",
            "about",
            "your",
            "what",
            "when",
            "where",
            "which",
            "would",
            "could",
            "should",
            "hello",
            "thanks",
            "thank",
        }
        query_tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z0-9_]+", user_input.lower())
            if len(tok) >= 4 and tok not in stopwords
        ]
        if not query_tokens:
            return []

        scored: List[tuple[int, Dict[str, str]]] = []
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            score = 0
            for token in query_tokens:
                if len(token) < 3:
                    continue
                score += content.count(token)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item for score, item in scored if score > 0]
        return selected[:max_chunks]

    def _format_local_knowledge(self, selected: List[Dict[str, str]], max_chars: int) -> str:
        if not selected:
            return ""

        lines: List[str] = []
        remaining = max_chars
        for chunk in selected:
            source = chunk.get("source", "")
            content = chunk.get("content", "")
            if not content:
                continue
            block = f"Source: {source}\n{content}\n"
            if len(block) > remaining:
                if remaining > 80:
                    lines.append(block[:remaining])
                break
            lines.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break
        return "\n".join(lines).strip()

    def _needs_live_data(self, user_input: str) -> bool:
        lowered = user_input.lower()
        live_markers = [
            "latest",
            "today",
            "current",
            "live",
            "price",
            "news",
            "now",
            "recent",
            "trend",
            "market",
            "stock",
            "bitcoin",
            "btc",
        ]
        return any(marker in lowered for marker in live_markers)

    def _local_context_is_relevant(self, user_input: str, local_context: str) -> bool:
        if not local_context:
            return False
        stopwords = {
            "the",
            "this",
            "that",
            "with",
            "from",
            "about",
            "your",
            "what",
            "when",
            "where",
            "which",
            "would",
            "could",
            "should",
            "hello",
            "thanks",
            "thank",
        }
        query_tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z0-9_]+", user_input.lower())
            if len(tok) >= 4 and tok not in stopwords
        ]
        if not query_tokens:
            return False
        haystack = local_context.lower()
        hits = sum(1 for token in set(query_tokens) if token in haystack)
        return hits >= 2

    def _coerce_tool_input(self, tool_name: str, tool_input: Any, user_input: str) -> Dict[str, Any]:
        data = tool_input if isinstance(tool_input, dict) else {}
        normalized: Dict[str, Any] = dict(data)

        if tool_name == "web_search":
            query = self._clean_text(normalized.get("query"))
            if not query:
                query = self._clean_text(user_input)
            normalized["query"] = query
            normalized["max_results"] = int(normalized.get("max_results", 4) or 4)
            normalized["workers"] = int(normalized.get("workers", 6) or 6)
            return normalized

        if tool_name == "local_docs_rag":
            query = self._clean_text(normalized.get("query"))
            if not query:
                query = self._clean_text(user_input)
            normalized["query"] = query
            return normalized

        if tool_name == "send_email":
            to_value = self._clean_text(normalized.get("to"))
            if not to_value:
                to_value = self._extract_first_email(user_input)

            subject = self._clean_text(normalized.get("subject"))
            body = self._clean_text(normalized.get("body"))
            if not subject:
                subject = "Message from AutoBot"
            if not body:
                body = "Hello and greetings."
                match = re.search(r"(?:saying|say|message|body)\s+(.+)$", user_input, re.IGNORECASE)
                if match:
                    body = self._clean_text(match.group(1)).strip("\"' ")
            normalized["to"] = to_value
            normalized["subject"] = subject
            normalized["body"] = body
            return normalized

        return normalized

    def _extract_first_email(self, text: str) -> str:
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return match.group(0) if match else ""

    def _conversation_context_text(self) -> str:
        if not self.chat_history:
            return "(empty)"

        lines: List[str] = []
        for item in self.chat_history[-5:]:
            lines.append(f"User: {item.get('user', '')}")
            lines.append(f"Assistant: {item.get('assistant', '')}")
        return "\n".join(lines)

    def _tool_manifest_text(self) -> str:
        lines: List[str] = []
        for tool in TOOL_SCHEMA:
            name = tool.get("name", "")
            description = tool.get("description", "")
            param_names = []
            params = tool.get("parameters", {}).get("properties", {})
            if isinstance(params, dict):
                param_names = sorted(params.keys())
            lines.append(f"- {name}: {description} | params={param_names}")
        lines.append(
            "- local_docs_rag: Use local document knowledge (non-latest/static information) "
            "and summarize via RAG before final answer | params=['query']"
        )
        return "\n".join(lines) if lines else "(no tools)"

    def _normalize_tool_name(self, tool_name: str) -> str:
        normalized = str(tool_name).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "websearch":
            return "web_search"
        if normalized in {"email", "sent_email", "sendemail"}:
            return "send_email"
        if normalized in {"local_docs", "local_doc", "local_rag", "rag"}:
            return "local_docs_rag"
        return normalized

    def _resolve_log_file(self) -> Path:
        configured = self.config.get("logging", {}).get("file", "./logs/autobot.log")
        path = Path(configured)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        return path

    def _read_recent_error_logs(self, limit: int = 60) -> str:
        if not self.log_file.exists():
            return ""

        try:
            text = self.log_file.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            recent = lines[-max(1, limit):]
            error_lines = [line for line in recent if "ERROR" in line or "Exception" in line or "Traceback" in line]
            selected = error_lines if error_lines else recent[-20:]
            return "\n".join(selected)
        except Exception as exc:
            self.logger.warning("Unable to read log file %s: %s", self.log_file, exc)
            return ""

    def _clean_text(self, text: Any) -> str:
        return str(text or "").strip()

    def _truncate_text(self, value: Any, max_chars: int) -> str:
        text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = (text or "").strip()
        if not stripped:
            return None

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(stripped[idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None
