"""
AutoBot Core Agentic Orchestrator

This orchestrator implements five agentic flows inspired by the reference
architectures in arc-files:
1. tool_use
2. react
3. planning
4. multi_agent
5. pev (planner -> executor -> verifier)
"""

import asyncio
import contextvars
import hashlib
import importlib.util
import inspect
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.intent_classifier import IntentClassifier
from core.llm_interface import LLMInterface
from core.planner import Planner
from interfaces.text_interface import TextInterface
from interfaces.voice_interface import VoiceInterface
from memory.memory_manager import MemoryManager
from tools.tool_registry import ToolRegistry

try:
    from tools.tool_registry import tools_json as TOOL_SCHEMA
except Exception:
    TOOL_SCHEMA = []

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.documents import Document as LCDocument
    from langchain_core.tools import StructuredTool
    LANGCHAIN_CORE_AVAILABLE = True
except Exception:
    PromptTemplate = None
    PydanticOutputParser = None
    BaseCallbackHandler = object  # type: ignore[assignment]
    LCDocument = None
    StructuredTool = None
    LANGCHAIN_CORE_AVAILABLE = False

try:
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_MEMORY_AVAILABLE = True
except Exception:
    ConversationBufferMemory = None
    LANGCHAIN_MEMORY_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_TEXT_SPLITTER_AVAILABLE = True
except Exception:
    RecursiveCharacterTextSplitter = None
    LANGCHAIN_TEXT_SPLITTER_AVAILABLE = False

try:
    from langchain_community.retrievers import BM25Retriever
    LANGCHAIN_RETRIEVERS_AVAILABLE = True
except Exception:
    BM25Retriever = None
    LANGCHAIN_RETRIEVERS_AVAILABLE = False

PyPDFLoader = None
CSVLoader = None
BSHTMLLoader = None
TextLoader = None
try:
    from langchain_community.document_loaders import PyPDFLoader as _PyPDFLoader

    PyPDFLoader = _PyPDFLoader
except Exception:
    PyPDFLoader = None
try:
    from langchain_community.document_loaders import CSVLoader as _CSVLoader

    CSVLoader = _CSVLoader
except Exception:
    CSVLoader = None
try:
    from langchain_community.document_loaders import BSHTMLLoader as _BSHTMLLoader

    BSHTMLLoader = _BSHTMLLoader
except Exception:
    BSHTMLLoader = None
try:
    from langchain_community.document_loaders import TextLoader as _TextLoader

    TextLoader = _TextLoader
except Exception:
    TextLoader = None

LANGCHAIN_LOADERS_AVAILABLE = any([PyPDFLoader, CSVLoader, BSHTMLLoader, TextLoader])

try:
    from langgraph.graph import StateGraph, END as LANGGRAPH_END
    LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    LANGGRAPH_END = None
    LANGGRAPH_AVAILABLE = False


class TaskComplexity(str, Enum):
    """Task complexity buckets used for flow routing."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"


class FlowType(str, Enum):
    """Supported agentic execution flows."""

    TOOL_USE = "tool_use"
    REACT = "react"
    PLANNING = "planning"
    MULTI_AGENT = "multi_agent"
    PEV = "pev"


class SpecialistType(str, Enum):
    """Specialist lanes used by the meta-controller."""

    GENERALIST = "generalist"
    RESEARCHER = "researcher"
    CODER = "coder"
    SUMMARIZER = "summarizer"
    ANALYST = "analyst"
    PLANNER = "planner"
    RETRIEVER = "retriever"
    COMPLIANCE_CHECKER = "compliance_checker"
    OPTIMIZER = "optimizer"
    EXPLAINER = "explainer"


@dataclass
class GoalProfile:
    """Normalized request profile used by the router."""

    goal: str
    complexity: TaskComplexity
    depth: str
    intent: str
    required_tools: List[str]
    flow: FlowType
    specialist: SpecialistType = SpecialistType.GENERALIST
    specialist_reasoning: str = ""
    requires_rag: bool = False
    requires_verification: bool = False
    confidence: float = 0.5


@dataclass
class FlowExecutionResult:
    """Standardized flow output."""

    response: str
    flow: FlowType
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    model_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoBotLangChainCallback(BaseCallbackHandler):
    """Minimal callback handler for langchain execution observability."""

    def __init__(self, logger: logging.Logger, debug_enabled: bool):
        self.logger = logger
        self.debug_enabled = debug_enabled

    def on_text(self, text: str, **kwargs: Any) -> Any:
        if self.debug_enabled:
            self.logger.info("[LC-CALLBACK] %s", text)


class GoalAssessmentSchema(BaseModel):
    goal: str = ""
    complexity: str = "medium"
    depth: str = "medium"
    required_tools: List[str] = Field(default_factory=list)
    recommended_flow: str = ""
    requires_rag: bool = False
    requires_verification: bool = False


class MetaControllerDecisionSchema(BaseModel):
    specialist: str = "generalist"
    reasoning: str = ""
    recommended_flow: str = ""
    required_tools: List[str] = Field(default_factory=list)
    requires_rag: bool = False
    requires_verification: bool = False


class ReactDecisionSchema(BaseModel):
    action: str = "final"
    thought: str = ""
    tool: str = ""
    tool_params: Dict[str, Any] = Field(default_factory=dict)
    memory_query: str = ""
    final_answer: str = ""


class PlanStepSchema(BaseModel):
    id: str = ""
    purpose: str = ""
    executor: str = "reasoning"
    input: Any = ""
    depends_on: List[str] = Field(default_factory=list)
    timeout_sec: int = 45


class PlanStepsSchema(BaseModel):
    steps: List[PlanStepSchema] = Field(default_factory=list)


class VerificationSchema(BaseModel):
    passed: bool = False
    score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    guidance: str = ""


class EmailExtractionSchema(BaseModel):
    to: str = ""
    subject: str = ""
    body: str = ""
    body_html: str = ""
    cc: str = ""
    bcc: str = ""
    attachments: List[str] = Field(default_factory=list)


class Orchestrator:
    """
    Central AutoBot controller with adaptive multi-agent orchestration.

    Flow selection:
    - simple tasks -> tool_use / react
    - medium tasks -> react / planning
    - high-depth tasks -> multi_agent / pev
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core subsystems
        self.memory = MemoryManager(config)
        self.llm = LLMInterface(config)
        self.intent_classifier = IntentClassifier(config, self.llm)
        self.planner = Planner(config, self.llm)
        self.tool_registry = ToolRegistry(config)

        # Interfaces (graceful degradation when optional deps are missing)
        self.text_interface = None
        if config["interfaces"]["text"]["enabled"]:
            try:
                self.text_interface = TextInterface(config)
            except Exception as exc:
                self.logger.warning("text interface disabled due to initialization error: %s", exc)
        self.voice_interface = None
        if config["interfaces"]["voice"]["enabled"]:
            try:
                self.voice_interface = VoiceInterface(config)
            except Exception as exc:
                self.logger.warning("voice interface disabled due to initialization error: %s", exc)

        # State
        self.running = True
        self.current_tasks: Dict[str, Dict[str, Any]] = {}
        self.execution_trace: List[Dict[str, Any]] = []

        # Performance
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = config.get("performance", {}).get("cache_ttl", 300)
        self.memory_batch: List[Dict[str, Any]] = []
        self.batch_size = config.get("performance", {}).get("batch_size", 5)

        # Agentic runtime tuning
        self.max_react_iterations = int(config.get("agentic", {}).get("react_max_iterations", 4))
        self.max_pev_attempts = int(config.get("agentic", {}).get("pev_max_attempts", 3))
        self.max_context_length = int(config.get("agentic", {}).get("max_context_length", 32768))
        self.max_tokens_hard_limit = int(config.get("agentic", {}).get("max_tokens_hard_limit", 4096))
        meta_cfg = config.get("agentic", {}).get("meta_controller", {})
        self.meta_controller_enabled = bool(meta_cfg.get("enabled", True))
        self.meta_controller_temperature = float(meta_cfg.get("temperature", 0.1))
        self.meta_controller_max_tokens = int(meta_cfg.get("max_tokens", 320))
        lc_cfg = config.get("agentic", {}).get("langchain", {})
        self.langchain_enabled = bool(lc_cfg.get("enabled", True))
        self.langchain_use_bm25_rerank = bool(lc_cfg.get("use_bm25_rerank", True))
        self.langchain_use_local_doc_loader = bool(lc_cfg.get("use_local_doc_loader", True))
        self.langchain_use_langgraph_multi_agent = bool(lc_cfg.get("use_langgraph_multi_agent", True))
        self.langchain_chunk_size = int(lc_cfg.get("chunk_size", 900))
        self.langchain_chunk_overlap = int(lc_cfg.get("chunk_overlap", 120))

        # Paths
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_root = self.project_root / "models"
        self.tool_root = self.project_root / "tools"

        # Device for direct model loaders
        self.device = self._resolve_device()

        # Dynamic modules + runtime model states
        self.dynamic_modules: Dict[str, Any] = {}
        self.specialized_models: Dict[str, Dict[str, Any]] = {
            "autobot_instruct": {"status": "uninitialized"},
            "autobot_thinking": {"status": "uninitialized"},
            "autobot_rag": {"status": "uninitialized"},
        }
        self.model_locks: Dict[str, asyncio.Lock] = {
            model_name: asyncio.Lock() for model_name in self.specialized_models
        }

        # Optional RAG runtime (lazy)
        self.rag_pipeline = None
        self.rag_available: Optional[bool] = None
        self._rag_lock = asyncio.Lock()

        # Debug instrumentation
        self.debug_enabled = bool(config.get("debug", {}).get("enabled", True))
        self._request_debug_ctx: contextvars.ContextVar = contextvars.ContextVar(
            "request_debug", default=None
        )
        self.langchain_callback = None
        self.langchain_memory = None
        self._langchain_available = self.langchain_enabled and LANGCHAIN_CORE_AVAILABLE
        if self._langchain_available:
            try:
                self.langchain_callback = AutoBotLangChainCallback(
                    logger=self.logger,
                    debug_enabled=self.debug_enabled,
                )
            except Exception:
                self.langchain_callback = None
        if (
            self.langchain_enabled
            and LANGCHAIN_MEMORY_AVAILABLE
            and ConversationBufferMemory is not None
        ):
            try:
                self.langchain_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    input_key="input",
                    output_key="output",
                    return_messages=False,
                )
            except Exception as exc:
                self.logger.warning("langchain memory init failed: %s", exc)
                self.langchain_memory = None
        self.langchain_tools: List[Any] = []
        if self._langchain_available and StructuredTool is not None:
            try:
                self.langchain_tools = self._build_langchain_tools()
            except Exception as exc:
                self.logger.warning("langchain tool registration failed: %s", exc)
                self.langchain_tools = []

    async def run_with_interface(self, interface):
        """Run AutoBot with the selected user interface."""
        self.logger.info("AutoBot orchestrator starting")

        await self.memory.initialize()
        await self.tool_registry.initialize()
        await self.llm.initialize()

        await interface.run(self.handle_input)
        self.logger.info("AutoBot orchestrator shutting down")

    async def handle_input(self, user_input: str, context: Optional[Dict] = None) -> str:
        """
        Main request pipeline:
        1. cache lookup
        2. intent + goal profiling
        3. flow selection
        4. flow execution
        5. memory + trace updates
        """
        context = context or {}
        self._emit_langchain_callback("request_start")
        lc_history = self._get_langchain_history()
        if lc_history and "langchain_history" not in context:
            context["langchain_history"] = lc_history
        start_time = time.time()
        trace_id = f"trace_{int(start_time * 1000)}"
        debug_token = self._start_request_debug(trace_id=trace_id, user_input=user_input)

        try:
            cache_key = self._generate_cache_key(user_input, context)
            cached = self._get_cached_response(cache_key)
            if cached:
                self.logger.info("cache hit for %s", cache_key[:8])
                self._mark_request_debug_event("cache_hit", {"cache_key": cache_key[:8]})
                return cached

            await self.memory.add_short_term(user_input)

            intent_json = await self.intent_classifier.classify(user_input)
            goal_profile = await self._build_goal_profile(user_input, intent_json, context)
            self._mark_request_debug_event(
                "flow_selected",
                {
                    "flow": goal_profile.flow.value,
                    "complexity": goal_profile.complexity.value,
                    "required_tools": goal_profile.required_tools,
                    "specialist": goal_profile.specialist.value,
                },
            )
            if self.debug_enabled:
                self.logger.info(
                    "[DEBUG] request=%s chosen_flow=%s chosen_complexity=%s chosen_specialist=%s",
                    trace_id,
                    goal_profile.flow.value,
                    goal_profile.complexity.value,
                    goal_profile.specialist.value,
                )
                print(
                    f"[DEBUG] request={trace_id} chosen_flow={goal_profile.flow.value} "
                    f"chosen_complexity={goal_profile.complexity.value} "
                    f"chosen_specialist={goal_profile.specialist.value}"
                )

            await self._record_agent_trace(
                {
                    "trace_id": trace_id,
                    "stage": "profiled",
                    "goal_profile": asdict(goal_profile),
                    "intent": intent_json,
                    "specialist": goal_profile.specialist.value,
                    "timestamp": time.time(),
                }
            )

            flow_result = await self._run_flow_with_replan(
                user_input=user_input,
                goal_profile=goal_profile,
                intent_json=intent_json,
                context=context,
                trace_id=trace_id,
            )
            response = self._clean_response(flow_result.response)
            if not response:
                response = "I could not produce a complete response for this request."

            self._cache_response(cache_key, response)
            await self._batch_memory_update(user_input, response, intent_json.get("intent", "unknown"))
            await self._update_episodic_semantic_memory(
                user_input=user_input,
                response=response,
                intent=intent_json.get("intent", "unknown"),
                goal_profile=goal_profile,
                flow_result=flow_result,
            )

            elapsed = time.time() - start_time
            debug_summary = self._get_request_debug_summary()
            await self._record_agent_trace(
                {
                    "trace_id": trace_id,
                    "stage": "completed",
                    "flow": goal_profile.flow.value,
                    "complexity": goal_profile.complexity.value,
                    "specialist": goal_profile.specialist.value,
                    "duration_sec": round(elapsed, 3),
                    "metadata": flow_result.metadata,
                    "debug": debug_summary,
                    "timestamp": time.time(),
                }
            )
            if self.debug_enabled:
                self.logger.info(
                    "[DEBUG] request=%s chosen_flow=%s chosen_specialist=%s chosen_models=%s",
                    trace_id,
                    goal_profile.flow.value,
                    goal_profile.specialist.value,
                    debug_summary.get("models_used", []),
                )
                print(
                    f"[DEBUG] request={trace_id} chosen_flow={goal_profile.flow.value} "
                    f"chosen_specialist={goal_profile.specialist.value} "
                    f"chosen_models={debug_summary.get('models_used', [])}"
                )

            self.logger.info(
                "request completed flow=%s specialist=%s complexity=%s in %.2fs",
                goal_profile.flow.value,
                goal_profile.specialist.value,
                goal_profile.complexity.value,
                elapsed,
            )
            self._save_langchain_history(user_input=user_input, response=response)
            self._emit_langchain_callback("request_end")
            return response

        except Exception as exc:
            self.logger.exception("handle_input failed: %s", exc)
            fallback = "I encountered an error while processing your request."
            await self._batch_memory_update(user_input, fallback, "error")
            self._emit_langchain_callback("request_error")
            return fallback
        finally:
            self._end_request_debug(debug_token)

    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backward-compatible public method used by existing code paths.
        Executes steps from plan with planning-style step execution.
        """
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            return {}
        pseudo_profile = GoalProfile(
            goal="execute_plan",
            complexity=TaskComplexity.MEDIUM,
            depth="medium",
            intent="general_conversation",
            required_tools=[],
            flow=FlowType.PLANNING,
        )
        return await self._execute_plan_steps(steps, "", pseudo_profile, {})

    async def generate_response(
        self,
        user_input: str,
        execution_results: Dict[str, Any],
        intent_json: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> str:
        """
        Backward-compatible public method. Uses the same final synthesizer used
        by the new flows.
        """
        pseudo_profile = GoalProfile(
            goal=user_input,
            complexity=TaskComplexity.MEDIUM,
            depth="medium",
            intent=intent_json.get("intent", "general_conversation"),
            required_tools=[],
            flow=FlowType.PLANNING,
        )
        return await self._synthesize_response(
            user_input=user_input,
            profile=pseudo_profile,
            flow=FlowType.PLANNING,
            artifacts={
                "intent": intent_json,
                "plan": plan,
                "execution_results": execution_results,
            },
        )

    async def _build_goal_profile(
        self, user_input: str, intent_json: Dict[str, Any], context: Dict[str, Any]
    ) -> GoalProfile:
        """Build normalized goal profile from meta-controller + model + heuristics."""
        intent = intent_json.get("intent", "general_conversation")
        confidence = float(intent_json.get("confidence", 0.5))
        heuristic_complexity, depth = self._estimate_complexity(user_input)
        required_tools = self._resolve_required_tools(user_input, intent_json, context)
        requires_rag = self._should_use_rag(user_input, intent_json, context)
        requires_verification = any(
            phrase in user_input.lower()
            for phrase in ["verify", "double-check", "double check", "audit", "accuracy", "critical"]
        )
        heuristic_specialist = self._infer_specialist_by_heuristics(user_input, intent_json, required_tools)
        specialist = heuristic_specialist
        specialist_reasoning = "heuristic routing"

        meta_decision: Dict[str, Any] = {}
        if self.meta_controller_enabled:
            meta_decision = await self._meta_controller_decision(
                user_input=user_input,
                intent_json=intent_json,
                heuristic_complexity=heuristic_complexity,
                required_tools=required_tools,
                requires_rag=requires_rag,
            )

        meta_specialist = self._parse_specialist(str(meta_decision.get("specialist", "")))
        if meta_specialist is not None:
            if self._specialist_decision_is_consistent(
                specialist=meta_specialist,
                user_input=user_input,
                intent_json=intent_json,
                required_tools=required_tools,
            ):
                specialist = meta_specialist
                specialist_reasoning = str(meta_decision.get("reasoning", "")).strip() or specialist_reasoning
            else:
                specialist = heuristic_specialist
                specialist_reasoning = (
                    "meta-controller decision adjusted by heuristic consistency checks"
                )

        if meta_decision.get("requires_rag") is True:
            requires_rag = True

        meta_tools = meta_decision.get("required_tools")
        if isinstance(meta_tools, list):
            merged_tools = [
                self._normalize_tool_name(str(tool))
                for tool in meta_tools
                if self._normalize_tool_name(str(tool))
            ]
            if merged_tools:
                required_tools = sorted(set(required_tools + merged_tools))

        if specialist == SpecialistType.RESEARCHER:
            if "web_search" not in required_tools:
                required_tools = sorted(set(required_tools + ["web_search"]))
            requires_rag = True
        if specialist == SpecialistType.RETRIEVER:
            requires_rag = True
        if specialist == SpecialistType.COMPLIANCE_CHECKER:
            requires_verification = True

        self._mark_request_debug_event(
            "meta_controller_selected",
            {
                "specialist": specialist.value,
                "reasoning": specialist_reasoning,
                "meta_decision": meta_decision,
            },
        )

        model_assessment = await self._model_goal_assessment(user_input, intent_json, required_tools)

        goal_text = user_input.strip()
        if isinstance(model_assessment.get("goal"), str) and model_assessment.get("goal").strip():
            goal_text = model_assessment["goal"].strip()
        elif isinstance(meta_decision.get("goal"), str) and str(meta_decision.get("goal")).strip():
            goal_text = str(meta_decision.get("goal")).strip()

        model_complexity = self._parse_complexity(str(model_assessment.get("complexity", "")))
        complexity = model_complexity or heuristic_complexity

        model_flow = self._parse_flow(str(model_assessment.get("recommended_flow", "")))
        if model_assessment.get("requires_verification") is True:
            requires_verification = True
        if model_assessment.get("requires_rag") is True:
            requires_rag = True

        model_tools = model_assessment.get("required_tools")
        if isinstance(model_tools, list):
            merged_tools = [
                self._normalize_tool_name(str(tool))
                for tool in model_tools
                if self._normalize_tool_name(str(tool))
            ]
            if merged_tools:
                required_tools = sorted(set(required_tools + merged_tools))

        preferred_flow = self._parse_flow(str(meta_decision.get("recommended_flow", ""))) or model_flow
        if preferred_flow is None:
            preferred_flow = self._derive_flow_from_specialist(
                specialist=specialist,
                complexity=complexity,
                required_tools=required_tools,
                requires_rag=requires_rag,
                requires_verification=requires_verification,
            )

        selected_flow = self._select_flow(
            complexity=complexity,
            required_tools=required_tools,
            requires_rag=requires_rag,
            requires_verification=requires_verification,
            preferred_flow=preferred_flow,
        )

        return GoalProfile(
            goal=goal_text,
            complexity=complexity,
            depth=depth,
            intent=intent,
            required_tools=required_tools,
            flow=selected_flow,
            specialist=specialist,
            specialist_reasoning=specialist_reasoning,
            requires_rag=requires_rag,
            requires_verification=requires_verification,
            confidence=confidence,
        )

    async def _model_goal_assessment(
        self, user_input: str, intent_json: Dict[str, Any], required_tools: List[str]
    ) -> Dict[str, Any]:
        """Use autobot-instruct to assess goal complexity and recommend flow."""
        format_instructions = (
            "Return JSON only with keys: goal, complexity, depth, required_tools, "
            "recommended_flow, requires_rag, requires_verification."
        )
        if self._langchain_available and PydanticOutputParser is not None:
            try:
                parser = PydanticOutputParser(pydantic_object=GoalAssessmentSchema)
                format_instructions = parser.get_format_instructions()
            except Exception:
                pass
        system_prompt = (
            "You are an orchestration classifier. Choose one flow among: "
            "tool_use, react, planning, multi_agent, pev.\n"
            "Return JSON only with keys: goal, complexity, depth, required_tools, "
            "recommended_flow, requires_rag, requires_verification."
        )
        prompt_template = (
            "User input: {user_input}\n"
            "Intent: {intent_json}\n"
            "Heuristic tools: {required_tools}\n"
            "Complexity values: simple, medium, high.\n"
            "Depth values: low, medium, high.\n"
            "required_tools can include web_search and send_email.\n"
            "{format_instructions}"
        )
        prompt = self._build_prompt_with_template(
            prompt_template,
            {
                "user_input": user_input,
                "intent_json": json.dumps(intent_json, default=str),
                "required_tools": required_tools,
                "format_instructions": format_instructions,
            },
        )

        response = await self._generate_with_model(
            model_name="autobot_instruct",
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=384,
            fallback_mode="intent",
        )
        parsed = self._parse_with_schema(response, GoalAssessmentSchema)
        return parsed if isinstance(parsed, dict) else {}

    async def _meta_controller_decision(
        self,
        user_input: str,
        intent_json: Dict[str, Any],
        heuristic_complexity: TaskComplexity,
        required_tools: List[str],
        requires_rag: bool,
    ) -> Dict[str, Any]:
        """
        Meta-controller router inspired by arc-files/11_meta_controller.ipynb.
        Chooses specialist lane and optional preferred flow.
        """
        specialists = {
            "generalist": "Handles casual conversation, greetings, and straightforward Q&A.",
            "researcher": (
                "Handles recent events, factual research, comparisons, and tasks requiring "
                "web search or knowledge retrieval."
            ),
            "coder": "Handles Python/code generation, debugging, refactoring, and algorithm requests.",
            "summarizer": "Condenses long text or documents into concise summaries.",
            "analyst": (
                "Interprets and reasons over structured/unstructured data to provide insights."
            ),
            "planner": "Breaks down complex tasks into smaller steps and orchestrates execution.",
            "retriever": "Fetches relevant documents or embeddings from a vector store.",
            "compliance_checker": (
                "Validates outputs against rules, safety, or domain-specific standards."
            ),
            "optimizer": (
                "Finds efficient solutions, tunes parameters, and balances resource usage."
            ),
            "explainer": "Provides step-by-step explanations for learning or debugging.",
        }
        specialist_descriptions = "\n".join(f"- {name}: {desc}" for name, desc in specialists.items())
        allowed_specialists = ", ".join(specialists.keys())
        format_instructions = (
            "Return strict JSON object: "
            '{"specialist":"...", "reasoning":"...", "recommended_flow":"...", '
            '"required_tools":["..."], "requires_rag":false, "requires_verification":false}'
        )
        if self._langchain_available and PydanticOutputParser is not None:
            try:
                parser = PydanticOutputParser(pydantic_object=MetaControllerDecisionSchema)
                format_instructions = parser.get_format_instructions()
            except Exception:
                pass

        system_prompt = (
            "You are AutoBot's meta-controller.\n"
            "Analyze the user request and route it to exactly one specialist.\n"
            f"Allowed specialists: {allowed_specialists}.\n"
            "Allowed flows: tool_use, react, planning, multi_agent, pev.\n"
            "Return JSON only with keys: specialist, reasoning, recommended_flow, "
            "required_tools, requires_rag, requires_verification."
        )
        prompt_template = (
            "Available specialists:\n{specialist_descriptions}\n\n"
            "Decide routing based primarily on the raw user request between tags.\n"
            "Ignore specialist descriptions when inferring intent.\n"
            "<user_request>\n{user_input}\n</user_request>\n"
            "Intent: {intent_json}\n"
            "Heuristic complexity: {heuristic_complexity}\n"
            "Heuristic required_tools: {required_tools}\n"
            "Heuristic requires_rag: {requires_rag}\n"
            "required_tools can include web_search and send_email.\n"
            "{format_instructions}"
        )
        prompt = self._build_prompt_with_template(
            prompt_template,
            {
                "specialist_descriptions": specialist_descriptions,
                "user_input": user_input,
                "intent_json": json.dumps(intent_json, default=str),
                "heuristic_complexity": heuristic_complexity.value,
                "required_tools": required_tools,
                "requires_rag": requires_rag,
                "format_instructions": format_instructions,
            },
        )
        self._emit_langchain_callback(
            "meta_controller_prompt_rendered",
            {"langchain_prompt_enabled": self._langchain_available},
        )
        try:
            response = await self._generate_with_model(
                model_name="autobot_instruct",
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.meta_controller_temperature,
                max_tokens=self.meta_controller_max_tokens,
                fallback_mode="intent",
            )
            parsed = self._parse_with_schema(response, MetaControllerDecisionSchema)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            self.logger.warning("meta-controller decision failed, falling back to heuristics: %s", exc)
            return {}

    def _infer_specialist_by_heuristics(
        self, user_input: str, intent_json: Dict[str, Any], required_tools: List[str]
    ) -> SpecialistType:
        """Heuristic specialist router when meta-controller output is unavailable."""
        text = user_input.lower()
        marker_map = {
            SpecialistType.COMPLIANCE_CHECKER: [
                "compliance",
                "policy check",
                "safety check",
                "validate against",
                "regulation",
                "rule check",
                "guardrail",
                "audit",
                "verify",
            ],
            SpecialistType.CODER: [
                "code",
                "python",
                "function",
                "class",
                "script",
                "debug",
                "bug",
                "algorithm",
                "sql",
                "api endpoint",
                "refactor",
            ],
            SpecialistType.SUMMARIZER: [
                "summarize",
                "summary",
                "tl;dr",
                "condense",
                "shorten",
                "key points",
            ],
            SpecialistType.PLANNER: [
                "plan",
                "roadmap",
                "steps",
                "break down",
                "workflow",
                "orchestrate",
            ],
            SpecialistType.RETRIEVER: [
                "retrieve",
                "fetch docs",
                "vector store",
                "embeddings",
                "knowledge base",
                "rag",
                "from memory",
            ],
            SpecialistType.ANALYST: [
                "analyze",
                "analysis",
                "insight",
                "trend",
                "dataset",
                "metrics",
                "interpret",
                "compare",
            ],
            SpecialistType.OPTIMIZER: [
                "optimize",
                "optimization",
                "tune",
                "faster",
                "efficient",
                "reduce latency",
                "reduce cost",
                "performance tuning",
            ],
            SpecialistType.EXPLAINER: [
                "explain",
                "walk through",
                "step by step",
                "teach me",
                "why",
                "how does",
            ],
            SpecialistType.RESEARCHER: [
                "latest",
                "news",
                "research",
                "find",
                "look up",
                "market",
                "current",
                "what happened",
            ],
        }

        # Strong signal: web retrieval intent should bias to researcher unless
        # a stronger specialist marker is present (compliance/coding).
        if "web_search" in required_tools or intent_json.get("intent") == "web_search":
            if any(marker in text for marker in marker_map[SpecialistType.COMPLIANCE_CHECKER]):
                return SpecialistType.COMPLIANCE_CHECKER
            if any(marker in text for marker in marker_map[SpecialistType.CODER]):
                return SpecialistType.CODER
            return SpecialistType.RESEARCHER

        # Priority matters: compliance/coding and strongly explicit asks first.
        priority = [
            SpecialistType.COMPLIANCE_CHECKER,
            SpecialistType.CODER,
            SpecialistType.RESEARCHER,
            SpecialistType.SUMMARIZER,
            SpecialistType.PLANNER,
            SpecialistType.RETRIEVER,
            SpecialistType.ANALYST,
            SpecialistType.OPTIMIZER,
            SpecialistType.EXPLAINER,
        ]
        for specialist in priority:
            if any(marker in text for marker in marker_map[specialist]):
                return specialist

        if "web_search" in required_tools or intent_json.get("intent") == "web_search":
            return SpecialistType.RESEARCHER

        return SpecialistType.GENERALIST

    def _specialist_decision_is_consistent(
        self,
        specialist: SpecialistType,
        user_input: str,
        intent_json: Dict[str, Any],
        required_tools: List[str],
    ) -> bool:
        """Guardrail to avoid obviously mismatched specialist routing decisions."""
        heuristic = self._infer_specialist_by_heuristics(user_input, intent_json, required_tools)
        if specialist == heuristic:
            return True

        text = user_input.lower()
        marker_map = {
            SpecialistType.CODER: ["python", "code", "function", "class", "debug", "bug", "algorithm", "script"],
            SpecialistType.RESEARCHER: ["latest", "news", "search", "find", "research", "market", "current", "trend"],
            SpecialistType.SUMMARIZER: ["summarize", "summary", "condense", "key points", "tldr"],
            SpecialistType.ANALYST: ["analyze", "analysis", "insight", "dataset", "metrics", "interpret", "compare"],
            SpecialistType.PLANNER: ["plan", "roadmap", "steps", "break down", "workflow", "orchestrate"],
            SpecialistType.RETRIEVER: ["retrieve", "vector store", "embeddings", "rag", "from memory", "knowledge base"],
            SpecialistType.COMPLIANCE_CHECKER: ["compliance", "policy", "validate against", "regulation", "rule check", "audit", "verify"],
            SpecialistType.OPTIMIZER: ["optimize", "optimization", "tune", "efficient", "reduce latency", "reduce cost"],
            SpecialistType.EXPLAINER: ["explain", "step by step", "teach me", "walk through", "how does", "why"],
        }

        if specialist == SpecialistType.RESEARCHER and (
            "web_search" in required_tools or intent_json.get("intent") == "web_search"
        ):
            return True
        if specialist == SpecialistType.RETRIEVER and (
            "rag" in text or "vector store" in text or "from memory" in text
        ):
            return True
        if specialist == SpecialistType.COMPLIANCE_CHECKER and (
            "audit" in text or "verify" in text
        ):
            return True

        if specialist in marker_map:
            return any(marker in text for marker in marker_map[specialist])
        if specialist == SpecialistType.GENERALIST:
            for markers in marker_map.values():
                if any(marker in text for marker in markers):
                    return False
            if "web_search" in required_tools or intent_json.get("intent") == "web_search":
                return False
            return True
        return False

    def _derive_flow_from_specialist(
        self,
        specialist: SpecialistType,
        complexity: TaskComplexity,
        required_tools: List[str],
        requires_rag: bool,
        requires_verification: bool,
    ) -> Optional[FlowType]:
        """Map specialist lane to a preferred flow when controller did not pick one."""
        if requires_verification:
            return FlowType.PEV

        if specialist == SpecialistType.RESEARCHER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.MULTI_AGENT
            if complexity == TaskComplexity.MEDIUM or requires_rag or len(required_tools) >= 2:
                return FlowType.PLANNING
            return FlowType.TOOL_USE

        if specialist == SpecialistType.CODER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.PEV
            if len(required_tools) >= 1 or requires_rag:
                return FlowType.PLANNING
            return FlowType.REACT

        if specialist == SpecialistType.SUMMARIZER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.PLANNING
            return FlowType.REACT

        if specialist == SpecialistType.ANALYST:
            if complexity == TaskComplexity.HIGH:
                return FlowType.MULTI_AGENT
            return FlowType.PLANNING

        if specialist == SpecialistType.PLANNER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.PEV
            return FlowType.PLANNING

        if specialist == SpecialistType.RETRIEVER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.PLANNING
            return FlowType.TOOL_USE

        if specialist == SpecialistType.COMPLIANCE_CHECKER:
            return FlowType.PEV

        if specialist == SpecialistType.OPTIMIZER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.PEV
            return FlowType.PLANNING

        if specialist == SpecialistType.EXPLAINER:
            if complexity == TaskComplexity.HIGH:
                return FlowType.PLANNING
            return FlowType.REACT

        if len(required_tools) >= 1:
            return FlowType.TOOL_USE
        return FlowType.REACT

    def _estimate_complexity(self, user_input: str) -> Tuple[TaskComplexity, str]:
        """Heuristic complexity and depth estimation."""
        text = user_input.lower()
        score = 1

        if len(text) > 140:
            score += 1
        if len(text) > 260:
            score += 1

        medium_markers = [
            "compare",
            "analyze",
            "explain why",
            "summarize",
            "plan",
            "steps",
            "and then",
            "after that",
        ]
        high_markers = [
            "comprehensive",
            "market analysis",
            "strategy",
            "verify",
            "audit",
            "critical",
            "high risk",
            "multi-agent",
            "multi step",
        ]

        if any(marker in text for marker in medium_markers):
            score += 1
        if any(marker in text for marker in high_markers):
            score += 2
        if text.count(" and ") >= 2:
            score += 1

        if score <= 2:
            return TaskComplexity.SIMPLE, "low"
        if score <= 4:
            return TaskComplexity.MEDIUM, "medium"
        return TaskComplexity.HIGH, "high"

    def _resolve_required_tools(
        self, user_input: str, intent_json: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Infer required tools from intent, query text, and context."""
        tools: List[str] = []
        for tool in intent_json.get("required_tools", []) or []:
            normalized = self._normalize_tool_name(str(tool))
            if normalized:
                tools.append(normalized)

        text = user_input.lower()
        if any(keyword in text for keyword in ["search", "latest", "news", "find", "look up", "research"]):
            tools.append("web_search")
        if any(keyword in text for keyword in ["email", "mail", "send to", "@"]):
            tools.append("send_email")
        if isinstance(context.get("email_payload"), dict):
            tools.append("send_email")
        if context.get("force_web_search") is True:
            tools.append("web_search")

        allowed = {"web_search", "send_email"}
        deduped = [tool for tool in sorted(set(tools)) if tool in allowed]
        return deduped

    def _should_use_rag(self, user_input: str, intent_json: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Decide whether a request should use memory/RAG retrieval."""
        if context.get("use_rag") is True:
            return True

        text = user_input.lower()
        rag_markers = [
            "knowledge base",
            "from memory",
            "from documents",
            "from docs",
            "rag",
            "use memory",
            "previous conversations",
        ]
        if any(marker in text for marker in rag_markers):
            return True

        if intent_json.get("intent") == "web_search":
            return True

        return False

    def _select_flow(
        self,
        complexity: TaskComplexity,
        required_tools: List[str],
        requires_rag: bool,
        requires_verification: bool,
        preferred_flow: Optional[FlowType],
    ) -> FlowType:
        """Choose final execution flow."""
        if preferred_flow is not None:
            return preferred_flow

        if requires_verification:
            return FlowType.PEV

        if complexity == TaskComplexity.SIMPLE:
            if required_tools:
                return FlowType.TOOL_USE
            return FlowType.REACT

        if complexity == TaskComplexity.MEDIUM:
            if len(required_tools) >= 2 or requires_rag:
                return FlowType.PLANNING
            return FlowType.REACT

        # High complexity
        if len(required_tools) >= 2 or requires_rag:
            return FlowType.MULTI_AGENT
        return FlowType.PEV

    async def _dispatch_flow(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FlowExecutionResult:
        """Dispatch to selected flow implementation."""
        self._mark_request_debug_event("flow_dispatch_start", {"flow": profile.flow.value})
        if profile.flow == FlowType.TOOL_USE:
            return await self._run_tool_use_flow(user_input, profile, intent_json, context)
        if profile.flow == FlowType.REACT:
            return await self._run_react_flow(user_input, profile, intent_json, context)
        if profile.flow == FlowType.PLANNING:
            return await self._run_planning_flow(user_input, profile, intent_json, context)
        if profile.flow == FlowType.MULTI_AGENT:
            return await self._run_multi_agent_flow(user_input, profile, intent_json, context)
        if profile.flow == FlowType.PEV:
            return await self._run_pev_flow(user_input, profile, intent_json, context)

        # Safe fallback
        return await self._run_react_flow(user_input, profile, intent_json, context)

    async def _run_flow_with_replan(
        self,
        user_input: str,
        goal_profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
        trace_id: str,
    ) -> FlowExecutionResult:
        """
        Execute selected flow and perform automatic replan with logs when:
        - flow raises exception
        - flow returns failed steps
        """
        try:
            flow_result = await self._dispatch_flow(user_input, goal_profile, intent_json, context)
            self._mark_request_debug_event(
                "flow_dispatch_completed",
                {"flow": goal_profile.flow.value, "failed_steps": self._count_failed_steps(flow_result.steps)},
            )
        except Exception as flow_exc:
            failure_logs = self._get_request_debug_summary()
            self._mark_request_debug_event(
                "flow_exception",
                {"flow": goal_profile.flow.value, "error": str(flow_exc)},
            )
            self.logger.exception(
                "flow execution failed, triggering replan | request=%s flow=%s",
                trace_id,
                goal_profile.flow.value,
            )
            return await self._replan_with_failure_logs(
                user_input=user_input,
                goal_profile=goal_profile,
                intent_json=intent_json,
                context=context,
                failure_reason=str(flow_exc),
                failure_logs=failure_logs,
            )

        if self._flow_result_failed(flow_result) and goal_profile.flow != FlowType.PEV:
            failure_logs = self._get_request_debug_summary()
            self._mark_request_debug_event(
                "flow_failed_steps",
                {
                    "flow": goal_profile.flow.value,
                    "failed_steps_count": self._count_failed_steps(flow_result.steps),
                },
            )
            self.logger.warning(
                "flow completed with failed steps, triggering replan | request=%s flow=%s",
                trace_id,
                goal_profile.flow.value,
            )
            return await self._replan_with_failure_logs(
                user_input=user_input,
                goal_profile=goal_profile,
                intent_json=intent_json,
                context=context,
                failure_reason="flow returned failed steps",
                failure_logs=failure_logs,
            )

        return flow_result

    async def _replan_with_failure_logs(
        self,
        user_input: str,
        goal_profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
        failure_reason: str,
        failure_logs: Dict[str, Any],
    ) -> FlowExecutionResult:
        """Replan tasks using PEV flow while passing failure logs as context."""
        replan_profile = replace(
            goal_profile,
            flow=FlowType.PEV,
            complexity=TaskComplexity.HIGH
            if goal_profile.complexity != TaskComplexity.HIGH
            else goal_profile.complexity,
            requires_verification=True,
        )
        replan_context = dict(context)
        replan_context.update(
            {
                "force_replan": True,
                "failure_reason": failure_reason,
                "failure_logs": failure_logs,
                "original_flow": goal_profile.flow.value,
                "original_specialist": goal_profile.specialist.value,
                "_replan_attempted": True,
            }
        )
        replan_query = (
            f"{user_input}\n\nFlow failure reason: {failure_reason}\n"
            "Use failure logs to generate a corrected plan."
        )

        try:
            self._mark_request_debug_event(
                "replan_started",
                {
                    "original_flow": goal_profile.flow.value,
                    "replan_flow": replan_profile.flow.value,
                },
            )
            result = await self._run_pev_flow(replan_query, replan_profile, intent_json, replan_context)
            result.metadata["replanned"] = True
            result.metadata["failure_reason"] = failure_reason
            result.metadata["failure_logs"] = failure_logs
            self._mark_request_debug_event("replan_completed", {"success": True})
            return result
        except Exception as exc:
            self.logger.exception("replan flow failed: %s", exc)
            self._mark_request_debug_event("replan_completed", {"success": False, "error": str(exc)})
            fallback_response = (
                "Primary flow failed and replan also failed.\n"
                f"Failure reason: {failure_reason}\n"
                "Please refine the task or provide additional constraints."
            )
            return FlowExecutionResult(
                response=fallback_response,
                flow=FlowType.PEV,
                steps=[],
                tool_outputs={},
                model_outputs={},
                metadata={
                    "replanned": True,
                    "replan_failed": True,
                    "failure_reason": failure_reason,
                    "replan_error": str(exc),
                    "failure_logs": failure_logs,
                },
            )

    def _flow_result_failed(self, flow_result: FlowExecutionResult) -> bool:
        """Detect flow failure from standardized step payloads."""
        if not isinstance(flow_result, FlowExecutionResult):
            return True

        if not flow_result.response:
            return True

        if self._count_failed_steps(flow_result.steps) > 0:
            return True
        return False

    def _count_failed_steps(self, steps: List[Dict[str, Any]]) -> int:
        """Count failures recursively in flow step outputs."""
        failed = 0
        for step in steps:
            if isinstance(step, dict):
                if step.get("success") is False:
                    failed += 1
                    continue
                verification = step.get("verification")
                if isinstance(verification, dict) and verification.get("passed") is False:
                    failed += 1
                    continue
                nested_steps = step.get("steps")
                if isinstance(nested_steps, list):
                    failed += self._count_failed_steps(nested_steps)
        return failed

    async def _run_tool_use_flow(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FlowExecutionResult:
        """Single-pass tool-use flow."""
        steps: List[Dict[str, Any]] = []
        tool_outputs: Dict[str, Any] = {}
        model_outputs: Dict[str, Any] = {}

        required_tools = profile.required_tools or self._resolve_required_tools(user_input, intent_json, context)
        if profile.requires_rag:
            required_tools = sorted(set(required_tools + ["rag_retrieve"]))

        if not required_tools:
            direct_response = await self._generate_with_model(
                model_name="autobot_instruct",
                system_prompt="You are AutoBot. Answer clearly and directly.",
                user_prompt=user_input,
                temperature=0.3,
                max_tokens=512,
            )
            steps.append({"step": "direct_generation", "success": True})
            model_outputs["autobot_instruct"] = direct_response
            return FlowExecutionResult(
                response=direct_response,
                flow=FlowType.TOOL_USE,
                steps=steps,
                tool_outputs=tool_outputs,
                model_outputs=model_outputs,
                metadata={"used_tools": []},
            )

        for tool_name in required_tools:
            if tool_name == "web_search":
                params = self._prepare_web_search_params({"query": user_input, "max_results": 8, "workers": 6})
                result = await self._execute_tool("web_search", params)
                tool_outputs["web_search"] = self._coerce_json(result)
                steps.append({"step": "web_search", "success": True, "params": params})

            elif tool_name == "send_email":
                payload = await self._build_email_payload(user_input, context)
                missing = payload.pop("_missing_required", [])
                if missing:
                    message = (
                        "I can send the email. Please provide these missing fields: "
                        + ", ".join(missing)
                        + "."
                    )
                    steps.append({"step": "send_email_validation", "success": False, "missing": missing})
                    return FlowExecutionResult(
                        response=message,
                        flow=FlowType.TOOL_USE,
                        steps=steps,
                        tool_outputs=tool_outputs,
                        model_outputs=model_outputs,
                        metadata={"used_tools": ["send_email"], "blocked": True},
                    )

                result = await self._execute_tool("send_email", payload)
                tool_outputs["send_email"] = self._coerce_json(result)
                steps.append({"step": "send_email", "success": True})

            elif tool_name == "rag_retrieve":
                result = await self._retrieve_knowledge(user_input, top_k=3, context=context)
                tool_outputs["rag_retrieve"] = result
                steps.append({"step": "rag_retrieve", "success": True, "count": result.get("count", 0)})

        synthesized = await self._synthesize_response(
            user_input=user_input,
            profile=profile,
            flow=FlowType.TOOL_USE,
            artifacts={"intent": intent_json, "tool_outputs": tool_outputs, "steps": steps},
        )
        return FlowExecutionResult(
            response=synthesized,
            flow=FlowType.TOOL_USE,
            steps=steps,
            tool_outputs=tool_outputs,
            model_outputs=model_outputs,
            metadata={"used_tools": required_tools},
        )

    async def _run_react_flow(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FlowExecutionResult:
        """ReAct loop: think -> act -> observe -> repeat."""
        steps: List[Dict[str, Any]] = []
        tool_outputs: Dict[str, Any] = {}
        observations: List[Dict[str, Any]] = []
        final_answer = ""

        for iteration in range(1, self.max_react_iterations + 1):
            decision = await self._react_decision(
                user_input=user_input,
                profile=profile,
                intent_json=intent_json,
                observations=observations,
                iteration=iteration,
            )
            action = str(decision.get("action", "final")).lower()
            steps.append({"iteration": iteration, "decision": decision})

            if action == "tool":
                tool_name = self._normalize_tool_name(str(decision.get("tool", "web_search")))
                params = decision.get("tool_params", {}) or {}

                if tool_name == "web_search":
                    if "query" not in params:
                        params["query"] = user_input
                    params = self._prepare_web_search_params(params)
                elif tool_name == "send_email":
                    email_payload = await self._build_email_payload(user_input, context)
                    email_payload.update(params)
                    params = email_payload
                    missing = params.pop("_missing_required", [])
                    if missing:
                        final_answer = (
                            "I can send the email, but I still need: "
                            + ", ".join(missing)
                            + "."
                        )
                        break

                tool_result = await self._execute_tool(tool_name, params)
                normalized_result = self._coerce_json(tool_result)
                tool_outputs[f"{tool_name}_{iteration}"] = normalized_result
                observations.append(
                    {
                        "type": "tool",
                        "tool": tool_name,
                        "result": self._truncate_text(normalized_result, 2200),
                    }
                )
                continue

            if action == "memory":
                memory_query = str(decision.get("memory_query", user_input))
                memory_result = await self._retrieve_knowledge(memory_query, top_k=3, context=context)
                observations.append({"type": "memory", "query": memory_query, "result": memory_result})
                continue

            if action == "final":
                candidate = str(decision.get("final_answer", "")).strip()
                if candidate:
                    final_answer = candidate
                    break

            reasoning = await self._generate_with_model(
                model_name="autobot_thinking",
                system_prompt="You are a ReAct reasoning agent. Continue reasoning from observations.",
                user_prompt=json.dumps({"query": user_input, "observations": observations}, default=str),
                temperature=0.2,
                max_tokens=320,
            )
            observations.append({"type": "reasoning", "result": self._truncate_text(reasoning, 1200)})

        if not final_answer:
            final_answer = await self._synthesize_response(
                user_input=user_input,
                profile=profile,
                flow=FlowType.REACT,
                artifacts={
                    "intent": intent_json,
                    "observations": observations,
                    "tool_outputs": tool_outputs,
                    "steps": steps,
                },
            )

        return FlowExecutionResult(
            response=final_answer,
            flow=FlowType.REACT,
            steps=steps,
            tool_outputs=tool_outputs,
            model_outputs={"react_observations": observations},
            metadata={"iterations": len(steps)},
        )

    async def _run_planning_flow(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FlowExecutionResult:
        """Planner -> executor -> synthesizer flow."""
        plan_steps = await self._generate_plan_steps(user_input, profile, context)
        execution_results = await self._execute_plan_steps(plan_steps, user_input, profile, context)
        response = await self._synthesize_response(
            user_input=user_input,
            profile=profile,
            flow=FlowType.PLANNING,
            artifacts={
                "intent": intent_json,
                "plan": plan_steps,
                "execution_results": execution_results,
            },
        )
        return FlowExecutionResult(
            response=response,
            flow=FlowType.PLANNING,
            steps=plan_steps,
            tool_outputs={"execution_results": execution_results},
            model_outputs={},
            metadata={"plan_size": len(plan_steps)},
        )

    async def _run_multi_agent_flow(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FlowExecutionResult:
        """
        Specialist team flow (5 agents):
        1) research_agent
        2) knowledge_agent
        3) analysis_agent
        4) planning_agent
        5) response_agent
        """
        langgraph_result = await self._run_multi_agent_flow_langgraph(
            user_input=user_input,
            profile=profile,
            intent_json=intent_json,
            context=context,
        )
        if langgraph_result is not None:
            return langgraph_result

        steps: List[Dict[str, Any]] = []
        tool_outputs: Dict[str, Any] = {}
        model_outputs: Dict[str, Any] = {}

        async def research_agent():
            params = self._prepare_web_search_params({"query": user_input, "max_results": 8, "workers": 6})
            return await self._execute_tool("web_search", params)

        async def knowledge_agent():
            return await self._retrieve_knowledge(user_input, top_k=4, context=context)

        research_raw, knowledge = await asyncio.gather(research_agent(), knowledge_agent())
        research = self._coerce_json(research_raw)
        tool_outputs["research_agent"] = research
        tool_outputs["knowledge_agent"] = knowledge
        steps.append({"agent": "research_agent", "success": True})
        steps.append({"agent": "knowledge_agent", "success": True})

        analysis_prompt = json.dumps(
            {
                "goal": profile.goal,
                "query": user_input,
                "research": research,
                "knowledge": knowledge,
            },
            default=str,
        )
        analysis = await self._generate_with_model(
            model_name="autobot_thinking",
            system_prompt="You are the analysis_agent. Produce a concise factual analysis.",
            user_prompt=analysis_prompt,
            temperature=0.2,
            max_tokens=700,
        )
        model_outputs["analysis_agent"] = analysis
        steps.append({"agent": "analysis_agent", "success": True})

        planning_json_prompt = (
            "Based on the analysis, decide if extra tools are needed.\n"
            "Return JSON with keys: tool_calls (array), notes.\n"
            "tool_calls entries: {\"tool\": \"web_search|send_email\", \"params\": {...}}.\n"
            f"Analysis:\n{analysis}"
        )
        planning_output = await self._generate_with_model(
            model_name="autobot_thinking",
            system_prompt="You are the planning_agent in a specialist team.",
            user_prompt=planning_json_prompt,
            temperature=0.2,
            max_tokens=380,
        )
        planning_data = self._extract_json_object(planning_output) or {}
        model_outputs["planning_agent"] = planning_data
        steps.append({"agent": "planning_agent", "success": True})

        extra_tool_calls = planning_data.get("tool_calls", []) if isinstance(planning_data, dict) else []
        if isinstance(extra_tool_calls, list):
            for idx, call in enumerate(extra_tool_calls, start=1):
                if not isinstance(call, dict):
                    continue
                tool_name = self._normalize_tool_name(str(call.get("tool", "")))
                params = call.get("params", {}) or {}
                if tool_name not in {"web_search", "send_email"}:
                    continue
                if tool_name == "web_search":
                    params = self._prepare_web_search_params(params)
                if tool_name == "send_email":
                    payload = await self._build_email_payload(user_input, context)
                    payload.update(params)
                    params = payload
                    missing = params.pop("_missing_required", [])
                    if missing:
                        continue
                output = await self._execute_tool(tool_name, params)
                tool_outputs[f"planning_agent_{tool_name}_{idx}"] = self._coerce_json(output)
                steps.append({"agent": "planning_agent", "tool": tool_name, "success": True})

        response = await self._synthesize_response(
            user_input=user_input,
            profile=profile,
            flow=FlowType.MULTI_AGENT,
            artifacts={
                "intent": intent_json,
                "analysis": analysis,
                "team_outputs": {
                    "research": research,
                    "knowledge": knowledge,
                    "planning": planning_data,
                    "extra_tool_outputs": tool_outputs,
                },
            },
        )
        steps.append({"agent": "response_agent", "success": True})

        return FlowExecutionResult(
            response=response,
            flow=FlowType.MULTI_AGENT,
            steps=steps,
            tool_outputs=tool_outputs,
            model_outputs=model_outputs,
            metadata={"agents_used": 5},
        )

    async def _run_multi_agent_flow_langgraph(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[FlowExecutionResult]:
        """
        Optional langgraph-backed multi-agent orchestration.
        Uses graph execution when available for predictable transitions.
        """
        if not (
            self.langchain_enabled
            and self.langchain_use_langgraph_multi_agent
            and LANGGRAPH_AVAILABLE
            and StateGraph
            and LANGGRAPH_END
        ):
            return None

        try:
            self._emit_langchain_callback("langgraph_multi_agent_start", {"query": user_input[:120]})

            async def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
                params = self._prepare_web_search_params({"query": state["query"], "max_results": 8, "workers": 6})
                research_raw, knowledge = await asyncio.gather(
                    self._execute_tool("web_search", params),
                    self._retrieve_knowledge(state["query"], top_k=4, context=context),
                )
                state["tool_outputs"]["research_agent"] = self._coerce_json(research_raw)
                state["tool_outputs"]["knowledge_agent"] = knowledge
                state["steps"].append({"agent": "research_agent", "success": True})
                state["steps"].append({"agent": "knowledge_agent", "success": True})
                return state

            async def analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
                analysis_prompt = json.dumps(
                    {
                        "goal": profile.goal,
                        "query": state["query"],
                        "research": state["tool_outputs"].get("research_agent"),
                        "knowledge": state["tool_outputs"].get("knowledge_agent"),
                    },
                    default=str,
                )
                analysis = await self._generate_with_model(
                    model_name="autobot_thinking",
                    system_prompt="You are the analysis_agent. Produce a concise factual analysis.",
                    user_prompt=analysis_prompt,
                    temperature=0.2,
                    max_tokens=700,
                )
                state["model_outputs"]["analysis_agent"] = analysis
                state["analysis"] = analysis
                state["steps"].append({"agent": "analysis_agent", "success": True})
                return state

            async def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
                planning_json_prompt = (
                    "Based on the analysis, decide if extra tools are needed.\n"
                    "Return JSON with keys: tool_calls (array), notes.\n"
                    "tool_calls entries: {\"tool\": \"web_search|send_email\", \"params\": {...}}.\n"
                    f"Analysis:\n{state['analysis']}"
                )
                planning_output = await self._generate_with_model(
                    model_name="autobot_thinking",
                    system_prompt="You are the planning_agent in a specialist team.",
                    user_prompt=planning_json_prompt,
                    temperature=0.2,
                    max_tokens=380,
                )
                planning_data = self._extract_json_object(planning_output) or {}
                state["model_outputs"]["planning_agent"] = planning_data
                state["planning"] = planning_data
                state["steps"].append({"agent": "planning_agent", "success": True})

                extra_tool_calls = planning_data.get("tool_calls", []) if isinstance(planning_data, dict) else []
                if isinstance(extra_tool_calls, list):
                    for idx, call in enumerate(extra_tool_calls, start=1):
                        if not isinstance(call, dict):
                            continue
                        tool_name = self._normalize_tool_name(str(call.get("tool", "")))
                        params = call.get("params", {}) or {}
                        if tool_name not in {"web_search", "send_email"}:
                            continue
                        if tool_name == "web_search":
                            params = self._prepare_web_search_params(params)
                        if tool_name == "send_email":
                            payload = await self._build_email_payload(state["query"], context)
                            payload.update(params)
                            params = payload
                            missing = params.pop("_missing_required", [])
                            if missing:
                                continue
                        output = await self._execute_tool(tool_name, params)
                        state["tool_outputs"][f"planning_agent_{tool_name}_{idx}"] = self._coerce_json(output)
                        state["steps"].append({"agent": "planning_agent", "tool": tool_name, "success": True})
                return state

            async def response_node(state: Dict[str, Any]) -> Dict[str, Any]:
                response = await self._synthesize_response(
                    user_input=state["query"],
                    profile=profile,
                    flow=FlowType.MULTI_AGENT,
                    artifacts={
                        "intent": intent_json,
                        "analysis": state.get("analysis"),
                        "team_outputs": {
                            "research": state["tool_outputs"].get("research_agent"),
                            "knowledge": state["tool_outputs"].get("knowledge_agent"),
                            "planning": state.get("planning"),
                            "extra_tool_outputs": state.get("tool_outputs"),
                        },
                    },
                )
                state["response"] = response
                state["steps"].append({"agent": "response_agent", "success": True})
                return state

            workflow = StateGraph(dict)
            workflow.add_node("research_agent", research_node)
            workflow.add_node("analysis_agent", analysis_node)
            workflow.add_node("planning_agent", planning_node)
            workflow.add_node("response_agent", response_node)
            workflow.set_entry_point("research_agent")
            workflow.add_edge("research_agent", "analysis_agent")
            workflow.add_edge("analysis_agent", "planning_agent")
            workflow.add_edge("planning_agent", "response_agent")
            workflow.add_edge("response_agent", LANGGRAPH_END)

            compiled = workflow.compile()
            initial_state = {
                "query": user_input,
                "tool_outputs": {},
                "model_outputs": {},
                "steps": [],
                "analysis": "",
                "planning": {},
                "response": "",
            }
            final_state = await compiled.ainvoke(initial_state)
            self._emit_langchain_callback(
                "langgraph_multi_agent_end",
                {"steps": len(final_state.get("steps", []))},
            )
            return FlowExecutionResult(
                response=str(final_state.get("response", "")),
                flow=FlowType.MULTI_AGENT,
                steps=final_state.get("steps", []),
                tool_outputs=final_state.get("tool_outputs", {}),
                model_outputs=final_state.get("model_outputs", {}),
                metadata={"agents_used": 5, "engine": "langgraph"},
            )
        except Exception as exc:
            self.logger.warning("langgraph multi-agent path failed, using fallback path: %s", exc)
            self._emit_langchain_callback("langgraph_multi_agent_error", {"error": str(exc)})
            return None

    async def _run_pev_flow(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FlowExecutionResult:
        """Planner -> executor -> verifier loop with retries."""
        attempts: List[Dict[str, Any]] = []
        working_query = user_input
        final_execution: Dict[str, Any] = {}
        verification: Dict[str, Any] = {"passed": False}

        for attempt in range(1, self.max_pev_attempts + 1):
            plan_steps = await self._generate_plan_steps(working_query, profile, context)
            execution_results = await self._execute_plan_steps(plan_steps, working_query, profile, context)
            verification = await self._verify_execution(working_query, plan_steps, execution_results)

            attempts.append(
                {
                    "attempt": attempt,
                    "plan": plan_steps,
                    "execution_results": execution_results,
                    "verification": verification,
                }
            )
            final_execution = execution_results

            if verification.get("passed") is True:
                break

            guidance = verification.get("guidance") or verification.get("issues") or ""
            working_query = f"{user_input}\nVerifier feedback: {guidance}"

        response = await self._synthesize_response(
            user_input=user_input,
            profile=profile,
            flow=FlowType.PEV,
            artifacts={
                "intent": intent_json,
                "attempts": attempts,
                "final_execution": final_execution,
                "final_verification": verification,
            },
        )
        return FlowExecutionResult(
            response=response,
            flow=FlowType.PEV,
            steps=attempts,
            tool_outputs={"final_execution": final_execution},
            model_outputs={"verification": verification},
            metadata={"attempts": len(attempts), "verified": bool(verification.get("passed"))},
        )

    async def _react_decision(
        self,
        user_input: str,
        profile: GoalProfile,
        intent_json: Dict[str, Any],
        observations: List[Dict[str, Any]],
        iteration: int,
    ) -> Dict[str, Any]:
        """Ask reasoning model for next ReAct step as JSON."""
        format_instructions = (
            "Return JSON object with keys: action, thought, tool, tool_params, memory_query, final_answer."
        )
        if self._langchain_available and PydanticOutputParser is not None:
            try:
                parser = PydanticOutputParser(pydantic_object=ReactDecisionSchema)
                format_instructions = parser.get_format_instructions()
            except Exception:
                pass
        system_prompt = (
            "You are a ReAct controller. Choose exactly one action: tool, memory, final.\n"
            "Return JSON only with keys: action, thought, tool, tool_params, memory_query, final_answer."
        )
        prompt_template = (
            "iteration={iteration}\n"
            "goal={goal}\n"
            "query={query}\n"
            "intent={intent}\n"
            "available_tools={available_tools}\n"
            "observations={observations}\n"
            "{format_instructions}"
        )
        prompt = self._build_prompt_with_template(
            prompt_template,
            {
                "iteration": iteration,
                "goal": profile.goal,
                "query": user_input,
                "intent": json.dumps(intent_json, default=str),
                "available_tools": [
                    t.name for t in self.langchain_tools if getattr(t, "name", None)
                ]
                or ["web_search", "send_email"],
                "observations": json.dumps(observations, default=str),
                "format_instructions": format_instructions,
            },
        )
        response = await self._generate_with_model(
            model_name="autobot_thinking",
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=320,
        )
        parsed = self._parse_with_schema(response, ReactDecisionSchema)
        if isinstance(parsed, dict):
            return parsed
        return {"action": "final", "final_answer": ""}

    async def _generate_plan_steps(
        self, user_input: str, profile: GoalProfile, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate structured plan steps."""
        format_instructions = (
            "Return JSON object with key `steps`, each step contains: "
            "id, purpose, executor, input, depends_on, timeout_sec."
        )
        if self._langchain_available and PydanticOutputParser is not None:
            try:
                parser = PydanticOutputParser(pydantic_object=PlanStepsSchema)
                format_instructions = parser.get_format_instructions()
            except Exception:
                pass
        system_prompt = (
            "You are a planning agent. Return JSON only with key `steps`.\n"
            "Each step keys: id, purpose, executor, input, depends_on, timeout_sec.\n"
            "Allowed executor values: web_search, send_email, rag_retrieve, reasoning, synthesis."
        )
        prompt_template = (
            "goal={goal}\n"
            "query={query}\n"
            "complexity={complexity}\n"
            "specialist={specialist}\n"
            "required_tools={required_tools}\n"
            "requires_rag={requires_rag}\n"
            "context={context}\n"
            "{format_instructions}"
        )
        prompt = self._build_prompt_with_template(
            prompt_template,
            {
                "goal": profile.goal,
                "query": user_input,
                "complexity": profile.complexity.value,
                "specialist": profile.specialist.value,
                "required_tools": profile.required_tools,
                "requires_rag": profile.requires_rag,
                "context": json.dumps(context, default=str),
                "format_instructions": format_instructions,
            },
        )
        model_response = await self._generate_with_model(
            model_name="autobot_thinking",
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=700,
        )
        parsed = self._parse_with_schema(model_response, PlanStepsSchema)
        if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
            normalized = self._normalize_plan_steps(parsed["steps"], user_input)
            if normalized:
                return normalized

        fallback_plan = await self.planner.create_plan(user_input, {"intent": profile.intent}, context)
        fallback_steps = fallback_plan.get("steps", []) if isinstance(fallback_plan, dict) else []
        normalized_fallback = self._normalize_plan_steps(fallback_steps, user_input)
        if normalized_fallback:
            return normalized_fallback

        return [
            {
                "id": "step_1",
                "purpose": "Reason about the user goal",
                "executor": "reasoning",
                "input": user_input,
                "depends_on": [],
                "timeout_sec": 45,
            }
        ]

    def _normalize_plan_steps(self, steps: List[Dict[str, Any]], default_query: str) -> List[Dict[str, Any]]:
        """Normalize planner outputs into the expected plan-step schema."""
        normalized_steps: List[Dict[str, Any]] = []
        for index, raw in enumerate(steps, start=1):
            if not isinstance(raw, dict):
                continue

            executor = raw.get("executor", raw.get("tool_or_model", "reasoning"))
            executor = str(executor).strip().lower().replace("-", "_")
            if executor == "llm":
                executor = "reasoning"
            if executor not in {"web_search", "send_email", "rag_retrieve", "reasoning", "synthesis"}:
                executor = "reasoning"

            raw_input = raw.get("input", raw.get("inputs", default_query))
            if isinstance(raw_input, list):
                step_input: Any = raw_input[0] if raw_input else default_query
            else:
                step_input = raw_input if raw_input else default_query

            depends_on = raw.get("depends_on", [])
            if not isinstance(depends_on, list):
                depends_on = []

            timeout_sec = raw.get("timeout_sec", 45)
            try:
                timeout_sec = int(timeout_sec)
            except Exception:
                timeout_sec = 45

            normalized_steps.append(
                {
                    "id": str(raw.get("id", f"step_{index}")),
                    "purpose": str(raw.get("purpose", "")).strip() or f"Execute {executor}",
                    "executor": executor,
                    "input": step_input,
                    "depends_on": depends_on,
                    "timeout_sec": max(10, min(timeout_sec, 180)),
                }
            )
        return normalized_steps

    async def _execute_plan_steps(
        self,
        steps: List[Dict[str, Any]],
        user_input: str,
        profile: GoalProfile,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute plan steps honoring dependencies."""
        if not steps:
            return {}

        pending: Dict[str, Dict[str, Any]] = {str(step.get("id", f"step_{idx}")): step for idx, step in enumerate(steps)}
        completed: Dict[str, Dict[str, Any]] = {}
        safety_counter = 0

        while pending and safety_counter < len(steps) * 3:
            safety_counter += 1
            progressed = False

            for step_id, step in list(pending.items()):
                deps = [str(dep) for dep in step.get("depends_on", [])]
                if deps and not all(dep in completed for dep in deps):
                    continue

                result = await self._execute_plan_step(step, user_input, profile, completed, context)
                completed[step_id] = result
                pending.pop(step_id, None)
                progressed = True

            if not progressed:
                step_id, step = next(iter(pending.items()))
                result = await self._execute_plan_step(step, user_input, profile, completed, context)
                completed[step_id] = result
                pending.pop(step_id, None)

        return completed

    async def _execute_plan_step(
        self,
        step: Dict[str, Any],
        user_input: str,
        profile: GoalProfile,
        completed: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single normalized plan step."""
        step_id = str(step.get("id", "unknown"))
        executor = str(step.get("executor", "reasoning"))
        purpose = str(step.get("purpose", ""))
        step_input = step.get("input", user_input)
        timeout_sec = int(step.get("timeout_sec", 45))
        started = time.time()

        async def _run() -> Dict[str, Any]:
            if executor == "web_search":
                params = self._prepare_web_search_params({"query": str(step_input)})
                output = await self._execute_tool("web_search", params)
                return {"success": True, "executor": executor, "output": self._coerce_json(output), "purpose": purpose}

            if executor == "send_email":
                payload = await self._build_email_payload(str(step_input), context)
                missing = payload.pop("_missing_required", [])
                if missing:
                    return {
                        "success": False,
                        "executor": executor,
                        "output": {"error": f"missing required fields: {missing}"},
                        "purpose": purpose,
                    }
                output = await self._execute_tool("send_email", payload)
                return {"success": True, "executor": executor, "output": self._coerce_json(output), "purpose": purpose}

            if executor == "rag_retrieve":
                output = await self._retrieve_knowledge(str(step_input), top_k=3, context=context)
                return {"success": True, "executor": executor, "output": output, "purpose": purpose}

            if executor in {"reasoning", "synthesis"}:
                model_name = "autobot_thinking" if executor == "reasoning" else "autobot_instruct"
                prompt_payload = {
                    "step_id": step_id,
                    "purpose": purpose,
                    "input": step_input,
                    "previous_results": completed,
                    "user_query": user_input,
                    "goal": profile.goal,
                }
                output = await self._generate_with_model(
                    model_name=model_name,
                    system_prompt=f"You are executing plan step {step_id}.",
                    user_prompt=json.dumps(prompt_payload, default=str),
                    temperature=0.2,
                    max_tokens=600 if executor == "reasoning" else 420,
                )
                return {"success": True, "executor": executor, "output": output, "purpose": purpose}

            output = await self._generate_with_model(
                model_name="autobot_thinking",
                system_prompt="You are a fallback executor.",
                user_prompt=str(step_input),
                temperature=0.2,
                max_tokens=320,
            )
            return {"success": True, "executor": "reasoning", "output": output, "purpose": purpose}

        try:
            result = await asyncio.wait_for(_run(), timeout=timeout_sec)
            result["duration_sec"] = round(time.time() - started, 3)
            return result
        except asyncio.TimeoutError:
            return {
                "success": False,
                "executor": executor,
                "output": {"error": f"timeout after {timeout_sec} seconds"},
                "purpose": purpose,
                "duration_sec": round(time.time() - started, 3),
            }
        except Exception as exc:
            self.logger.exception("plan step failed id=%s executor=%s: %s", step_id, executor, exc)
            return {
                "success": False,
                "executor": executor,
                "output": {"error": str(exc)},
                "purpose": purpose,
                "duration_sec": round(time.time() - started, 3),
            }

    async def _verify_execution(
        self, user_input: str, plan_steps: List[Dict[str, Any]], execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verifier step for PEV flow."""
        format_instructions = (
            "Return JSON object: {\"passed\": bool, \"score\": 0-1, \"issues\": [], \"guidance\": \"...\"}."
        )
        if self._langchain_available and PydanticOutputParser is not None:
            try:
                parser = PydanticOutputParser(pydantic_object=VerificationSchema)
                format_instructions = parser.get_format_instructions()
            except Exception:
                pass
        system_prompt = (
            "You are a verifier. Validate whether execution solved the task.\n"
            "Return JSON only: {\"passed\": bool, \"score\": 0-1, \"issues\": [], \"guidance\": \"...\"}."
        )
        prompt_template = (
            "query={query}\n"
            "plan_steps={plan_steps}\n"
            "execution_results={execution_results}\n"
            "{format_instructions}"
        )
        prompt = self._build_prompt_with_template(
            prompt_template,
            {
                "query": user_input,
                "plan_steps": json.dumps(plan_steps, default=str),
                "execution_results": json.dumps(execution_results, default=str),
                "format_instructions": format_instructions,
            },
        )
        response = await self._generate_with_model(
            model_name="autobot_thinking",
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=380,
        )
        parsed = self._parse_with_schema(response, VerificationSchema)
        if isinstance(parsed, dict):
            return parsed

        all_success = True
        for result in execution_results.values():
            if isinstance(result, dict) and result.get("success") is False:
                all_success = False
                break
        return {
            "passed": all_success,
            "score": 1.0 if all_success else 0.4,
            "issues": [] if all_success else ["one or more plan steps failed"],
            "guidance": "replan failed steps" if not all_success else "none",
        }

    async def _synthesize_response(
        self,
        user_input: str,
        profile: GoalProfile,
        flow: FlowType,
        artifacts: Dict[str, Any],
    ) -> str:
        """Final response synthesis from aggregated artifacts."""
        serialized_artifacts = self._truncate_text(artifacts, 7000)
        system_prompt = (
            "You are AutoBot. Produce a clear final answer for the user.\n"
            "Use available evidence. Mention tool results when relevant.\n"
            "If email was sent, include status clearly.\n"
            "If evidence is insufficient, state that and suggest the next required input."
        )
        if profile.specialist == SpecialistType.RESEARCHER:
            system_prompt += (
                "\nAs the researcher specialist, prioritize factual accuracy and include concise "
                "source-aware references when available."
            )
        elif profile.specialist == SpecialistType.CODER:
            system_prompt += (
                "\nAs the coding specialist, provide clean implementation output. "
                "Use fenced code blocks for code and keep explanations concise."
            )
        elif profile.specialist == SpecialistType.SUMMARIZER:
            system_prompt += (
                "\nAs the summarizer specialist, produce concise summaries that preserve key facts."
            )
        elif profile.specialist == SpecialistType.ANALYST:
            system_prompt += (
                "\nAs the analyst specialist, present structured insights, assumptions, and conclusions."
            )
        elif profile.specialist == SpecialistType.PLANNER:
            system_prompt += (
                "\nAs the planner specialist, provide an ordered execution plan with clear dependencies."
            )
        elif profile.specialist == SpecialistType.RETRIEVER:
            system_prompt += (
                "\nAs the retriever specialist, ground output in retrieved context and indicate gaps."
            )
        elif profile.specialist == SpecialistType.COMPLIANCE_CHECKER:
            system_prompt += (
                "\nAs the compliance checker specialist, explicitly validate constraints and highlight violations."
            )
        elif profile.specialist == SpecialistType.OPTIMIZER:
            system_prompt += (
                "\nAs the optimizer specialist, prioritize efficient solutions and discuss tradeoffs."
            )
        elif profile.specialist == SpecialistType.EXPLAINER:
            system_prompt += (
                "\nAs the explainer specialist, provide clear step-by-step reasoning suitable for learning."
            )
        prompt = (
            f"User query: {user_input}\n"
            f"Goal: {profile.goal}\n"
            f"Flow: {flow.value}\n"
            f"Specialist: {profile.specialist.value}\n"
            f"Complexity: {profile.complexity.value}\n"
            f"Artifacts:\n{serialized_artifacts}"
        )

        model_name = "autobot_instruct"
        if profile.requires_rag:
            model_name = "autobot_rag"
        elif profile.complexity == TaskComplexity.HIGH:
            model_name = "autobot_thinking"

        response = await self._generate_with_model(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=800,
        )
        return response.strip()

    async def _build_email_payload(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize email payload for send_email tool."""
        payload: Dict[str, Any] = {}
        if isinstance(context.get("email_payload"), dict):
            payload.update(context["email_payload"])

        email_matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_input)
        if email_matches and "to" not in payload:
            payload["to"] = ", ".join(sorted(set(email_matches)))

        extraction_prompt = (
            "Extract email fields from the user request.\n"
            "Return JSON only with keys: to, subject, body, body_html, cc, bcc, attachments.\n"
            "Use empty string or [] when unknown."
        )
        extraction_response = await self._generate_with_model(
            model_name="autobot_instruct",
            system_prompt=extraction_prompt,
            user_prompt=user_input,
            temperature=0.1,
            max_tokens=320,
            fallback_mode="intent",
        )
        extracted = self._parse_with_schema(extraction_response, EmailExtractionSchema)
        if isinstance(extracted, dict):
            for key in ["to", "subject", "body", "body_html", "cc", "bcc", "attachments"]:
                value = extracted.get(key)
                if value not in [None, "", []]:
                    payload[key] = value

        required = ["to", "subject", "body"]
        missing = [field for field in required if not payload.get(field)]
        payload["_missing_required"] = missing
        return payload

    async def _retrieve_knowledge(
        self, query: str, top_k: int = 3, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve context from RAG pipeline + episodic/semantic memory."""
        if not query:
            return {"type": "memory_recall", "query": "", "count": 0, "context": "", "sources": []}
        context = context or {}

        episodic_semantic = await self.memory.recall_episodic_with_semantic(
            query=query, episodic_limit=max(2, top_k), semantic_limit=max(3, top_k)
        )
        local_doc_chunks: List[str] = []
        local_doc_paths = context.get("local_documents", [])
        if isinstance(local_doc_paths, list) and local_doc_paths:
            try:
                loop = asyncio.get_running_loop()
                local_doc_chunks = await loop.run_in_executor(
                    None, lambda: self._load_local_documents_via_langchain(local_doc_paths)
                )
            except Exception as exc:
                self.logger.warning("local document loading failed: %s", exc)

        episodic_lines = episodic_semantic.get("episodic", [])
        semantic_lines = [
            f"{fact.get('key')}: {fact.get('value')}" for fact in episodic_semantic.get("semantic", [])
        ]
        bm25_corpus = list(episodic_lines) + list(semantic_lines) + list(local_doc_chunks)
        bm25_reranked = self._bm25_rerank_texts(query=query, texts=bm25_corpus, top_k=top_k)

        pipeline = await self._ensure_rag_pipeline()
        if pipeline is not None:
            try:
                loop = asyncio.get_running_loop()
                context_obj, results = await loop.run_in_executor(
                    None,
                    lambda: pipeline.retrieve_context(
                        query=query,
                        top_k=top_k,
                        max_context_chunks=min(4, max(1, top_k)),
                        filters=None,
                    ),
                )
                if context_obj:
                    merged_context_parts = [context_obj.context_text]
                    if episodic_lines:
                        merged_context_parts.append(
                            "Episodic Memory:\n"
                            + "\n".join(f"- {item}" for item in episodic_lines)
                        )
                    if semantic_lines:
                        merged_context_parts.append(
                            "Semantic Memory:\n"
                            + "\n".join(f"- {line}" for line in semantic_lines)
                        )
                    if local_doc_chunks:
                        merged_context_parts.append(
                            "Local Documents:\n"
                            + "\n".join(f"- {self._truncate_text(chunk, 260)}" for chunk in local_doc_chunks[:top_k])
                        )
                    if bm25_reranked:
                        merged_context_parts.append(
                            "BM25 Reranked Context:\n"
                            + "\n".join(f"- {self._truncate_text(chunk, 260)}" for chunk in bm25_reranked)
                        )
                    return {
                        "type": "rag_plus_memory",
                        "query": query,
                        "count": len(results),
                        "context": "\n\n".join(merged_context_parts),
                        "sources": context_obj.sources,
                        "results": [r.to_dict() for r in results],
                        "episodic_semantic": episodic_semantic,
                        "bm25_reranked": bm25_reranked,
                        "local_doc_chunks": len(local_doc_chunks),
                    }
            except Exception as exc:
                self.logger.warning("rag retrieval failed, using memory fallback: %s", exc)

        memories = await self.memory.recall(query, limit=top_k)
        combined_recall_chunks = list(memories) + bm25_reranked
        combined_recall_chunks = self._bm25_rerank_texts(query, combined_recall_chunks, top_k=top_k)
        context_parts = []
        if combined_recall_chunks:
            context_parts.append("\n".join(combined_recall_chunks))
        if episodic_lines:
            context_parts.append("Episodic Memory:\n" + "\n".join(f"- {line}" for line in episodic_lines))
        if semantic_lines:
            context_parts.append("Semantic Memory:\n" + "\n".join(f"- {line}" for line in semantic_lines))
        if local_doc_chunks:
            context_parts.append(
                "Local Documents:\n"
                + "\n".join(f"- {self._truncate_text(chunk, 260)}" for chunk in local_doc_chunks[:top_k])
            )
        return {
            "type": "memory_recall_with_episodic_semantic",
            "query": query,
            "count": len(combined_recall_chunks) + len(episodic_lines) + len(semantic_lines),
            "context": "\n\n".join(context_parts),
            "sources": [],
            "results": combined_recall_chunks,
            "episodic_semantic": episodic_semantic,
            "bm25_reranked": bm25_reranked,
            "local_doc_chunks": len(local_doc_chunks),
        }

    async def _ensure_rag_pipeline(self):
        """Lazy RAG pipeline initialization."""
        if self.rag_available is False:
            return None
        if self.rag_pipeline is not None:
            return self.rag_pipeline

        async with self._rag_lock:
            if self.rag_pipeline is not None:
                return self.rag_pipeline
            if self.rag_available is False:
                return None

            try:
                from memory.rag_pipeline import RAGPipeline

                store_path = self.config.get("memory", {}).get("vector_store", "./memory/vector_store")
                loop = asyncio.get_running_loop()
                self.rag_pipeline = await loop.run_in_executor(
                    None, lambda: RAGPipeline(store_path=store_path)
                )
                self.rag_available = True
                self.logger.info("RAG pipeline initialized from %s", store_path)
            except Exception as exc:
                self.rag_available = False
                self.rag_pipeline = None
                self.logger.warning("RAG pipeline unavailable: %s", exc)
            return self.rag_pipeline

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Tool dispatcher for web_search and send_email."""
        normalized = self._normalize_tool_name(tool_name)
        if normalized == "web_search":
            return await self.tool_registry.execute_tool("web_search", self._prepare_web_search_params(params))
        if normalized == "send_email":
            return await self._execute_send_email(params)
        if normalized == "rag_retrieve":
            query = str(params.get("query", ""))
            top_k = int(params.get("top_k", 3))
            retrieval_context = params.get("context", {})
            if not isinstance(retrieval_context, dict):
                retrieval_context = {}
            return await self._retrieve_knowledge(query, top_k=top_k, context=retrieval_context)
        raise ValueError(f"unsupported tool: {tool_name}")

    async def _execute_send_email(self, params: Dict[str, Any]) -> Any:
        """Execute send_email from tools/send-email/send-email.py."""
        module_path = self.tool_root / "send-email" / "send-email.py"
        module = self._load_module_from_path("send_email_tool_module", module_path)
        send_email_fn = getattr(module, "send_email", None)
        if send_email_fn is None:
            raise RuntimeError("send_email function not found in tools/send-email/send-email.py")

        payload = {
            "to": params.get("to", ""),
            "subject": params.get("subject", ""),
            "body": params.get("body", ""),
            "body_html": params.get("body_html"),
            "attachments": params.get("attachments", []),
            "cc": params.get("cc"),
            "bcc": params.get("bcc"),
        }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: send_email_fn(
                to=payload["to"],
                subject=payload["subject"],
                body=payload["body"],
                body_html=payload["body_html"],
                attachments=payload["attachments"],
                cc=payload["cc"],
                bcc=payload["bcc"],
            ),
        )

    def _prepare_web_search_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize web search parameters."""
        query = ""
        max_results = 8
        workers = 6

        if isinstance(params, dict):
            query = str(params.get("query", params.get("input", ""))).strip()
            max_results = int(params.get("max_results", max_results))
            workers = int(params.get("workers", workers))
        else:
            query = str(params).strip()

        max_results = max(1, min(max_results, 20))
        workers = max(1, min(workers, 12))
        return {"query": query, "max_results": max_results, "workers": workers}

    async def _generate_with_model(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        messages: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        fallback_mode: str = "reasoning",
    ) -> str:
        """
        Unified generation entry:
        - try specialized model (autobot-instruct / autobot-thinking / autobot-rag)
        - fallback to existing LLMInterface
        """
        model_name = model_name.strip().lower()
        max_tokens = max(32, min(int(max_tokens), self.max_tokens_hard_limit))
        self._register_model_usage(model_name)
        self._emit_langchain_callback(
            "model_generation_start",
            {"model": model_name, "max_tokens": max_tokens, "temperature": temperature},
        )

        try:
            runtime = await self._ensure_specialized_model(model_name)
            if runtime is not None:
                payload = await self._run_specialized_generation(
                    model_name=model_name,
                    runtime=runtime,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                    tools=tools,
                )
                if isinstance(payload, dict):
                    if payload.get("success") is False and model_name == "autobot_thinking":
                        raise RuntimeError(payload.get("error", "autobot_thinking generation failed"))
                    text = payload.get("text") or payload.get("answer") or ""
                else:
                    text = str(payload)
                text = str(text).strip()
                if text:
                    self._mark_request_debug_event(
                        "model_generation_success",
                        {"model": model_name, "used_specialized_runtime": True},
                    )
                    self._emit_langchain_callback(
                        "model_generation_success",
                        {"model": model_name, "specialized_runtime": True},
                    )
                    return text
        except Exception as exc:
            self.logger.warning("specialized model %s failed, using fallback: %s", model_name, exc)
            self._mark_request_debug_event(
                "model_generation_fallback",
                {"model": model_name, "error": str(exc)},
            )
            self._emit_langchain_callback(
                "model_generation_fallback",
                {"model": model_name, "error": str(exc)},
            )

        fallback_text = await self._generate_with_llm_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            mode=fallback_mode,
        )
        self._mark_request_debug_event(
            "model_generation_success",
            {"model": f"llm_fallback:{fallback_mode}", "used_specialized_runtime": False},
        )
        self._emit_langchain_callback(
            "model_generation_success",
            {"model": f"llm_fallback:{fallback_mode}", "specialized_runtime": False},
        )
        return fallback_text

    async def _ensure_specialized_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model runtime lazily (once) and cache it."""
        if model_name not in self.specialized_models:
            return None

        state = self.specialized_models[model_name]
        if state.get("status") == "ready":
            return state
        if state.get("status") == "failed":
            return None

        lock = self.model_locks[model_name]
        async with lock:
            state = self.specialized_models[model_name]
            if state.get("status") == "ready":
                return state
            if state.get("status") == "failed":
                return None

            try:
                runtime = await self._load_model_runtime(model_name)
                runtime["status"] = "ready"
                self.specialized_models[model_name] = runtime
                self.logger.info("loaded specialized model runtime: %s", model_name)
                return runtime
            except Exception as exc:
                self.specialized_models[model_name] = {"status": "failed", "error": str(exc)}
                self.logger.warning("failed to load specialized model %s: %s", model_name, exc)
                return None

    async def _load_model_runtime(self, model_name: str) -> Dict[str, Any]:
        """Load model runtime per model type."""
        if model_name == "autobot_instruct":
            return await self._load_autobot_instruct_runtime()
        if model_name == "autobot_thinking":
            return await self._load_autobot_thinking_runtime()
        if model_name == "autobot_rag":
            return await self._load_autobot_rag_runtime()
        raise ValueError(f"unknown model runtime: {model_name}")

    async def _load_autobot_instruct_runtime(self) -> Dict[str, Any]:
        loader_module = self._load_module_from_path(
            "autobot_instruct_loader", self.model_root / "load-autobot-instruct.py"
        )
        generator_module = self._load_module_from_path(
            "autobot_instruct_generator", self.model_root / "generate-autobot-instruct.py"
        )
        load_fn = getattr(loader_module, "load_autobot_instruct")
        generate_fn = getattr(generator_module, "generate_autobot_instruct")

        loop = asyncio.get_running_loop()
        tokenizer, model, model_dir = await loop.run_in_executor(
            None, lambda: load_fn(base_dir=str(self.project_root), device=self.device)
        )
        return {
            "tokenizer": tokenizer,
            "model": model,
            "generator": generate_fn,
            "model_dir": model_dir,
        }

    async def _load_autobot_thinking_runtime(self) -> Dict[str, Any]:
        loader_module = self._load_module_from_path(
            "autobot_thinking_loader", self.model_root / "load_autobot_thinking.py"
        )
        generator_module = self._load_module_from_path(
            "autobot_thinking_generator", self.model_root / "generate_autobot_thinking.py"
        )
        load_fn = getattr(loader_module, "load_autobot_thinking_model")
        generate_fn = getattr(generator_module, "generate_autobot_thinking")

        if inspect.iscoroutinefunction(load_fn):
            tokenizer, model, success = await load_fn()
        else:
            loop = asyncio.get_running_loop()
            tokenizer, model, success = await loop.run_in_executor(None, load_fn)

        if not success or tokenizer is None or model is None:
            raise RuntimeError("load_autobot_thinking_model reported failure")

        return {
            "tokenizer": tokenizer,
            "model": model,
            "generator": generate_fn,
            "model_dir": str(self.model_root / "autobot-thinking"),
        }

    async def _load_autobot_rag_runtime(self) -> Dict[str, Any]:
        loader_module = self._load_module_from_path(
            "autobot_rag_loader", self.model_root / "load-autobot-rag.py"
        )
        generator_module = self._load_module_from_path(
            "autobot_rag_generator", self.model_root / "generate-autobot-rag.py"
        )
        load_fn = getattr(loader_module, "load_autobot_rag")
        generate_fn = getattr(generator_module, "generate_autobot_response")

        model_path = str(self.model_root / "autobot-rag")
        loop = asyncio.get_running_loop()
        tokenizer, model = await loop.run_in_executor(
            None, lambda: load_fn(model_path=model_path, device=self.device)
        )
        if tokenizer is None or model is None:
            raise RuntimeError("load_autobot_rag returned empty model state")

        return {
            "tokenizer": tokenizer,
            "model": model,
            "generator": generate_fn,
            "model_dir": model_path,
        }

    async def _run_specialized_generation(
        self,
        model_name: str,
        runtime: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        messages: Optional[List[Dict[str, str]]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Any:
        """Run generation against a loaded specialized runtime."""
        tokenizer = runtime["tokenizer"]
        model = runtime["model"]
        generate_fn = runtime["generator"]
        loop = asyncio.get_running_loop()

        if model_name == "autobot_instruct":
            tool_payload = tools if tools is not None else TOOL_SCHEMA
            return await loop.run_in_executor(
                None,
                lambda: generate_fn(
                    model=model,
                    tokenizer=tokenizer,
                    system_message=system_prompt,
                    user_prompt=user_prompt,
                    device=self.device,
                    max_context_length=self.max_context_length,
                    max_tokens=max_tokens,
                    max_tokens_hard_limit=self.max_tokens_hard_limit,
                    temperature=temperature,
                    tools_json=tool_payload,
                    messages=messages,
                ),
            )

        if model_name == "autobot_thinking":
            tool_payload = json.dumps(tools if tools is not None else TOOL_SCHEMA)
            return await loop.run_in_executor(
                None,
                lambda: generate_fn(
                    model=model,
                    tokenizer=tokenizer,
                    system_message=system_prompt,
                    user_prompt=user_prompt,
                    device=self.device,
                    max_context_length=self.max_context_length,
                    max_tokens=max_tokens,
                    max_tokens_hard_limit=self.max_tokens_hard_limit,
                    temperature=temperature,
                    tools_json=tool_payload,
                ),
            )

        if model_name == "autobot_rag":
            return await loop.run_in_executor(
                None,
                lambda: generate_fn(
                    model=model,
                    tokenizer=tokenizer,
                    system_message=system_prompt,
                    user_prompt=user_prompt,
                    device=self.device,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_context_length=self.max_context_length,
                    max_tokens_hard_limit=self.max_tokens_hard_limit,
                    model_label="autobot-rag",
                ),
            )

        raise ValueError(f"unsupported model for generation: {model_name}")

    async def _generate_with_llm_fallback(self, system_prompt: str, user_prompt: str, mode: str) -> str:
        """Fallback generation using existing LLMInterface."""
        if mode == "intent":
            return await self.llm.generate_with_intent_model(user_prompt, system_prompt)
        return await self.llm.generate_with_reasoning_model(user_prompt, system_prompt)

    def _load_module_from_path(self, module_key: str, path: Path):
        """Load and cache a python module from an explicit file path."""
        cache_key = f"{module_key}:{path.resolve()}"
        if cache_key in self.dynamic_modules:
            return self.dynamic_modules[cache_key]

        if not path.exists():
            raise FileNotFoundError(f"module path not found: {path}")

        spec = importlib.util.spec_from_file_location(module_key, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load module from {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.dynamic_modules[cache_key] = module
        return module

    def _start_request_debug(self, trace_id: str, user_input: str):
        """Initialize request-scoped debug payload."""
        payload = {
            "trace_id": trace_id,
            "user_input": user_input[:500],
            "started_at": time.time(),
            "chosen_flow": None,
            "chosen_specialist": None,
            "models_used": [],
            "events": [],
        }
        return self._request_debug_ctx.set(payload)

    def _end_request_debug(self, token):
        """Reset request-scoped debug payload context variable."""
        try:
            self._request_debug_ctx.reset(token)
        except Exception:
            pass

    def _mark_request_debug_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """Append one debug event for active request."""
        payload = self._request_debug_ctx.get()
        if not isinstance(payload, dict):
            return
        event = {
            "event": event_name,
            "timestamp": time.time(),
            "data": data or {},
        }
        payload["events"].append(event)

        # Keep only last 100 events
        if len(payload["events"]) > 100:
            payload["events"] = payload["events"][-100:]

        if event_name == "flow_selected":
            payload["chosen_flow"] = str(event.get("data", {}).get("flow"))
            payload["chosen_specialist"] = str(event.get("data", {}).get("specialist"))
        if event_name == "meta_controller_selected":
            payload["chosen_specialist"] = str(event.get("data", {}).get("specialist"))

    def _register_model_usage(self, model_name: str):
        """Track model usage for current request."""
        payload = self._request_debug_ctx.get()
        if not isinstance(payload, dict):
            return
        models_used = payload.get("models_used", [])
        if model_name not in models_used:
            models_used.append(model_name)
        payload["models_used"] = models_used

    def _get_request_debug_summary(self) -> Dict[str, Any]:
        """Return a compact debug summary for active request."""
        payload = self._request_debug_ctx.get()
        if not isinstance(payload, dict):
            return {}
        return {
            "trace_id": payload.get("trace_id"),
            "chosen_flow": payload.get("chosen_flow"),
            "chosen_specialist": payload.get("chosen_specialist"),
            "models_used": payload.get("models_used", []),
            "events": payload.get("events", []),
            "started_at": payload.get("started_at"),
        }

    async def get_debug_status(self, limit: int = 20) -> Dict[str, Any]:
        """
        Debug endpoint-like status payload.
        Can be exposed by any API/interface layer.
        """
        trace = self.execution_trace[-max(1, limit):]
        working_debug = await self.memory.get_working_memory("agent_trace", [])
        return {
            "active_request": self._get_request_debug_summary(),
            "recent_trace": trace,
            "working_debug_events": working_debug[-max(1, limit):] if isinstance(working_debug, list) else [],
            "cache_size": len(self.response_cache),
            "langchain": {
                "enabled": self.langchain_enabled,
                "core_available": LANGCHAIN_CORE_AVAILABLE,
                "memory_enabled": self.langchain_memory is not None,
                "bm25_enabled": bool(
                    self.langchain_enabled and self.langchain_use_bm25_rerank and LANGCHAIN_RETRIEVERS_AVAILABLE
                ),
                "loaders_enabled": bool(
                    self.langchain_enabled and self.langchain_use_local_doc_loader and LANGCHAIN_LOADERS_AVAILABLE
                ),
                "langgraph_multi_agent_enabled": bool(
                    self.langchain_enabled and self.langchain_use_langgraph_multi_agent and LANGGRAPH_AVAILABLE
                ),
                "langgraph_available": LANGGRAPH_AVAILABLE,
                "registered_tools": [t.name for t in self.langchain_tools if getattr(t, "name", None)],
            },
        }

    async def _record_agent_trace(self, event: Dict[str, Any]):
        """Store execution trace in memory manager working memory."""
        self.execution_trace.append(event)
        if len(self.execution_trace) > 200:
            self.execution_trace = self.execution_trace[-200:]

        try:
            await self.memory.set_working_memory("last_agent_event", event)
            await self.memory.append_working_memory("agent_trace", event, max_items=200)
        except Exception:
            # Trace persistence should never block response execution.
            pass

    async def shutdown(self):
        """Graceful shutdown."""
        if self.memory_batch:
            await self._process_memory_batch()

        self.running = False
        await self.memory.shutdown()
        await self.tool_registry.shutdown()

    async def _batch_memory_update(self, user_input: str, response: str, intent: str):
        """Batch memory updates to reduce write overhead."""
        self.memory_batch.append(
            {
                "user_input": user_input,
                "response": response,
                "intent": intent,
                "timestamp": time.time(),
            }
        )

        if len(self.memory_batch) >= self.batch_size:
            await self._process_memory_batch()

    async def _update_episodic_semantic_memory(
        self,
        user_input: str,
        response: str,
        intent: str,
        goal_profile: GoalProfile,
        flow_result: FlowExecutionResult,
    ):
        """
        Persist episodic + semantic memory per interaction using the
        architecture in arc-files/08_episodic_with_semantic.ipynb.
        """
        try:
            episode_summary = await self._create_episode_summary(user_input, response, intent, goal_profile)
            semantic_facts = await self._extract_semantic_facts(user_input, response)
            metadata = {
                "flow": goal_profile.flow.value,
                "complexity": goal_profile.complexity.value,
                "models_used": self._get_request_debug_summary().get("models_used", []),
                "result_metadata": flow_result.metadata,
            }
            episodic_result = await self.memory.add_episodic_memory(
                summary=episode_summary,
                source_user_input=user_input,
                source_response=response,
                intent=intent,
                metadata=metadata,
            )
            semantic_result = await self.memory.add_semantic_memories(
                facts=semantic_facts,
                source=f"flow:{goal_profile.flow.value}",
            )
            await self._record_agent_trace(
                {
                    "stage": "episodic_semantic_memory_updated",
                    "episodic": episodic_result,
                    "semantic": semantic_result,
                    "timestamp": time.time(),
                }
            )
        except Exception as exc:
            self.logger.warning("episodic+semantic memory update failed: %s", exc)

    async def _create_episode_summary(
        self, user_input: str, response: str, intent: str, goal_profile: GoalProfile
    ) -> str:
        """Create concise episodic memory summary for one interaction."""
        summarization_prompt = (
            "Create one concise sentence summarizing the interaction.\n"
            "Include user goal, assistant outcome, and key constraint/preference if present."
        )
        payload = (
            f"Intent: {intent}\n"
            f"Flow: {goal_profile.flow.value}\n"
            f"User: {user_input}\n"
            f"Assistant: {response}"
        )
        summary = await self._generate_with_model(
            model_name="autobot_instruct",
            system_prompt=summarization_prompt,
            user_prompt=payload,
            temperature=0.1,
            max_tokens=120,
            fallback_mode="intent",
        )
        cleaned = self._clean_response(summary).replace("\n", " ").strip()
        if cleaned:
            return cleaned[:500]
        return f"User asked about '{user_input[:140]}'; assistant responded with intent '{intent}'."

    async def _extract_semantic_facts(self, user_input: str, response: str) -> List[Dict[str, Any]]:
        """Extract stable semantic facts from interaction."""
        system_prompt = (
            "Extract durable semantic facts from this interaction.\n"
            "Return JSON only: {\"facts\": [{\"key\": \"...\", \"value\": \"...\", \"confidence\": 0.0-1.0}]}\n"
            "Only include facts likely useful in future tasks (preferences, identities, constraints, recurring goals)."
        )
        prompt = f"User: {user_input}\nAssistant: {response}"
        model_out = await self._generate_with_model(
            model_name="autobot_instruct",
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=260,
            fallback_mode="intent",
        )
        parsed = self._extract_json_object(model_out)
        facts: List[Dict[str, Any]] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
            for fact in parsed["facts"]:
                if isinstance(fact, dict) and fact.get("key") and fact.get("value"):
                    facts.append(
                        {
                            "key": str(fact["key"]).strip()[:200],
                            "value": str(fact["value"]).strip()[:500],
                            "confidence": float(fact.get("confidence", 0.7)),
                            "metadata": {"source": "model_extraction"},
                        }
                    )
        return facts

    async def _process_memory_batch(self):
        """Persist pending interactions in long-term memory."""
        if not self.memory_batch:
            return

        try:
            for interaction in self.memory_batch:
                await self.memory.add_interaction(
                    interaction["user_input"],
                    interaction["response"],
                    interaction["intent"],
                )
            self.logger.info("processed memory batch: %d interactions", len(self.memory_batch))
            self.memory_batch.clear()
        except Exception as exc:
            self.logger.error("memory batch update failed: %s", exc)
            self.memory_batch.clear()

    def _generate_cache_key(self, user_input: str, context: Optional[Dict] = None) -> str:
        """Create cache key from input + lightweight context fingerprint."""
        content = user_input.lower().strip()
        if context:
            content += "|" + self._serialize_for_cache(context)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _serialize_for_cache(self, context: Dict[str, Any]) -> str:
        try:
            return json.dumps(context, sort_keys=True, default=str)
        except Exception:
            return str(context)

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Return cached response if still valid."""
        item = self.response_cache.get(cache_key)
        if not item:
            return None
        if time.time() - item["timestamp"] >= self.cache_ttl:
            self.response_cache.pop(cache_key, None)
            return None
        return item["response"]

    def _cache_response(self, cache_key: str, response: str):
        """Cache response and keep cache bounded."""
        self.response_cache[cache_key] = {"response": response, "timestamp": time.time()}
        if len(self.response_cache) > 1000:
            oldest = sorted(self.response_cache, key=lambda key: self.response_cache[key]["timestamp"])[:100]
            for key in oldest:
                self.response_cache.pop(key, None)

    def _clean_response(self, response: str) -> str:
        """Remove obvious technical prefixes from final answer."""
        if not response:
            return ""
        lines = [line.strip() for line in str(response).splitlines()]
        cleaned = []
        for line in lines:
            lower = line.lower()
            if lower.startswith("intent classified"):
                continue
            if lower.startswith("tool results:"):
                continue
            if lower.startswith("debug:"):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _emit_langchain_callback(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        """Emit lightweight callback events when langchain callbacks are enabled."""
        if not self.langchain_callback:
            return
        payload = payload or {}
        try:
            text = f"{event_name}: {self._truncate_text(payload, 500)}"
            self.langchain_callback.on_text(text)
        except Exception:
            return

    def _get_langchain_history(self) -> str:
        """Return conversation history from langchain memory if configured."""
        if not self.langchain_memory:
            return ""
        try:
            memory_vars = self.langchain_memory.load_memory_variables({})
            history = memory_vars.get("chat_history", "")
            return str(history)
        except Exception:
            return ""

    def _save_langchain_history(self, user_input: str, response: str):
        """Persist one turn in langchain memory, if enabled."""
        if not self.langchain_memory:
            return
        try:
            self.langchain_memory.save_context({"input": user_input}, {"output": response})
        except Exception:
            return

    def _build_langchain_tools(self) -> List[Any]:
        """Define reusable langchain tool objects for local environment interactions."""
        if StructuredTool is None:
            return []

        def web_search_tool(query: str, max_results: int = 8, workers: int = 6) -> str:
            payload = self._prepare_web_search_params(
                {"query": query, "max_results": max_results, "workers": workers}
            )
            return json.dumps(payload, default=str)

        def send_email_tool(to: str, subject: str, body: str) -> str:
            payload = {"to": to, "subject": subject, "body": body}
            return json.dumps(payload, default=str)

        return [
            StructuredTool.from_function(
                name="web_search",
                description="Search the web for up-to-date information.",
                func=web_search_tool,
            ),
            StructuredTool.from_function(
                name="send_email",
                description="Send an email with recipient, subject, and body.",
                func=send_email_tool,
            ),
        ]

    def _build_prompt_with_template(self, template: str, values: Dict[str, Any]) -> str:
        """Render prompts via langchain PromptTemplate when available."""
        if self._langchain_available and PromptTemplate is not None:
            try:
                prompt = PromptTemplate.from_template(template)
                return prompt.format(**values)
            except Exception:
                pass
        safe_values = {k: str(v) for k, v in values.items()}
        try:
            return template.format(**safe_values)
        except Exception:
            return template

    def _parse_with_schema(self, model_out: str, schema_cls: Any) -> Optional[Dict[str, Any]]:
        """
        Parse model output using langchain output parsers + pydantic schema.
        Falls back to manual JSON extraction.
        """
        if self._langchain_available and PydanticOutputParser is not None:
            try:
                parser = PydanticOutputParser(pydantic_object=schema_cls)
                parsed_obj = parser.parse(model_out)
                if hasattr(parsed_obj, "model_dump"):
                    return parsed_obj.model_dump()
                return dict(parsed_obj)
            except Exception:
                pass

        extracted = self._extract_json_object(model_out)
        if isinstance(extracted, dict):
            try:
                parsed_obj = schema_cls.model_validate(extracted)
                return parsed_obj.model_dump()
            except Exception:
                return extracted
        return None

    def _split_text_for_retrieval(self, text: str) -> List[str]:
        """Split long text using langchain text splitters when available."""
        if not text:
            return []
        if self.langchain_enabled and LANGCHAIN_TEXT_SPLITTER_AVAILABLE and RecursiveCharacterTextSplitter:
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.langchain_chunk_size,
                    chunk_overlap=self.langchain_chunk_overlap,
                )
                chunks = splitter.split_text(text)
                if chunks:
                    return chunks
            except Exception:
                pass
        return [text]

    def _bm25_rerank_texts(self, query: str, texts: List[str], top_k: int) -> List[str]:
        """Rerank texts with BM25 retriever from langchain-community."""
        if not texts:
            return []
        if (
            self.langchain_enabled
            and self.langchain_use_bm25_rerank
            and LANGCHAIN_RETRIEVERS_AVAILABLE
            and BM25Retriever is not None
            and LCDocument is not None
        ):
            try:
                docs = [LCDocument(page_content=t, metadata={"source": "bm25"}) for t in texts if t]
                retriever = BM25Retriever.from_documents(docs)
                retriever.k = max(1, top_k)
                ranked_docs = retriever.invoke(query)
                ranked_texts = [d.page_content for d in ranked_docs if getattr(d, "page_content", "")]
                if ranked_texts:
                    return ranked_texts[:top_k]
            except Exception:
                pass
        return texts[:top_k]

    def _load_local_documents_via_langchain(self, file_paths: List[str]) -> List[str]:
        """Load local docs through langchain-community loaders when suitable."""
        loaded_texts: List[str] = []
        if not file_paths:
            return loaded_texts
        if (
            not self.langchain_enabled
            or not self.langchain_use_local_doc_loader
            or not LANGCHAIN_LOADERS_AVAILABLE
        ):
            return loaded_texts

        for raw_path in file_paths:
            try:
                path = Path(str(raw_path))
                if not path.exists() or not path.is_file():
                    continue
                suffix = path.suffix.lower()
                docs = []
                if suffix == ".pdf" and PyPDFLoader is not None:
                    docs = PyPDFLoader(str(path)).load()
                elif suffix == ".csv" and CSVLoader is not None:
                    docs = CSVLoader(str(path)).load()
                elif suffix in {".html", ".htm"} and BSHTMLLoader is not None:
                    docs = BSHTMLLoader(str(path)).load()
                elif TextLoader is not None:
                    docs = TextLoader(str(path), encoding="utf-8").load()

                for doc in docs:
                    content = getattr(doc, "page_content", "")
                    if content:
                        loaded_texts.extend(self._split_text_for_retrieval(content))
            except Exception as exc:
                self.logger.warning("langchain local doc load failed for %s: %s", raw_path, exc)
        return loaded_texts

    def _normalize_tool_name(self, tool_name: str) -> str:
        normalized = tool_name.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "websearch":
            return "web_search"
        if normalized.startswith("web_") and "search" in normalized:
            return "web_search"
        if normalized == "email":
            return "send_email"
        return normalized

    def _parse_complexity(self, value: str) -> Optional[TaskComplexity]:
        value = value.strip().lower()
        mapping = {
            "simple": TaskComplexity.SIMPLE,
            "low": TaskComplexity.SIMPLE,
            "medium": TaskComplexity.MEDIUM,
            "moderate": TaskComplexity.MEDIUM,
            "high": TaskComplexity.HIGH,
            "complex": TaskComplexity.HIGH,
        }
        return mapping.get(value)

    def _parse_specialist(self, value: str) -> Optional[SpecialistType]:
        value = value.strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "generalist": SpecialistType.GENERALIST,
            "general": SpecialistType.GENERALIST,
            "chatbot": SpecialistType.GENERALIST,
            "researcher": SpecialistType.RESEARCHER,
            "research": SpecialistType.RESEARCHER,
            "coder": SpecialistType.CODER,
            "coding": SpecialistType.CODER,
            "developer": SpecialistType.CODER,
            "programmer": SpecialistType.CODER,
            "summarizer": SpecialistType.SUMMARIZER,
            "summary": SpecialistType.SUMMARIZER,
            "analyst": SpecialistType.ANALYST,
            "analysis": SpecialistType.ANALYST,
            "planner": SpecialistType.PLANNER,
            "planning": SpecialistType.PLANNER,
            "retriever": SpecialistType.RETRIEVER,
            "retrieval": SpecialistType.RETRIEVER,
            "compliance_checker": SpecialistType.COMPLIANCE_CHECKER,
            "compliance": SpecialistType.COMPLIANCE_CHECKER,
            "validator": SpecialistType.COMPLIANCE_CHECKER,
            "safety_checker": SpecialistType.COMPLIANCE_CHECKER,
            "optimizer": SpecialistType.OPTIMIZER,
            "optimiser": SpecialistType.OPTIMIZER,
            "explainer": SpecialistType.EXPLAINER,
            "teacher": SpecialistType.EXPLAINER,
        }
        return mapping.get(value)

    def _parse_flow(self, value: str) -> Optional[FlowType]:
        value = value.strip().lower().replace("-", "_")
        mapping = {
            "tool_use": FlowType.TOOL_USE,
            "react": FlowType.REACT,
            "planning": FlowType.PLANNING,
            "multi_agent": FlowType.MULTI_AGENT,
            "multiagent": FlowType.MULTI_AGENT,
            "pev": FlowType.PEV,
            "planner_executor_verifier": FlowType.PEV,
        }
        return mapping.get(value)

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract first valid JSON object from model text."""
        if not text:
            return None
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    def _coerce_json(self, value: Any) -> Any:
        """Parse JSON strings to Python objects where possible."""
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value
            try:
                return json.loads(stripped)
            except Exception:
                extracted = self._extract_json_object(stripped)
                return extracted if extracted is not None else value
        return value

    def _truncate_text(self, value: Any, max_chars: int) -> str:
        """Convert payload to string and truncate for prompts/logs."""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, default=str, indent=2)
            except Exception:
                text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"

    def _resolve_device(self) -> str:
        """Detect inference device for direct model loaders."""
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
