"""
DevMentor AI — Autonomous Agent Loop
=======================================
Production-grade agent execution engine with:

- Multi-model LLM support via ModelRouter
- Sandboxed tool execution (whitelist only)
- Structured ReAct loop (Reason → Act → Observe)
- Async execution with timeout enforcement
- Per-iteration and total timeout limits
- Memory management with sliding window
- Token and cost tracking per run
- Streaming thought output via callbacks
- Multi-agent orchestration
- Graceful error recovery
- Full audit logging per agent run

How it works (ReAct pattern):
    1. THINK: LLM reasons about the objective
    2. ACT:   LLM calls a tool if needed
    3. OBSERVE: Tool result is fed back to LLM
    4. REPEAT until objective complete or max iterations

Usage:
    from agent_loop import Agent, run_agent

    result = await run_agent(
        objective="Research quantum computing and summarize",
        user_id="u123",
        user_tier="pro",
    )
    print(result.final_answer)
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from config import get_settings
from agent_tools import ToolCall, ToolResult, parse_tool_calls, tool_registry

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()


# ===========================================================================
# Data Classes
# ===========================================================================

class AgentStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    TIMEOUT   = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    iteration: int
    thought: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    tokens_used: int = 0
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentResult:
    """Final result of an agent run."""
    run_id: str
    objective: str
    status: AgentStatus
    final_answer: Optional[str]
    steps: List[AgentStep]
    total_iterations: int
    total_tokens: int
    total_cost_usd: float
    duration_ms: int
    error: Optional[str] = None
    user_id: str = "anonymous"
    model_used: str = "unknown"

    @property
    def success(self) -> bool:
        return self.status == AgentStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "objective": self.objective,
            "status": self.status,
            "final_answer": self.final_answer,
            "total_iterations": self.total_iterations,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "success": self.success,
            "steps": [
                {
                    "iteration": s.iteration,
                    "thought": s.thought[:500],
                    "tool_calls": [tc.tool for tc in s.tool_calls],
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
        }


# ===========================================================================
# System Prompt Builder
# ===========================================================================

def _build_system_prompt(available_tools: List[str]) -> str:
    """Build the agent system prompt with tool descriptions."""
    tool_schemas = tool_registry.get_tool_schemas()
    available_schemas = [
        s for s in tool_schemas
        if s["function"]["name"] in available_tools
    ]

    tools_desc = ""
    for schema in available_schemas:
        fn = schema["function"]
        params = fn.get("parameters", {}).get("properties", {})
        param_desc = ", ".join(
            f"{k}: {v.get('description', '')}"
            for k, v in params.items()
        )
        tools_desc += f"\n- **{fn['name']}**({param_desc}): {fn['description']}"

    return f"""You are an autonomous AI agent. You reason step by step and use tools to complete objectives.

## Available Tools
{tools_desc}

## Tool Usage Format
To call a tool, include in your response:
[[tool:tool_name({{"param": "value"}})]]

## Instructions
1. Think carefully about what you need to do
2. Use tools when you need real-time information or calculations
3. Analyze tool results and continue reasoning
4. When the objective is complete, say "FINAL ANSWER:" followed by your complete response
5. Be concise but thorough
6. Never make up information — use tools to verify facts

## Completion Signal
When done, write: "FINAL ANSWER: <your complete answer here>"
"""


def _build_user_prompt(
    objective: str,
    memory: List[Dict[str, str]],
    iteration: int,
    max_iterations: int,
) -> str:
    """Build the user prompt for each iteration."""
    history = ""
    if memory:
        history_items = memory[-12:]  # Last 12 messages for context
        history = "\n".join(
            f"[{msg['role'].upper()}]: {msg['content'][:800]}"
            for msg in history_items
        )

    return f"""## Objective
{objective}

## Progress ({iteration}/{max_iterations} iterations)
{history if history else "This is the first iteration."}

## Your Turn
Continue working toward the objective. Use tools if needed.
If objective is complete, write "FINAL ANSWER: <answer>".
"""


# ===========================================================================
# Completion Detection
# ===========================================================================

_COMPLETION_SIGNALS = [
    "final answer:",
    "objective is complete",
    "task completed",
    "task is complete",
    "i have completed",
    "i have finished",
    "the answer is complete",
    "in conclusion,",
    "to summarize everything",
]


def _detect_completion(thought: str) -> bool:
    """Detect if the agent has signaled completion."""
    thought_lower = thought.lower()
    return any(signal in thought_lower for signal in _COMPLETION_SIGNALS)


def _extract_final_answer(thought: str) -> str:
    """Extract the final answer from a completion thought."""
    lower = thought.lower()
    if "final answer:" in lower:
        idx = lower.index("final answer:")
        return thought[idx + len("final answer:"):].strip()
    return thought.strip()


# ===========================================================================
# Agent Class
# ===========================================================================

class Agent:
    """
    Autonomous ReAct agent that reasons and uses tools to complete objectives.

    Each Agent instance handles one objective run. Create a new instance
    for each run — do not reuse across objectives.
    """

    def __init__(
        self,
        user_id: str = "anonymous",
        user_tier: str = "free",
        max_iterations: Optional[int] = None,
        total_timeout: Optional[int] = None,
        iteration_timeout: int = 60,
        memory_limit: int = 20,
        allowed_tools: Optional[List[str]] = None,
        on_step: Optional[Callable[[AgentStep], None]] = None,
    ):
        """
        Args:
            user_id:           User identifier for cost tracking and logging
            user_tier:         Subscription tier for model routing
            max_iterations:    Max ReAct loop iterations (default from settings)
            total_timeout:     Total run timeout in seconds (default from settings)
            iteration_timeout: Per-iteration timeout in seconds
            memory_limit:      Max messages to keep in context window
            allowed_tools:     Whitelist of tools this agent can use
            on_step:           Callback called after each step (for streaming)
        """
        self.run_id = str(uuid.uuid4())
        self.user_id = user_id
        self.user_tier = user_tier
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self.total_timeout = total_timeout or settings.agent_timeout
        self.iteration_timeout = iteration_timeout
        self.memory_limit = memory_limit
        self.allowed_tools = allowed_tools or tool_registry.list_tools()
        self.on_step = on_step

        # Runtime state
        self.memory: List[Dict[str, str]] = []
        self.steps: List[AgentStep] = []
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.status: AgentStatus = AgentStatus.PENDING
        self._cancelled: bool = False

        logger.info(
            "Agent created: run_id=%s user=%s tier=%s tools=%s",
            self.run_id, user_id, user_tier, self.allowed_tools,
        )

    def cancel(self) -> None:
        """Cancel the agent run gracefully."""
        self._cancelled = True
        self.status = AgentStatus.CANCELLED
        logger.info("Agent run %s cancelled", self.run_id)

    def _add_memory(self, role: str, content: str) -> None:
        """Add a message to memory with sliding window."""
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > self.memory_limit:
            # Keep system context + last N messages
            self.memory = self.memory[-self.memory_limit:]

    async def _think(
        self,
        objective: str,
        iteration: int,
        model_router,
    ) -> tuple:
        """
        Call LLM for one reasoning step.

        Returns:
            Tuple of (thought_text, prompt_tokens, completion_tokens, cost)
        """
        system_prompt = _build_system_prompt(self.allowed_tools)
        user_prompt = _build_user_prompt(
            objective, self.memory, iteration, self.max_iterations
        )

        try:
            response = await asyncio.wait_for(
                model_router.route(
                    prompt=user_prompt,
                    user_id=self.user_id,
                    user_tier=self.user_tier,
                    system_prompt=system_prompt,
                ),
                timeout=self.iteration_timeout,
            )

            return (
                response.response,
                response.prompt_tokens,
                response.completion_tokens,
                response.cost_usd,
            )

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Iteration {iteration} exceeded {self.iteration_timeout}s timeout"
            )

    def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute all tool calls from a thought.
        Only allowed tools are executed.
        """
        results = []

        for call in tool_calls:
            # Enforce tool whitelist
            if call.tool not in self.allowed_tools:
                logger.warning(
                    "Agent %s attempted to call non-whitelisted tool: %s",
                    self.run_id, call.tool,
                )
                results.append(ToolResult(
                    tool_name=call.tool,
                    success=False,
                    output="",
                    error=f"Tool '{call.tool}' is not available for this agent",
                ))
                continue

            result = tool_registry.execute(
                tool_name=call.tool,
                parameters=call.parameters,
                user_id=self.user_id,
            )
            results.append(result)

            logger.info(
                "Agent %s tool result: %s → success=%s",
                self.run_id, call.tool, result.success,
            )

        return results

    async def run(self, objective: str) -> AgentResult:
        """
        Execute the agent loop for a given objective.

        Args:
            objective: The task for the agent to complete

        Returns:
            AgentResult with final answer and execution stats
        """
        if not objective or not objective.strip():
            raise ValueError("Objective cannot be empty")

        self.status = AgentStatus.RUNNING
        start_time = time.time()
        objective = objective.strip()

        logger.info(
            "Agent run %s starting: user=%s objective=%s...",
            self.run_id, self.user_id, objective[:100],
        )

        # Add objective to memory
        self._add_memory("user", objective)

        # Get model router
        try:
            from model_router import get_model_router
            model_router = get_model_router()
        except Exception as exc:
            logger.error("Failed to get model router: %s", exc)
            return self._make_result(
                objective, AgentStatus.FAILED,
                None, start_time,
                error=f"Model router unavailable: {exc}",
            )

        final_answer = None

        for iteration in range(1, self.max_iterations + 1):
            # Check cancellation
            if self._cancelled:
                return self._make_result(
                    objective, AgentStatus.CANCELLED,
                    None, start_time,
                    error="Agent run was cancelled",
                )

            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed > self.total_timeout:
                logger.warning(
                    "Agent %s exceeded total timeout (%ds)",
                    self.run_id, self.total_timeout,
                )
                return self._make_result(
                    objective, AgentStatus.TIMEOUT,
                    None, start_time,
                    error=f"Total timeout exceeded ({self.total_timeout}s)",
                )

            iter_start = time.time()
            logger.info("Agent %s iteration %d/%d", self.run_id, iteration, self.max_iterations)

            try:
                # THINK
                thought, prompt_tokens, completion_tokens, cost = await self._think(
                    objective, iteration, model_router
                )

                self.total_tokens += prompt_tokens + completion_tokens
                self.total_cost_usd += cost

                # Add thought to memory
                self._add_memory("assistant", thought)

                # ACT — parse and execute tool calls
                tool_calls = parse_tool_calls(thought)
                tool_results = []

                if tool_calls:
                    tool_results = self._execute_tools(tool_calls)

                    # Add tool results to memory
                    for tc, tr in zip(tool_calls, tool_results):
                        if tr.success:
                            self._add_memory(
                                "system",
                                f"Tool '{tc.tool}' result: {tr.output}",
                            )
                        else:
                            self._add_memory(
                                "system",
                                f"Tool '{tc.tool}' failed: {tr.error}",
                            )

                # Record step
                step = AgentStep(
                    iteration=iteration,
                    thought=thought,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    tokens_used=prompt_tokens + completion_tokens,
                    duration_ms=int((time.time() - iter_start) * 1000),
                )
                self.steps.append(step)

                # Notify callback
                if self.on_step:
                    try:
                        self.on_step(step)
                    except Exception:
                        pass

                # CHECK COMPLETION
                if _detect_completion(thought):
                    final_answer = _extract_final_answer(thought)
                    logger.info(
                        "Agent %s completed at iteration %d",
                        self.run_id, iteration,
                    )
                    return self._make_result(
                        objective, AgentStatus.COMPLETED,
                        final_answer, start_time,
                    )

            except TimeoutError as exc:
                logger.warning("Agent %s iteration timeout: %s", self.run_id, exc)
                self.steps.append(AgentStep(
                    iteration=iteration,
                    thought=f"[TIMEOUT] {exc}",
                    duration_ms=int((time.time() - iter_start) * 1000),
                ))
                continue

            except Exception as exc:
                logger.error(
                    "Agent %s iteration %d failed: %s",
                    self.run_id, iteration, exc, exc_info=True,
                )
                return self._make_result(
                    objective, AgentStatus.FAILED,
                    None, start_time,
                    error=str(exc),
                )

        # Max iterations reached
        logger.warning(
            "Agent %s reached max iterations (%d)",
            self.run_id, self.max_iterations,
        )
        return self._make_result(
            objective, AgentStatus.FAILED,
            "Max iterations reached without completing the objective.",
            start_time,
            error="Max iterations exceeded",
        )

    def _make_result(
        self,
        objective: str,
        status: AgentStatus,
        final_answer: Optional[str],
        start_time: float,
        error: Optional[str] = None,
    ) -> AgentResult:
        """Build the final AgentResult."""
        self.status = status
        duration_ms = int((time.time() - start_time) * 1000)

        # Log analytics
        self._log_analytics(status, duration_ms)

        return AgentResult(
            run_id=self.run_id,
            objective=objective,
            status=status,
            final_answer=final_answer,
            steps=self.steps,
            total_iterations=len(self.steps),
            total_tokens=self.total_tokens,
            total_cost_usd=self.total_cost_usd,
            duration_ms=duration_ms,
            error=error,
            user_id=self.user_id,
        )

    def _log_analytics(self, status: AgentStatus, duration_ms: int) -> None:
        """Log agent run to analytics (non-blocking)."""
        try:
            from analytics import get_analytics
            analytics = get_analytics()
            analytics.track_agent_run(
                user_id=self.user_id,
                run_id=self.run_id,
                status=status,
                iterations=len(self.steps),
                total_tokens=self.total_tokens,
                total_cost_usd=self.total_cost_usd,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            logger.debug("Analytics logging failed (non-critical): %s", exc)


# ===========================================================================
# Multi-Agent Orchestrator
# ===========================================================================

class MultiAgentOrchestrator:
    """
    Manages multiple specialized agents running concurrently.

    Provides pre-configured agents for common task types:
    - Research agent: web search + Wikipedia
    - Coding agent: calculator + code assistance
    - Analysis agent: data analysis tasks
    - General agent: all available tools
    """

    def __init__(self):
        self._active_runs: Dict[str, Agent] = {}

    async def run_research_agent(
        self,
        query: str,
        user_id: str = "anonymous",
        user_tier: str = "free",
    ) -> AgentResult:
        """Research agent with web search and Wikipedia tools."""
        agent = Agent(
            user_id=user_id,
            user_tier=user_tier,
            max_iterations=8,
            allowed_tools=["web_search", "wikipedia", "current_time"],
        )
        self._active_runs[agent.run_id] = agent
        try:
            result = await agent.run(
                f"Research the following topic thoroughly and provide a comprehensive summary: {query}"
            )
        finally:
            self._active_runs.pop(agent.run_id, None)
        return result

    async def run_coding_agent(
        self,
        task: str,
        language: str = "python",
        user_id: str = "anonymous",
        user_tier: str = "free",
    ) -> AgentResult:
        """Coding agent with calculator and JSON tools."""
        agent = Agent(
            user_id=user_id,
            user_tier=user_tier,
            max_iterations=10,
            allowed_tools=["calculator", "json_formatter", "unit_converter"],
        )
        self._active_runs[agent.run_id] = agent
        try:
            result = await agent.run(
                f"Write {language} code to accomplish the following task: {task}"
            )
        finally:
            self._active_runs.pop(agent.run_id, None)
        return result

    async def run_analysis_agent(
        self,
        data: str,
        question: str,
        user_id: str = "anonymous",
        user_tier: str = "free",
    ) -> AgentResult:
        """Analysis agent for data and math tasks."""
        agent = Agent(
            user_id=user_id,
            user_tier=user_tier,
            max_iterations=8,
            allowed_tools=["calculator", "unit_converter", "json_formatter"],
        )
        self._active_runs[agent.run_id] = agent
        try:
            result = await agent.run(
                f"Analyze the following data and answer the question.\n\nData:\n{data}\n\nQuestion: {question}"
            )
        finally:
            self._active_runs.pop(agent.run_id, None)
        return result

    async def run_general_agent(
        self,
        objective: str,
        user_id: str = "anonymous",
        user_tier: str = "free",
        max_iterations: Optional[int] = None,
    ) -> AgentResult:
        """General purpose agent with all available tools."""
        agent = Agent(
            user_id=user_id,
            user_tier=user_tier,
            max_iterations=max_iterations,
        )
        self._active_runs[agent.run_id] = agent
        try:
            result = await agent.run(objective)
        finally:
            self._active_runs.pop(agent.run_id, None)
        return result

    def cancel_run(self, run_id: str) -> bool:
        """Cancel an active agent run by ID."""
        agent = self._active_runs.get(run_id)
        if agent:
            agent.cancel()
            return True
        return False

    def get_active_runs(self) -> List[str]:
        """Get list of currently active run IDs."""
        return list(self._active_runs.keys())


# ===========================================================================
# Convenience Function
# ===========================================================================

async def run_agent(
    objective: str,
    user_id: str = "anonymous",
    user_tier: str = "free",
    max_iterations: Optional[int] = None,
    allowed_tools: Optional[List[str]] = None,
) -> AgentResult:
    """
    Simple convenience function to run an agent.

    Args:
        objective:     What the agent should accomplish
        user_id:       User identifier
        user_tier:     Subscription tier
        max_iterations: Max iterations override
        allowed_tools:  Tool whitelist override

    Returns:
        AgentResult with final answer and stats
    """
    agent = Agent(
        user_id=user_id,
        user_tier=user_tier,
        max_iterations=max_iterations,
        allowed_tools=allowed_tools,
    )
    return await agent.run(objective)


# ===========================================================================
# Singletons
# ===========================================================================
orchestrator = MultiAgentOrchestrator()