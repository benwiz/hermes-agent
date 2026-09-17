"""Content-free turn receipts and independent request admission budgets.

Limits are opt-in and checked before each main-loop provider attempt, including retries.
They do not cancel in-flight requests, tool execution, or auxiliary provider calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import threading
from collections import Counter
from dataclasses import dataclass, field

from hermes_constants import get_hermes_home

TOKEN_KEYS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens")
ESCALATION_REASONS = frozenset({"interactive_web_required", "authenticated_session_required",
    "visual_verification_required", "structured_api_unavailable", "desktop_only_workflow"})


def schema_snapshot(tools):
    encoded = json.dumps(tools or [], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"signature": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded),
            "names": sorted(t["function"]["name"] for t in (tools if isinstance(tools, list) else []) if "function" in t)}


def settings(config):
    raw = (config.get("agent") or {}).get("efficiency")
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("agent.efficiency must be a mapping")
    if "receipts" in raw and not isinstance(raw["receipts"], bool):
        raise ValueError("agent.efficiency.receipts must be a boolean")
    result = {"receipts": raw.get("receipts") is True}
    for key in ("max_model_calls", "checkpoint_model_calls", "max_seconds"):
        value = raw.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"agent.efficiency.{key} must be positive and finite")
            if key.endswith("calls") and int(value) != value:
                raise ValueError(f"agent.efficiency.{key} must be an integer")
        result[key] = value
    checkpoint_model_calls = result.get("checkpoint_model_calls")
    max_model_calls = result.get("max_model_calls")
    if (checkpoint_model_calls is not None and max_model_calls is not None
            and checkpoint_model_calls > max_model_calls):
        raise ValueError("checkpoint_model_calls must be less than or equal to max_model_calls")
    return result


@dataclass
class TurnEfficiency:
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    tools: Counter = field(default_factory=Counter)
    schemas: dict = field(default_factory=dict)
    token_start: dict = field(default_factory=dict)
    stop_reason: str | None = None
    escalation_reasons: set = field(default_factory=set)
    lock: object = field(default_factory=threading.RLock)
    shared_calls: int = 0
    deadline: float | None = None
    timer: object = None


def start_turn(agent):
    agent._efficiency_turn = TurnEfficiency(token_start={
        key: getattr(agent, f"session_{key}", 0) for key in TOKEN_KEYS})
    state = agent._efficiency_turn
    seconds = getattr(agent, "_efficiency_settings", {}).get("max_seconds")
    state.deadline = state.started + seconds if seconds else None
    inherited = getattr(agent, "_efficiency_parent_budget", None)
    if inherited is not None and inherited.deadline is not None:
        state.deadline = min(state.deadline, inherited.deadline) if state.deadline else inherited.deadline
    if state.deadline is not None and callable(getattr(agent, "interrupt", None)):
        def expire():
            with state.lock:
                if getattr(agent, "_efficiency_turn", None) is state:
                    state.stop_reason = "wall_time_budget"
                    agent.interrupt(hard_cancel=True, tool_reason="wall_time_budget")
        state.timer = threading.Timer(max(0, state.deadline - time.monotonic()), expire)
        state.timer.daemon = True
        state.timer.start()


def stop_watchdog(agent):
    state = getattr(agent, "_efficiency_turn", None)
    if isinstance(state, TurnEfficiency) and state.timer is not None:
        state.timer.cancel()
        if state.timer is not threading.current_thread():
            state.timer.join()
        state.timer = None


def deadline_expired(agent):
    state = getattr(agent, "_efficiency_turn", None)
    if not isinstance(state, TurnEfficiency):
        return False
    if state.deadline is not None and time.monotonic() >= state.deadline:
        state.stop_reason = "wall_time_budget"
        return True
    return False


def admit_request(agent, tools):
    state = getattr(agent, "_efficiency_turn", None)
    if not isinstance(state, TurnEfficiency):
        return
    cfg = getattr(agent, "_efficiency_settings", {})
    limits = (("max_seconds", time.monotonic() - state.started, "wall_time_budget"),
              ("max_model_calls", state.model_calls, "model_call_budget"),
              ("checkpoint_model_calls", state.model_calls, "continuation_required"))
    for key, used, reason in limits:
        if cfg.get(key) and used >= cfg[key]:
            state.stop_reason = reason
            raise InterruptedError(reason)
    root = getattr(agent, "_efficiency_parent_budget", None) or state
    root_cfg = getattr(agent, "_efficiency_parent_settings", None) or cfg
    with root.lock:
        if root_cfg.get("max_model_calls") and root.shared_calls >= root_cfg["max_model_calls"]:
            state.stop_reason = "model_call_budget"
            raise InterruptedError(state.stop_reason)
        if deadline_expired(agent):
            raise InterruptedError(state.stop_reason)
        root.shared_calls += 1
    state.model_calls += 1
    snapshot = schema_snapshot(tools)
    state.schemas[snapshot["signature"]] = snapshot


def record_tool(agent, name, *, blocked=False):
    state = getattr(agent, "_efficiency_turn", None)
    if isinstance(state, TurnEfficiency) and not blocked:
        state.tools[name] += 1


def finish_turn(agent, result, turn_id):
    state = getattr(agent, "_efficiency_turn", None)
    if not isinstance(state, TurnEfficiency):
        return
    reason = state.stop_reason or result["turn_exit_reason"]
    receipt = {"version": 1, "turn_id": turn_id, "session_id": agent.session_id,
        "model": agent.model, "provider": agent.provider,
        "route": getattr(agent, "_efficiency_route", {}),
        "toolset_signature": getattr(agent, "_session_tool_signature", None),
        "shared_model_calls": state.shared_calls,
        "model_calls": state.model_calls, "tool_calls": sum(state.tools.values()),
        "tools": dict(state.tools), "tool_schemas": list(state.schemas.values()),
        "tokens": {key: max(0, getattr(agent, f"session_{key}", 0) - state.token_start[key]) for key in TOKEN_KEYS},
        "elapsed_seconds": round(time.monotonic() - state.started, 3),
        "stop_reason": reason, "success": "ungraded",
        "completed": bool(result.get("completed")), "failed": bool(result.get("failed")),
        "escalation_reasons": sorted(state.escalation_reasons), "provider_cost_usd": None,
        "browser_calls": sum(n for name, n in state.tools.items() if name.startswith("browser_")),
        "computer_calls": sum(n for name, n in state.tools.items() if name.startswith("computer"))}
    result["efficiency"] = receipt
    if getattr(agent, "_efficiency_settings", {}).get("receipts"):
        path = get_hermes_home() / "logs" / "efficiency.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND and one write prevent concurrent turn writers from sharing a seek offset.
        payload = (json.dumps(receipt, ensure_ascii=True) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("short efficiency receipt write")
        finally:
            os.close(fd)
