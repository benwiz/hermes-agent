"""Explicit capability transitions: finite leases and real conversation boundaries."""
from copy import copy
import asyncio
from weakref import WeakValueDictionary
import re
import time


_TRANSITION_LOCKS = WeakValueDictionary()


def transition_lock(runner, source):
    key = (id(runner), runner._session_key_for_source(source))
    lock = _TRANSITION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _TRANSITION_LOCKS[key] = lock
    return lock


_CONTROL = re.compile(
    r"(?:please\s+)?(?:(enter|enable|use|exit|disable|leave)\s+heavy\s+mode|"
    r"(?:turn|switch)\s+(on|off)\s+heavy\s+mode|"
    r"heavy\s+mode\s+(on|off))(?:\s+please)?[.!]?", re.IGNORECASE)


def normalize_mode_control(event):
    """Only standalone imperatives become commands, before either busy guard."""
    if not event.allow_gateway_control or event.internal:
        return
    match = _CONTROL.fullmatch((event.text or "").strip())
    if match:
        verb = next(value for value in match.groups() if value).lower()
        event.text = "/mode " + ("off" if verb in {"exit", "disable", "leave", "off"} else "heavy")


def _config(runner, source):
    from hermes_cli.config import load_config
    from gateway.run import _get_channel_override
    config = load_config()
    routing = (config.get("session_routing") or {}).get(source.platform.value)
    explicit = _get_channel_override(runner.config, source.platform, source.chat_id,
        thread_id=source.thread_id, parent_id=source.parent_chat_id)
    return config, routing, explicit


def _valid_toolsets(names):
    from tools.registry import discover_builtin_tools
    from toolsets import validate_toolset
    discover_builtin_tools()
    return isinstance(names, list) and all(validate_toolset(name) for name in names)


def recovery_reason(runner, source, routing, explicit, now=None):
    """Stale saved capabilities must rotate, never silently change within a session."""
    key = runner._session_key_for_source(source)
    entry = runner.session_store.get_or_create_session(source)
    mode = runner.session_store.get_session_metadata(key, "session_mode")
    if isinstance(mode, dict) and mode.get("session_id") == entry.session_id:
        tier = mode.get("tier")
        if tier not in routing["tiers"] or (explicit and explicit.allowed_tiers is not None
                and tier not in explicit.allowed_tiers):
            return "saved mode is no longer available"
        if tier == "heavy":
            deadline = mode.get("expires_at")
            if not isinstance(deadline, (int, float)) or deadline <= (time.time() if now is None else now):
                return "heavy-mode lease expired"
    policy = runner.session_store.get_session_metadata(key, "tool_policy")
    if isinstance(policy, dict) and policy.get("session_id") == entry.session_id:
        if policy.get("enabled") is not None and not _valid_toolsets(policy["enabled"]):
            return "saved tool policy contains unavailable toolsets"
    return None


async def _rotate(runner, event, tier=None, expires_at=None, reason="requested mode change"):
    key = runner._session_key_for_source(event.source)
    old = runner.session_store.get_or_create_session(event.source)
    old_sid = old.session_id
    transcript = runner.session_store.load_transcript(old_sid)
    # Only recent plain conversation text, never tool calls/results or full old prompts.
    snippets = [f"{m['role']}: {m['content'][:1000]}" for m in transcript
        if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)
        and not m.get("tool_calls")][-4:]
    reset_event = copy(event)
    reset_event.text = "/new"
    await runner._handle_reset_command(reset_event)
    event._mode_rotated = True
    entry = runner.session_store.get_or_create_session(event.source)
    if tier is not None:
        if not runner.session_store.set_session_metadata(key, "session_mode", {
                "session_id": entry.session_id, "tier": tier, "expires_at": expires_at}):
            raise RuntimeError("Could not persist session mode")
    if snippets or reason:
        # One assistant message leaves the next real user turn alternating normally.
        runner.session_store.append_to_transcript(entry.session_id, {
            "role": "assistant", "content":
            f"Session boundary: {reason}. Previous conversation context "
            f"(archived session {old_sid}; reference only):\n"
            + "\n".join(snippets)[:4000]})
    return entry


async def recover_before_turn(runner, event):
    config, routing, explicit = _config(runner, event.source)
    if not routing:
        return
    reason = recovery_reason(runner, event.source, routing, explicit)
    if reason:
        await _rotate(runner, event, reason=reason)


async def handle_mode(runner, event):
    async with transition_lock(runner, event.source):
        return await _handle_mode(runner, event)


async def _handle_mode(runner, event):
    from gateway.session_routing import resolve_policy
    config, routing, explicit = _config(runner, event.source)
    if not routing:
        return "Session modes are not configured for this platform."
    key = runner._session_key_for_source(event.source)
    entry = runner.session_store.get_or_create_session(event.source)
    requested = event.get_command_args().strip().lower()
    mode = runner.session_store.get_session_metadata(key, "session_mode")
    mode = mode if isinstance(mode, dict) else {}
    if mode.get("session_id") != entry.session_id:
        mode = {}
    reason = recovery_reason(runner, event.source, routing, explicit)
    if requested in {"", "status"}:
        _, default_tier = resolve_policy(runner, config, event.source, ignore_saved=True)
        current = mode.get("tier", default_tier)
        lease = ""
        if current == "heavy" and isinstance(mode.get("expires_at"), (int, float)):
            lease = f"; {max(0, int(mode['expires_at'] - time.time()))} seconds remaining"
        pending = f"; recovery pending before next turn: {reason}" if reason else ""
        allowed = explicit.allowed_tiers if explicit and explicit.allowed_tiers is not None else list(routing["tiers"])
        off_tier = routing.get("off_tier") or (
            "off" if "off" in routing.get("tiers", {}) else None
        )
        off_action = f"starts {off_tier}" if off_tier else f"restores {default_tier}"
        return f"Mode: {current}{lease}{pending}. Available: {', '.join(allowed)}. /mode off {off_action}."
    off_tier = routing.get("off_tier") or (
        "off" if "off" in routing.get("tiers", {}) else None
    )
    target = off_tier if requested == "off" else (None if requested == "default" else requested)
    if target is not None:
        if target not in routing["tiers"]:
            return "Unknown mode. Available: " + ", ".join(routing["tiers"]) + ", off, status."
        if explicit and explicit.allowed_tiers is not None and target not in explicit.allowed_tiers:
            return "That mode is not allowed in this channel."
        names = explicit.toolsets if explicit and explicit.toolsets is not None else routing["tiers"][target].get("toolsets")
        if not _valid_toolsets(names):
            return "This mode needs an explicit allowlist of registered toolsets. No session was changed."
    else:
        policy, _ = resolve_policy(runner, config, event.source, ignore_saved=True)
        if not _valid_toolsets(policy.toolsets):
            return "Channel policy needs registered toolsets. No session was changed."
    # /off must be available even when the saved tier has disappeared.
    deadline = time.time() + max(60, min(86400, int(routing.get("heavy_lease_seconds", 1800)))) if target == "heavy" else None
    _, default_tier = resolve_policy(runner, config, event.source, ignore_saved=True)
    same_mode = mode.get("tier") == target or (not mode and target == default_tier and target != "heavy")
    if not reason and same_mode:
        if target == "heavy":
            mode["expires_at"] = deadline
            if not runner.session_store.set_session_metadata(key, "session_mode", mode):
                raise RuntimeError("Could not renew session mode")
        return f"Already in {target or 'channel policy'} mode." + (" Heavy lease renewed." if deadline else "")
    await _rotate(runner, event, target, deadline)
    label = target or "channel policy"
    lease = " Heavy mode is temporary; /mode status shows remaining time and /mode off exits early." if deadline else ""
    return f"Started a new {label} session with recent conversation context.{lease}"
