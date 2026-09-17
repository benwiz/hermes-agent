"""Channel capability selection and durable session-boundary policy snapshots."""
from __future__ import annotations

from agent.skill_utils import parse_config_string_list


def channel_toolsets(runner, config, source, platform_key, enabled, disabled):
    from gateway.session_routing import resolve_policy
    override, _ = resolve_policy(runner, config, source)
    if override is None:
        return enabled, disabled
    if override.toolsets is not None:
        # Bypass platform convenience expansion: native/plugin defaults are not an allowlist.
        enabled = list(override.toolsets)
    disabled = sorted(set(disabled or []) | set(parse_config_string_list(override.disabled_toolsets)))
    return enabled, disabled or None


def pin_session_policy(store, session_key, session_id, enabled, disabled):
    """Policy changes take effect after /new; a restart keeps the prior selection.

    Stamp session_id because reset paths can carry unrelated metadata forward.
    """
    key = "tool_policy"
    policy = store.get_session_metadata(session_key, key)
    if not isinstance(policy, dict) or policy.get("session_id") != session_id:
        policy = {"session_id": session_id, "enabled": list(enabled) if enabled is not None else None, "disabled": list(disabled or [])}
        if not store.set_session_metadata(session_key, key, policy):
            raise RuntimeError("Could not persist session tool policy")
    return (list(policy["enabled"]) if policy["enabled"] is not None else None), list(policy["disabled"]) or None


def channel_efficiency(runner, config, source):
    from agent.efficiency import settings
    from gateway.session_routing import resolve_policy
    override, _ = resolve_policy(runner, config, source)
    raw = dict((config.get("agent") or {}).get("efficiency") or {})
    if override is not None and override.efficiency is not None:
        raw.update(override.efficiency)
    return settings({"agent": {"efficiency": raw}})


def pin_efficiency(store, session_key, session_id, settings):
    policy = store.get_session_metadata(session_key, "efficiency_policy")
    if not isinstance(policy, dict) or policy.get("session_id") != session_id:
        policy = {"session_id": session_id, "settings": settings}
        if not store.set_session_metadata(session_key, "efficiency_policy", policy):
            raise RuntimeError("Could not persist session efficiency policy")
    return dict(policy["settings"])
