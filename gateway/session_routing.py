"""Opt-in channel tiers resolved once, before constructing a gateway agent."""
from fnmatch import fnmatchcase

from gateway.config import ChannelOverride


def resolve_policy(runner, config, source, *, ignore_saved=False):
    from gateway.run import _get_channel_override
    explicit = _get_channel_override(runner.config, source.platform, source.chat_id,
        thread_id=source.thread_id, parent_id=source.parent_chat_id)
    routing = (config.get("session_routing") or {}).get(source.platform.value)
    if not routing:
        return explicit, None
    tiers = routing["tiers"]
    tier = routing["default"]
    # Only adapter-supplied parent names may route a thread. Its editable title is not a channel.
    name = source.parent_chat_name if source.parent_chat_id else source.channel_name
    for rule in routing.get("channels", []):
        if name and fnmatchcase(name, rule["pattern"]):
            tier = rule["tier"]
            break
    if explicit and explicit.tier:
        tier = explicit.tier
    key = runner._session_key_for_source(source)
    entry = runner.session_store.get_or_create_session(source)
    mode = runner.session_store.get_session_metadata(key, "session_mode")
    if not ignore_saved and isinstance(mode, dict) and mode.get("session_id") == entry.session_id:
        requested = mode["tier"]
        if explicit and explicit.allowed_tiers is not None and requested not in explicit.allowed_tiers:
            raise ValueError("Requested mode is not allowed in this channel")
        tier = requested
    if tier not in tiers:
        raise ValueError("Unknown session routing tier")
    raw = dict(tiers[tier])
    if "toolsets" not in raw:
        raise ValueError("Session routing tiers require an explicit toolsets allowlist")
    if explicit:
        raw.update({k: v for k, v in explicit.to_dict().items() if k != "tier"})
    return ChannelOverride.from_dict(raw), tier


def pin_model_route(store, key, sid, route, policy):
    """Persist only public route identity; credentials are always freshly resolved."""
    saved = store.get_session_metadata(key, "model_policy")
    runtime = route["runtime"]
    identity = {k: runtime.get(k) for k in ("provider", "base_url", "api_mode")}
    if isinstance(saved, dict) and saved.get("session_id") == sid:
        if saved["identity"] != identity:
            raise ValueError("Session provider changed; use /new to apply the new route")
        route["model"] = saved["model"]
        return saved["reason"]
    reason = "configured_route"
    candidates = policy.model_candidates if policy else None
    if candidates:
        from hermes_cli.models import fetch_api_models
        try:
            advertised = fetch_api_models(runtime.get("api_key"), runtime.get("base_url"),
                api_mode=runtime.get("api_mode")) or []
        except Exception:
            advertised = []
        selected = next((name for name in candidates if name in advertised), None)
        if selected:
            route["model"] = selected
            reason = "advertised_candidate"
        else:
            reason = "catalog_unavailable_or_candidate_absent"
    saved = {"session_id": sid, "model": route["model"], "identity": identity, "reason": reason}
    if not store.set_session_metadata(key, "model_policy", saved):
        raise RuntimeError("Could not persist session model route")
    return reason


def pin_schema(store, key, sid, agent):
    from agent.efficiency import schema_snapshot
    from agent.tool_executor import _tool_search_scoped_names
    from model_tools import get_tool_definitions
    full = getattr(agent, "_session_resolved_schemas", None)
    if full is None:
        full = get_tool_definitions(enabled_toolsets=agent.enabled_toolsets,
        disabled_toolsets=agent.disabled_toolsets, quiet_mode=True, skip_tool_search_assembly=True)
    agent._session_resolved_schemas = full
    snapshot = schema_snapshot({"advertised": agent.tools, "resolved": full})
    agent._session_deferred_names = _tool_search_scoped_names(agent)
    saved = store.get_session_metadata(key, "tool_schema")
    if isinstance(saved, dict) and saved.get("session_id") == sid:
        if saved["signature"] != snapshot["signature"]:
            raise ValueError("Session tool schema changed; use /new to apply the new capabilities")
    elif not store.set_session_metadata(key, "tool_schema", {"session_id": sid, **snapshot}):
        raise RuntimeError("Could not persist session schema signature")
    agent._session_tool_signature = snapshot["signature"]
    agent._skip_mcp_refresh = True
