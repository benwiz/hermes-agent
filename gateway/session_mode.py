"""Explicit capability escalation uses the existing session reset lifecycle."""
async def handle_mode(runner, event):
    from hermes_cli.config import load_config
    from gateway.run import _get_channel_override
    from gateway.config import ChannelOverride
    source = event.source
    config = load_config()
    routing = (config.get("session_routing") or {}).get(source.platform.value)
    if not routing:
        return "Session modes are not configured for this platform."
    key = runner._session_key_for_source(source)
    requested = event.get_command_args().strip()
    if not requested:
        mode = runner.session_store.get_session_metadata(key, "session_mode") or {}
        schema = runner.session_store.get_session_metadata(key, "tool_schema") or {}
        return f"Mode: {mode.get('tier', 'channel policy')}. Tool signature: {schema.get('signature', 'not initialized')}. Available: {', '.join(routing['tiers'])}."
    if requested not in routing["tiers"]:
        return "Unknown mode. Available: " + ", ".join(routing["tiers"])
    override = _get_channel_override(runner.config, source.platform, source.chat_id,
        thread_id=source.thread_id, parent_id=source.parent_chat_id)
    if override and override.allowed_tiers is not None and requested not in override.allowed_tiers:
        return "That mode is not allowed in this channel."
    # Validate before destroying a conversation; no model request is needed to rotate.
    policy = ChannelOverride.from_dict(routing["tiers"][requested])
    if policy.toolsets is None:
        return "This mode needs an explicit toolsets allowlist."
    await runner._handle_reset_command(event)
    entry = runner.session_store.get_or_create_session(source)
    if not runner.session_store.set_session_metadata(key, "session_mode", {
        "session_id": entry.session_id, "tier": requested,
    }):
        raise RuntimeError("Could not persist session mode")
    return f"Started a new {requested} session. The tool signature will be recorded on its first turn."
