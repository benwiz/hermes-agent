"""Channel allowlists flow through real resolution and survive session reloads."""
from types import SimpleNamespace

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, SessionStore
from gateway.session_tool_policy import channel_toolsets, pin_session_policy


def test_channel_schemas_are_isolated_and_gated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("tools:\n  tool_search: false\n")
    from tools.registry import registry
    from model_tools import get_tool_definitions
    for name, available in (("policy_home", True), ("policy_coding", True), ("policy_unavailable", False)):
        registry.register(name=name, toolset=name, schema={"name": name, "description": "fixture", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: "ok", check_fn=lambda available=available: available)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(channel_overrides={
        "home": ChannelOverride(toolsets=["policy_home", "policy_unavailable"]),
        "project": ChannelOverride(toolsets=["policy_coding"]),
    })})
    runner = SimpleNamespace(config=config)
    user_config = {"mcp_servers": {"unrelated": {"command": "unused", "enabled": True}}}
    for channel, expected in (("home", "policy_home"), ("project", "policy_coding")):
        source = SessionSource(platform=Platform.DISCORD, chat_id="thread", parent_chat_id=channel)
        enabled, disabled = channel_toolsets(runner, user_config, source, "discord", ["browser"], None)
        assert "unrelated" not in enabled
        schemas = get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True)
        assert {t["function"]["name"] for t in schemas} == {expected}
    assert ChannelOverride.from_dict(config.platforms[Platform.DISCORD].channel_overrides["home"].to_dict()).toolsets == ["policy_home", "policy_unavailable"]


def test_pinned_policy_survives_restart_but_new_session_can_escalate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.DISCORD, chat_id="home")
    entry = store.get_or_create_session(source)
    assert pin_session_policy(store, entry.session_key, entry.session_id, ["web"], ["browser"]) == (["web"], ["browser"])
    reloaded = SessionStore(tmp_path / "sessions", config)
    entry = reloaded.get_or_create_session(source)
    assert pin_session_policy(reloaded, entry.session_key, entry.session_id, ["browser"], []) == (["web"], ["browser"])
    fresh = reloaded.get_or_create_session(source, force_new=True)
    assert pin_session_policy(reloaded, fresh.session_key, fresh.session_id, ["browser"], []) == (["browser"], None)
