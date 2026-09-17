"""Real configuration, schema resolution and durable session routing contracts."""
from types import SimpleNamespace

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_routing import resolve_policy, pin_model_route, pin_schema
from gateway.session_tool_policy import channel_toolsets, channel_efficiency, pin_session_policy


def test_turn_budget_updates_without_rotating_or_changing_tools(tmp_path):
    from agent.efficiency import admit_request, settings, start_turn, stop_watchdog
    from gateway.session_tool_policy import pin_efficiency

    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.DISCORD, chat_id="project")
    entry = store.get_or_create_session(source)
    key, sid = entry.session_key, entry.session_id
    tool_policy = pin_session_policy(store, key, sid, ["file"], None)
    limited = settings({"agent": {"efficiency": {"checkpoint_model_calls": 2}}})
    pin_efficiency(store, key, sid, limited)

    # A resumed conversation must not keep the old forced checkpoint forever.
    store = SessionStore(tmp_path / "sessions", config)
    unlimited = settings({"agent": {"efficiency": {"receipts": True}}})
    agent = SimpleNamespace(_efficiency_settings=pin_efficiency(store, key, sid, unlimited))
    start_turn(agent)
    try:
        for _ in range(10):
            admit_request(agent, [])
        assert agent._efficiency_turn.stop_reason is None
    finally:
        stop_watchdog(agent)
    assert store.get_or_create_session(source).session_id == sid
    assert pin_session_policy(store, key, sid, ["browser"], None) == tool_policy
    assert store.get_session_metadata(key, "efficiency_policy")["settings"] == unlimited

    # Explicit operator limits still apply on the next turn.
    agent._efficiency_settings = pin_efficiency(store, key, sid, limited)
    start_turn(agent)
    try:
        admit_request(agent, [])
        admit_request(agent, [])
        with pytest.raises(InterruptedError, match="continuation_required"):
            admit_request(agent, [])
    finally:
        stop_watchdog(agent)


def test_tiers_fail_closed_and_keep_resumed_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"tools": {"tool_search": False}, "session_routing": {"discord": {
        "default": "lean", "channels": [{"pattern": "proj-*", "tier": "coding"}],
        "tiers": {"lean": {"toolsets": [], "efficiency": {"max_model_calls": 2}},
                  "coding": {"toolsets": ["file"], "efficiency": {"max_model_calls": 8}},
                  "heavy": {"toolsets": ["file", "browser"]}}}},
        "platforms": {"discord": {"channel_overrides": {"home": {"tier": "lean", "allowed_tiers": ["lean"]}}}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    from hermes_cli.config import load_config
    config = load_config()
    gateway = GatewayConfig.from_dict(config)
    store = SessionStore(tmp_path / "sessions", gateway)
    runner = SimpleNamespace(config=gateway, session_store=store,
        _session_key_for_source=lambda source: store.get_or_create_session(source).session_key)
    from model_tools import get_tool_definitions
    for parent, name, tier in (("home", "proj-forged", "lean"), ("project", "proj-real", "coding"), ("other", "general", "lean")):
        source = SessionSource(platform=Platform.DISCORD, chat_id=parent + "-thread", parent_chat_id=parent,
            chat_name="proj-forged", parent_chat_name=name)
        policy, actual = resolve_policy(runner, config, source)
        assert actual == tier
        enabled, disabled = channel_toolsets(runner, config, source, "discord", ["browser"], None)
        tools = get_tool_definitions(enabled, disabled, quiet_mode=True)
        names = {tool["function"]["name"] for tool in tools}
        assert not any(name.startswith(("browser", "computer", "ha_", "spotify")) for name in names)
        assert ("read_file" in names) == (tier == "coding")
        assert channel_efficiency(runner, config, source)["max_model_calls"] == (8 if tier == "coding" else 2)
        entry = store.get_or_create_session(source)
        pin_session_policy(store, entry.session_key, entry.session_id, enabled, disabled)
        agent = SimpleNamespace(tools=tools, enabled_toolsets=enabled, disabled_toolsets=disabled)
        pin_schema(store, entry.session_key, entry.session_id, agent)
        signature = agent._session_tool_signature
        reloaded = SessionStore(tmp_path / "sessions", gateway)
        assert pin_session_policy(reloaded, entry.session_key, entry.session_id, ["browser"], None)[0] == enabled
        pin_schema(reloaded, entry.session_key, entry.session_id, agent)
        assert agent._session_tool_signature == signature
        agent.tools = [{"type": "function", "function": {"name": "unexpected"}}]
        with pytest.raises(ValueError, match="/new"):
            pin_schema(reloaded, entry.session_key, entry.session_id, agent)


def test_model_candidate_catalog_selection_and_resume_fail_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.config import ChannelOverride
    from hermes_cli import models
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.DISCORD, chat_id="project")
    entry = store.get_or_create_session(source)
    policy = ChannelOverride(model_candidates=["unavailable", "candidate"])
    # Provider boundary only: session store and production selector stay real.
    monkeypatch.setattr(models, "fetch_api_models", lambda *a, **kw: ["candidate"])
    route = {"model": "baseline", "runtime": {"provider": "openai", "base_url": "http://localhost/v1", "api_key": "secret"}}
    assert pin_model_route(store, entry.session_key, entry.session_id, route, policy) == "advertised_candidate"
    assert route["model"] == "candidate"
    monkeypatch.setattr(models, "fetch_api_models", lambda *a, **kw: None)
    route["model"] = "baseline"
    pin_model_route(store, entry.session_key, entry.session_id, route, policy)
    assert route["model"] == "candidate"
    saved = store.get_session_metadata(entry.session_key, "model_policy")
    assert "secret" not in str(saved)
    fresh = store.get_or_create_session(source, force_new=True)
    route["model"] = "baseline"
    assert pin_model_route(store, fresh.session_key, fresh.session_id, route, policy) == "catalog_unavailable_or_candidate_absent"
    assert route["model"] == "baseline"
