"""Modes exercise real reset, disk persistence and adapter control dispatch."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from gateway.session_mode import handle_mode, normalize_mode_control, recover_before_turn


@pytest.fixture
def modes(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {'session_routing': {'discord': {'default': 'coding', 'tiers': {
        'coding': {'toolsets': ['file']}, 'heavy': {'toolsets': ['file', 'terminal', 'browser']}}}}}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config))
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig.from_dict(config)
    runner.session_store = SessionStore(tmp_path / 'sessions', runner.config)
    runner.adapters = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._voice_mode = {}
    source = SessionSource(platform=Platform.DISCORD, chat_id='project', chat_type='channel')
    return runner, source


@pytest.mark.asyncio
async def test_lifecycle_lease_restart_and_handoff(modes):
    runner, source = modes
    store = runner.session_store
    old = store.get_or_create_session(source)
    old_sid, key = old.session_id, old.session_key
    store.append_to_transcript(old_sid, {'role': 'user', 'content': 'finish the task'})
    store.append_to_transcript(old_sid, {'role': 'tool', 'content': 'secret tool output'})
    result = await handle_mode(runner, MessageEvent('/mode heavy', source=source))
    assert 'new heavy session' in result
    sid = store.get_or_create_session(source).session_id
    assert sid != old_sid
    assert 'finish the task' in str(store.load_transcript(sid))
    assert 'secret tool output' not in str(store.load_transcript(sid))
    await handle_mode(runner, MessageEvent('/mode heavy', source=source))
    assert store.get_or_create_session(source).session_id == sid
    runner.session_store = SessionStore(store.sessions_dir, runner.config)
    runner._async_session_store = None
    store = runner.session_store
    assert 'heavy' in await handle_mode(runner, MessageEvent('/mode status', source=source))
    saved = store.get_session_metadata(key, 'session_mode')
    saved['expires_at'] = 1
    store.set_session_metadata(key, 'session_mode', saved)
    await recover_before_turn(runner, MessageEvent('next task', source=source))
    assert store.get_or_create_session(source).session_id != sid
    from gateway.session_routing import resolve_policy
    from hermes_cli.config import load_config
    assert resolve_policy(runner, load_config(), source)[1] == 'coding'
    await handle_mode(runner, MessageEvent('/mode heavy', source=source))
    await handle_mode(runner, MessageEvent('/mode off', source=source))
    assert resolve_policy(runner, load_config(), source)[1] == 'coding'


@pytest.mark.asyncio
async def test_invalid_status_and_old_alias_recovery(modes):
    runner, source = modes
    store = runner.session_store
    old = store.get_or_create_session(source)
    sid = old.session_id
    for command in ('/mode status', '/mode typo'):
        await handle_mode(runner, MessageEvent(command, source=source))
        assert store.get_or_create_session(source).session_id == sid
    store.set_session_metadata(old.session_key, 'tool_policy', {
        'session_id': sid, 'enabled': ['sasha-discord-coding']})
    await recover_before_turn(runner, MessageEvent('continue', source=source))
    assert store.get_or_create_session(source).session_id != sid


@pytest.mark.parametrize('text,expected', [
    ('Please use heavy mode.', '/mode heavy'), ('exit heavy mode!', '/mode off'),
    ('How do I use heavy mode?', None), ('Do not use heavy mode.', None),
    ('"Use heavy mode"', None), ('Use heavy mode to delete everything', None)])
def test_unambiguous_controls(text, expected):
    event = MessageEvent(text)
    normalize_mode_control(event)
    assert event.text == (expected or text)
    internal = MessageEvent(text, internal=True)
    normalize_mode_control(internal)
    assert internal.text == text


@pytest.mark.asyncio
async def test_disallowed_and_unknown_saved_modes_recover_without_unlocking(modes):
    from gateway.config import ChannelOverride, PlatformConfig
    runner, source = modes
    runner.config.platforms[Platform.DISCORD] = PlatformConfig(channel_overrides={
        'project': ChannelOverride(tier='coding', allowed_tiers=['coding'])})
    store = runner.session_store
    entry = store.get_or_create_session(source)
    old_sid = entry.session_id
    assert 'not allowed' in await handle_mode(runner, MessageEvent('/mode heavy', source=source))
    assert store.get_or_create_session(source).session_id == old_sid
    for tier in ('removed', 'heavy'):
        entry = store.get_or_create_session(source)
        old_sid = entry.session_id
        store.set_session_metadata(entry.session_key, 'session_mode', {
            'session_id': old_sid, 'tier': tier, 'expires_at': 9999999999})
        await recover_before_turn(runner, MessageEvent('continue', source=source))
        assert store.get_or_create_session(source).session_id != old_sid
        assert 'Mode: coding' in await handle_mode(runner, MessageEvent('/mode status', source=source))
