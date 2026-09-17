"""Receipt privacy, independent admission limits, and real loop integration."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.efficiency import admit_request, finish_turn, settings, start_turn


def test_receipts_measure_each_turn_and_limits_never_refund(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(session_id="session", model="model", provider="provider",
        session_input_tokens=10, _efficiency_settings=settings({"agent": {"efficiency": {
            "receipts": True, "max_model_calls": 2, "max_seconds": 60}}}))
    start_turn(agent)
    schema = [{"type": "function", "function": {"name": "read_file", "description": "private schema description"}}]
    admit_request(agent, schema)
    admit_request(agent, schema)
    with pytest.raises(InterruptedError, match="model_call_budget"):
        admit_request(agent, schema)
    agent.session_input_tokens = 30
    result = {"turn_exit_reason": "interrupted", "completed": False}
    finish_turn(agent, result, "turn")
    raw = (tmp_path / "logs/efficiency.jsonl").read_text()
    receipt = json.loads(raw)
    assert "private" not in raw
    assert receipt["tokens"]["input_tokens"] == 20
    assert receipt["model_calls"] == 2
    assert len(receipt["tool_schemas"]) == 1
    assert receipt["stop_reason"] == "model_call_budget"
    start_turn(agent)
    agent._efficiency_turn.started -= 61
    with pytest.raises(InterruptedError, match="wall_time_budget"):
        admit_request(agent, schema)
    assert agent._efficiency_turn.model_calls == 0


def test_real_turn_stops_at_checkpoint_and_persists_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("tools:\n  tool_search: false\nagent:\n  efficiency:\n    receipts: true\n    checkpoint_model_calls: 1\n")
    from run_agent import AIAgent
    from tools.registry import registry
    registry.register(name="efficiency_probe", toolset="efficiency_probe",
        schema={"name": "efficiency_probe", "description": "Probe", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: '{"ok": true}', check_fn=lambda: True)
    agent = AIAgent(model="test-model", provider="openai", api_key="test-key",
        base_url="http://localhost:9/v1", enabled_toolsets=["efficiency_probe"],
        quiet_mode=True, skip_memory=True, skip_context_files=True,
        skip_background_review=True, max_iterations=10)
    tool = SimpleNamespace(id="probe-1", type="function", function=SimpleNamespace(name="efficiency_probe", arguments="{}"))
    message = SimpleNamespace(content=None, tool_calls=[tool], reasoning_content=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], usage=None)
    agent.client = MagicMock()
    call = MagicMock(return_value=response)
    monkeypatch.setattr(agent, "_interruptible_api_call", call)
    result = agent.run_conversation("Do the probe. Sensitive user text.")
    assert call.call_count == 1
    assert result["turn_exit_reason"] == "continuation_required"
    assert not result["completed"]
    assert "continue" in result["final_response"]
    receipt = json.loads((tmp_path / "logs/efficiency.jsonl").read_text().splitlines()[-1])
    assert receipt["model_calls"] == 1
    assert receipt["tool_calls"] == 1
    assert "Sensitive" not in json.dumps(receipt)


def test_checkpoint_model_calls_cannot_exceed_max_model_calls():
    import pytest
    from agent.efficiency import settings

    with pytest.raises(ValueError, match="checkpoint_model_calls.*max_model_calls"):
        settings({"agent": {"efficiency": {"checkpoint_model_calls": 11, "max_model_calls": 10}}})
