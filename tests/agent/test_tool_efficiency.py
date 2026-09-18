from agent.tool_efficiency import annotate_schemas, consume_escalation


def _schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "A tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def test_heavy_tools_are_guided_not_availability_gated():
    schema = annotate_schemas([_schema("browser_exec")])[0]["function"]
    assert "judiciously" in schema["description"]
    assert "escalation_reason" in schema["parameters"]["properties"]
    assert "escalation_reason" not in schema["parameters"]["required"]


def test_escalation_metadata_is_optional_but_validated_when_present():
    args = {}
    assert consume_escalation(None, "browser_exec", args) is None
    assert args == {}

    args = {"escalation_reason": "desktop_only_workflow"}
    assert consume_escalation(None, "computer_use", args) is None
    assert args == {}
