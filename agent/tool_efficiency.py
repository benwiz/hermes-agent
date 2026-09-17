"""Structured-tool-first guidance and bounded escalation metadata, without new tools."""
from copy import deepcopy

from agent.efficiency import ESCALATION_REASONS


def escalation_tool(name):
    return name.startswith(("browser_", "computer"))


def annotate_schemas(schemas):
    result = []
    for original in schemas:
        name = original["function"]["name"]
        if not escalation_tool(name):
            result.append(original)
            continue
        schema = deepcopy(original)
        fn = schema["function"]
        if "escalation_reason" in fn.get("parameters", {}).get("properties", {}):
            result.append(schema)
            continue
        fn["description"] = fn.get("description", "") + (
            " Prefer available domain APIs for Home/media actions, file/search/terminal tools for local work, "
            "and public search/extraction for documents. Use a browser only for interaction, authentication, "
            "JavaScript-only data or visual verification; use computer control only for desktop-only work "
            "or unavailable structured/browser paths. Record the prerequisite in escalation_reason."
        )
        params = fn.setdefault("parameters", {"type": "object"})
        params.setdefault("properties", {})["escalation_reason"] = {
            "type": "string", "enum": sorted(allowed_reasons(name)),
            "description": "Why structured tools cannot complete this step; no page content or secrets.",
        }
        if "escalation_reason" not in params.setdefault("required", []):
            params["required"].append("escalation_reason")
        result.append(schema)
    return result


def consume_escalation(agent, name, args):
    """Strip runtime metadata before vendor dispatch. Return a blocking reason if absent."""
    if not escalation_tool(name):
        return None
    reason = args.pop("escalation_reason", None)
    allowed = allowed_reasons(name)
    if not isinstance(reason, str) or reason not in allowed:
        return "Provide a valid escalation_reason explaining why structured tools cannot complete this step."
    state = getattr(agent, "_efficiency_turn", None)
    if state is not None:
        state.escalation_reasons.add(reason)
    return None


def allowed_reasons(name):
    if name.startswith("computer"):
        return {"desktop_only_workflow", "structured_api_unavailable"}
    return ESCALATION_REASONS - {"desktop_only_workflow"}
