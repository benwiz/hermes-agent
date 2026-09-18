import tools.computer_use.cua_backend as cua_backend
import tools.computer_use.tool as computer_tool


def test_unrestricted_permission_mode_is_explicit_and_supported(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"permission_mode": "unrestricted"})
    assert cua_backend._cua_configured_permission_mode() == "unrestricted"


def test_unrestricted_permission_mode_skips_repetitive_action_consent(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"permission_mode": "unrestricted"})
    assert computer_tool._request_approval("click", {"x": 1, "y": 1}) is None


def test_invalid_permission_mode_fails_closed(monkeypatch):
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"permission_mode": "typo"})
    assert cua_backend._cua_configured_permission_mode() == "standard"
