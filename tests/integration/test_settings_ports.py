"""ServiceSettings env loading + the InfrelayTextPorts adapter (httpx-gated: needs InfrelayClient)."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from kinoforge.service.settings import ServiceSettings  # noqa: E402
from kinoforge.service.runtimes.text_ports import InfrelayTextPorts  # noqa: E402


# --- ServiceSettings ------------------------------------------------------

def test_from_env_reads_route_and_paths(monkeypatch):
    monkeypatch.setenv("INFRELAY_URL", "http://infrelay:8090")
    monkeypatch.setenv("INFRELAY_SERVICE_TOKEN", "tok")
    monkeypatch.setenv("KINOFORGE_SHARED_ROOT", "/data/output")
    monkeypatch.delenv("KINOFORGE_CACHE_DIR", raising=False)
    s = ServiceSettings.from_env()
    assert s.infrelay_url == "http://infrelay:8090"
    assert s.infrelay_token == "tok"
    assert str(s.shared_root) == "/data/output"
    assert str(s.cache_dir) == "/data/output/.kinoforge-cache"


def test_from_env_defaults_when_unset(monkeypatch):
    for key in ("INFRELAY_URL", "INFRELAY_SERVICE_TOKEN", "KINOFORGE_SHARED_ROOT",
                "KINOFORGE_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    s = ServiceSettings.from_env()
    assert s.infrelay_url == "" and s.infrelay_token == ""
    assert str(s.shared_root) == "/app/output"


def test_infrelay_builds_client_for_owner():
    client = ServiceSettings(infrelay_url="http://x/", infrelay_token="t").infrelay("owner1")
    assert client._url == "http://x"  # trailing slash trimmed
    assert client._tenant == "owner1"


# --- InfrelayTextPorts ----------------------------------------------------

class _FakeInfrelay:
    def __init__(self):
        self.calls = []

    def text(self, provider, model, system, user, *, temperature, max_tokens):
        self.calls.append((provider, model))
        return '{"episodes": [1, 2]}', {"usage": 3}


def _ports(default_pick, *, budgets=None, contract_key=None, language_key=None):
    return InfrelayTextPorts(
        _FakeInfrelay(),
        renderer=object(),  # unused in the paths tested here
        default_pick=default_pick,
        budgets=budgets or {},
        label="story",
        contract_key=contract_key,
        language_key=language_key,
    )


def test_complete_routes_through_default_pick():
    p = _ports({"provider": "op", "model": "m"})
    text, usage = p.complete("x", "sys", "usr", pick={}, project="", temperature=0.3,
                             max_tokens=100, bill=True)
    assert text == '{"episodes": [1, 2]}'
    assert usage == {"usage": 3}
    assert p._infrelay.calls == [("op", "m")]


def test_pick_overrides_default():
    p = _ports({"provider": "op", "model": "m"})
    p.complete("x", "s", "u", pick={"model": "m2"}, project="", temperature=0.1, max_tokens=10,
               bill=True)
    assert p._infrelay.calls == [("op", "m2")]


def test_complete_without_provider_raises():
    p = _ports({})
    with pytest.raises(RuntimeError, match="story text route is missing a provider"):
        p.complete("x", "s", "u", pick={}, project="", temperature=0.1, max_tokens=10, bill=True)


def test_complete_json_parses_result():
    p = _ports({"provider": "op", "model": "m"})
    assert p.complete_json("x", "s", "u", pick={}, temperature=0.2, max_tokens=50) == {
        "episodes": [1, 2]
    }


def test_parse_json_strips_code_fences():
    assert InfrelayTextPorts.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert InfrelayTextPorts.parse_json('noise {"a": 2} trailing') == {"a": 2}


def test_story_budget_clamps_override_to_limit():
    p = _ports({}, budgets={"story": 100})
    assert p.story_budget() == 100
    assert p.story_budget("50") == 50
    assert p.story_budget("200") == 100  # capped
    assert p.story_budget("0") == 100    # non-positive ignored
    assert p.story_budget("bad") == 100  # unparseable ignored


def test_missing_fragment_keys_raise():
    p = _ports({}, contract_key=None, language_key=None)
    with pytest.raises(RuntimeError, match="contract fragment is not available"):
        p.contract(6)
    with pytest.raises(RuntimeError, match="language fragment is not available"):
        p.language_rule("title", "fr")
