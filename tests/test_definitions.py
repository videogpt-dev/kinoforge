"""Definition bundle parsing, engine-version gating, lookup, and template rendering."""

from __future__ import annotations

import pytest

from kinoforge.definitions.models import (
    Definition,
    DefinitionBundle,
    DefinitionError,
    DefinitionKind,
)
from kinoforge.definitions.renderer import DefinitionRenderer


def _agent(key="writer", body="Write $subject."):
    return {"type": "agent", "key": key, "body": body}


def _bundle_mapping(defs=None, *, minimum="0.1.0", maximum=""):
    return {
        "schema_version": 1,
        "id": "cat1",
        "version": "1.0.0",
        "engine": {"minimum": minimum, "maximum_exclusive": maximum},
        "definitions": defs if defs is not None else [_agent()],
    }


# --- Definition.from_mapping ---------------------------------------------

def test_definition_from_mapping_reads_fields():
    d = Definition.from_mapping({"type": "preset", "key": "k", "name": "N", "body": "B"})
    assert d.kind is DefinitionKind.PRESET
    assert d.key == "k" and d.name == "N" and d.body == "B"


def test_definition_bad_type_raises():
    with pytest.raises(DefinitionError):
        Definition.from_mapping({"type": "nope", "key": "k"})


def test_definition_missing_key_raises():
    with pytest.raises(DefinitionError):
        Definition.from_mapping({"type": "agent", "key": "  "})


def test_definition_non_dict_extras_raises():
    with pytest.raises(DefinitionError):
        Definition.from_mapping({"type": "agent", "key": "k", "extras": ["x"]})


# --- DefinitionBundle.from_mapping ---------------------------------------

def test_bundle_parses_and_holds_definitions():
    bundle = DefinitionBundle.from_mapping(_bundle_mapping(), engine_version="0.2.0")
    assert bundle.catalog_id == "cat1"
    assert len(bundle.definitions) == 1


def test_bundle_wrong_schema_raises():
    with pytest.raises(DefinitionError):
        DefinitionBundle.from_mapping({"schema_version": 2, "id": "x", "version": "1"},
                                      engine_version="0.1.0")


def test_bundle_requires_id_and_version():
    m = _bundle_mapping()
    m["id"] = ""
    with pytest.raises(DefinitionError):
        DefinitionBundle.from_mapping(m, engine_version="0.1.0")


def test_bundle_rejects_engine_below_minimum():
    with pytest.raises(DefinitionError):
        DefinitionBundle.from_mapping(_bundle_mapping(minimum="0.5.0"), engine_version="0.1.0")


def test_bundle_rejects_engine_at_or_above_maximum():
    with pytest.raises(DefinitionError):
        DefinitionBundle.from_mapping(
            _bundle_mapping(maximum="0.2.0"), engine_version="0.2.0"
        )


def test_bundle_accepts_engine_below_maximum():
    bundle = DefinitionBundle.from_mapping(
        _bundle_mapping(maximum="0.9.0"), engine_version="0.2.0"
    )
    assert bundle.engine_maximum_exclusive == "0.9.0"


def test_bundle_rejects_duplicate_kind_and_key():
    m = _bundle_mapping([_agent("dup"), _agent("dup")])
    with pytest.raises(DefinitionError):
        DefinitionBundle.from_mapping(m, engine_version="0.1.0")


def test_bundle_bad_engine_version_string_raises():
    with pytest.raises(DefinitionError):
        DefinitionBundle.from_mapping(_bundle_mapping(), engine_version="not-a-version")


# --- require --------------------------------------------------------------

def test_require_returns_matching_definition():
    bundle = DefinitionBundle.from_mapping(_bundle_mapping(), engine_version="0.1.0")
    assert bundle.require(DefinitionKind.AGENT, "writer").key == "writer"


def test_require_missing_raises():
    bundle = DefinitionBundle.from_mapping(_bundle_mapping(), engine_version="0.1.0")
    with pytest.raises(DefinitionError):
        bundle.require(DefinitionKind.PRESET, "writer")


# --- DefinitionRenderer ---------------------------------------------------

def _renderer(body="Write $subject."):
    bundle = DefinitionBundle.from_mapping(_bundle_mapping([_agent(body=body)]),
                                           engine_version="0.1.0")
    return DefinitionRenderer(bundle)


def test_render_substitutes_params():
    assert _renderer().render(DefinitionKind.AGENT, "writer", subject="a cat") == "Write a cat."


def test_render_none_param_becomes_empty_string():
    assert _renderer().render(DefinitionKind.AGENT, "writer", subject=None) == "Write ."


def test_render_empty_body_raises():
    with pytest.raises(DefinitionError):
        _renderer(body="   ").render(DefinitionKind.AGENT, "writer", subject="x")


def test_render_missing_placeholder_raises():
    with pytest.raises(DefinitionError):
        _renderer().render(DefinitionKind.AGENT, "writer")  # no subject supplied


def test_call_shortcut_targets_agent_kind():
    assert _renderer()("writer", subject="dogs") == "Write dogs."
