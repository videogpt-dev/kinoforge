from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, Mapping, Tuple


class DefinitionError(ValueError):
    pass


class DefinitionKind(StrEnum):
    SCREENWRITER = "screenwriter"
    AGENT = "agent"
    PRESET = "preset"
    FRAGMENT = "fragment"


def _version(value: str) -> Tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise DefinitionError(f"invalid engine version: {value}")
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


@dataclass(frozen=True)
class Definition:
    kind: DefinitionKind
    key: str
    subtype: str = ""
    name: str = ""
    category: str = ""
    body: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Definition:
        try:
            kind = DefinitionKind(str(value["type"]))
        except (KeyError, ValueError) as exc:
            raise DefinitionError("definition type is invalid") from exc
        key = str(value.get("key") or "").strip()
        if not key:
            raise DefinitionError("definition key is required")
        extras = value.get("extras") or {}
        if not isinstance(extras, dict):
            raise DefinitionError(f"definition extras must be an object: {key}")
        return cls(
            kind=kind,
            key=key,
            subtype=str(value.get("subtype") or ""),
            name=str(value.get("name") or ""),
            category=str(value.get("category") or ""),
            body=str(value.get("body") or ""),
            extras=dict(extras),
        )


@dataclass(frozen=True)
class DefinitionBundle:
    schema_version: int
    catalog_id: str
    version: str
    engine_minimum: str
    engine_maximum_exclusive: str = ""
    definitions: Tuple[Definition, ...] = ()

    @classmethod
    def empty(cls) -> DefinitionBundle:
        return cls(
            schema_version=1,
            catalog_id="empty",
            version="0",
            engine_minimum="0.1.0",
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        engine_version: str,
    ) -> DefinitionBundle:
        if int(value.get("schema_version") or 0) != 1:
            raise DefinitionError("unsupported definition bundle schema")
        catalog_id = str(value.get("id") or "").strip()
        version = str(value.get("version") or "").strip()
        if not catalog_id or not version:
            raise DefinitionError("definition bundle id and version are required")
        engine = value.get("engine") or {}
        if not isinstance(engine, dict):
            raise DefinitionError("definition bundle engine must be an object")
        minimum = str(engine.get("minimum") or "").strip()
        maximum = str(engine.get("maximum_exclusive") or "").strip()
        if not minimum:
            raise DefinitionError("definition bundle engine.minimum is required")
        current = _version(engine_version)
        if current < _version(minimum):
            raise DefinitionError(
                f"definition bundle requires Kinoforge {minimum} or newer"
            )
        if maximum and current >= _version(maximum):
            raise DefinitionError(
                f"definition bundle requires Kinoforge older than {maximum}"
            )

        raw_definitions = value.get("definitions") or []
        if not isinstance(raw_definitions, list):
            raise DefinitionError("definition bundle definitions must be an array")
        definitions = tuple(Definition.from_mapping(item) for item in raw_definitions)
        identities = [(item.kind, item.key) for item in definitions]
        if len(identities) != len(set(identities)):
            raise DefinitionError("definition bundle contains duplicate type and key")
        return cls(
            schema_version=1,
            catalog_id=catalog_id,
            version=version,
            engine_minimum=minimum,
            engine_maximum_exclusive=maximum,
            definitions=definitions,
        )

    def require(self, kind: DefinitionKind, key: str) -> Definition:
        for definition in self.definitions:
            if definition.kind is kind and definition.key == key:
                return definition
        raise DefinitionError(f"required {kind.value} definition is unavailable: {key}")
