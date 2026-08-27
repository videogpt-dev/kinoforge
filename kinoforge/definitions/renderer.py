from string import Template
from typing import Mapping

from kinoforge.definitions.models import DefinitionBundle, DefinitionError, DefinitionKind


class DefinitionRenderer:
    def __init__(self, bundle: DefinitionBundle) -> None:
        self._bundle = bundle

    def __call__(self, key: str, **params: object) -> str:
        return self.render(DefinitionKind.AGENT, key, **params)

    def render(self, kind: DefinitionKind, key: str, **params: object) -> str:
        definition = self._bundle.require(kind, key)
        if not definition.body.strip():
            raise DefinitionError(f"required {kind.value} definition is empty: {key}")
        values: Mapping[str, str] = {
            name: "" if value is None else str(value) for name, value in params.items()
        }
        try:
            return Template(definition.body).substitute(values)
        except (KeyError, ValueError) as exc:
            raise DefinitionError(f"could not render definition {key}: {exc}") from exc
