"""Single source of service configuration read from the environment.

The runtimes used to each read INFRELAY_URL / INFRELAY_SERVICE_TOKEN (and the clips runtime
the shared/cache paths) in their own module-level factory. That env access now lives here, in
one frozen settings object every runtime builds from with `.from_env()`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kinoforge.inference import InfrelayClient

_DEFAULT_SHARED_ROOT = "/app/output"


@dataclass(frozen=True)
class ServiceSettings:
    infrelay_url: str = ""
    infrelay_token: str = ""
    shared_root: Path = Path(_DEFAULT_SHARED_ROOT)
    cache_dir: Path = Path(_DEFAULT_SHARED_ROOT) / ".kinoforge-cache"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        shared_root = Path(os.getenv("KINOFORGE_SHARED_ROOT") or _DEFAULT_SHARED_ROOT)
        return cls(
            infrelay_url=os.getenv("INFRELAY_URL") or "",
            infrelay_token=os.getenv("INFRELAY_SERVICE_TOKEN") or "",
            shared_root=shared_root,
            cache_dir=Path(os.getenv("KINOFORGE_CACHE_DIR") or shared_root / ".kinoforge-cache"),
        )

    def infrelay(self, owner: str = "") -> InfrelayClient:
        """An Infrelay client for one caller/tenant."""
        return InfrelayClient(self.infrelay_url, self.infrelay_token, owner)
