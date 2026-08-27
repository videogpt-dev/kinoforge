import hmac
import os
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

service_token = HTTPBearer(
    auto_error=False,
    scheme_name="ServiceToken",
    bearerFormat="shared secret",
    description="Core-to-Kinoforge token from KINOFORGE_SERVICE_TOKEN.",
)


def require_service(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(service_token),
    ],
) -> None:
    expected = os.getenv("KINOFORGE_SERVICE_TOKEN") or ""
    if not expected:
        return
    supplied = credentials.credentials if credentials else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid service token")
