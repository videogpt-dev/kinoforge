"""HTTP request logging middleware: one line in, one line out, per request."""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from fastapi import Request, Response

from kinoforge.observ import build_logger, shape


def _summarize_body(raw: bytes) -> Tuple[Optional[str], Any]:
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
    except ValueError:
        return None, {"raw_bytes": len(raw)}
    if not isinstance(data, dict):
        return None, {"type": type(data).__name__}
    return data.get("log_level"), shape(data)


async def log_requests(request: Request, call_next: Any) -> Response:
    # Reading the body caches it, so route handlers still parse it.
    level, body = _summarize_body(await request.body())
    logger = build_logger(segment="http", level=level)
    logger.debug(
        f"-> {request.method} {request.url.path}",
        query=dict(request.query_params) or None, body=body,
    )
    response = await call_next(request)
    logger.debug(f"<- {request.method} {request.url.path} {response.status_code}",
                 status=response.status_code)
    return response
