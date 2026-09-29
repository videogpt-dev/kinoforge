"""Kinoforge HTTP entrypoint. Builds the FastAPI app and mounts the segment routers.

Routes live in kinoforge/api/routers/{meta,clips,story,series}.py; shared-volume path guards
in kinoforge/api/paths.py; incoming-request logging in kinoforge/api/observe.py. Boot:
uvicorn kinoforge.api.app:app."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from kinoforge.api.observe import log_requests
from kinoforge.api.routers import ALL as ROUTERS
from kinoforge.service.executions import ExecutionConflict

app = FastAPI(
    title="Kinoforge API",
    summary="VideoGPT generation engine",
    description=(
        "Source-available clips, story, and series generation engine. API exposes clips"
        " execution, Story writing and edits, plus supporting media operations. "
        "Runs on a trusted internal network; the caller is the only client."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "Service", "description": "Health and service metadata."},
        {"name": "Discovery", "description": "Engine segments and their implementation status."},
        {"name": "Clips", "description": "Execute complete long-video to short-clips jobs."},
        {"name": "Story", "description": "Write story plans and generate scene media."},
        {"name": "Series", "description": "Plan connected episodes for a recurring show."},
        {"name": "Media", "description": "Probe, render, and aspect-format shared-volume media."},
        {"name": "Transcription", "description": "Transcribe and align narration for captions."},
    ],
)

app.middleware("http")(log_requests)


@app.exception_handler(ExecutionConflict)
async def _execution_conflict(_request: Request, exc: ExecutionConflict) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


for _router in ROUTERS:
    app.include_router(_router)
