# Kinoforge

Source-available generation engine behind VideoGPT. Internal segments: clips, series, and
story. Cloud core invokes engine through REST. Kinoforge never imports closed `app.*` code.

Current implementation:

- Clips pipeline: transcription, moment discovery/scoring, rendering, formatting, and project
  record generation.
- REST service on port `8100`.
- Inference through `INFRELAY_URL`.
- Optional bearer authentication through `KINOFORGE_SERVICE_TOKEN`.
- Shared-volume media adapter for current cloud deployment.
- Meter events returned to caller; cloud core performs billing.
- Cloud core owns durable jobs, projects, queue, retry, cancellation, and final artifact
  placement.
- Dataless execution boundary: caller supplies frozen resolved definitions; Kinoforge owns no
  catalog, prompt bodies, database, or durable queue.

## Definition boundary

Public community definitions live in separate versioned catalog repository. Cloud private
professional definitions remain in closed cloud database. Cloud core or self-host thin runtime
resolves selected agent stages, screenwriters, presets, and fragments into immutable bundle before
calling Kinoforge.

Self-host consumes pinned community releases from disk and can select or assign definitions. It
does not store prompt bodies in database or expose definition authoring.

See [ADR 0001](docs/decisions/0001-dataless-engine-and-definition-catalogs.md) for accepted
ownership and catalog rules.

See [clips reference flow](docs/clips-reference-flow.md) for Cloud submission, queue, definition
snapshot, REST execution, cancellation, persistence, artifacts, and billing path.

Run service:

```bash
docker build -t kinoforge apps/kinoforge
docker run --rm -p 8100:8100 \
  -e KINOFORGE_SHARED_ROOT=/app/output \
  -e INFRELAY_URL=http://infrelay:8090 \
  -v ./output:/app/output \
  kinoforge
```

Health endpoint: `GET /health`.

API explorer: `GET /docs`. ReDoc: `GET /redoc`. OpenAPI schema: `GET /openapi.json`.

Main execution endpoint: `POST /v1/segments/{code_name}/execute`. Segment-scoped media probe,
base render, aspect-format, and transcription-alignment endpoints support cloud
editor/publishing flows without Python imports.

Segment discovery: `GET /v1/segments` returns stable names, labels, icons, descriptions,
implementation status, API versions, and execution paths for clips, story, and series.

Status: clips cloud boundary implemented. Series/story extraction, assets URL transport,
self-host thin runtime, licensing, and public repository split remain.
