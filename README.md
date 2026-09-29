# Kinoforge

The generation engine behind VideoGPT. It runs the clip, story, and series pipelines and is
driven over REST.

## What it does

- **Clips** turn a long video into short vertical clips: transcribe, find the best moments,
  render, and format.
- **Story** writes a scene-by-scene script and generates the image, video, music, and voice for
  each scene through Infrelay.
- **Series** plans a recurring show with a consistent cast, then runs each approved episode
  through the Story pipeline.

## How it fits in

Kinoforge is stateless by design. The caller resolves the definitions a run needs, such as agents,
screenwriters, and presets, into a frozen bundle and passes it in

Community definitions about prompts(screenwriters,presets,etc) live in a separate repo Videogpt-catalog.

## Running it

```bash
docker build -t kinoforge apps/kinoforge
docker run --rm -p 8100:8100 \
  -e KINOFORGE_SHARED_ROOT=/app/output \
  -e INFRELAY_URL=http://infrelay:8090 \
  -v ./output:/app/output \
  kinoforge
```

Inference runs through `INFRELAY_URL`. Kinoforge runs on a trusted internal network reachable only by the caller; it has no auth of its own.

## API

- `GET /health`, `GET /docs`, `GET /redoc`, `GET /openapi.json`
- `GET /v1/segments` lists the segments (clips, story, series) with their labels and status.
- `POST /v1/segments/{code_name}/execute` runs one.

## Clips request

A clips request splits into `options` (user choices: `clip_count`, `min_length`, `max_length`,
`formats`, `quality`, `generate_captions`, `min_interest_score`, `whisper_model`, `language`)
and `config` (resolved engine settings: `transcription`, `scoring`, `rendering`, `limits`,
`context_window`, `moment_finder`, `moment_route`). The engine merges them as
`{**config, **options}`, so a key belongs to one side only.

Moment finding is config-only:

- `moment_finder`: `auto` (AI if a route is set, else offline), `ai`, or `offline`.
- `moment_route`: `{"provider", "model"}` for AI finding.

`ai` without a route warns and falls back to offline.

`log_level` on the request sets that run's console verbosity; unset uses `KINOFORGE_LOG_LEVEL`
(default `error`). Returned `logs` are complete regardless.
