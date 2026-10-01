FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KINOFORGE_SHARED_ROOT=/app/output

COPY . .
RUN pip install -e . && pip install "debugpy>=1.8"

RUN mkdir -p /app/output /app/.cache

EXPOSE 8100
# VS Code attach point (see .vscode/launch.json); only listens when KINOFORGE_DEBUG=1.
EXPOSE 5678

# --reload is harmless without a bind mount (nothing to watch) and picks up live edits when
# docker-compose.yml mounts ./apps/kinoforge over /app for local dev.
#
# KINOFORGE_DEBUG=1 swaps in debugpy instead, single worker, no --reload (the two don't mix
# reliably: reload restarts the process the debugger just attached to). KINOFORGE_DEBUG_WAIT=1
# additionally blocks startup until VS Code attaches, so you can catch import-time bugs too.
CMD ["sh", "-c", "\
    if [ \"$KINOFORGE_DEBUG\" = \"1\" ]; then \
        wait_flag=''; \
        [ \"$KINOFORGE_DEBUG_WAIT\" = \"1\" ] && wait_flag='--wait-for-client'; \
        exec python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 $wait_flag \
            -m uvicorn kinoforge.api.app:app --host 0.0.0.0 --port 8100; \
    else \
        exec uvicorn kinoforge.api.app:app --host 0.0.0.0 --port 8100 --reload; \
    fi"]
