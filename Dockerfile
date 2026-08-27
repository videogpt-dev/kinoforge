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
RUN pip install -e .

RUN mkdir -p /app/output /app/.cache

EXPOSE 8100

CMD ["uvicorn", "kinoforge.service.app:app", "--host", "0.0.0.0", "--port", "8100"]
