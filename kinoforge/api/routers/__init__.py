"""FastAPI routers, one module per segment plus service metadata. app.py mounts them all."""

from kinoforge.api.routers import clips, meta, series, story

ALL = (meta.router, clips.router, story.router, series.router)

__all__ = ["ALL", "clips", "meta", "series", "story"]
