"""Service metadata: health and segment discovery."""

from __future__ import annotations

from fastapi import APIRouter

from kinoforge.schemas import HealthResponse, SegmentsResponse
from kinoforge.service.discovery import SegmentCatalog

router = APIRouter()


@router.get("/health", tags=["Service"], summary="Service health", response_model=HealthResponse)
def health() -> dict:
    return {"ok": True, "service": "kinoforge", "version": "0.1.0"}


@router.get(
    "/v1/segments",
    tags=["Discovery"],
    summary="List generation segments",
    description=(
        "Returns stable metadata for every Kinoforge segment, including planned segments. "
        "Consumers must inspect status before using execute_path."
    ),
    response_model=SegmentsResponse,
)
def list_segments() -> SegmentsResponse:
    return SegmentCatalog.list()
