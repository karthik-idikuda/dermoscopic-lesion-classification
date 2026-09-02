"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeOptions(BaseModel):
    """Pipeline switches accepted by ``POST /api/analyze``.

    Sent as a JSON string in the ``options`` form field, because the request
    itself is ``multipart/form-data`` carrying the image.
    """

    hair_removal: bool = True
    color_constancy: bool = True
    vignette_crop: bool = True
    segmentation: bool = True
    morphometry: bool = True
    gradcam: bool = True
    gradcam_method: Literal["gradcam", "gradcam++"] = "gradcam++"
    colormap: Literal["jet", "turbo", "inferno", "magma"] = "jet"
    tta: bool = True
    mc_dropout: bool = True
    include_images: bool = True
    narrative: bool = True
    persist: bool = Field(
        default=True, description="Store the case in the local SQLite history."
    )


class CompareRequest(BaseModel):
    """Metadata for a two-image longitudinal comparison."""

    baseline_date: str | None = None
    followup_date: str | None = None
    frame_width_mm: float | None = Field(default=None, gt=0, le=200)
    include_images: bool = True

    @field_validator("baseline_date", "followup_date")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        return value or None


class NotesUpdate(BaseModel):
    """Free-text clinician annotation on a stored case."""

    notes: str | None = Field(default=None, max_length=5000)


class HealthResponse(BaseModel):
    # `model_loaded` collides with pydantic v2's reserved `model_` namespace,
    # which emits a UserWarning at import time. The field name is the clearer
    # one for an API consumer, so opt out of the protection instead.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    weights_status: str
    device: str
    warnings: list[str] = []


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None


class BatchItem(BaseModel):
    filename: str
    ok: bool
    error: str | None = None
    result: dict[str, Any] | None = None


__all__ = [
    "AnalyzeOptions",
    "BatchItem",
    "CompareRequest",
    "ErrorResponse",
    "HealthResponse",
    "NotesUpdate",
]
