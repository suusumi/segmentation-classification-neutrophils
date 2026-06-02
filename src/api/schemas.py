"""Схемы API на Pydantic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Ответ эндпоинта проверки состояния."""
    status: str = "ok"
    service: str
    version: str


class ErrorResponse(BaseModel):
    """Единообразная полезная нагрузка ошибки."""
    error: str
    message: str


class AnalysisResponse(BaseModel):
    """Ответ эндпоинта анализа."""
    analysis_id: str
    status: str
    input_filename: str
    image_width_px: int
    image_height_px: int
    features: dict[str, Any] = Field(default_factory=dict)
    classification: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
