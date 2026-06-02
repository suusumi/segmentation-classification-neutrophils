"""Доменные исключения, используемые API и пайплайном."""

from __future__ import annotations


class AnalysisError(Exception):
    """Базовый класс для ожидаемых ошибок анализа."""

    status_code = 500
    error_code = "analysis_error"


class UploadValidationError(AnalysisError):
    """Возникает, когда загруженный файл нельзя принять."""

    status_code = 400
    error_code = "invalid_upload"


class PipelineError(AnalysisError):
    """Возникает при сбое пайплайна анализа."""

    status_code = 500
    error_code = "pipeline_error"


class AnalysisNotFoundError(AnalysisError):
    """Возникает, когда идентификатор анализа неизвестен."""

    status_code = 404
    error_code = "analysis_not_found"

