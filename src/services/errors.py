"""Domain exceptions used by the API and pipeline."""

from __future__ import annotations


class AnalysisError(Exception):
    """Base class for expected analysis failures."""

    status_code = 500
    error_code = "analysis_error"


class UploadValidationError(AnalysisError):
    """Raised when an uploaded file cannot be accepted."""

    status_code = 400
    error_code = "invalid_upload"


class PipelineError(AnalysisError):
    """Raised when the analysis pipeline fails."""

    status_code = 500
    error_code = "pipeline_error"


class AnalysisNotFoundError(AnalysisError):
    """Raised when an analysis id is unknown."""

    status_code = 404
    error_code = "analysis_not_found"

