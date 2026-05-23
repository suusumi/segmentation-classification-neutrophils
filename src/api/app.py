"""FastAPI application for neutrophil analysis."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.api.schemas import AnalysisResponse, ErrorResponse, HealthResponse
from src.core.paths import PATHS, project_path
from src.core.settings import SETTINGS
from src.services.analysis_service import AnalysisService
from src.services.errors import AnalysisError, AnalysisNotFoundError
from src.storage.sqlite_repository import SQLiteAnalysisRepository


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    PATHS.ensure_runtime_dirs()
    repository = SQLiteAnalysisRepository(SETTINGS.analysis_db_path)
    service = AnalysisService(repository=repository)

    fastapi_app = FastAPI(
        title=SETTINGS.app_name,
        version=SETTINGS.api_version,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    fastapi_app.state.analysis_service = service
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.exception_handler(AnalysisError)
    async def handle_analysis_error(_, exc: AnalysisError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": str(exc)},
        )

    @fastapi_app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(service=SETTINGS.app_name, version=SETTINGS.api_version)

    @fastapi_app.post("/analysis", response_model=AnalysisResponse)
    async def analyze_image(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        return service.run_uploaded_image(
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
        )

    @fastapi_app.get("/analysis/{analysis_id}")
    async def get_analysis(analysis_id: str) -> dict:
        return service.get_analysis(analysis_id)

    @fastapi_app.get("/analysis/{analysis_id}/logs", response_class=PlainTextResponse)
    async def get_analysis_logs(analysis_id: str) -> str:
        record = service.get_analysis(analysis_id)
        result = record.get("result")
        if not result:
            raise AnalysisNotFoundError(f"Analysis result is not available: {analysis_id}")
        log_path = project_path(result["artifacts"]["log_file"])
        if not log_path.is_file():
            raise HTTPException(status_code=404, detail="Analysis log file not found.")
        return log_path.read_text(encoding="utf-8")

    @fastapi_app.get("/analysis/{analysis_id}/report", response_class=PlainTextResponse)
    async def get_analysis_report(analysis_id: str) -> str:
        record = service.get_analysis(analysis_id)
        result = record.get("result")
        if not result:
            raise AnalysisNotFoundError(f"Analysis result is not available: {analysis_id}")
        report_path = project_path(result["artifacts"]["report_markdown"])
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Analysis report file not found.")
        return report_path.read_text(encoding="utf-8")

    return fastapi_app


app = create_app()

