"""FastAPI application for neutrophil analysis."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

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

    @fastapi_app.post("/analysis/lobe-debug", response_model=AnalysisResponse)
    async def analyze_lobes_with_curated_mask(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        return service.run_uploaded_image_with_curated_nucleus_mask(
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
        )

    @fastapi_app.post("/analysis/yolo", response_model=AnalysisResponse)
    async def analyze_lobes_with_yolo(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        return service.run_uploaded_image_with_yolo_lobes(
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

    @fastapi_app.get("/artifacts/{artifact_path:path}")
    async def get_artifact(artifact_path: str) -> FileResponse:
        requested_path = Path(artifact_path)
        if requested_path.parts[:1] == ("outputs",):
            artifact_file = project_path(requested_path)
        else:
            artifact_file = PATHS.outputs / requested_path

        artifact_file = artifact_file.resolve()
        try:
            artifact_file.relative_to(PATHS.outputs.resolve())
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Artifact not found.") from error

        if not artifact_file.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(artifact_file)

    return fastapi_app


app = create_app()
