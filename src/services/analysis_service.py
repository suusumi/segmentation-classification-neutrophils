"""Application service for upload validation and pipeline execution."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from src.core.paths import PATHS
from src.core.settings import SETTINGS
from src.pipeline.pipeline import NeutrophilAnalysisPipeline
from src.services.errors import PipelineError, UploadValidationError
from src.storage.sqlite_repository import SQLiteAnalysisRepository
from src.utils.logger import get_logger


class AnalysisService:
    """Coordinates upload storage, pipeline execution, and result persistence."""

    def __init__(
        self,
        repository: SQLiteAnalysisRepository,
        pipeline: NeutrophilAnalysisPipeline | None = None,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline or NeutrophilAnalysisPipeline()
        self.logger = get_logger("analysis_service", log_file=PATHS.logs / "api.log")

    def run_uploaded_image(
        self,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> dict:
        """Save an uploaded image, run analysis, and return a JSON result."""

        self._validate_upload_metadata(
            filename=filename,
            content_type=content_type,
            content=content,
        )
        safe_filename = self._safe_filename(filename)

        analysis_id = uuid4().hex
        analysis_dir = PATHS.analyses / analysis_id
        input_dir = analysis_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        extension = Path(safe_filename).suffix.lower()
        image_path = input_dir / f"original{extension}"
        image_path.write_bytes(content)
        self._validate_image_bytes(image_path)

        self.repository.create(
            analysis_id=analysis_id,
            input_filename=safe_filename,
            status="running",
        )
        self.logger.info("Created analysis %s for %s", analysis_id, safe_filename)

        try:
            result = self.pipeline.run(
                analysis_id=analysis_id,
                image_path=image_path,
                output_dir=analysis_dir,
                input_filename=safe_filename,
            )
        except Exception as error:
            self.repository.update_status(analysis_id=analysis_id, status="failed")
            self.logger.exception("Analysis %s failed", analysis_id)
            raise PipelineError(str(error)) from error

        result_payload = result.to_dict()
        self.repository.update_result(
            analysis_id=analysis_id,
            status="completed",
            result=result_payload,
        )
        return result_payload

    def get_analysis(self, analysis_id: str) -> dict:
        """Return persisted analysis metadata and result."""

        return self.repository.get(analysis_id)

    def _validate_upload_metadata(
        self,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> None:
        if not filename:
            raise UploadValidationError("Uploaded file must have a filename.")
        extension = Path(self._safe_filename(filename)).suffix.lower()
        if extension not in SETTINGS.allowed_image_extensions:
            raise UploadValidationError(f"Unsupported image extension: {extension}")
        if content_type and content_type not in SETTINGS.allowed_content_types:
            raise UploadValidationError(f"Unsupported content type: {content_type}")
        max_size = SETTINGS.max_upload_size_mb * 1024 * 1024
        if len(content) > max_size:
            raise UploadValidationError(
                f"Uploaded file is too large. Max size is {SETTINGS.max_upload_size_mb} MB."
            )
        if not content:
            raise UploadValidationError("Uploaded file is empty.")

    @staticmethod
    def _safe_filename(filename: str) -> str:
        """Normalize browser-supplied filenames from Windows or POSIX clients."""

        windows_name = PureWindowsPath(filename).name
        safe_name = Path(windows_name).name
        if not safe_name:
            raise UploadValidationError("Uploaded file must have a valid filename.")
        return safe_name

    @staticmethod
    def _validate_image_bytes(image_path: Path) -> None:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except UnidentifiedImageError as error:
            raise UploadValidationError("Uploaded file is not a readable image.") from error
