"""Tests for the FastAPI analysis endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src.api.app import create_app


def _image_bytes(path: Path) -> bytes:
    image = Image.new("RGB", (96, 96), color=(236, 220, 218))
    draw = ImageDraw.Draw(image)
    draw.ellipse((32, 34, 64, 66), fill=(50, 40, 110))
    image.save(path)
    return path.read_bytes()


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analysis_endpoint_accepts_image(tmp_path: Path) -> None:
    image_path = tmp_path / "cell.png"
    content = _image_bytes(image_path)
    client = TestClient(create_app())

    response = client.post(
        "/analysis",
        files={"file": ("cell.png", content, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["classification"]["label"] in {"normal", "hypersegmentation", "unknown"}
    assert payload["artifacts"]["mask_image"].endswith("nucleus_mask.png")
    assert payload["metadata"]["segmenter_name"] == "threshold"
