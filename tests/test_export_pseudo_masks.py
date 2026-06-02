"""Тесты утилит экспорта псевдомасок."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.export_pseudo_masks import (
    collect_image_paths,
    generate_pseudo_masks,
    save_manifest,
)


def _create_test_cell(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 96), color=(235, 220, 214))
    draw = ImageDraw.Draw(image)
    draw.ellipse((34, 36, 62, 62), fill=(45, 35, 115))
    image.save(path)


def test_collect_image_paths_respects_limit(tmp_path: Path) -> None:
    _create_test_cell(tmp_path / "cells" / "a.jpg")
    _create_test_cell(tmp_path / "cells" / "b.png")
    (tmp_path / "cells" / ".DS_fake.jpg").write_text("not an image", encoding="utf-8")

    image_paths = collect_image_paths(tmp_path / "cells")

    assert len(image_paths) == 2
    assert {path.name for path in image_paths} == {"a.jpg", "b.png"}


def test_generate_pseudo_masks_writes_annotation_assets(tmp_path: Path) -> None:
    image_path = tmp_path / "source" / "SNE_test.jpg"
    output_dir = tmp_path / "pseudo_labels"
    _create_test_cell(image_path)

    records = generate_pseudo_masks(
        image_paths=[image_path],
        output_dir=output_dir,
        split_map={image_path.resolve(): "train"},
        overwrite=True,
    )
    manifest_path = save_manifest(records, output_dir)

    mask_path = output_dir / "pseudo_masks" / "SNE_test.png"
    overlay_path = output_dir / "overlays" / "SNE_test_overlay.png"
    copied_image_path = output_dir / "images" / "SNE_test.jpg"
    mask = np.asarray(Image.open(mask_path))

    assert len(records) == 1
    assert records[0].split == "train"
    assert copied_image_path.is_file()
    assert mask_path.is_file()
    assert overlay_path.is_file()
    assert manifest_path.is_file()
    assert mask.max() == 255
