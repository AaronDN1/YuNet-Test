from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path).lower())


def load_image(path: Path) -> np.ndarray:
    """Load an image with OpenCV while supporting Windows Unicode paths.

    OpenCV applies EXIF orientation for common formats unless explicitly told to
    ignore it, so the detector sees the same orientation that users normally see.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode this image format")
    return image


def unique_output_path(output_root: Path, relative_path: Path) -> Path:
    candidate = output_root / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent
    index = 1
    while True:
        numbered = parent / f"{stem}_{index:03d}{suffix}"
        if not numbered.exists():
            return numbered
        index += 1


def save_image_strip_metadata(path: Path, image: np.ndarray) -> None:
    """Save pixels only. cv2.imencode/imwrite does not copy source EXIF metadata."""
    extension = path.suffix.lower()
    params: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif extension == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise ValueError(f"OpenCV could not encode output as {extension}")
    encoded.tofile(str(path))
