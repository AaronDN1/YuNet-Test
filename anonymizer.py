from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MODE_FILL = "Solid average-color fill"
MODE_BLUR = "Strong blur"
MODE_PIXELATE = "Pixelation/mosaic"
ANONYMIZATION_MODES = (MODE_FILL, MODE_BLUR, MODE_PIXELATE)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0


def clamp_box(box: Box, width: int, height: int) -> Box | None:
    x1 = max(0, min(width, int(round(box.x))))
    y1 = max(0, min(height, int(round(box.y))))
    x2 = max(0, min(width, int(round(box.x + box.w))))
    y2 = max(0, min(height, int(round(box.y + box.h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2 - x1, y2 - y1, box.score)


def expand_face_box(box: Box, image_width: int, image_height: int) -> Box | None:
    """Expand detection to cover forehead, chin, ears, and detector edge misses."""
    width_growth = box.w * 0.25
    height_growth = box.h * 0.35
    new_w = box.w + width_growth
    new_h = box.h + height_growth
    x = box.x - width_growth / 2
    y = box.y - height_growth * 0.58
    return clamp_box(Box(int(x), int(y), int(new_w), int(new_h), box.score), image_width, image_height)


def anonymize_faces(image: np.ndarray, boxes: list[Box], mode: str) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    for box in boxes:
        expanded = expand_face_box(box, width, height)
        if expanded is None:
            continue
        x, y, w, h = expanded.x, expanded.y, expanded.w, expanded.h
        roi = output[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        if mode == MODE_BLUR:
            output[y : y + h, x : x + w] = _strong_blur(roi)
        elif mode == MODE_PIXELATE:
            output[y : y + h, x : x + w] = _pixelate(roi)
        else:
            average = roi.reshape(-1, 3).mean(axis=0)
            output[y : y + h, x : x + w] = average.astype(np.uint8)

    return output


def _strong_blur(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    kernel = max(31, (min(w, h) // 2) | 1)
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    # A second pass makes the blur deliberately privacy-first, not cosmetic.
    return cv2.GaussianBlur(blurred, (kernel, kernel), 0)


def _pixelate(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    pixel_size = max(12, min(w, h) // 8)
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
