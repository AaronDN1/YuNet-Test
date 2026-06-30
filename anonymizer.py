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
            effect = _strong_blur(roi)
            result = _rounded_composite(roi, effect)
        elif mode == MODE_PIXELATE:
            result = _pixelate(roi)
        else:
            average = roi.reshape(-1, 3).mean(axis=0)
            result = np.full_like(roi, average.astype(np.uint8))

        output[y : y + h, x : x + w] = result

    return output


def _strong_blur(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    kernel = max(31, (min(w, h) // 2) | 1)
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    strong = cv2.GaussianBlur(blurred, (kernel, kernel), 0)

    # Blend toward the strongest blur at the center for a smoother visual finish.
    yy, xx = np.ogrid[-1.0:1.0:complex(h), -1.0:1.0:complex(w)]
    center_weight = np.clip(1.0 - np.sqrt(xx * xx + yy * yy), 0.0, 1.0)
    center_weight = (0.45 + 0.55 * center_weight)[..., None]
    return np.clip(blurred * (1.0 - center_weight) + strong * center_weight, 0, 255).astype(np.uint8)


def _rounded_composite(original: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Apply an effect through a rounded mask with a subtle graduated edge."""
    h, w = original.shape[:2]
    radius = min(max(4, int(min(w, h) * 0.10)), max(1, (min(w, h) - 1) // 2))
    feather = max(2, int(min(w, h) * 0.035))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (w - 1, h - radius - 1), 255, -1)
    for center in (
        (radius, radius),
        (w - radius - 1, radius),
        (radius, h - radius - 1),
        (w - radius - 1, h - radius - 1),
    ):
        cv2.circle(mask, center, radius, 255, -1)

    kernel = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.clip(effect * alpha + original * (1.0 - alpha), 0, 255).astype(np.uint8)


def _pixelate(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    pixel_size = max(12, min(w, h) // 8)
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
