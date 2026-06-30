from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box, clamp_box


SCORE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000
SCALES = (1.0, 1.5, 2.0, 0.75, 0.5)
MAX_DETECTION_SIDE = 1800
TILE_OVERLAP = 0.20
ENABLE_ROTATED_PASSES = False


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n\n"
                "Download face_detection_yunet_2023mar.onnx and place it in the models folder."
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        candidates: list[Box] = []

        # Full-frame multi-scale detection catches faces at very different sizes.
        for scale in SCALES:
            candidates.extend(self._detect_scaled(image, scale, 0, 0))

        # Overlapping tiles improve small/difficult faces without forcing the
        # entire image through YuNet at a huge resolution.
        candidates.extend(self._detect_tiles(image, rows=2, cols=2))
        if max(width, height) >= 1800:
            candidates.extend(self._detect_tiles(image, rows=3, cols=3))

        if ENABLE_ROTATED_PASSES:
            candidates.extend(self._detect_rotated(image))

        # Final full-image pass after tiles catches faces that cross tile borders.
        candidates.extend(self._detect_scaled(image, 1.0, 0, 0))
        clamped = [box for box in (clamp_box(b, width, height) for b in candidates) if box]
        return nms(clamped, NMS_THRESHOLD)

    def _detect_scaled(self, image: np.ndarray, requested_scale: float, x_offset: int, y_offset: int) -> list[Box]:
        height, width = image.shape[:2]
        scaled_w = max(1, int(round(width * requested_scale)))
        scaled_h = max(1, int(round(height * requested_scale)))
        limit_scale = min(1.0, MAX_DETECTION_SIDE / max(scaled_w, scaled_h))
        effective_scale = requested_scale * limit_scale

        if abs(effective_scale - 1.0) < 0.001:
            detect_image = image
        else:
            detect_size = (max(1, int(round(width * effective_scale))), max(1, int(round(height * effective_scale))))
            interpolation = cv2.INTER_CUBIC if effective_scale > 1.0 else cv2.INTER_AREA
            detect_image = cv2.resize(image, detect_size, interpolation=interpolation)

        dh, dw = detect_image.shape[:2]
        if dw < 20 or dh < 20:
            return []

        self.detector.setInputSize((dw, dh))
        _, faces = self.detector.detect(detect_image)
        if faces is None:
            return []

        boxes: list[Box] = []
        for face in faces:
            x, y, w, h = face[:4]
            score = float(face[-1])
            if w <= 1 or h <= 1:
                continue
            boxes.append(
                Box(
                    int(round(x / effective_scale)) + x_offset,
                    int(round(y / effective_scale)) + y_offset,
                    int(round(w / effective_scale)),
                    int(round(h / effective_scale)),
                    score,
                )
            )
        return boxes

    def _detect_tiles(self, image: np.ndarray, rows: int, cols: int) -> list[Box]:
        height, width = image.shape[:2]
        tile_w = width / cols
        tile_h = height / rows
        overlap_x = tile_w * TILE_OVERLAP
        overlap_y = tile_h * TILE_OVERLAP
        boxes: list[Box] = []

        for row in range(rows):
            for col in range(cols):
                x1 = int(max(0, col * tile_w - overlap_x))
                y1 = int(max(0, row * tile_h - overlap_y))
                x2 = int(min(width, (col + 1) * tile_w + overlap_x))
                y2 = int(min(height, (row + 1) * tile_h + overlap_y))
                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                boxes.extend(self._detect_scaled(tile, 1.0, x1, y1))
        return boxes

    def _detect_rotated(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        rotations = (
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_180, 180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
        )
        boxes: list[Box] = []
        for rotate_code, angle in rotations:
            rotated = cv2.rotate(image, rotate_code)
            for box in self._detect_scaled(rotated, 1.0, 0, 0):
                boxes.append(_box_from_rotated(box, width, height, angle))
        return boxes


def _box_from_rotated(box: Box, original_w: int, original_h: int, angle: int) -> Box:
    x1, y1 = box.x, box.y
    x2, y2 = box.x + box.w, box.y + box.h
    if angle == 90:
        points = [(y1, original_h - x2), (y2, original_h - x1)]
    elif angle == 180:
        points = [(original_w - x2, original_h - y2), (original_w - x1, original_h - y1)]
    else:
        points = [(original_w - y2, x1), (original_w - y1, x2)]
    nx1 = min(p[0] for p in points)
    ny1 = min(p[1] for p in points)
    nx2 = max(p[0] for p in points)
    ny2 = max(p[1] for p in points)
    return Box(nx1, ny1, nx2 - nx1, ny2 - ny1, box.score)


def nms(boxes: list[Box], threshold: float) -> list[Box]:
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: boxes[i].score, reverse=True)
    keep: list[Box] = []
    suppressed: set[int] = set()

    for i in order:
        if i in suppressed:
            continue
        current = boxes[i]
        keep.append(current)
        for j in order:
            if j == i or j in suppressed:
                continue
            if _iou(current, boxes[j]) > threshold:
                suppressed.add(j)
    return keep


def _iou(a: Box, b: Box) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0
