from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box, clamp_box


SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.30
TOP_K = 5000
SCALES = (1.0, 1.5, 2.0, 0.75, 0.5)
MAX_DETECTION_SIDE = 1800
TILE_OVERLAP = 0.20
ENHANCED_SCALES = (1.0, 1.5, 2.0)
ENHANCEMENT_MAX_SIDE = 1800
ENABLE_ROTATED_PASSES = True
LOW_LIGHT_MEAN = 105.0
LOW_CONTRAST_STDDEV = 42.0
SOFT_IMAGE_LAPLACIAN_VARIANCE = 115.0


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

        # Enhanced views recover evidence lost to darkness, flat contrast,
        # sensor noise, compression, and mild focus blur. The original passes
        # above remain authoritative for normal, well-exposed photographs.
        candidates.extend(self._detect_enhanced_views(image))

        # Overlapping tiles improve small/difficult faces without forcing the
        # entire image through YuNet at a huge resolution.
        candidates.extend(self._detect_tiles(image, rows=2, cols=2))
        if max(width, height) >= 1800:
            candidates.extend(self._detect_tiles(image, rows=3, cols=3))

        # Mixed-orientation group photos can contain both upright and sideways
        # faces, so rotated views must not depend on the upright result count.
        if ENABLE_ROTATED_PASSES:
            candidates.extend(self._detect_rotated(image))

        # Final full-image pass after tiles catches faces that cross tile borders.
        candidates.extend(self._detect_scaled(image, 1.0, 0, 0))
        clamped = [box for box in (clamp_box(b, width, height) for b in candidates) if box]
        return nms(clamped, NMS_THRESHOLD)

    def _detect_enhanced_views(self, image: np.ndarray) -> list[Box]:
        prepared, coordinate_scale = _bounded_copy(image, ENHANCEMENT_MAX_SIDE)
        variants = _enhancement_variants(prepared)
        boxes: list[Box] = []

        for variant in variants:
            for scale in ENHANCED_SCALES:
                detected = self._detect_scaled(variant, scale, 0, 0)
                boxes.extend(_rescale_boxes(detected, coordinate_scale))

            # One tiled enhanced pass gives small degraded faces more pixels
            # without repeating the full original-image tile pyramid.
            tiled = self._detect_tiles(variant, rows=2, cols=2)
            boxes.extend(_rescale_boxes(tiled, coordinate_scale))
        return boxes

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


def _bounded_copy(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 0.999:
        return image, 1.0
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def _enhancement_variants(image: np.ndarray) -> list[np.ndarray]:
    """Build a small adaptive set of detection-only image enhancements."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    brightness = float(mean[0, 0])
    contrast = float(stddev[0, 0])
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    variants: list[np.ndarray] = []

    # Local luminance contrast is useful across normal, backlit, and unevenly
    # illuminated scenes while preserving color information expected by YuNet.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clip_limit = 3.0 if contrast < LOW_CONTRAST_STDDEV else 2.0
    lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lightness)
    variants.append(cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR))

    if brightness < LOW_LIGHT_MEAN:
        # Gamma below one lifts shadow detail without flattening highlights.
        gamma = float(np.clip(0.48 + brightness / 500.0, 0.48, 0.68))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        lifted = cv2.LUT(image, lut)
        lifted_lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
        lifted_l, lifted_a, lifted_b = cv2.split(lifted_lab)
        lifted_l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lifted_l)
        variants.append(cv2.cvtColor(cv2.merge((lifted_l, lifted_a, lifted_b)), cv2.COLOR_LAB2BGR))

    if sharpness < SOFT_IMAGE_LAPLACIAN_VARIANCE or contrast < LOW_CONTRAST_STDDEV:
        # Gentle denoising before unsharp masking avoids magnifying block and
        # sensor noise in low-quality files.
        denoised = cv2.bilateralFilter(image, 7, 35, 35)
        smooth = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        restored = cv2.addWeighted(denoised, 1.65, smooth, -0.65, 0)
        variants.append(restored)

    return variants


def _rescale_boxes(boxes: list[Box], coordinate_scale: float) -> list[Box]:
    if coordinate_scale >= 0.999:
        return boxes
    inverse = 1.0 / coordinate_scale
    return [
        Box(
            int(round(box.x * inverse)),
            int(round(box.y * inverse)),
            int(round(box.w * inverse)),
            int(round(box.h * inverse)),
            box.score,
        )
        for box in boxes
    ]


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
