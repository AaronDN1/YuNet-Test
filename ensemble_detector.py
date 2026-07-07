"""Lean multi-model face detection ensemble.

Design goals: fast and accurate on batch/CPU.

Stage 1 (recall): each detector (YuNet, CenterFace, and optionally YOLOX-face)
runs a small, fixed number of passes -- a base pass, one low-light-enhanced pass
only when the image is dark/flat, and one tiled pass only on large images for
small faces. Everything runs at a single capped resolution, computed once.

Stage 2 (precision): cross-model agreement. A detection is kept if it is
confident on its own OR corroborated by another architecturally independent
model. A medium-confidence face seen by only one model is still recoverable
through agreement rather than silently dropped, which matters for a privacy tool
where a missed face is a leak. More independent voters make agreement stronger.

All bundled models are MIT/Apache-2.0 and safe for commercial use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box, clamp_box
from boxops import containment, iou, nms
from centerface_detector import CenterFaceDetector
from yunet_detector import NMS_THRESHOLD, YuNetFaceDetector, bounded_copy

# One capped resolution for all passes keeps cost predictable and bounded.
MAX_DETECTION_SIDE = 1600
TILE_TRIGGER_SIDE = 1600

# YOLOX is fixed-640 and heavy; tiling it is accurate but slow. Off by default.
YOLOX_TILE_ENABLED = False

# Only spend a second detection pass on enhancement when the image needs it.
LOW_LIGHT_MEAN = 110.0
LOW_CONTRAST_STDDEV = 48.0

# Per-model thresholds: accept alone above `trust`; below it (but above `min`)
# only with corroboration from another model.
CENTERFACE_TRUST, CENTERFACE_MIN = 0.45, 0.20
YUNET_TRUST, YUNET_MIN = 0.85, 0.40
YOLOX_TRUST, YOLOX_MIN = 0.50, 0.30

AGREEMENT_IOU = 0.30
AGREEMENT_CONTAINMENT = 0.60


@dataclass
class _ModelGroup:
    name: str
    boxes: list[Box]
    trust: float
    min_agree: float


class EnsembleFaceDetector:
    def __init__(
        self,
        yunet_model_path: Path,
        second_model_path: Path,
        yolox_model_path: Path | None = None,
    ) -> None:
        self.yunet = YuNetFaceDetector(yunet_model_path)
        self.second = CenterFaceDetector(second_model_path, max_side=MAX_DETECTION_SIDE)
        self.yolox = None
        if yolox_model_path is not None:
            from yolox_detector import YoloxFaceDetector

            self.yolox = YoloxFaceDetector(yolox_model_path)

    def detect(self, image: np.ndarray) -> list[Box]:
        return self.detect_debug(image)["faces"]

    def detect_debug(self, image: np.ndarray) -> dict[str, list[Box]]:
        """Run detection and also return each model's raw boxes.

        The extra fields let the visualization tool show what each model proposed
        versus what the ensemble accepted, which is how you diagnose misses
        (recall) and false blurs (precision) on real images.
        """
        height, width = image.shape[:2]

        # Cap resolution once; run every pass in this space, map back at the end.
        work, scale = bounded_copy(image, MAX_DETECTION_SIDE)
        large = max(work.shape[:2]) >= TILE_TRIGGER_SIDE

        views = [work]
        if _needs_low_light(work):
            views.append(_enhance_low_light(work))

        second_boxes: list[Box] = []
        yunet_boxes: list[Box] = []
        yolox_boxes: list[Box] = []
        for view in views:
            second_boxes.extend(self.second.detect(view))
            yunet_boxes.extend(self.yunet.detect_simple(view))
            if self.yolox is not None:
                yolox_boxes.extend(self.yolox.detect(view))

        # Large images: one tiled pass recovers small faces missed when the whole
        # frame is downscaled.
        if large:
            second_boxes.extend(self.second.detect_tiles(work, rows=2, cols=2))
            yunet_boxes.extend(self.yunet.detect_tiles(work, rows=2, cols=2))
            if self.yolox is not None and YOLOX_TILE_ENABLED:
                yolox_boxes.extend(self.yolox.detect_tiles(work, rows=2, cols=2))

        second_boxes = nms(_rescale_boxes(second_boxes, scale), self.second.nms_threshold)
        yunet_boxes = nms(_rescale_boxes(yunet_boxes, scale), NMS_THRESHOLD)

        groups = [
            _ModelGroup("centerface", second_boxes, CENTERFACE_TRUST, CENTERFACE_MIN),
            _ModelGroup("yunet", yunet_boxes, YUNET_TRUST, YUNET_MIN),
        ]
        if self.yolox is not None:
            yolox_boxes = nms(_rescale_boxes(yolox_boxes, scale), NMS_THRESHOLD)
            groups.append(_ModelGroup("yolox", yolox_boxes, YOLOX_TRUST, YOLOX_MIN))

        accepted = _fuse(groups)
        clamped = [box for box in (clamp_box(b, width, height) for b in accepted) if box]
        faces = nms(clamped, NMS_THRESHOLD)
        return {
            "faces": faces,
            "centerface": second_boxes,
            "yunet": yunet_boxes,
            "yolox": yolox_boxes,
        }


def _fuse(groups: list[_ModelGroup]) -> list[Box]:
    accepted: list[Box] = []
    for index, group in enumerate(groups):
        others = [g for j, g in enumerate(groups) if j != index]
        for box in group.boxes:
            if box.score >= group.trust:
                accepted.append(box)
            elif box.score >= group.min_agree and _corroborated_by_any(box, others):
                accepted.append(box)
    return accepted


def _corroborated_by_any(box: Box, others: list[_ModelGroup]) -> bool:
    return any(_corroborated(box, group.boxes, group.min_agree) for group in others)


def _corroborated(box: Box, others: list[Box], min_score: float) -> bool:
    for other in others:
        if other.score < min_score:
            continue
        if iou(box, other) >= AGREEMENT_IOU or containment(box, other) >= AGREEMENT_CONTAINMENT:
            return True
    return False


def _needs_low_light(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    return float(mean[0, 0]) < LOW_LIGHT_MEAN or float(stddev[0, 0]) < LOW_CONTRAST_STDDEV


def _enhance_low_light(image: np.ndarray) -> np.ndarray:
    """Fast detection-only enhancement: CLAHE on luminance plus a shadow-lifting
    gamma when the frame is dark. No bilateral filtering (too slow, and it
    magnifies compression noise into false positives)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    brightness = float(lightness.mean())
    lightness = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    if brightness < LOW_LIGHT_MEAN:
        gamma = float(np.clip(0.5 + brightness / 500.0, 0.5, 0.75))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)

    return enhanced


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
