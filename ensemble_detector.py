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
from image_quality import HardImageSignals, assess_image_quality, build_hard_image_variants
from yunet_detector import NMS_THRESHOLD, YuNetFaceDetector, bounded_copy

# One capped resolution for all passes keeps cost predictable and bounded.
MAX_DETECTION_SIDE = 1600
TILE_TRIGGER_SIDE = 1600

# YOLOX is fixed-640 and heavy; tiling it is accurate but slow. Off by default
# in the normal fast path, but recovery can still use it.
YOLOX_TILE_ENABLED = False

RECOVERY_ENABLED = True
RECOVERY_UPSCALE = 2.0
RECOVERY_CENTERFACE_SCORE = 0.25
RECOVERY_YOLOX_SCORE = 0.22
RECOVERY_AGREEMENT_IOU = 0.35

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

    def detect_debug(self, image: np.ndarray) -> dict[str, object]:
        """Run detection and also return each model's raw boxes.

        The extra fields let the visualization tool show what each model proposed
        versus what the ensemble accepted, which is how you diagnose misses
        (recall) and false blurs (precision) on real images.
        """
        height, width = image.shape[:2]

        # Cap resolution once; run every pass in this space, map back at the end.
        work, scale = bounded_copy(image, MAX_DETECTION_SIDE)
        large = max(work.shape[:2]) >= TILE_TRIGGER_SIDE
        signals = assess_image_quality(work)

        raw_second_boxes, raw_yunet_boxes, raw_yolox_boxes = self._collect_primary_proposals(work, signals, large)
        groups = self._build_groups(raw_second_boxes, raw_yunet_boxes, raw_yolox_boxes, scale)
        second_boxes = groups[0].boxes
        yunet_boxes = groups[1].boxes
        yolox_boxes = groups[2].boxes if len(groups) > 2 else []

        accepted = _fuse(groups)
        recovery_used = False
        if RECOVERY_ENABLED and signals.is_hard and not accepted:
            recovery_used = True
            second_boxes, yunet_boxes, yolox_boxes, accepted = self._recover_zero_face(
                work,
                scale,
                existing_second=raw_second_boxes,
                existing_yunet=raw_yunet_boxes,
                existing_yolox=raw_yolox_boxes,
            )

        clamped = [box for box in (clamp_box(b, width, height) for b in accepted) if box]
        faces = nms(clamped, NMS_THRESHOLD)
        return {
            "faces": faces,
            "centerface": second_boxes,
            "yunet": yunet_boxes,
            "yolox": yolox_boxes,
            "is_hard": signals.is_hard,
            "recovery_used": recovery_used,
            "quality": signals,
        }

    def _collect_primary_proposals(
        self, work: np.ndarray, signals: HardImageSignals, large: bool
    ) -> tuple[list[Box], list[Box], list[Box]]:
        second_boxes: list[Box] = []
        yunet_boxes: list[Box] = []
        yolox_boxes: list[Box] = []

        self._append_view_boxes(second_boxes, yunet_boxes, yolox_boxes, work)
        if signals.is_dark or signals.is_low_contrast:
            # Preserve the old lean behavior for images that are simply dim/flat.
            # The more aggressive denoise/upscale path is reserved for zero-face
            # recovery so already-detected images keep their current accuracy.
            self._append_view_boxes(second_boxes, yunet_boxes, yolox_boxes, _enhance_low_light(work))

        # Large images still get one tiled pass in the normal fast path.
        if large:
            second_boxes.extend(self.second.detect_tiles(work, rows=2, cols=2))
            yunet_boxes.extend(self.yunet.detect_tiles(work, rows=2, cols=2))
            if self.yolox is not None and YOLOX_TILE_ENABLED:
                yolox_boxes.extend(self.yolox.detect_tiles(work, rows=2, cols=2))

        return second_boxes, yunet_boxes, yolox_boxes

    def _recover_zero_face(
        self,
        work: np.ndarray,
        scale: float,
        existing_second: list[Box],
        existing_yunet: list[Box],
        existing_yolox: list[Box],
    ) -> tuple[list[Box], list[Box], list[Box], list[Box]]:
        second_boxes = list(existing_second)
        yunet_boxes = list(existing_yunet)
        yolox_boxes = list(existing_yolox)

        for view in build_hard_image_variants(work):
            self._append_view_boxes(
                second_boxes,
                yunet_boxes,
                yolox_boxes,
                view,
                centerface_threshold=RECOVERY_CENTERFACE_SCORE,
                yolox_threshold=RECOVERY_YOLOX_SCORE,
            )

        upscaled = _resized(work, RECOVERY_UPSCALE)
        self._append_view_boxes(
            second_boxes,
            yunet_boxes,
            yolox_boxes,
            upscaled,
            coordinate_scale=RECOVERY_UPSCALE,
            centerface_threshold=RECOVERY_CENTERFACE_SCORE,
            yolox_threshold=RECOVERY_YOLOX_SCORE,
        )

        second_boxes.extend(
            self.second.detect_tiles(work, rows=2, cols=2, score_threshold=RECOVERY_CENTERFACE_SCORE)
        )
        yunet_boxes.extend(self.yunet.detect_tiles(work, rows=2, cols=2))
        if self.yolox is not None:
            yolox_boxes.extend(
                self.yolox.detect_tiles(work, rows=2, cols=2, score_threshold=RECOVERY_YOLOX_SCORE)
            )

        groups = self._build_groups(second_boxes, yunet_boxes, yolox_boxes, scale)
        accepted = _fuse(
            groups,
            allow_trust=False,
            agreement_iou=RECOVERY_AGREEMENT_IOU,
        )
        return (
            _dedupe_group(second_boxes, self.second.nms_threshold, scale),
            _dedupe_group(yunet_boxes, NMS_THRESHOLD, scale),
            _dedupe_group(yolox_boxes, NMS_THRESHOLD, scale),
            accepted,
        )

    def _append_view_boxes(
        self,
        second_boxes: list[Box],
        yunet_boxes: list[Box],
        yolox_boxes: list[Box],
        view: np.ndarray,
        coordinate_scale: float = 1.0,
        centerface_threshold: float | None = None,
        yolox_threshold: float | None = None,
    ) -> None:
        second = self.second.detect(view, score_threshold=centerface_threshold)
        yunet = self.yunet.detect_simple(view)
        yolox = self.yolox.detect(view, score_threshold=yolox_threshold) if self.yolox is not None else []

        if coordinate_scale > 1.001:
            second = _rescale_boxes(second, coordinate_scale)
            yunet = _rescale_boxes(yunet, coordinate_scale)
            yolox = _rescale_boxes(yolox, coordinate_scale)

        second_boxes.extend(second)
        yunet_boxes.extend(yunet)
        yolox_boxes.extend(yolox)

    def _build_groups(
        self, second_boxes: list[Box], yunet_boxes: list[Box], yolox_boxes: list[Box], scale: float
    ) -> list[_ModelGroup]:
        groups = [
            _ModelGroup(
                "centerface",
                _dedupe_group(second_boxes, self.second.nms_threshold, scale),
                CENTERFACE_TRUST,
                CENTERFACE_MIN,
            ),
            _ModelGroup("yunet", _dedupe_group(yunet_boxes, NMS_THRESHOLD, scale), YUNET_TRUST, YUNET_MIN),
        ]
        if self.yolox is not None:
            groups.append(_ModelGroup("yolox", _dedupe_group(yolox_boxes, NMS_THRESHOLD, scale), YOLOX_TRUST, YOLOX_MIN))
        return groups


def _fuse(
    groups: list[_ModelGroup],
    allow_trust: bool = True,
    agreement_iou: float = AGREEMENT_IOU,
    agreement_containment: float = AGREEMENT_CONTAINMENT,
) -> list[Box]:
    accepted: list[Box] = []
    for index, group in enumerate(groups):
        others = [g for j, g in enumerate(groups) if j != index]
        for box in group.boxes:
            if allow_trust and box.score >= group.trust:
                accepted.append(box)
            elif box.score >= group.min_agree and _corroborated_by_any(
                box, others, agreement_iou, agreement_containment
            ):
                accepted.append(box)
    return accepted


def _corroborated_by_any(
    box: Box, others: list[_ModelGroup], agreement_iou: float, agreement_containment: float
) -> bool:
    return any(
        _corroborated(box, group.boxes, group.min_agree, agreement_iou, agreement_containment)
        for group in others
    )


def _corroborated(
    box: Box,
    others: list[Box],
    min_score: float,
    agreement_iou: float,
    agreement_containment: float,
) -> bool:
    for other in others:
        if other.score < min_score:
            continue
        if iou(box, other) >= agreement_iou or containment(box, other) >= agreement_containment:
            return True
    return False


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


def _enhance_low_light(image: np.ndarray) -> np.ndarray:
    """Fast lean-path enhancement for dim or flat images."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    brightness = float(lightness.mean())
    lightness = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    if brightness < 105.0:
        gamma = float(np.clip(0.5 + brightness / 500.0, 0.5, 0.75))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)

    return enhanced


def _resized(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


def _dedupe_group(boxes: list[Box], nms_threshold: float, work_scale: float) -> list[Box]:
    return nms(_rescale_boxes(boxes, work_scale), nms_threshold)
