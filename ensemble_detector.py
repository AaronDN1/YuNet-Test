"""Lean multi-model face detection ensemble.

Design goals: fast and accurate on batch/CPU.

Stage 1 (recall): each detector (YuNet, CenterFace, and optionally YOLOX-face)
runs a small, fixed number of passes -- a base pass, one quality-triggered
enhanced pass only when the image is dark/flat/soft, and one tiled pass only on
large images for small faces. Everything runs at a single capped resolution,
computed once.

Stage 2 (precision): cross-model agreement. A detection is kept if it is
confident on its own OR corroborated by another architecturally independent
model. Widened solo bands for YOLOX and YuNet are gated by face-like geometry
and (for medium YOLOX) soft-image / multi-view consistency so noisy clinical
misses can be recovered without a blanket threshold cut.

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
SOFT_IMAGE_LAPLACIAN_VARIANCE = 115.0

# Per-model thresholds: accept alone above `trust`; below it (but above `min`)
# only with corroboration from another model.
CENTERFACE_TRUST, CENTERFACE_MIN = 0.45, 0.20
# YuNet: >= high_trust unconditional; [trust, high_trust) needs face-like geometry.
YUNET_TRUST, YUNET_HIGH_TRUST, YUNET_MIN = 0.65, 0.85, 0.40
# YOLOX: >= trust solo; [medium_min, trust) needs soft/multi-view + geometry.
YOLOX_TRUST, YOLOX_MEDIUM_MIN, YOLOX_MIN = 0.45, 0.35, 0.30

AGREEMENT_IOU = 0.30
AGREEMENT_CONTAINMENT = 0.60

# Geometry gate for widened solo paths (not applied to high-trust / agreement).
FACE_ASPECT_MIN = 0.55
FACE_ASPECT_MAX = 1.85
FACE_MIN_SIDE_FRAC = 0.02
FACE_MAX_AREA_FRAC = 0.45


@dataclass
class _ModelGroup:
    name: str
    boxes: list[Box]
    trust: float
    min_agree: float
    high_trust: float | None = None
    medium_min: float | None = None


@dataclass(frozen=True)
class _ImageQuality:
    needs_low_light: bool
    is_soft: bool

    @property
    def degraded(self) -> bool:
        """True when an extra enhancement pass is justified or medium YOLOX may fire."""
        return self.needs_low_light or self.is_soft


@dataclass
class _FuseContext:
    image_width: int
    image_height: int
    degraded: bool
    yolox_base: list[Box]
    yolox_enhanced: list[Box]


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
        quality = _probe_quality(work)

        views = [work]
        if quality.needs_low_light:
            views.append(_enhance_low_light(work))
        elif quality.is_soft:
            views.append(_enhance_soft(work))

        second_boxes: list[Box] = []
        yunet_boxes: list[Box] = []
        yolox_base: list[Box] = []
        yolox_enhanced: list[Box] = []
        for view_index, view in enumerate(views):
            second_boxes.extend(self.second.detect(view))
            yunet_boxes.extend(self.yunet.detect_simple(view))
            if self.yolox is not None:
                detected = self.yolox.detect(view)
                if view_index == 0:
                    yolox_base.extend(detected)
                else:
                    yolox_enhanced.extend(detected)

        # Large images: one tiled pass recovers small faces missed when the whole
        # frame is downscaled. Tiled YOLOX stays on the base view only.
        if large:
            second_boxes.extend(self.second.detect_tiles(work, rows=2, cols=2))
            yunet_boxes.extend(self.yunet.detect_tiles(work, rows=2, cols=2))
            if self.yolox is not None and YOLOX_TILE_ENABLED:
                yolox_base.extend(self.yolox.detect_tiles(work, rows=2, cols=2))

        second_boxes = nms(_rescale_boxes(second_boxes, scale), self.second.nms_threshold)
        yunet_boxes = nms(_rescale_boxes(yunet_boxes, scale), NMS_THRESHOLD)

        yolox_boxes: list[Box] = []
        yolox_base_rescaled: list[Box] = []
        yolox_enhanced_rescaled: list[Box] = []
        if self.yolox is not None:
            yolox_base_rescaled = _rescale_boxes(yolox_base, scale)
            yolox_enhanced_rescaled = _rescale_boxes(yolox_enhanced, scale)
            yolox_boxes = nms(yolox_base_rescaled + yolox_enhanced_rescaled, NMS_THRESHOLD)

        groups = [
            _ModelGroup("centerface", second_boxes, CENTERFACE_TRUST, CENTERFACE_MIN),
            _ModelGroup(
                "yunet",
                yunet_boxes,
                YUNET_TRUST,
                YUNET_MIN,
                high_trust=YUNET_HIGH_TRUST,
            ),
        ]
        if self.yolox is not None:
            groups.append(
                _ModelGroup(
                    "yolox",
                    yolox_boxes,
                    YOLOX_TRUST,
                    YOLOX_MIN,
                    medium_min=YOLOX_MEDIUM_MIN,
                )
            )

        context = _FuseContext(
            image_width=width,
            image_height=height,
            degraded=quality.degraded,
            yolox_base=yolox_base_rescaled,
            yolox_enhanced=yolox_enhanced_rescaled,
        )
        accepted = _fuse(groups, context)
        clamped = [box for box in (clamp_box(b, width, height) for b in accepted) if box]
        faces = nms(clamped, NMS_THRESHOLD)
        return {
            "faces": faces,
            "centerface": second_boxes,
            "yunet": yunet_boxes,
            "yolox": yolox_boxes,
        }


def _fuse(groups: list[_ModelGroup], context: _FuseContext) -> list[Box]:
    accepted: list[Box] = []
    for index, group in enumerate(groups):
        others = [g for j, g in enumerate(groups) if j != index]
        for box in group.boxes:
            if _accept_box(box, group, others, context):
                accepted.append(box)
    return accepted


def _accept_box(
    box: Box,
    group: _ModelGroup,
    others: list[_ModelGroup],
    context: _FuseContext,
) -> bool:
    high_trust = group.high_trust if group.high_trust is not None else group.trust

    # Unconditional high-confidence solo (YuNet >= 0.85, or models without a mid band).
    if box.score >= high_trust:
        return True

    # Mid-band solo: YuNet [0.65, 0.85) needs face-like geometry.
    # If geometry fails, still allow cross-model agreement below.
    if (
        group.high_trust is not None
        and box.score >= group.trust
        and _looks_like_face(box, context.image_width, context.image_height)
    ):
        return True

    # Cross-model agreement (unchanged).
    if box.score >= group.min_agree and _corroborated_by_any(box, others):
        return True

    # Medium YOLOX solo: [0.35, 0.45) on soft/noisy images or multi-view consistent.
    if (
        group.medium_min is not None
        and box.score >= group.medium_min
        and box.score < group.trust
        and _looks_like_face(box, context.image_width, context.image_height)
        and (context.degraded or _yolox_multiview_consistent(box, context))
    ):
        return True

    return False


def _yolox_multiview_consistent(box: Box, context: _FuseContext) -> bool:
    """True when the same region appears in both base and enhanced YOLOX passes."""
    if not context.yolox_base or not context.yolox_enhanced:
        return False
    # Box is from the fused YOLOX list; require overlap with the *other* view.
    in_base = _corroborated(box, context.yolox_base, YOLOX_MEDIUM_MIN)
    in_enhanced = _corroborated(box, context.yolox_enhanced, YOLOX_MEDIUM_MIN)
    return in_base and in_enhanced


def _corroborated_by_any(box: Box, others: list[_ModelGroup]) -> bool:
    return any(_corroborated(box, group.boxes, group.min_agree) for group in others)


def _corroborated(box: Box, others: list[Box], min_score: float) -> bool:
    for other in others:
        if other.score < min_score:
            continue
        if iou(box, other) >= AGREEMENT_IOU or containment(box, other) >= AGREEMENT_CONTAINMENT:
            return True
    return False


def _looks_like_face(box: Box, image_width: int, image_height: int) -> bool:
    """Reject obvious non-face blobs on widened solo paths only."""
    if box.w <= 0 or box.h <= 0 or image_width <= 0 or image_height <= 0:
        return False

    aspect = box.w / float(box.h)
    if aspect < FACE_ASPECT_MIN or aspect > FACE_ASPECT_MAX:
        return False

    shorter = min(image_width, image_height)
    if min(box.w, box.h) < shorter * FACE_MIN_SIDE_FRAC:
        return False

    if (box.w * box.h) > image_width * image_height * FACE_MAX_AREA_FRAC:
        return False

    return True


def _probe_quality(image: np.ndarray) -> _ImageQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    brightness = float(mean[0, 0])
    contrast = float(stddev[0, 0])
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    needs_low_light = brightness < LOW_LIGHT_MEAN or contrast < LOW_CONTRAST_STDDEV
    is_soft = sharpness < SOFT_IMAGE_LAPLACIAN_VARIANCE
    return _ImageQuality(needs_low_light=needs_low_light, is_soft=is_soft)


def _needs_low_light(image: np.ndarray) -> bool:
    return _probe_quality(image).needs_low_light


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


def _enhance_soft(image: np.ndarray) -> np.ndarray:
    """Cheap soft/noisy pass: mild Gaussian then CLAHE (no bilateral)."""
    smoothed = cv2.GaussianBlur(image, (0, 0), 1.0)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


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
