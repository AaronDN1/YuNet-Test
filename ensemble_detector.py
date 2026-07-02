"""Two-stage face detection ensemble.

Stage 1 (recall): a loose YuNet proposer and an independent CenterFace detector
each produce candidate faces, including on contrast/low-light enhanced views.

Stage 2 (precision): candidates are accepted only when they are either strong on
their own or corroborated by the *other* model. Because the two detectors have
different architectures and training (YuNet is anchor-based; CenterFace is an
anchor-free CenterNet), their agreement is genuinely independent evidence, which
breaks the precision/recall tradeoff that a single model (run many times over
correlated enhancements) cannot escape.

Both models are MIT-licensed and safe for commercial use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anonymizer import Box, clamp_box
from boxops import containment, iou, nms
from centerface_detector import CenterFaceDetector
from yunet_detector import (
    ENHANCEMENT_MAX_SIDE,
    NMS_THRESHOLD,
    YuNetFaceDetector,
    bounded_copy,
    enhancement_variants,
)

# Accept a single model's detection on its own only when it is confident.
SECOND_TRUST_THRESHOLD = 0.45
YUNET_TRUST_THRESHOLD = 0.88

# Weaker detections are accepted only with independent corroboration.
SECOND_MIN_FOR_AGREEMENT = 0.25
YUNET_MIN_FOR_AGREEMENT = 0.50
AGREEMENT_IOU = 0.30
AGREEMENT_CONTAINMENT = 0.60


class EnsembleFaceDetector:
    def __init__(self, yunet_model_path: Path, second_model_path: Path) -> None:
        self.yunet = YuNetFaceDetector(yunet_model_path)
        self.second = CenterFaceDetector(second_model_path)

    def detect(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]

        second_boxes = self._second_candidates(image)
        yunet_boxes = self.yunet.detect_candidates(image)

        accepted = _fuse(second_boxes, yunet_boxes)
        clamped = [box for box in (clamp_box(b, width, height) for b in accepted) if box]
        return nms(clamped, NMS_THRESHOLD)

    def _second_candidates(self, image: np.ndarray) -> list[Box]:
        boxes = self.second.detect_candidates(image)

        # Run the second detector on the same contrast/low-light enhancements
        # YuNet uses, so faces lost to darkness or flat contrast get a strong
        # second look from an independent model.
        prepared, coordinate_scale = bounded_copy(image, ENHANCEMENT_MAX_SIDE)
        for variant in enhancement_variants(prepared):
            detected = self.second.detect(variant)
            boxes.extend(_rescale_boxes(detected, coordinate_scale))

        return nms(boxes, self.second.nms_threshold)


def _fuse(second_boxes: list[Box], yunet_boxes: list[Box]) -> list[Box]:
    accepted: list[Box] = []

    for box in second_boxes:
        if box.score >= SECOND_TRUST_THRESHOLD:
            accepted.append(box)

    for box in yunet_boxes:
        if box.score >= YUNET_TRUST_THRESHOLD:
            accepted.append(box)

    for second_box in second_boxes:
        if not (SECOND_MIN_FOR_AGREEMENT <= second_box.score < SECOND_TRUST_THRESHOLD):
            continue
        if _corroborated(second_box, yunet_boxes):
            accepted.append(second_box)

    return accepted


def _corroborated(second_box: Box, yunet_boxes: list[Box]) -> bool:
    for yunet_box in yunet_boxes:
        if yunet_box.score < YUNET_MIN_FOR_AGREEMENT:
            continue
        if iou(second_box, yunet_box) >= AGREEMENT_IOU or containment(second_box, yunet_box) >= AGREEMENT_CONTAINMENT:
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
