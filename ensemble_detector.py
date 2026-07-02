"""Two-stage face detection ensemble.

Stage 1 (recall): a loose YuNet proposer and an independent SCRFD detector each
produce candidate faces, including on contrast/low-light enhanced views.

Stage 2 (precision): candidates are accepted only when they are either strong on
their own or corroborated by the *other* model. Because the two detectors have
different architectures and training, their agreement is genuinely independent
evidence, which breaks the precision/recall tradeoff that a single model (run
many times over correlated enhancements) cannot escape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anonymizer import Box, clamp_box
from boxops import containment, iou, nms
from scrfd_detector import ScrfdFaceDetector
from yunet_detector import (
    ENHANCEMENT_MAX_SIDE,
    NMS_THRESHOLD,
    YuNetFaceDetector,
    bounded_copy,
    enhancement_variants,
)

# Accept a single model's detection on its own only when it is confident.
SCRFD_TRUST_THRESHOLD = 0.50
YUNET_TRUST_THRESHOLD = 0.88

# Weaker detections are accepted only with independent corroboration.
SCRFD_MIN_FOR_AGREEMENT = 0.30
YUNET_MIN_FOR_AGREEMENT = 0.50
AGREEMENT_IOU = 0.30
AGREEMENT_CONTAINMENT = 0.60


class EnsembleFaceDetector:
    def __init__(self, yunet_model_path: Path, scrfd_model_path: Path) -> None:
        self.yunet = YuNetFaceDetector(yunet_model_path)
        self.scrfd = ScrfdFaceDetector(scrfd_model_path)

    def detect(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]

        scrfd_boxes = self._scrfd_candidates(image)
        yunet_boxes = self.yunet.detect_candidates(image)

        accepted = _fuse(scrfd_boxes, yunet_boxes)
        clamped = [box for box in (clamp_box(b, width, height) for b in accepted) if box]
        return nms(clamped, NMS_THRESHOLD)

    def _scrfd_candidates(self, image: np.ndarray) -> list[Box]:
        boxes = self.scrfd.detect_candidates(image)

        # Run SCRFD on the same contrast/low-light enhancements YuNet uses, so
        # faces lost to darkness or flat contrast get a strong second look.
        prepared, coordinate_scale = bounded_copy(image, ENHANCEMENT_MAX_SIDE)
        for variant in enhancement_variants(prepared):
            detected = self.scrfd.detect(variant)
            boxes.extend(_rescale_boxes(detected, coordinate_scale))

        return nms(boxes, self.scrfd.nms_threshold)


def _fuse(scrfd_boxes: list[Box], yunet_boxes: list[Box]) -> list[Box]:
    accepted: list[Box] = []

    for box in scrfd_boxes:
        if box.score >= SCRFD_TRUST_THRESHOLD:
            accepted.append(box)

    for box in yunet_boxes:
        if box.score >= YUNET_TRUST_THRESHOLD:
            accepted.append(box)

    for scrfd_box in scrfd_boxes:
        if not (SCRFD_MIN_FOR_AGREEMENT <= scrfd_box.score < SCRFD_TRUST_THRESHOLD):
            continue
        if _corroborated(scrfd_box, yunet_boxes):
            accepted.append(scrfd_box)

    return accepted


def _corroborated(scrfd_box: Box, yunet_boxes: list[Box]) -> bool:
    for yunet_box in yunet_boxes:
        if yunet_box.score < YUNET_MIN_FOR_AGREEMENT:
            continue
        if iou(scrfd_box, yunet_box) >= AGREEMENT_IOU or containment(scrfd_box, yunet_box) >= AGREEMENT_CONTAINMENT:
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
