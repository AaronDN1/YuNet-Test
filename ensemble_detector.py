"""Lean multi-model face detection ensemble with a CenterFace-first escalation ladder.

Stage 0: CenterFace base pass only. Confident faces return immediately.
Stage 1: Add YuNet (and optional YOLOX) on the base frame; fuse.
Stage 2: One enhanced view (low-light or CLAHE); CenterFace + YuNet (+ YOLOX).
Stage 3: 2x2 tiles on large images when earlier stages found nothing.

Normal clear photos pay ~one CenterFace pass. Low-quality quarantine cases get
targeted escalation without running every model on every view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box, clamp_box
from boxops import containment, iou, nms
from centerface_detector import CenterFaceDetector
from yunet_detector import NMS_THRESHOLD, YuNetFaceDetector, bounded_copy

MAX_DETECTION_SIDE = 1600
TILE_TRIGGER_SIDE = 1600

# YOLOX is heavy; off by default. Set True to load yoloxs_face.onnx as a voter.
ENABLE_YOLOX_VOTER = False

LOW_LIGHT_MEAN = 110.0
LOW_CONTRAST_STDDEV = 48.0

# Lower proposal threshold for CenterFace during escalation only (fusion unchanged).
ESCALATION_CENTERFACE_SCORE = 0.28

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
        if yolox_model_path is not None and ENABLE_YOLOX_VOTER:
            from yolox_detector import YoloxFaceDetector

            self.yolox = YoloxFaceDetector(yolox_model_path)

    def detect(self, image: np.ndarray) -> list[Box]:
        return self.detect_debug(image)["faces"]

    def detect_debug(self, image: np.ndarray) -> dict[str, object]:
        height, width = image.shape[:2]
        work, scale = bounded_copy(image, MAX_DETECTION_SIDE)
        large = max(work.shape[:2]) >= TILE_TRIGGER_SIDE

        second_boxes: list[Box] = []
        yunet_boxes: list[Box] = []
        yolox_boxes: list[Box] = []
        escalation_stage = 0
        stage_timings: dict[str, float] = {}

        # Stage 0: CenterFace only — fast exit when clearly confident.
        t0 = time.perf_counter()
        second_boxes.extend(self.second.detect(work))
        stage_timings["stage0_centerface"] = (time.perf_counter() - t0) * 1000.0
        confident = [b for b in second_boxes if b.score >= CENTERFACE_TRUST]
        if confident:
            faces = self._output(confident, width, height, scale)
            return self._debug_result(
                faces,
                second_boxes,
                yunet_boxes,
                yolox_boxes,
                scale,
                escalation_stage,
                stage_timings,
            )

        # Stage 1: corroborate with YuNet (+ optional YOLOX) on the base frame.
        escalation_stage = 1
        t0 = time.perf_counter()
        yunet_boxes.extend(self.yunet.detect_simple(work))
        if self.yolox is not None:
            yolox_boxes.extend(self.yolox.detect(work))
        stage_timings["stage1_yunet_yolox"] = (time.perf_counter() - t0) * 1000.0

        accepted = self._fuse_groups(second_boxes, yunet_boxes, yolox_boxes, scale)
        if accepted:
            faces = self._output(accepted, width, height, scale)
            return self._debug_result(
                faces,
                second_boxes,
                yunet_boxes,
                yolox_boxes,
                scale,
                escalation_stage,
                stage_timings,
            )

        # Stage 2: one enhanced view for low-quality / noisy misses.
        escalation_stage = 2
        t0 = time.perf_counter()
        enhanced = (
            _enhance_low_light(work) if _needs_low_light(work) else _enhance_clahe(work)
        )
        second_boxes.extend(
            self.second.detect(enhanced, score_threshold=ESCALATION_CENTERFACE_SCORE)
        )
        yunet_boxes.extend(self.yunet.detect_simple(enhanced))
        if self.yolox is not None:
            yolox_boxes.extend(self.yolox.detect(enhanced))
        stage_timings["stage2_enhanced"] = (time.perf_counter() - t0) * 1000.0

        accepted = self._fuse_groups(second_boxes, yunet_boxes, yolox_boxes, scale)
        if accepted:
            faces = self._output(accepted, width, height, scale)
            return self._debug_result(
                faces,
                second_boxes,
                yunet_boxes,
                yolox_boxes,
                scale,
                escalation_stage,
                stage_timings,
            )

        # Stage 3: tiled last resort on large images only.
        if large:
            escalation_stage = 3
            t0 = time.perf_counter()
            second_boxes.extend(
                self.second.detect_tiles(
                    work, rows=2, cols=2, score_threshold=ESCALATION_CENTERFACE_SCORE
                )
            )
            yunet_boxes.extend(self.yunet.detect_tiles(work, rows=2, cols=2))
            if self.yolox is not None:
                yolox_boxes.extend(self.yolox.detect_tiles(work, rows=2, cols=2))
            stage_timings["stage3_tiles"] = (time.perf_counter() - t0) * 1000.0

            accepted = self._fuse_groups(second_boxes, yunet_boxes, yolox_boxes, scale)
            if accepted:
                faces = self._output(accepted, width, height, scale)
                return self._debug_result(
                    faces,
                    second_boxes,
                    yunet_boxes,
                    yolox_boxes,
                    scale,
                    escalation_stage,
                    stage_timings,
                )

        faces = self._output([], width, height, scale)
        return self._debug_result(
            faces, second_boxes, yunet_boxes, yolox_boxes, scale, escalation_stage, stage_timings
        )

    def _fuse_groups(
        self,
        second_boxes: list[Box],
        yunet_boxes: list[Box],
        yolox_boxes: list[Box],
        scale: float,
    ) -> list[Box]:
        second = nms(_rescale_boxes(second_boxes, scale), self.second.nms_threshold)
        yunet = nms(_rescale_boxes(yunet_boxes, scale), NMS_THRESHOLD)
        groups = [
            _ModelGroup("centerface", second, CENTERFACE_TRUST, CENTERFACE_MIN),
            _ModelGroup("yunet", yunet, YUNET_TRUST, YUNET_MIN),
        ]
        if self.yolox is not None:
            yolox = nms(_rescale_boxes(yolox_boxes, scale), NMS_THRESHOLD)
            groups.append(_ModelGroup("yolox", yolox, YOLOX_TRUST, YOLOX_MIN))
        return _fuse(groups)

    def _output(
        self, accepted: list[Box], width: int, height: int, work_scale: float
    ) -> list[Box]:
        if work_scale < 0.999:
            accepted = _rescale_boxes(accepted, work_scale)
        clamped = [box for box in (clamp_box(b, width, height) for b in accepted) if box]
        return nms(clamped, NMS_THRESHOLD)

    def _debug_result(
        self,
        faces: list[Box],
        second_boxes: list[Box],
        yunet_boxes: list[Box],
        yolox_boxes: list[Box],
        scale: float,
        escalation_stage: int,
        stage_timings: dict[str, float] | None = None,
    ) -> dict[str, object]:
        return {
            "faces": faces,
            "centerface": nms(_rescale_boxes(second_boxes, scale), self.second.nms_threshold),
            "yunet": nms(_rescale_boxes(yunet_boxes, scale), NMS_THRESHOLD),
            "yolox": nms(_rescale_boxes(yolox_boxes, scale), NMS_THRESHOLD) if yolox_boxes else [],
            "escalation_stage": escalation_stage,
            "stage_timings_ms": stage_timings or {},
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


def _enhance_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lightness)
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
