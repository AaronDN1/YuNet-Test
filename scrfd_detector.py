"""Self-contained SCRFD face detector running on onnxruntime (CPU).

This wrapper implements the standard InsightFace SCRFD post-processing so it
works with the common SCRFD ONNX exports (with or without keypoints) without
depending on the ``insightface`` package, whose hosted model weights carry a
non-commercial license. Supply your own commercially-licensed SCRFD ``.onnx``.

The detector deliberately returns candidates down to a low score. Precision is
applied later by the ensemble via agreement with an independent model, so this
stage optimizes for recall on hard, low-light, and low-quality faces.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box
from boxops import nms

SCORE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.40
DEFAULT_INPUT_SIZE = (640, 640)
TILE_TRIGGER_SIDE = 1100
TILE_OVERLAP = 0.20


class ScrfdFaceDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        try:
            import onnxruntime  # noqa: WPS433 (optional dependency, imported lazily)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "onnxruntime is required for SCRFD detection. Install it with "
                "'pip install onnxruntime'."
            ) from exc

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.input_size = _resolve_input_size(self.session, input_size)
        self._configure_outputs(len(self.output_names))
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _configure_outputs(self, output_count: int) -> None:
        """Match the SCRFD output layout to feature strides and anchor count."""
        if output_count == 6:
            self.fmc, self.feat_strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, False
        elif output_count == 9:
            self.fmc, self.feat_strides, self.num_anchors, self.use_kps = 3, [8, 16, 32], 2, True
        elif output_count == 10:
            self.fmc, self.feat_strides, self.num_anchors, self.use_kps = 5, [8, 16, 32, 64, 128], 1, False
        elif output_count == 15:
            self.fmc, self.feat_strides, self.num_anchors, self.use_kps = 5, [8, 16, 32, 64, 128], 1, True
        else:
            raise ValueError(
                f"Unsupported SCRFD model: expected 6, 9, 10, or 15 outputs, got {output_count}."
            )

    def detect(self, image: np.ndarray) -> list[Box]:
        """Single full-frame letterboxed detection pass."""
        return self._detect_padded(image, 0, 0)

    def detect_candidates(self, image: np.ndarray) -> list[Box]:
        """Full-frame pass plus tiled passes on large images for small faces."""
        height, width = image.shape[:2]
        boxes = self._detect_padded(image, 0, 0)

        if max(width, height) >= TILE_TRIGGER_SIDE:
            boxes.extend(self._detect_tiles(image, rows=2, cols=2))

        return nms(boxes, self.nms_threshold)

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
                boxes.extend(self._detect_padded(tile, x1, y1))
        return boxes

    def _detect_padded(self, image: np.ndarray, x_offset: int, y_offset: int) -> list[Box]:
        height, width = image.shape[:2]
        if width < 8 or height < 8:
            return []

        input_w, input_h = self.input_size
        image_ratio = height / width
        model_ratio = input_h / input_w
        if image_ratio > model_ratio:
            new_h = input_h
            new_w = int(round(new_h / image_ratio))
        else:
            new_w = input_w
            new_h = int(round(new_w * image_ratio))
        new_w = max(1, min(new_w, input_w))
        new_h = max(1, min(new_h, input_h))

        det_scale = new_h / height
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (input_w, input_h), (127.5, 127.5, 127.5), swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self._decode(outputs, input_h, input_w, det_scale, x_offset, y_offset)

    def _decode(
        self,
        outputs: list[np.ndarray],
        input_h: int,
        input_w: int,
        det_scale: float,
        x_offset: int,
        y_offset: int,
    ) -> list[Box]:
        scores_all: list[np.ndarray] = []
        boxes_all: list[np.ndarray] = []

        for index, stride in enumerate(self.feat_strides):
            scores = _squeeze_batch(outputs[index]).reshape(-1)
            bbox_preds = _squeeze_batch(outputs[index + self.fmc]).reshape(-1, 4) * stride

            grid_h = input_h // stride
            grid_w = input_w // stride
            centers = self._anchor_centers(grid_h, grid_w, stride)

            keep = np.where(scores >= self.score_threshold)[0]
            if keep.size == 0:
                continue

            boxes = _distance_to_box(centers[keep], bbox_preds[keep])
            scores_all.append(scores[keep])
            boxes_all.append(boxes)

        if not boxes_all:
            return []

        scores = np.concatenate(scores_all)
        boxes = np.concatenate(boxes_all) / det_scale

        result: list[Box] = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            result.append(
                Box(int(round(x1)) + x_offset, int(round(y1)) + y_offset, int(round(w)), int(round(h)), float(score))
            )
        return result

    def _anchor_centers(self, grid_h: int, grid_w: int, stride: int) -> np.ndarray:
        key = (grid_h, grid_w, stride)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached

        centers = np.stack(np.mgrid[:grid_h, :grid_w][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)
        if self.num_anchors > 1:
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
        self._center_cache[key] = centers
        return centers


def _resolve_input_size(session, fallback: tuple[int, int]) -> tuple[int, int]:
    """Use the model's fixed input size when present, otherwise the fallback."""
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int) and shape[2] > 0 and shape[3] > 0:
        return (shape[3], shape[2])
    return fallback


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array[0]
    return array


def _distance_to_box(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)
