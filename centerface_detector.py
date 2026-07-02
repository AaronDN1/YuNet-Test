"""CenterFace detector (MIT) running through OpenCV's DNN module.

CenterFace is an anchor-free CenterNet-style face detector. Its weights are
MIT-licensed (Star-Clouds/CenterFace), so they are safe for commercial use and
can ship with this repository. The model runs via ``cv2.dnn`` (no onnxruntime
needed); OpenCV re-infers layer shapes, so the fixed-shape ONNX export still
accepts arbitrary input sizes that are multiples of 32.

As with the YuNet proposer, this detector returns candidates down to a low score
on purpose: precision comes later from cross-model agreement in the ensemble.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box
from boxops import nms

SCORE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.30
MAX_SIDE = 1600
TILE_TRIGGER_SIDE = 1100
TILE_OVERLAP = 0.20
STRIDE = 4
OUTPUT_NAMES = ("537", "538", "539", "540")  # heatmap, scale, offset, landmarks


class CenterFaceDetector:
    def __init__(
        self,
        model_path: Path,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        max_side: int = MAX_SIDE,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"CenterFace model not found: {model_path}")

        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_side = max_side

    def detect(self, image: np.ndarray) -> list[Box]:
        return self._detect_scaled(image, 0, 0)

    def detect_candidates(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        boxes = self._detect_scaled(image, 0, 0)

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
                boxes.extend(self._detect_scaled(tile, x1, y1))
        return boxes

    def _detect_scaled(self, image: np.ndarray, x_offset: int, y_offset: int) -> list[Box]:
        height, width = image.shape[:2]
        if width < 8 or height < 8:
            return []

        cap = min(1.0, self.max_side / max(width, height))
        net_w = max(32, int(np.ceil(width * cap / 32) * 32))
        net_h = max(32, int(np.ceil(height * cap / 32) * 32))

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(net_w, net_h), mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        heatmap, scale, offset, _ = self.net.forward(list(OUTPUT_NAMES))

        scale_w = net_w / width
        scale_h = net_h / height
        return self._decode(heatmap, scale, offset, net_w, net_h, scale_w, scale_h, x_offset, y_offset)

    def _decode(
        self,
        heatmap: np.ndarray,
        scale: np.ndarray,
        offset: np.ndarray,
        net_w: int,
        net_h: int,
        scale_w: float,
        scale_h: float,
        x_offset: int,
        y_offset: int,
    ) -> list[Box]:
        heat = heatmap[0, 0]
        scale_h_map, scale_w_map = scale[0, 0], scale[0, 1]
        offset_y_map, offset_x_map = offset[0, 0], offset[0, 1]

        rows, cols = np.where(heat > self.score_threshold)
        if rows.size == 0:
            return []

        scores = heat[rows, cols]
        box_h = np.exp(scale_h_map[rows, cols]) * STRIDE
        box_w = np.exp(scale_w_map[rows, cols]) * STRIDE
        off_y = offset_y_map[rows, cols]
        off_x = offset_x_map[rows, cols]

        x1 = (cols + off_x + 0.5) * STRIDE - box_w / 2
        y1 = (rows + off_y + 0.5) * STRIDE - box_h / 2
        x1 = np.clip(x1, 0, net_w)
        y1 = np.clip(y1, 0, net_h)

        result: list[Box] = []
        for bx, by, bw, bh, score in zip(x1, y1, box_w, box_h, scores):
            w = bw / scale_w
            h = bh / scale_h
            if w <= 1 or h <= 1:
                continue
            result.append(
                Box(int(round(bx / scale_w)) + x_offset, int(round(by / scale_h)) + y_offset, int(round(w)), int(round(h)), float(score))
            )
        return result
