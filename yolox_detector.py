"""YOLOX-face detector (ONNX Runtime).

Third independent voter in the ensemble. The model is an end-to-end YOLOX-S
export trained on WIDER FACE with NMS baked into the graph, so a forward pass
returns already-deduplicated boxes.

License: YOLOX is Apache-2.0 (code and Megvii weights); this WIDER-FACE export
is distributed under Apache-2.0 as well. Safe for commercial use.

Preprocessing is standard YOLOX: letterbox to 640x640 padded with 114, BGR,
raw 0-255 values (no /255 normalization), channels-first.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anonymizer import Box

INPUT_SIZE = 640
PAD_VALUE = 114
SCORE_THRESHOLD = 0.35


class YoloxFaceDetector:
    def __init__(self, model_path: Path, score_threshold: float = SCORE_THRESHOLD) -> None:
        import onnxruntime as ort

        self.score_threshold = score_threshold
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> list[Box]:
        blob, ratio = self._preprocess(image)
        num_dets, boxes, scores, _classes = self.session.run(None, {self.input_name: blob})

        count = int(num_dets[0, 0])
        if count <= 0:
            return []

        results: list[Box] = []
        img_h, img_w = image.shape[:2]
        for index in range(count):
            score = float(scores[0, index])
            if score < self.score_threshold:
                continue
            x1, y1, x2, y2 = boxes[0, index]
            x1, y1, x2, y2 = x1 / ratio, y1 / ratio, x2 / ratio, y2 / ratio
            x = int(round(max(0.0, x1)))
            y = int(round(max(0.0, y1)))
            w = int(round(min(x2, img_w) - x1))
            h = int(round(min(y2, img_h) - y1))
            if w > 0 and h > 0:
                results.append(Box(x, y, w, h, score))
        return results

    def detect_tiles(self, image: np.ndarray, rows: int, cols: int) -> list[Box]:
        height, width = image.shape[:2]
        tile_h, tile_w = height // rows, width // cols
        overlap_y, overlap_x = tile_h // 5, tile_w // 5

        results: list[Box] = []
        for row in range(rows):
            for col in range(cols):
                y0 = max(0, row * tile_h - overlap_y)
                x0 = max(0, col * tile_w - overlap_x)
                y1 = min(height, (row + 1) * tile_h + overlap_y)
                x1 = min(width, (col + 1) * tile_w + overlap_x)
                tile = image[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                for box in self.detect(tile):
                    results.append(Box(box.x + x0, box.y + y0, box.w, box.h, box.score))
        return results

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
        new_h, new_w = int(round(height * ratio)), int(round(width * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        return blob, ratio
