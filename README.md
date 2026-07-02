# Local Face Anonymizer

A simple local desktop program for privacy-focused face anonymization. It uses OpenCV YuNet through `cv2.FaceDetectorYN` to detect visible faces, then anonymizes expanded face regions before saving processed copies to an output folder.

This is not a face recognition app. It does not identify, compare, label, classify, store, or upload people or faces.

## Features

- Local Tkinter desktop UI.
- Two-model detection ensemble: OpenCV YuNet plus SCRFD (ONNX via onnxruntime).
- Cross-model agreement for precision: weak detections are kept only when an
  independent model corroborates them, which reduces false positives on
  low-quality/low-light images without lowering recall.
- Automatic fallback to YuNet-only if no SCRFD model or onnxruntime is present.
- Multi-scale detection passes.
- Adaptive CLAHE, low-light gamma correction, denoising, and sharpening detection passes (applied to both detectors).
- Overlapping 2x2 and 3x3 tiled detection for difficult images.
- IoU non-maximum suppression to merge duplicate detections.
- Expanded anonymization boxes to cover forehead, chin, ears, and face edges.
- Three anonymization modes:
  - Solid average-color fill, the default and strongest privacy option.
  - Strong blur.
  - Pixelation/mosaic.
- Recursive folder processing with relative folder structure preserved.
- Unique output filenames when a destination already exists.
- EXIF/metadata stripped from saved images.
- Log file written to the output folder.
- Optional input-folder deletion only after all valid images are processed successfully and the user confirms.

## Setup

1. Install Python 3.10 or newer.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `models` folder beside `main.py`, or let the app create it the first time you press Start.
4. Download the YuNet ONNX model named:

```text
face_detection_yunet_2023mar.onnx
```

Place it here:

```text
models/face_detection_yunet_2023mar.onnx
```

The model is published by OpenCV Zoo. If your downloaded file is named `face_detection_yunet_2023.mar.onnx`, that filename is accepted too. If the file is missing, the app will show a clear startup error when processing begins.

5. (Recommended) Add a SCRFD ONNX model to enable the detection ensemble. Place any of these filenames in the `models` folder (the app also matches any `scrfd*.onnx`):

```text
models/scrfd_10g_bnkps.onnx        (most accurate, slower)
models/scrfd_2.5g_bnkps.onnx       (balanced)
models/scrfd_500m_bnkps.onnx       (fastest)
```

If a SCRFD model is present, the app runs the YuNet + SCRFD ensemble automatically. If it is absent (or onnxruntime is not installed), the app logs a note and continues with YuNet-only detection.

### Licensing note for commercial use

The SCRFD *architecture/code* (InsightFace) is MIT-licensed, but the model
**weights** hosted by InsightFace are published for non-commercial research use.
For commercial deployment, ensure the specific `.onnx` weights file you ship is
covered by a license that permits commercial use. This applies to the actual
weights file, not just the source repository. This note is not legal advice.

## Run

```bash
python main.py
```

Then:

1. Select an input folder.
2. Select a different output folder.
3. Choose the anonymization mode.
4. Start processing.

Supported input extensions are `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, and `webp` when your OpenCV build supports them.

## Privacy Notes

- Images with no detected faces after all detection passes are isolated under `Quarantine`, rather than being mixed with anonymized results.

- The app runs locally.
- It makes no network requests during image processing.
- It never places a zero-detection image in the normal anonymized output tree; those images are isolated under `Quarantine` as a fail-safe.
- Output files are newly encoded with OpenCV, which strips source EXIF/metadata.
- The optional deletion step happens only when every valid image file was processed and written successfully, and only after a final confirmation dialog.
- If `send2trash` happens to be installed, deletion moves the input folder to the recycle bin. Otherwise Python permanently deletes the folder with `shutil.rmtree`.
- The app does not claim secure deletion. SSDs, backups, sync tools, and modern filesystems may retain recoverable data.

## Detection Defaults

YuNet pass settings are configurable in `yunet_detector.py`:

- `SCORE_THRESHOLD = 0.45`
- `NMS_THRESHOLD = 0.30`
- `TOP_K = 5000`
- `SCALES = (1.0, 1.5, 2.0, 0.75, 0.5)`
- `MAX_DETECTION_SIDE = 1800`
- `TILE_OVERLAP = 0.20`
- `ENHANCED_SCALES = (1.0, 1.5, 2.0)`
- `ENABLE_ROTATED_PASSES = True`

SCRFD settings are in `scrfd_detector.py` (`SCORE_THRESHOLD`, `NMS_THRESHOLD`, `DEFAULT_INPUT_SIZE`, `TILE_TRIGGER_SIDE`).

The ensemble fusion knobs live in `ensemble_detector.py` and are how you tune the precision/recall balance:

- `SCRFD_TRUST_THRESHOLD = 0.50` — accept SCRFD alone above this score.
- `YUNET_TRUST_THRESHOLD = 0.88` — accept YuNet alone above this score.
- `SCRFD_MIN_FOR_AGREEMENT = 0.30` / `YUNET_MIN_FOR_AGREEMENT = 0.50` — minimum scores that can qualify for cross-model agreement.
- `AGREEMENT_IOU = 0.30` / `AGREEMENT_CONTAINMENT = 0.60` — overlap needed to count as agreement.

To catch more faces (higher recall), lower `SCRFD_TRUST_THRESHOLD` and the agreement minimums. To cut false positives (higher precision), raise the trust thresholds so more detections must be corroborated by both models.

Enhancements are detection-only; anonymization is still applied to the original-resolution image. Normal-image passes always run, while extra low-light and restoration variants are selected from measured brightness, contrast, and sharpness.
