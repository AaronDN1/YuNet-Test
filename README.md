# Local Face Anonymizer

A simple local desktop program for privacy-focused face anonymization. It uses OpenCV YuNet through `cv2.FaceDetectorYN` to detect visible faces, then anonymizes expanded face regions before saving processed copies to an output folder.

This is not a face recognition app. It does not identify, compare, label, classify, store, or upload people or faces.

## Features

- Local Tkinter desktop UI.
- Multi-model detection ensemble: OpenCV YuNet (MIT) + CenterFace (MIT), with an
  optional YOLOX-face third voter (Apache-2.0, via onnxruntime). All weights are
  commercial-safe. YOLOX is **off by default** for speed.
- Cross-model agreement for precision: weak detections are kept only when an
  independent model corroborates them, which reduces false positives on
  low-quality/low-light images without lowering recall. More independent voters
  make agreement stronger.
- Automatic fallback to YuNet-only if the CenterFace model is missing.
- **Escalation ladder** for speed on clear photos and targeted recall on hard
  images: CenterFace-only fast exit when confident, YuNet corroboration only when
  needed, one enhanced pass for low-quality misses, and 2x2 tiling only as a last
  resort on large images.
- Adaptive CLAHE + shadow-lifting gamma for low-light images (applied to both detectors).
- IoU non-maximum suppression to merge duplicate detections.
- Expanded anonymization boxes to cover forehead, chin, ears, and face edges.
- `visualize_detections.py` tool to inspect and tune detection on your own images.
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

5. The CenterFace model (`models/centerface.onnx`, MIT-licensed) ships with this
repository, so the YuNet + CenterFace ensemble runs automatically. If that file
is missing, the app logs a note and continues with YuNet-only detection.

6. A YOLOX-face model (`models/yoloxs_face.onnx`, Apache-2.0) ships with the
repository but is **not used by default**. Set `ENABLE_YOLOX_VOTER = True` in
`ensemble_detector.py` to load it as an optional third voter (requires
`onnxruntime` in `requirements.txt`).

### Licensing note for commercial use

Only use detector weights whose license permits commercial use. In particular,
**InsightFace's SCRFD/RetinaFace pretrained weights (including `buffalo_*` /
`det_10g.onnx`) are for non-commercial research only** and must not be used or
redistributed in a commercial product. The InsightFace *code* is MIT, but the
*weights* are not. Commercially usable, permissively licensed alternatives
include OpenCV YuNet (MIT), CenterFace (MIT), YOLOX / YOLOX-face (Apache-2.0),
and MediaPipe BlazeFace (Apache-2.0). This note is not legal advice.

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

The ensemble uses a **CenterFace-first escalation ladder** in
`ensemble_detector.py`:

| Stage | When | What runs |
|-------|------|-----------|
| **0** | Always | CenterFace on capped frame; **fast exit** if any box ≥ `CENTERFACE_TRUST` |
| **1** | No stage-0 exit | YuNet (+ optional YOLOX) on base frame; fuse with CenterFace |
| **2** | Still zero faces | One enhanced view (low-light or CLAHE); CenterFace + YuNet |
| **3** | Still zero, large image | 2×2 tiles for CenterFace + YuNet only |

Clear photos typically finish at **stage 0** (~one CenterFace pass). Low-quality
quarantine misses get stage 2 enhancement; large difficult images may reach
stage 3 tiles.

YuNet pass settings are configurable in `yunet_detector.py`:

- `SCORE_THRESHOLD = 0.45`
- `NMS_THRESHOLD = 0.30`
- `TOP_K = 5000`
- `SCALES = (1.0, 1.5, 2.0, 0.75, 0.5)`
- `MAX_DETECTION_SIDE = 1800`
- `TILE_OVERLAP = 0.20`
- `ENHANCED_SCALES = (1.0, 1.5, 2.0)`
- `ENABLE_ROTATED_PASSES = True`

Detection runs the escalation ladder above instead of running every model on
every view. CenterFace settings are in `centerface_detector.py`
(`SCORE_THRESHOLD`, `NMS_THRESHOLD`, `INPUT_SIZE`); it always runs at a fixed
letterboxed input size for correctness across mixed-size batches. YOLOX settings
are in `yolox_detector.py` (`SCORE_THRESHOLD`).

The ensemble knobs live in `ensemble_detector.py` and are how you tune the
precision/recall balance and speed:

- `ENABLE_YOLOX_VOTER = False` — set `True` to load YOLOX as a third voter (slower).
- `MAX_DETECTION_SIDE = 1600` — full-frame passes run at this cap. Lower is faster but misses small faces.
- `TILE_TRIGGER_SIDE = 1600` — images at/above this may get a tiled pass in stage 3 only.
- `ESCALATION_CENTERFACE_SCORE = 0.28` — lower proposal threshold for CenterFace in stages 2–3 only.
- `CENTERFACE_TRUST = 0.45` / `YUNET_TRUST = 0.85` / `YOLOX_TRUST = 0.50` — accept a detection from one model alone above these scores.
- `CENTERFACE_MIN = 0.20` / `YUNET_MIN = 0.40` / `YOLOX_MIN = 0.30` — weaker detections need corroboration from another model.
- `AGREEMENT_IOU = 0.30` / `AGREEMENT_CONTAINMENT = 0.60` — overlap needed to count as agreement.

To catch more faces (higher recall), lower the trust thresholds and agreement minimums. To cut false positives (higher precision), raise the trust thresholds so more detections must be corroborated by another model.

### Tuning on your own images

Run the visualization tool to see exactly what each model proposes and what the
ensemble accepts, then adjust the knobs above:

```bash
python visualize_detections.py path/to/input_folder
```

It writes annotated copies (green = accepted/blurred, blue = CenterFace, red = YuNet, with scores) plus a `_summary.txt` showing `stage=0|1|2|3` per image.

To profile per-stage timings on a single image:

```bash
python profile_one_image.py path/to/image.jpg
```

Enhancements are detection-only; anonymization is still applied to the original-resolution image.
