# Local Face Anonymizer

A simple local desktop program for privacy-focused face anonymization. It uses OpenCV YuNet through `cv2.FaceDetectorYN` to detect visible faces, then anonymizes expanded face regions before saving processed copies to an output folder.

This is not a face recognition app. It does not identify, compare, label, classify, store, or upload people or faces.

## Features

- Local Tkinter desktop UI.
- OpenCV YuNet primary face detector.
- Multi-scale detection passes.
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

- Images with no detected faces are stored under `Quarantine` in the output folder for manual review, rather than being mixed with anonymized results.

- The app runs locally.
- It makes no network requests during image processing.
- It never places a zero-detection image in the normal anonymized output tree; those images are isolated under `Quarantine` for manual review.
- Output files are newly encoded with OpenCV, which strips source EXIF/metadata.
- The optional deletion step happens only when every valid image file was processed and written successfully, and only after a final confirmation dialog.
- If `send2trash` happens to be installed, deletion moves the input folder to the recycle bin. Otherwise Python permanently deletes the folder with `shutil.rmtree`.
- The app does not claim secure deletion. SSDs, backups, sync tools, and modern filesystems may retain recoverable data.

## Detection Defaults

Detection settings are configurable in `yunet_detector.py`:

- `SCORE_THRESHOLD = 0.60`
- `NMS_THRESHOLD = 0.30`
- `TOP_K = 5000`
- `SCALES = (1.0, 1.5, 2.0, 0.75, 0.5)`
- `MAX_DETECTION_SIDE = 1800`
- `TILE_OVERLAP = 0.20`
- `ENABLE_ROTATED_PASSES = False`

Rotated passes are disabled by default for speed. Enable them in code if your image set commonly contains sideways or upside-down images.
