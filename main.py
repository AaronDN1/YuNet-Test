from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

from anonymizer import ANONYMIZATION_MODES, MODE_FILL, anonymize_faces
from deletion import delete_input_folder
from image_io import iter_image_files, load_image, save_image_strip_metadata, unique_output_path
from processing_log import ProcessingLog
from yunet_detector import YuNetFaceDetector


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODEL_FILENAMES = (
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2023.mar.onnx",
)


@dataclass(frozen=True)
class JobConfig:
    input_dir: Path
    output_dir: Path
    model_path: Path
    recursive: bool
    delete_after_success: bool
    mode: str


class FaceAnonymizerApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Local Face Anonymizer")
        self.root.geometry("760x520")
        self.root.minsize(680, 460)

        self.input_dir = StringVar()
        self.output_dir = StringVar()
        self.mode = StringVar(value=MODE_FILL)
        self.recursive = BooleanVar(value=True)
        self.delete_after_success = BooleanVar(value=False)
        self.status_text = StringVar(value="Select folders to begin.")

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.total_files = 0
        self.processed = 0
        self.failed = 0
        self.quarantined = 0

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)

        frame = ttk.Frame(self.root, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="Select input folder", command=self._select_input).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(frame, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=10)

        ttk.Button(frame, text="Select output folder", command=self._select_output).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(frame, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=10)

        options = ttk.LabelFrame(frame, text="Options", padding=12)
        options.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 8))
        options.columnconfigure(1, weight=1)

        ttk.Label(options, text="Anonymization mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(options, values=ANONYMIZATION_MODES, textvariable=self.mode, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=(12, 0)
        )
        ttk.Checkbutton(options, text="Process subfolders recursively", variable=self.recursive).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Checkbutton(
            options,
            text="Delete original input folder after successful processing",
            variable=self.delete_after_success,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.start_button = ttk.Button(frame, text="Start", command=self._start)
        self.start_button.grid(row=3, column=0, sticky="w", pady=(10, 8))

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=3, column=1, sticky="ew", padx=10, pady=(10, 8))

        ttk.Label(frame, textvariable=self.status_text).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        self.log_box = None
        import tkinter as tk

        self.log_box = tk.Text(frame, height=14, wrap="word")
        self.log_box.grid(row=5, column=0, columnspan=2, sticky="nsew")
        frame.rowconfigure(5, weight=1)

    def _select_input(self) -> None:
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_dir.set(folder)

    def _select_output(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir.set(folder)

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Folders required", "Select both an input folder and an output folder.")
            return

        input_dir = Path(self.input_dir.get()).resolve()
        output_dir = Path(self.output_dir.get()).resolve()
        model_path = _find_yunet_model()

        if input_dir == output_dir:
            messagebox.showerror("Unsafe folder choice", "Input and output folders must be different.")
            return
        if _is_relative_to(output_dir, input_dir):
            messagebox.showerror("Unsafe folder choice", "Output folder cannot be inside the input folder.")
            return
        if model_path is None:
            expected_model_path = MODELS_DIR / MODEL_FILENAMES[0]
            messagebox.showerror(
                "YuNet model missing",
                f"Missing YuNet model file.\n\nExpected location:\n{expected_model_path}\n\n"
                "Create the models folder beside main.py and place the YuNet ONNX model there.\n"
                "Accepted filenames:\n"
                "- face_detection_yunet_2023mar.onnx\n"
                "- face_detection_yunet_2023.mar.onnx",
            )
            return

        config = JobConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=model_path,
            recursive=self.recursive.get(),
            delete_after_success=self.delete_after_success.get(),
            mode=self.mode.get(),
        )

        self.start_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=1)
        self.status_text.set("Starting...")
        self._clear_log()
        self.worker = threading.Thread(target=self._run_job, args=(config,), daemon=True)
        self.worker.start()

    def _run_job(self, config: JobConfig) -> None:
        try:
            files = iter_image_files(config.input_dir, config.recursive)
            self.events.put(("total", len(files)))
            log = ProcessingLog(config.output_dir, config.input_dir)
            detector = YuNetFaceDetector(config.model_path)

            for index, source in enumerate(files, start=1):
                self.events.put(("file", f"[{index}/{len(files)}] {source}"))
                try:
                    relative = source.relative_to(config.input_dir)
                    image = load_image(source)
                    boxes = detector.detect(image)
                    quarantined = len(boxes) == 0
                    destination_root = config.output_dir / "Quarantine" if quarantined else config.output_dir
                    destination = unique_output_path(destination_root, relative)
                    output = image if quarantined else anonymize_faces(image, boxes, config.mode)
                    save_image_strip_metadata(destination, output)
                    log.record_success(source, destination, len(boxes), quarantined=quarantined)
                    self.events.put(("success", (len(boxes), quarantined)))
                except Exception as exc:
                    log.record_failure(source, f"{exc}\n{traceback.format_exc()}")
                    self.events.put(("failure", f"{source}: {exc}"))

            self.events.put(("finished_processing", log))
            self.events.put(("delete_check", (config, log)))
        except Exception as exc:
            self.events.put(("fatal", f"{exc}\n{traceback.format_exc()}"))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._handle_event(event, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: str, payload: object) -> None:
        if event == "total":
            self.total_files = int(payload)
            self.processed = 0
            self.failed = 0
            self.quarantined = 0
            self.progress.configure(value=0, maximum=max(1, self.total_files))
            self._append_log(f"Found {self.total_files} image file(s).")
        elif event == "file":
            self.status_text.set(str(payload))
            self._append_log(str(payload))
        elif event == "success":
            faces, quarantined = payload  # type: ignore[misc]
            self.processed += 1
            if quarantined:
                self.quarantined += 1
                self._append_log("No face detected; saved to Quarantine for manual review.")
            self.progress.configure(value=self.processed + self.failed)
            self.status_text.set(
                f"Processed: {self.processed} | Quarantined: {self.quarantined} | "
                f"Failed: {self.failed} | Faces in last file: {faces}"
            )
        elif event == "failure":
            self.failed += 1
            self.progress.configure(value=self.processed + self.failed)
            self._append_log(f"FAILED: {payload}")
            self.status_text.set(f"Processed: {self.processed} | Failed: {self.failed}")
        elif event == "delete_check":
            config, log = payload  # type: ignore[misc]
            self._maybe_delete(config, log)
        elif event == "finished_processing":
            self._append_log("Processing completed. Checking deletion settings...")
        elif event == "fatal":
            self.start_button.configure(state="normal")
            messagebox.showerror("Processing failed", str(payload))
            self.status_text.set("Processing failed.")

    def _maybe_delete(self, config: JobConfig, log: ProcessingLog) -> None:
        deletion_performed = False
        if config.delete_after_success and self.total_files > 0 and self.failed == 0 and self.processed == self.total_files:
            confirmed = messagebox.askyesno(
                "Confirm input folder deletion",
                "All valid image files were processed and written successfully.\n\n"
                f"Delete the original input folder?\n{config.input_dir}\n\n"
                "If send2trash is installed, the folder will be moved to the recycle bin. "
                "Otherwise it will be permanently deleted. This is not secure deletion.",
            )
            if confirmed:
                ok, message = delete_input_folder(config.input_dir)
                deletion_performed = ok
                self._append_log(message)
                if not ok:
                    messagebox.showerror("Deletion failed", message)

        log.finish(deletion_performed)
        self.start_button.configure(state="normal")
        self.status_text.set(
            f"Complete. Processed: {self.processed} | Quarantined: {self.quarantined} | "
            f"Failed: {self.failed} | Log: {log.path}"
        )
        messagebox.showinfo(
            "Complete",
            f"Processed {self.processed} file(s), quarantined {self.quarantined}, "
            f"failed {self.failed}.\n\nLog:\n{log.path}",
        )

    def _append_log(self, text: str) -> None:
        if self.log_box is None:
            return
        self.log_box.insert("end", f"{text}\n")
        self.log_box.see("end")

    def _clear_log(self) -> None:
        if self.log_box is not None:
            self.log_box.delete("1.0", "end")


def main() -> None:
    root = Tk()
    FaceAnonymizerApp(root)
    root.mainloop()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _find_yunet_model() -> Path | None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILENAMES:
        candidate = MODELS_DIR / filename
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
