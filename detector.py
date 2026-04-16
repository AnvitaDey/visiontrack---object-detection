"""
Real-Time Object Detection & Counting using YOLOv8 + OpenCV
============================================================
Author  : Your Name
Model   : YOLOv8n (pretrained on COCO – 80 classes, no extra dataset needed)
Usage   : python detector.py              → webcam (device 0)
          python detector.py --source video.mp4
          python detector.py --source 0 --track   → enable object tracking
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Colour palette (BGR) – one colour per class index ──────────────────────
np.random.seed(42)
PALETTE = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)


# ── Utility helpers ─────────────────────────────────────────────────────────

def get_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a given class id."""
    c = PALETTE[class_id % len(PALETTE)]
    return int(c[0]), int(c[1]), int(c[2])


def draw_box(frame: np.ndarray,
             x1: int, y1: int, x2: int, y2: int,
             label: str,
             color: tuple[int, int, int],
             conf: float) -> None:
    """Draw a filled-label bounding box on *frame* in-place."""
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)

    # Label text (white on coloured background)
    cv2.putText(frame, text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_stats(frame: np.ndarray,
               counts: dict[str, int],
               fps: float,
               total: int) -> None:
    """Overlay FPS, total count, and per-class counts in the top-left corner."""
    h, w = frame.shape[:2]
    panel_w = 220
    panel_h = 30 + 22 * (len(counts) + 2)          # dynamic height

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 28
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"FPS : {fps:5.1f}", (14, y), font, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"Total objects : {total}", (14, y), font, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    y += 22

    for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        cv2.putText(frame, f"  {cls_name:<16} {cnt:>3}",
                    (14, y), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20


# ── Core detector class ──────────────────────────────────────────────────────

class ObjectDetector:
    """
    Wraps a YOLOv8 model and processes frames one at a time.

    Parameters
    ----------
    model_path : str
        Path or Ultralytics hub name for the YOLOv8 weights file.
        'yolov8n.pt' is downloaded automatically on first run (~6 MB).
    conf      : float  – confidence threshold (0–1)
    iou       : float  – NMS IoU threshold   (0–1)
    device    : str    – 'cpu', '0' (GPU), or '' (auto)
    track     : bool   – use SORT tracker (yolo.track) instead of detect
    """

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf: float = 0.40,
                 iou: float = 0.45,
                 device: str = "",
                 track: bool = False):
        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.track = track
        self.names: dict[int, str] = self.model.names   # {0: 'person', 1: 'bicycle', …}

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, int], int]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        annotated  : np.ndarray   – frame with boxes drawn
        counts     : dict         – {class_name: count}
        total      : int          – total detections
        """
        if self.track:
            results = self.model.track(
                frame, conf=self.conf, iou=self.iou,
                device=self.device, persist=True, verbose=False
            )
        else:
            results = self.model(
                frame, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False
            )

        counts: dict[str, int] = defaultdict(int)
        annotated = frame.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id          = int(box.cls[0])
                conf_val        = float(box.conf[0])
                cls_name        = self.names.get(cls_id, str(cls_id))
                color           = get_color(cls_id)

                draw_box(annotated, x1, y1, x2, y2, cls_name, color, conf_val)
                counts[cls_name] += 1

        total = sum(counts.values())
        return annotated, dict(counts), total


# ── Video loop ───────────────────────────────────────────────────────────────

def run(source,
        model_path: str = "yolov8n.pt",
        conf: float = 0.40,
        iou: float = 0.45,
        device: str = "",
        track: bool = False,
        save: bool = False,
        output_path: str = "output.mp4") -> None:
    """
    Open *source* (webcam index or video file path) and run detection.

    Press  Q  or  ESC  to quit.
    """
    detector = ObjectDetector(model_path, conf, iou, device, track)

    # Open capture
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source: {source}  |  Resolution: {w}×{h}  |  FPS: {fps_in:.1f}")

    # Optional writer
    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))
        print(f"[INFO] Saving output to: {output_path}")

    fps_calc = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # ── Inference ──────────────────────────────────────────────────────
        annotated, counts, total = detector.process_frame(frame)

        # ── Overlay stats ──────────────────────────────────────────────────
        fps_calc.tick()
        draw_stats(annotated, counts, fps_calc.fps, total)

        # ── Display ────────────────────────────────────────────────────────
        cv2.imshow("YOLOv8 Object Detection", annotated)
        if writer:
            writer.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or ESC
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ── FPS counter helper ───────────────────────────────────────────────────────

class FPSCounter:
    """Smoothed FPS counter using a rolling window."""

    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window
        self.fps: float = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            self.fps = (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ── CLI entry-point ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detector")
    p.add_argument("--source",  default="0",
                   help="Webcam index (0,1,…) or path to video file")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="YOLOv8 weights: yolov8n/s/m/l/x.pt  (default: yolov8n.pt)")
    p.add_argument("--conf",    type=float, default=0.40,
                   help="Confidence threshold  (default: 0.40)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold     (default: 0.45)")
    p.add_argument("--device",  default="",
                   help="Inference device: '' auto | 'cpu' | '0' GPU")
    p.add_argument("--track",   action="store_true",
                   help="Enable object tracking (SORT)")
    p.add_argument("--save",    action="store_true",
                   help="Save annotated video to --output")
    p.add_argument("--output",  default="output.mp4",
                   help="Output video file name  (default: output.mp4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source     = args.source,
        model_path = args.model,
        conf       = args.conf,
        iou        = args.iou,
        device     = args.device,
        track      = args.track,
        save       = args.save,
        output_path= args.output,
    )
