"""
Real-time webcam inference engine with ROI targeting guide and FPS tracking.
"""
import cv2
import time
import threading
import numpy as np
from PIL import Image
from typing import Optional, Callable, Dict, Any

from .model_engine import TrafficSignModel

class WebcamEngine:
    def __init__(self, model_engine: TrafficSignModel):
        self.model_engine = model_engine
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_result: Optional[Dict[str, Any]] = None
        self.fps = 0.0
        self.camera_index = 0

    def start(self, camera_index: int = 0, on_frame_callback: Optional[Callable[[Image.Image, Optional[Dict[str, Any]], float], None]] = None):
        if self.is_running:
            return
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else 0)
        if not self.cap or not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, args=(on_frame_callback,), daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self, callback: Optional[Callable]):
        prev_time = time.time()
        inference_interval = 0.1  # Run inference every 100ms for smoothness
        last_inference_time = 0

        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            prev_time = now

            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Calculate Center ROI box (e.g. 200x200 square)
            box_size = min(h, w) // 2
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Run inference periodically on the cropped ROI
            if (now - last_inference_time) >= inference_interval and self.model_engine.is_loaded():
                roi = frame[y1:y2, x1:x2]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                try:
                    self.latest_result = self.model_engine.predict(roi_rgb, top_k=3)
                except Exception:
                    pass
                last_inference_time = now

            # Draw visual overlay on frame
            color = (0, 255, 127)  # Bright emerald green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Target Traffic Sign Here", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if self.latest_result:
                best = self.latest_result["best"]
                label_txt = f"{best['name']} ({best['confidence_str']})"
                cv2.putText(frame, label_txt, (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Convert full frame to PIL for UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            if callback:
                try:
                    callback(pil_img, self.latest_result, self.fps)
                except Exception:
                    break

            time.sleep(0.01)
