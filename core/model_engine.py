"""
Model loading and inference engine for Traffic Sign Recognition.
Supports top-K probability distribution, timing, and robust preprocessing.
"""
import os
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

# Suppress noisy TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

from .sign_metadata import get_sign_info, SignInfo

class TrafficSignModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = None
        self.input_shape = (30, 30)
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = load_model(model_path)
        self.model_path = model_path
        # Warmup prediction to initialize GPU/CPU pipeline
        dummy = np.zeros((1, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        return self

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess_image(self, image_input) -> np.ndarray:
        """
        Preprocesses a PIL Image, numpy array, or file path to (1, 30, 30, 3) uint8/float32 format.
        Converts RGBA / Grayscale to 3-channel RGB.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # If BGR (OpenCV)
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image = Image.fromarray(image_input)
            else:
                image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # Ensure RGB (strip alpha or expand grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to GTSRB model dimensions (30x30)
        image = image.resize(self.input_shape, Image.Resampling.BILINEAR)
        img_array = np.array(image, dtype=np.float32)
        # Add batch dimension -> shape (1, 30, 30, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_input, top_k: int = 5) -> Dict[str, Any]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load() first.")

        t0 = time.perf_counter()
        tensor = self.preprocess_image(image_input)
        probabilities = self.model.predict(tensor, verbose=0)[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions: List[Dict[str, Any]] = []
        for idx in top_indices:
            idx = int(idx)
            conf = float(probabilities[idx]) * 100.0
            info = get_sign_info(idx)
            top_predictions.append({
                "class_id": idx,
                "confidence": conf,
                "confidence_str": f"{conf:.1f}%",
                "name": info.name,
                "category": info.category,
                "badge_color": info.badge_color,
                "action_instruction": info.action_instruction,
                "shape": info.shape
            })

        best = top_predictions[0]
        return {
            "best": best,
            "top_k": top_predictions,
            "latency_ms": round(inference_time_ms, 2),
            "all_probabilities": probabilities.tolist()
        }
