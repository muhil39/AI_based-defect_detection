"""
Surface Crack Detection AI Module (CNN 2).
Supports both:
1. TensorFlow/Keras (.keras format)
2. Standalone Convolutional Neural Network weights (.npz format)
Outputs: 'CRACK DETECTED' or 'NO VISIBLE CRACK DETECTED'
"""

import os
import numpy as np
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crack_detector.keras")
NPZ_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crack_model_weights.npz")


def conv2d_bank(img_rgb: np.ndarray) -> np.ndarray:
    """Extracts spatial convolutional feature maps using directional and edge kernels."""
    gray = (0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1] + 0.1140 * img_rgb[..., 2])
    gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)

    k_horiz = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    k_vert = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    k_diag1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
    k_diag2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
    k_laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    k_sharp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

    kernels = [k_horiz, k_vert, k_diag1, k_diag2, k_laplace, k_sharp]
    feature_maps = []
    h, w = gray.shape
    for k in kernels:
        padded = np.pad(gray, 1, mode="reflect")
        conv = (
            k[0, 0] * padded[0:h, 0:w] + k[0, 1] * padded[0:h, 1:w+1] + k[0, 2] * padded[0:h, 2:w+2] +
            k[1, 0] * padded[1:h+1, 0:w] + k[1, 1] * padded[1:h+1, 1:w+1] + k[1, 2] * padded[1:h+1, 2:w+2] +
            k[2, 0] * padded[2:h+2, 0:w] + k[2, 1] * padded[2:h+2, 1:w+1] + k[2, 2] * padded[2:h+2, 2:w+2]
        )
        feature_maps.append(np.maximum(0, conv))

    features = []
    for f in feature_maps:
        features.append(np.mean(f))
        features.append(np.max(f))
        features.append(np.percentile(f, 95))

    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    saturation = np.std(img_rgb, axis=-1)
    features.append(np.mean(saturation))
    features.append(np.std(saturation))
    features.append(np.mean(r) - np.mean(b))
    features.append(np.std(gray))

    return np.array(features, dtype=np.float32)


class CrackDetector:
    """
    CNN Crack Detector supporting Keras & Standalone CNN backends.
    """

    def __init__(self):
        self.model_type = None
        self.keras_model = None
        self.npz_weights = None
        self.is_loaded = False
        self.error_message: Optional[str] = None
        self._load_model()

    def _load_model(self) -> bool:
        if os.path.exists(KERAS_MODEL_PATH):
            try:
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
                self.model_type = "KERAS"
                self.is_loaded = True
                self.error_message = None
                return True
            except Exception as e:
                pass

        if os.path.exists(NPZ_MODEL_PATH):
            try:
                data = np.load(NPZ_MODEL_PATH)
                self.npz_weights = {
                    "W": data["W"],
                    "b": data["b"],
                    "mean": data["mean"],
                    "std": data["std"],
                    "classes": data["classes"]
                }
                self.model_type = "STANDALONE_CNN"
                self.is_loaded = True
                self.error_message = None
                return True
            except Exception as e:
                self.error_message = f"Failed to load crack CNN weights: {e}"

        self.error_message = (
            "Crack detector CNN model not found. "
            "Please train using 'python training/train_crack.py' or 'python training/train_lightweight_cnn.py'."
        )
        self.is_loaded = False
        return False

    def predict(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            if not self._load_model():
                return {
                    "predicted_class": "INSPECTION NOT PERFORMED",
                    "surface_condition": "NOT EVALUATED",
                    "confidence": 0.0,
                    "is_model_loaded": False,
                    "error": self.error_message
                }

        try:
            if self.model_type == "KERAS":
                predictions = self.keras_model.predict(preprocessed_image, verbose=0)
                if predictions.shape[-1] == 1:
                    prob_crack = float(predictions[0][0])
                else:
                    prob_crack = float(predictions[0][0])
                    prob_no_crack = float(predictions[0][1])
            else:
                img_rgb = preprocessed_image[0]
                feats = conv2d_bank(img_rgb)
                norm_feats = (feats - self.npz_weights["mean"]) / self.npz_weights["std"]
                logits = np.dot(norm_feats, self.npz_weights["W"]) + self.npz_weights["b"]
                exp_logits = np.exp(logits - np.max(logits))
                scores = (exp_logits / np.sum(exp_logits))[0]
                # classes: ["crack", "no_crack"] -> index 0 is crack, index 1 is clean
                prob_crack = float(scores[0])
                prob_no_crack = float(scores[1])

            if prob_crack >= 0.5:
                predicted_class = "CRACK DETECTED"
                surface_cond = "DEFECTIVE"
                conf = prob_crack
            else:
                predicted_class = "NO VISIBLE CRACK DETECTED"
                surface_cond = "ACCEPTABLE"
                conf = 1.0 - prob_crack

            return {
                "predicted_class": predicted_class,
                "surface_condition": surface_cond,
                "confidence": conf,
                "model_engine": self.model_type,
                "is_model_loaded": True,
                "error": None
            }
        except Exception as e:
            return {
                "predicted_class": "INSPECTION NOT PERFORMED",
                "surface_condition": "ERROR",
                "confidence": 0.0,
                "is_model_loaded": False,
                "error": f"Inference failure: {str(e)}"
            }


crack_detector = CrackDetector()
