"""
Image processing module for AI-Based Surface Defect Detection.
Handles image validation, base64 decoding, OpenCV preprocessing,
and disk storage for inspection images.
"""

import os
import re
import base64
import uuid
from datetime import datetime
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import io

# We will dynamically import cv2 if available, or fall back to PIL for image manipulation
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DEFAULT_TARGET_SIZE = (224, 224)


def validate_image_extension(filename: str) -> bool:
    """Checks whether the filename has an allowed extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def decode_base64_image(base64_str: str) -> bytes:
    """
    Decodes a base64 string or data URL (e.g., 'data:image/jpeg;base64,...')
    into raw image bytes.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    # Remove any whitespace or newline
    base64_str = re.sub(r"\s+", "", base64_str)
    return base64.b64decode(base64_str)


def save_image_to_disk(image_bytes: bytes, upload_folder: str, prefix: str = "capture") -> Tuple[str, str]:
    """
    Saves image bytes to the upload folder with a unique timestamped filename.
    Returns (filename, absolute_filepath).
    """
    os.makedirs(upload_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{timestamp}_{unique_id}.jpg"
    filepath = os.path.join(upload_folder, filename)

    # Use PIL to ensure standard JPEG format and orientation
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(filepath, format="JPEG", quality=95)

    return filename, filepath


def preprocess_image_for_model(
    image_input,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """
    Preprocesses an image (file path, raw bytes, or numpy array) for CNN inference:
    1. Loads as RGB
    2. Resizes to target_size (width, height)
    3. Normalizes pixel values to [0.0, 1.0]
    4. Expands dimensions to (1, height, width, 3)

    Returns:
        np.ndarray: Batch array with shape (1, target_size[1], target_size[0], 3)
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, np.ndarray):
        # Convert BGR/RGB numpy array to PIL
        if len(image_input.shape) == 2:
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 3:
            pil_img = Image.fromarray(image_input)
        else:
            raise ValueError("Unsupported numpy image array dimensions")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize using high quality bicubic interpolation
    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to float numpy array normalized between 0 and 1
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0

    # Expand batch dimension: (1, H, W, 3)
    batch_arr = np.expand_dims(arr, axis=0)
    return batch_arr


def assess_image_quality(image_bytes: bytes) -> dict:
    """
    Optional OpenCV check to evaluate if image is too blurry or pitch black.
    """
    quality_info = {"is_valid": True, "message": "OK", "blur_score": 0.0}
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        if HAS_CV2:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"is_valid": False, "message": "Failed to decode image data."}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Variance of Laplacian for blur estimation
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_info["blur_score"] = float(laplacian_var)

            # Check mean brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 10:
                quality_info["message"] = "Image appears too dark or lens covered."
            elif laplacian_var < 15.0:
                quality_info["message"] = "Image might be blurry. Ensure component is in focus."
        else:
            img = Image.open(io.BytesIO(image_bytes))
            if img.size[0] < 50 or img.size[1] < 50:
                return {"is_valid": False, "message": "Image resolution too low."}
    except Exception as e:
        return {"is_valid": False, "message": f"Image validation error: {str(e)}"}

    return quality_info

