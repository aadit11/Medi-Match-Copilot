import logging
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from core.config import USE_CLAHE_PREPROCESSING

logger = logging.getLogger(__name__)


def to_uint8_rgb(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert a PIL image or array to uint8 RGB."""
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image, dtype=np.uint8)

    array = np.asarray(image)
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255.0).clip(0, 255)
        else:
            array = array.clip(0, 255)
        array = array.astype(np.uint8)

    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return array


def apply_clahe(rgb_uint8: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB color space for medical images."""
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def enhance_image_contrast(img_array: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE when enabled, otherwise min-max normalization."""
    try:
        rgb_uint8 = to_uint8_rgb(img_array)
        if USE_CLAHE_PREPROCESSING:
            return apply_clahe(rgb_uint8)
        img_float = rgb_uint8.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        img_norm = (img_float - min_val) / (max_val - min_val + 1e-8)
        return np.clip(img_norm * 255, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Error enhancing image contrast: {e}")
        return to_uint8_rgb(img_array)


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = False,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """
    Preprocess an image for classical CV feature extraction.

    Returns uint8 RGB by default so OpenCV/skimage pipelines receive valid pixel data.
    """
    try:
        rgb_uint8 = to_uint8_rgb(image)
        rgb_uint8 = np.array(
            Image.fromarray(rgb_uint8).resize(target_size, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

        if enhance_contrast:
            rgb_uint8 = enhance_image_contrast(rgb_uint8)

        if normalize:
            return rgb_uint8.astype(np.float32) / 255.0
        return rgb_uint8

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def preprocess_for_analysis(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    enhance_contrast: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Single-pass preprocessing that produces arrays reused by the feature pipeline.

    Returns:
        rgb_uint8: Contrast-enhanced RGB image for color/shape analysis
        grayscale: Grayscale copy for texture/shape analysis
    """
    rgb_uint8 = preprocess_image(
        image,
        target_size=target_size,
        normalize=False,
        enhance_contrast=enhance_contrast,
    )
    grayscale = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return {
        "rgb_uint8": rgb_uint8,
        "grayscale": grayscale,
    }


def validate_image(image: Union[str, Image.Image, np.ndarray]) -> Tuple[bool, str]:
    """Validate an image for medical image analysis."""
    try:
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(to_uint8_rgb(image))
        else:
            return False, "Invalid image format"

        width, height = img.size
        if width < 100 or height < 100:
            return False, "Image dimensions too small"

        if img.getbbox() is None:
            return False, "Image appears to be empty or corrupted"

        return True, ""

    except Exception as e:
        return False, f"Error validating image: {str(e)}"
