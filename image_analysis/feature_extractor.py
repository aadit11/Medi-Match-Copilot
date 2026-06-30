import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage import feature

from image_analysis.preprocessor import to_uint8_rgb

logger = logging.getLogger(__name__)

BODY_AREA_FEATURE_FOCUS = {
    "foot": {"texture": True, "shape": True, "color": True},
    "skin": {"texture": True, "shape": True, "color": True},
    "chest": {"texture": True, "shape": True, "color": False},
    "hand": {"texture": True, "shape": True, "color": True},
    "eye": {"texture": True, "shape": True, "color": True},
}


def _resolve_feature_flags(body_area: Optional[str]) -> Dict[str, bool]:
    if not body_area:
        return {"texture": True, "shape": True, "color": True}
    return BODY_AREA_FEATURE_FOCUS.get(body_area.lower(), {
        "texture": True,
        "shape": True,
        "color": True,
    })


def extract_features(
    image: np.ndarray,
    body_area: Optional[str] = None,
    grayscale: Optional[np.ndarray] = None,
    extract_texture: Optional[bool] = None,
    extract_shape: Optional[bool] = None,
    extract_color: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Extract texture, shape, and color features from a preprocessed image."""
    try:
        rgb_uint8 = to_uint8_rgb(image)
        gray = grayscale if grayscale is not None else cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

        flags = _resolve_feature_flags(body_area)
        use_texture = extract_texture if extract_texture is not None else flags["texture"]
        use_shape = extract_shape if extract_shape is not None else flags["shape"]
        use_color = extract_color if extract_color is not None else flags["color"]

        features: Dict[str, Any] = {"body_area": body_area or "unspecified"}

        if use_texture:
            features.update(extract_texture_features(gray))
        if use_shape:
            features.update(extract_shape_features(gray))
        if use_color and rgb_uint8.ndim == 3:
            features.update(extract_color_features(rgb_uint8))

        features["feature_summary"] = summarize_features(features)
        return features

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None


def summarize_features(features: Dict[str, Any]) -> str:
    """Create a concise, human-readable summary for downstream classification."""
    parts: List[str] = []

    texture_std = features.get("texture_std")
    texture_entropy = features.get("texture_entropy")
    if texture_std is not None:
        texture_level = "high" if texture_std > 30 else "moderate" if texture_std > 15 else "low"
        parts.append(f"Texture variation is {texture_level}")
    if texture_entropy is not None:
        parts.append(f"texture entropy {texture_entropy:.2f}")

    circularity = features.get("circularity")
    aspect_ratio = features.get("aspect_ratio")
    if circularity is not None:
        shape_desc = "rounded" if circularity > 0.7 else "irregular" if circularity < 0.4 else "moderately irregular"
        parts.append(f"Lesion shape appears {shape_desc}")
    if aspect_ratio is not None:
        parts.append(f"aspect ratio {aspect_ratio:.2f}")

    color_diversity = features.get("color_diversity")
    if color_diversity is not None:
        color_desc = "high" if color_diversity > 4.0 else "moderate" if color_diversity > 2.5 else "low"
        parts.append(f"Color diversity is {color_desc}")

    red_mean = features.get("red_mean")
    if red_mean is not None:
        if red_mean > 150:
            parts.append("pronounced redness in tissue")
        elif red_mean < 80:
            parts.append("relatively pale tissue tones")

    return "; ".join(parts) if parts else "Limited visual feature signal detected"


def extract_texture_features(gray: np.ndarray) -> Dict[str, float]:
    try:
        gray_uint8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
        lbp = feature.local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        hist = hist.astype(float) / (hist.sum() + 1e-10)

        return {
            "texture_mean": float(np.mean(lbp)),
            "texture_std": float(np.std(lbp)),
            "texture_entropy": float(-np.sum(hist * np.log2(hist + 1e-10))),
        }
    except Exception as e:
        logger.error(f"Error extracting texture features: {e}")
        return {}


def extract_shape_features(gray: np.ndarray) -> Dict[str, float]:
    try:
        gray_uint8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
        _, binary = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0,
            "aspect_ratio": calculate_aspect_ratio(largest_contour),
        }
    except Exception as e:
        logger.error(f"Error extracting shape features: {e}")
        return {}


def extract_color_features(rgb_uint8: np.ndarray) -> Dict[str, float]:
    try:
        if rgb_uint8.ndim != 3:
            return {}

        color_features: Dict[str, float] = {}
        for i, color in enumerate(["red", "green", "blue"]):
            channel = rgb_uint8[:, :, i]
            color_features.update({
                f"{color}_mean": float(np.mean(channel)),
                f"{color}_std": float(np.std(channel)),
                f"{color}_median": float(np.median(channel)),
            })

        color_features["color_diversity"] = calculate_color_diversity(rgb_uint8)
        return color_features

    except Exception as e:
        logger.error(f"Error extracting color features: {e}")
        return {}


def calculate_aspect_ratio(contour: np.ndarray) -> float:
    try:
        _, _, w, h = cv2.boundingRect(contour)
        return float(w) / h if h > 0 else 0.0
    except Exception:
        return 0.0


def calculate_color_diversity(rgb_uint8: np.ndarray) -> float:
    try:
        hist = cv2.calcHist([rgb_uint8], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        return float(-np.sum(hist * np.log2(hist + 1e-10)))
    except Exception:
        return 0.0
