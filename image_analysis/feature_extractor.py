import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2
from skimage import feature, measure

logger = logging.getLogger(__name__)

def extract_features(
    image: np.ndarray,
    extract_texture: bool = True,
    extract_shape: bool = True,
    extract_color: bool = True
) -> Optional[Dict[str, Any]]:
    
    try:
        features = {}
        
        if extract_texture:
            texture_features = extract_texture_features(image)
            features.update(texture_features)
        
        if extract_shape:
            shape_features = extract_shape_features(image)
            features.update(shape_features)
        
        if extract_color:
            color_features = extract_color_features(image)
            features.update(color_features)
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def extract_texture_features(image: np.ndarray) -> Dict[str, float]:
    
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        hist = hist.astype(float) / hist.sum()
        
        texture_features = {
            'texture_mean': np.mean(lbp),
            'texture_std': np.std(lbp),
            'texture_entropy': -np.sum(hist * np.log2(hist + 1e-10))
        }
        
        return texture_features
    
    except Exception as e:
        logger.error(f"Error extracting texture features: {e}")
        return {}

def extract_shape_features(image: np.ndarray) -> Dict[str, float]:
    
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        shape_features = {
            'area': area,
            'perimeter': perimeter,
            'circularity': 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0,
            'aspect_ratio': calculate_aspect_ratio(largest_contour)
        }
        
        return shape_features
    
    except Exception as e:
        logger.error(f"Error extracting shape features: {e}")
        return {}

def extract_color_features(image: np.ndarray) -> Dict[str, float]:
    
    try:
        if len(image.shape) != 3:
            return {}
        
        color_features = {}
        for i, color in enumerate(['red', 'green', 'blue']):
            channel = image[:, :, i]
            color_features.update({
                f'{color}_mean': np.mean(channel),
                f'{color}_std': np.std(channel),
                f'{color}_median': np.median(channel)
            })
        
        color_features['color_diversity'] = calculate_color_diversity(image)
        
        return color_features
    
    except Exception as e:
        logger.error(f"Error extracting color features: {e}")
        return {}

def calculate_aspect_ratio(contour: np.ndarray) -> float:
    
    try:
        x, y, w, h = cv2.boundingRect(contour)
        return float(w) / h if h > 0 else 0
    except Exception:
        return 0

def calculate_color_diversity(image: np.ndarray) -> float:
    
    try:
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 1, 0, 1, 0, 1])
        hist = hist.flatten() / hist.sum()
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    except Exception:
        return 0