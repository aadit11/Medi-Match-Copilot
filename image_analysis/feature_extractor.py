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
    """
    Extract various features from an input image including texture, shape, and color features.
    
    Args:
        image (np.ndarray): Input image as a numpy array. Can be grayscale or RGB.
        extract_texture (bool, optional): Whether to extract texture features. Defaults to True.
        extract_shape (bool, optional): Whether to extract shape features. Defaults to True.
        extract_color (bool, optional): Whether to extract color features. Defaults to True.
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing the extracted features. Returns None if extraction fails.
        The dictionary contains the following keys based on enabled features:
        - Texture features: 'texture_mean', 'texture_std', 'texture_entropy'
        - Shape features: 'area', 'perimeter', 'circularity', 'aspect_ratio'
        - Color features: 'red_mean', 'red_std', 'red_median', 'green_mean', 'green_std', 
                         'green_median', 'blue_mean', 'blue_std', 'blue_median', 'color_diversity'
    """
    
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
    """
    Extract texture features from an image using Local Binary Pattern (LBP).
    
    Args:
        image (np.ndarray): Input image as a numpy array. Can be grayscale or RGB.
    
    Returns:
        Dict[str, float]: Dictionary containing texture features:
            - 'texture_mean': Mean value of LBP features
            - 'texture_std': Standard deviation of LBP features
            - 'texture_entropy': Entropy of LBP histogram
        Returns empty dictionary if extraction fails.
    """
    
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
    """
    Extract shape features from an image by analyzing its contours.
    
    Args:
        image (np.ndarray): Input image as a numpy array. Can be grayscale or RGB.
    
    Returns:
        Dict[str, float]: Dictionary containing shape features:
            - 'area': Area of the largest contour
            - 'perimeter': Perimeter of the largest contour
            - 'circularity': Measure of how circular the shape is (4π * area / perimeter²)
            - 'aspect_ratio': Ratio of width to height of the bounding rectangle
        Returns empty dictionary if extraction fails or no contours are found.
    """
    
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
    """
    Extract color features from an RGB image.
    
    Args:
        image (np.ndarray): Input image as a numpy array. Must be RGB (3 channels).
    
    Returns:
        Dict[str, float]: Dictionary containing color features for each channel (red, green, blue):
            - '{color}_mean': Mean value of the channel
            - '{color}_std': Standard deviation of the channel
            - '{color}_median': Median value of the channel
            - 'color_diversity': Entropy-based measure of color diversity
        Returns empty dictionary if extraction fails or image is not RGB.
    """
    
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
    """
    Calculate the aspect ratio of a contour using its bounding rectangle.
    
    Args:
        contour (np.ndarray): Contour points as a numpy array.
    
    Returns:
        float: Aspect ratio (width/height) of the bounding rectangle.
        Returns 0 if calculation fails or height is 0.
    """
    
    try:
        x, y, w, h = cv2.boundingRect(contour)
        return float(w) / h if h > 0 else 0
    except Exception:
        return 0

def calculate_color_diversity(image: np.ndarray) -> float:
    """
    Calculate color diversity of an image using histogram entropy.
    
    Args:
        image (np.ndarray): Input RGB image as a numpy array.
    
    Returns:
        float: Entropy-based measure of color diversity.
        Returns 0 if calculation fails.
    """
    
    try:
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 1, 0, 1, 0, 1])
        hist = hist.flatten() / hist.sum()
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    except Exception:
        return 0