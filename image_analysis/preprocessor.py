import logging
import numpy as np
from PIL import Image
from typing import Tuple, Union

logger = logging.getLogger(__name__)

def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True,
    enhance_contrast: bool = True
) -> np.ndarray:
    """Preprocess an image for medical image analysis.
    
    This function performs a series of preprocessing steps on an input image:
    1. Converts the image to RGB mode if it isn't already
    2. Resizes the image to the target dimensions using Lanczos resampling
    3. Optionally enhances the image contrast
    4. Optionally normalizes pixel values to [0, 1] range
    
    Args:
        image (Image.Image): Input PIL Image to preprocess
        target_size (Tuple[int, int], optional): Target dimensions (width, height). Defaults to (512, 512).
        normalize (bool, optional): Whether to normalize pixel values to [0, 1]. Defaults to True.
        enhance_contrast (bool, optional): Whether to enhance image contrast. Defaults to True.
    
    Returns:
        np.ndarray: Preprocessed image as a numpy array
        
    Raises:
        Exception: If any preprocessing step fails
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        
        if enhance_contrast:
            img_array = enhance_image_contrast(img_array)
        
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def enhance_image_contrast(img_array: np.ndarray) -> np.ndarray:
    """Enhance the contrast of an image using min-max normalization.
    
    This function enhances image contrast by:
    1. Converting the image to float32
    2. Finding the minimum and maximum pixel values
    3. Normalizing the pixel values to [0, 255] range
    4. Clipping values to ensure they stay within valid range
    
    Args:
        img_array (np.ndarray): Input image as a numpy array
        
    Returns:
        np.ndarray: Contrast-enhanced image as a numpy array of type uint8.
                   If enhancement fails, returns the original image.
    """
    try:
        img_float = img_array.astype(np.float32)
        
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        
        img_norm = (img_float - min_val) / (max_val - min_val + 1e-8)
        
        img_eq = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
        
        return img_eq
    
    except Exception as e:
        logger.error(f"Error enhancing image contrast: {e}")
        return img_array

def validate_image(image: Union[str, Image.Image, np.ndarray]) -> Tuple[bool, str]:
    """Validate an image for medical image analysis.
    
    This function performs several validation checks on the input image:
    1. Verifies the image format is valid (string path, PIL Image, or numpy array)
    2. Checks if image dimensions are at least 100x100 pixels
    3. Verifies the image is not empty or corrupted
    
    Args:
        image (Union[str, Image.Image, np.ndarray]): Input image as either:
            - A string path to an image file
            - A PIL Image object
            - A numpy array
    
    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if image is valid, False otherwise
            - str: Empty string if valid, error message if invalid
    """
    try:
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
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
