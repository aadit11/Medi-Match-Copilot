import logging
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True,
    enhance_contrast: bool = True
) -> np.ndarray:
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
