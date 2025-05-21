import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from core.config import MEDICAL_KB_DIR
from models.text_models import TextModelClient

logger = logging.getLogger(__name__)

class ConditionClassifier:
    def __init__(
        self,
        model_client: Optional[TextModelClient] = None,
        conditions_db_path: Optional[Path] = None
    ):
        self.model_client = model_client or TextModelClient()
        self.conditions_db_path = conditions_db_path or MEDICAL_KB_DIR / "conditions_database.json"
        self.conditions_db = self._load_conditions_database()
    
    def _load_conditions_database(self) -> Dict[str, Any]:
        try:
            if self.conditions_db_path.exists():
                with open(self.conditions_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading conditions database: {e}")
            return {}
    
    def classify_condition(
        self,
        image_features: Dict[str, Any],
        body_area: Optional[str] = None,
        min_confidence: float = 0.4
    ) -> List[Dict[str, Any]]:
        
        try:
            if not image_features:
                return []
            
            feature_desc = self._format_features(image_features)
            
            prompt = self._create_classification_prompt(feature_desc, body_area)
            
            response = self.model_client.generate_text(
                system_prompt=self._get_system_prompt(),
                prompt=prompt
            )
            
            classifications = self._extract_classifications(response)
            
            filtered_classifications = [
                c for c in classifications
                if c.get("confidence", 0) >= min_confidence
            ]
            
            return filtered_classifications
        
        except Exception as e:
            logger.error(f"Error classifying conditions: {e}")
            return []
    
    def _format_features(self, features: Dict[str, Any]) -> str:
        feature_parts = []
        
        if any(k.startswith('texture_') for k in features):
            texture_desc = "Texture characteristics:\n"
            for k, v in features.items():
                if k.startswith('texture_'):
                    texture_desc += f"- {k.replace('texture_', '')}: {v:.3f}\n"
            feature_parts.append(texture_desc)
        
        if any(k in ['area', 'perimeter', 'circularity', 'aspect_ratio'] for k in features):
            shape_desc = "Shape characteristics:\n"
            for k in ['area', 'perimeter', 'circularity', 'aspect_ratio']:
                if k in features:
                    shape_desc += f"- {k}: {features[k]:.3f}\n"
            feature_parts.append(shape_desc)
        
        if any(k.startswith(('red_', 'green_', 'blue_')) for k in features):
            color_desc = "Color characteristics:\n"
            for color in ['red', 'green', 'blue']:
                if f'{color}_mean' in features:
                    color_desc += f"- {color} channel: mean={features[f'{color}_mean']:.3f}, "
                    color_desc += f"std={features[f'{color}_std']:.3f}\n"
            feature_parts.append(color_desc)
        
        return "\n".join(feature_parts)
    
    def _create_classification_prompt(
        self,
        feature_desc: str,
        body_area: Optional[str]
    ) -> str:
        prompt = "Based on the following image features, identify possible medical conditions:\n\n"
        prompt += feature_desc
        
        if body_area:
            prompt += f"\nBody area: {body_area}\n"
        
        prompt += "\nFor each condition, provide:\n"
        prompt += "1. Condition name\n"
        prompt += "2. Confidence level (0-1)\n"
        prompt += "3. Brief explanation of why this condition matches the features\n"
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        return """
        You are a medical image analysis expert. Analyze the provided image features
        and identify possible medical conditions. Consider both the visual characteristics
        and the body area context. Provide your assessment with confidence levels and
        explanations for each potential condition.
        """
    
    def _extract_classifications(self, response: str) -> List[Dict[str, Any]]:
        try:
            system_prompt = """
            Extract the medical conditions from the analysis text.
            For each condition, extract:
            1. The name of the condition
            2. The confidence level (convert to decimal: 80% = 0.8)
            3. The explanation/reasoning
            
            Return the data in this JSON format:
            [
                {
                    "condition": "Condition name",
                    "confidence": 0.8,
                    "explanation": "Explanation text"
                }
            ]
            """
            
            extraction_result = self.model_client.generate_text(
                system_prompt=system_prompt,
                prompt=f"Extract conditions from this analysis:\n\n{response}"
            )
            
            import re
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", extraction_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                classifications = json.loads(json_str)
            else:
                json_match = re.search(r"(\[\s*\{.*?\}\s*\])", extraction_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    classifications = json.loads(json_str)
                else:
                    logger.warning("Could not extract structured classifications from model response")
                    return []
            
            return classifications
        
        except Exception as e:
            logger.error(f"Error extracting classifications: {e}")
            return []

def create_condition_classifier() -> ConditionClassifier:
    return ConditionClassifier()

def classify_condition(
    image_features: Dict[str, Any],
    body_area: Optional[str] = None,
    min_confidence: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Standalone function to classify conditions from image features.
    This is a convenience wrapper around ConditionClassifier.classify_condition.
    """
    classifier = create_condition_classifier()
    return classifier.classify_condition(
        image_features=image_features,
        body_area=body_area,
        min_confidence=min_confidence
    )
