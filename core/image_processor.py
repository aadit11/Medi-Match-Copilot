import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from PIL import Image
import numpy as np

from core.config import (
    IMAGE_SIZE,
    ALLOWED_IMAGE_EXTENSIONS,
    IMAGE_CONFIDENCE_THRESHOLD
)
from models.vision_models import VisionModelClient
from image_analysis.preprocessor import preprocess_image
from image_analysis.feature_extractor import extract_features
from image_analysis.condition_classifier import classify_condition
from utils.medical_validators import validate_medical_image
from utils.prompt_engineering import create_image_analysis_prompt

logger = logging.getLogger(__name__)

class ImageProcessor:
    
    def __init__(
        self,
        vision_model: Optional[VisionModelClient] = None,
        confidence_threshold: float = IMAGE_CONFIDENCE_THRESHOLD
    ):
        
        self.vision_model = vision_model or VisionModelClient()
        self.confidence_threshold = confidence_threshold
        
    def analyze_image(
        self,
        image_path: str,
        primary_concern: Optional[str] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        body_area: Optional[str] = None
    ) -> Dict[str, Any]:
      
        
        is_valid, error_msg = validate_medical_image(image_path)
        if not is_valid:
            logger.error(f"Invalid medical image: {error_msg}")
            return {"error": error_msg}
        
        try:
            image = Image.open(image_path)
            preprocessed_image = preprocess_image(image, target_size=IMAGE_SIZE)
            
            image_features = extract_features(preprocessed_image)
            
            classification_results = []
            if image_features is not None:
                classification_results = classify_condition(
                    image_features, 
                    body_area=body_area
                )
                
                classification_results = [
                    result for result in classification_results
                    if result.get("confidence", 0) >= self.confidence_threshold
                ]
            
            prompt = create_image_analysis_prompt(
                image_path=image_path,
                primary_concern=primary_concern,
                patient_info=patient_info,
                body_area=body_area
            )
            
            detailed_analysis = self.vision_model.analyze_image(
                image_path=image_path, 
                prompt=prompt
            )
            
            return {
                "image_path": image_path,
                "patient_info": patient_info or {},
                "primary_concern": primary_concern,
                "body_area": body_area,
                "classifications": classification_results,
                "detailed_analysis": detailed_analysis
            }
        
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return {
                "error": "Failed to analyze medical image",
                "details": str(e)
            }
    
    def _extract_findings_from_analysis(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Internal method to extract findings from image analysis."""
        if "error" in analysis:
            return []
        
        detailed_text = analysis.get("detailed_analysis", "")
        if not detailed_text:
            return []
        
        try:
            system_prompt = """
            Extract the key medical findings from the image analysis text.
            For each finding, extract:
            1. The description of the finding
            2. The confidence level (convert to decimal: 80% = 0.8)
            3. The potential diagnosis associated with this finding
            4. Whether it requires urgent attention (true/false)
            
            Return the data in this JSON format:
            [
                {
                    "finding": "Irregular border on lesion",
                    "confidence": 0.8,
                    "potential_diagnosis": "Melanoma",
                    "urgency_required": true
                }
            ]
            """
            
            user_prompt = f"Extract findings from this medical image analysis:\n\n{detailed_text}"
            
            extraction_result = self.vision_model.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            import re
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", extraction_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                findings = json.loads(json_str)
                return findings
            else:
                json_match = re.search(r"(\[\s*\{.*?\}\s*\])", extraction_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    findings = json.loads(json_str)
                    return findings
                
            logger.warning("Could not extract structured findings from analysis")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting findings: {e}")
            return []
    
    def combine_with_symptom_data(
        self,
        image_analysis: Dict[str, Any],
        symptom_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        if "error" in image_analysis or "error" in symptom_data:
            errors = []
            if "error" in image_analysis:
                errors.append(f"Image analysis error: {image_analysis['error']}")
            if "error" in symptom_data:
                errors.append(f"Symptom analysis error: {symptom_data['error']}")
                
            return {"errors": errors}
        
        image_diagnoses = []
        for classification in image_analysis.get("classifications", []):
            image_diagnoses.append({
                "name": classification.get("condition", "Unknown condition"),
                "confidence": classification.get("confidence", 0),
                "source": "image_analysis"
            })
        
        symptom_diagnoses = []
        for diagnosis in symptom_data.get("diagnoses", []):
            symptom_diagnoses.append({
                "name": diagnosis.get("name", "Unknown condition"),
                "confidence": diagnosis.get("confidence", 0),
                "source": "symptom_analysis"
            })
        
        try:
            system_prompt = """
            You are a medical diagnostic assistant. Create a comprehensive assessment
            based on both image analysis and symptom analysis. Consider how the visual
            evidence and reported symptoms may indicate particular conditions.
            Provide an integrated analysis that considers both sources of information.
            """
            
            user_prompt = f"""
            # Image Analysis Results
            {json.dumps(image_analysis.get("classifications", []), indent=2)}
            
            Detailed visual assessment:
            {image_analysis.get("detailed_analysis", "")}
            
            # Symptom Analysis Results
            {json.dumps(symptom_diagnoses, indent=2)}
            
            Raw diagnostic assessment:
            {symptom_data.get("raw_assessment", "")}
            
            Based on both the image analysis and symptom assessment, provide an integrated
            diagnostic analysis. Highlight where the image findings confirm or contradict
            the symptom-based diagnoses.
            """
            
            combined_assessment = self.vision_model.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            return {
                "image_diagnoses": image_diagnoses,
                "symptom_diagnoses": symptom_diagnoses,
                "combined_assessment": combined_assessment,
                "is_urgent": symptom_data.get("is_urgent", False) or any(
                    finding.get("urgency_required", False) 
                    for finding in self._extract_findings_from_analysis(image_analysis)
                )
            }
            
        except Exception as e:
            logger.error(f"Error creating combined assessment: {e}")
            return {
                "image_diagnoses": image_diagnoses,
                "symptom_diagnoses": symptom_diagnoses,
                "error": "Failed to create combined assessment"
            }

def create_image_processor() -> ImageProcessor:
    
    vision_model = VisionModelClient()
    
    return ImageProcessor(
        vision_model=vision_model,
        confidence_threshold=IMAGE_CONFIDENCE_THRESHOLD
    )