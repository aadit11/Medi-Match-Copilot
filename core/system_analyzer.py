import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from core.diagnosis_engine import DiagnosisEngine, create_diagnosis_engine
from core.image_processor import ImageProcessor, create_image_processor
from utils.medical_validators import sanitize_patient_data

logger = logging.getLogger(__name__)

class SystemAnalyzer:
    
    
    def __init__(
        self,
        diagnosis_engine: Optional[DiagnosisEngine] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        self.diagnosis_engine = diagnosis_engine or create_diagnosis_engine()
        self.image_processor = image_processor or create_image_processor()
    
    def analyze_case(
        self,
        primary_symptom: str,
        secondary_symptoms: Optional[List[str]] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        medical_history: Optional[List[str]] = None,
        duration_days: Optional[int] = None,
        image_path: Optional[str] = None,
        body_area: Optional[str] = None
    ) -> Dict[str, Any]:

        patient_info = sanitize_patient_data(patient_info or {})
        
        symptom_analysis = self.diagnosis_engine.analyze_symptoms(
            primary_symptom=primary_symptom,
            secondary_symptoms=secondary_symptoms,
            patient_info=patient_info,
            medical_history=medical_history,
            duration_days=duration_days
        )
        
        result = {
            "symptom_analysis": symptom_analysis,
            "is_urgent": symptom_analysis.get("is_urgent", False),
            "patient_info": patient_info
        }
        
        if image_path:
            image_analysis = self.image_processor.analyze_image(
                image_path=image_path,
                primary_concern=primary_symptom,
                patient_info=patient_info,
                body_area=body_area
            )
            
            if "error" not in image_analysis:
                combined_analysis = self.image_processor.combine_with_symptom_data(
                    image_analysis=image_analysis,
                    symptom_data=symptom_analysis
                )
                
                result.update({
                    "image_analysis": image_analysis,
                    "combined_analysis": combined_analysis,
                    "is_urgent": result["is_urgent"] or combined_analysis.get("is_urgent", False)
                })
            else:
                result["image_analysis_error"] = image_analysis["error"]
        
        return result
    
    def get_condition_details(self, condition_name: str) -> Dict[str, Any]:
        return self.diagnosis_engine.get_condition_details(condition_name)
    
    def get_symptom_guidance(self, symptom: str) -> Dict[str, Any]:
        return self.diagnosis_engine.get_symptom_guidance(symptom)
    
    def extract_findings_from_image(self, image_path: str) -> List[Dict[str, Any]]:
        analysis = self.image_processor.analyze_image(image_path)
        return self.image_processor.extract_findings_from_analysis(analysis)

def create_system_analyzer() -> SystemAnalyzer:
    return SystemAnalyzer()
