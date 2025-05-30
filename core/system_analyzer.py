import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from core.diagnosis_engine import DiagnosisEngine, create_diagnosis_engine
from core.image_processor import ImageProcessor, create_image_processor
from utils.medical_validators import sanitize_patient_data

logger = logging.getLogger(__name__)

class SystemAnalyzer:
    """
    A class that coordinates the analysis of medical cases by combining symptom analysis
    and image processing capabilities.
    
    This class serves as the main orchestrator for analyzing medical cases, combining
    both symptom-based diagnosis and image analysis when available. It manages the
    interaction between the diagnosis engine and image processor components.
    """
    
    def __init__(
        self,
        diagnosis_engine: Optional[DiagnosisEngine] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        """
        Initialize the SystemAnalyzer with optional diagnosis engine and image processor.
        
        Args:
            diagnosis_engine (Optional[DiagnosisEngine]): The diagnosis engine instance.
                If None, a new instance will be created.
            image_processor (Optional[ImageProcessor]): The image processor instance.
                If None, a new instance will be created.
        """
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
        """
        Analyze a medical case by combining symptom analysis and optional image processing.
        
        Args:
            primary_symptom (str): The main symptom or concern of the patient.
            secondary_symptoms (Optional[List[str]]): Additional symptoms reported by the patient.
            patient_info (Optional[Dict[str, Any]]): Dictionary containing patient demographic information.
            medical_history (Optional[List[str]]): List of relevant medical conditions or history.
            duration_days (Optional[int]): Number of days the symptoms have been present.
            image_path (Optional[str]): Path to medical image if available.
            body_area (Optional[str]): The body area associated with the image if provided.
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - symptom_analysis: Results from symptom-based diagnosis
                - is_urgent: Boolean indicating if the case requires urgent attention
                - patient_info: Sanitized patient information
                - image_analysis: (Optional) Results from image analysis
                - combined_analysis: (Optional) Combined results from both analyses
                - image_analysis_error: (Optional) Error message if image analysis failed
        """
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
            try:
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
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                result["image_analysis_error"] = f"Failed to process image: {str(e)}"
        
        return result

def create_system_analyzer() -> SystemAnalyzer:
    """
    Factory function to create a new SystemAnalyzer instance.
    
    Returns:
        SystemAnalyzer: A new instance of the SystemAnalyzer class with default components.
    """
    return SystemAnalyzer()
