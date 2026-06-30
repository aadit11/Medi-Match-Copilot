import logging
from typing import Dict, List, Any, Optional
from core.diagnosis_engine import DiagnosisEngine, create_diagnosis_engine
from core.image_processor import ImageProcessor, create_image_processor
from utils.medical_validators import sanitize_patient_data

logger = logging.getLogger(__name__)


class SystemAnalyzer:
    """Coordinates symptom analysis and optional image processing for a medical case."""

    def __init__(
        self,
        diagnosis_engine: Optional[DiagnosisEngine] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        self.diagnosis_engine = diagnosis_engine or create_diagnosis_engine()
        self.image_processor = image_processor or create_image_processor()

    def _build_image_context(self, symptom_analysis: Dict[str, Any]) -> str:
        context_parts = []
        diagnoses = symptom_analysis.get("diagnoses", [])
        if diagnoses:
            top_names = [d.get("name", "Unknown") for d in diagnoses[:3]]
            context_parts.append(f"Top symptom-based diagnoses: {', '.join(top_names)}")
        if symptom_analysis.get("primary_symptom"):
            context_parts.append(f"Primary symptom: {symptom_analysis['primary_symptom']}")
        if symptom_analysis.get("context"):
            context_parts.append(symptom_analysis["context"][:600])
        return "\n".join(context_parts)

    def analyze_case(
        self,
        primary_symptom: str,
        secondary_symptoms: Optional[List[str]] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        medical_history: Optional[List[str]] = None,
        duration_days: Optional[int] = None,
        image_path: Optional[str] = None,
        body_area: Optional[str] = None,
        image_type: Optional[str] = None,
        additional_notes: Optional[str] = None,
        analysis_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        patient_info = sanitize_patient_data(patient_info or {})
        medical_history = medical_history or []
        analysis_params = analysis_params or {}

        symptom_analysis = self.diagnosis_engine.analyze_symptoms(
            primary_symptom=primary_symptom,
            secondary_symptoms=secondary_symptoms,
            patient_info=patient_info,
            medical_history=medical_history,
            duration_days=duration_days,
            additional_notes=additional_notes,
            analysis_params=analysis_params
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
                    body_area=body_area,
                    image_type=image_type,
                    relevant_context=self._build_image_context(symptom_analysis),
                    medical_history=medical_history
                )

                if "error" not in image_analysis:
                    combined_analysis = self.image_processor.combine_with_symptom_data(
                        image_analysis=image_analysis,
                        symptom_data=symptom_analysis,
                        medical_history=medical_history
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
    return SystemAnalyzer()
