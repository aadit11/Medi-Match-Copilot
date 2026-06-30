import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image

from core.config import (
    IMAGE_SIZE,
    IMAGE_CONFIDENCE_THRESHOLD
)
from models.vision_models import VisionModelClient
from models.text_models import TextModelClient
from image_analysis.preprocessor import preprocess_for_analysis
from image_analysis.feature_extractor import extract_features
from image_analysis.condition_classifier import classify_condition
from utils.medical_validators import validate_medical_image
from utils.prompt_engineering import create_image_analysis_prompt

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process and analyze medical images, then combine results with symptom data."""

    def __init__(
        self,
        vision_model: Optional[VisionModelClient] = None,
        text_model: Optional[TextModelClient] = None,
        confidence_threshold: float = IMAGE_CONFIDENCE_THRESHOLD
    ):
        self.vision_model = vision_model or VisionModelClient()
        self.text_model = text_model or TextModelClient()
        self.confidence_threshold = confidence_threshold

    def analyze_image(
        self,
        image_path: str,
        primary_concern: Optional[str] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        body_area: Optional[str] = None,
        image_type: Optional[str] = None,
        relevant_context: Optional[str] = None,
        medical_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        is_valid, error_msg = validate_medical_image(image_path)
        if not is_valid:
            logger.error(f"Invalid medical image: {error_msg}")
            return {"error": error_msg}

        try:
            image = Image.open(image_path)
            processed = preprocess_for_analysis(image, target_size=IMAGE_SIZE)
            image_features = extract_features(
                processed["rgb_uint8"],
                body_area=body_area,
                grayscale=processed["grayscale"],
            )

            classification_results = []
            if image_features is not None:
                classification_results = classify_condition(
                    image_features,
                    body_area=body_area,
                )
                classification_results = [
                    result for result in classification_results
                    if result.get("confidence", 0) >= self.confidence_threshold
                ]

            context = {}
            if patient_info:
                context["patient_info"] = patient_info
            if medical_history:
                context["medical_history"] = medical_history
            if relevant_context:
                context["symptom_context"] = relevant_context
            if classification_results:
                context["preliminary_classifications"] = classification_results
            if image_features and image_features.get("feature_summary"):
                context["visual_feature_summary"] = image_features["feature_summary"]
            if body_area:
                context["body_area"] = body_area
            if primary_concern:
                context["primary_concern"] = primary_concern

            structured = self.vision_model.analyze_medical_image(
                image_path=image_path,
                image_type=image_type or "medical image",
                context=context
            )

            if structured.get("error") and not structured.get("findings"):
                prompt = create_image_analysis_prompt(
                    image_path=image_path,
                    primary_concern=primary_concern or "",
                    patient_info=patient_info,
                    relevant_context=relevant_context or "",
                    body_area=body_area or "",
                    medical_history=medical_history
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
                    "findings": [],
                    "detailed_analysis": detailed_analysis,
                    "overall_assessment": detailed_analysis,
                    "recommendations": [],
                    "limitations": structured.get("limitations", [])
                }

            findings = structured.get("findings", [])
            overall_assessment = structured.get("overall_assessment", "")

            return {
                "image_path": image_path,
                "patient_info": patient_info or {},
                "primary_concern": primary_concern,
                "body_area": body_area,
                "classifications": classification_results,
                "findings": findings,
                "detailed_analysis": overall_assessment,
                "overall_assessment": overall_assessment,
                "recommendations": structured.get("recommendations", []),
                "limitations": structured.get("limitations", [])
            }

        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return {
                "error": "Failed to analyze medical image",
                "details": str(e)
            }

    def _image_findings_indicate_urgency(self, image_analysis: Dict[str, Any]) -> bool:
        for finding in image_analysis.get("findings", []):
            if not isinstance(finding, dict):
                continue
            if str(finding.get("significance", "")).lower() == "high":
                return True
            if finding.get("urgency_required"):
                return True
        return False

    def combine_with_symptom_data(
        self,
        image_analysis: Dict[str, Any],
        symptom_data: Dict[str, Any],
        medical_history: Optional[List[str]] = None
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

        for finding in image_analysis.get("findings", []):
            if not isinstance(finding, dict):
                continue
            description = finding.get("description", "")
            if description:
                image_diagnoses.append({
                    "name": description,
                    "confidence": float(finding.get("confidence", 0.5)),
                    "source": "image_finding"
                })

        symptom_diagnoses = []
        for diagnosis in symptom_data.get("diagnoses", []):
            symptom_diagnoses.append({
                "name": diagnosis.get("name", "Unknown condition"),
                "confidence": diagnosis.get("confidence", 0),
                "source": "symptom_analysis"
            })

        try:
            history_text = ""
            if medical_history:
                history_text = "\nMedical history:\n" + "\n".join(f"- {item}" for item in medical_history)

            system_prompt = """
            You are a medical diagnostic assistant. Create a comprehensive assessment
            based on both image analysis and symptom analysis. Consider how the visual
            evidence and reported symptoms may indicate particular conditions.
            Provide an integrated analysis that considers both sources of information.
            Highlight agreements, contradictions, and what additional workup is needed.
            """

            user_prompt = f"""
            # Image Analysis Results
            Structured findings:
            {json.dumps(image_analysis.get("findings", []), indent=2)}

            Preliminary classifications:
            {json.dumps(image_analysis.get("classifications", []), indent=2)}

            Overall visual assessment:
            {image_analysis.get("overall_assessment", image_analysis.get("detailed_analysis", ""))}

            Image recommendations:
            {json.dumps(image_analysis.get("recommendations", []), indent=2)}

            # Symptom Analysis Results
            {json.dumps(symptom_diagnoses, indent=2)}

            Overall symptom assessment:
            {symptom_data.get("raw_assessment", "")}
            {history_text}

            Based on both the image analysis and symptom assessment, provide an integrated
            diagnostic analysis. Highlight where the image findings confirm or contradict
            the symptom-based diagnoses.
            """

            combined_assessment = self.text_model.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.25
            )

            return {
                "image_diagnoses": image_diagnoses,
                "symptom_diagnoses": symptom_diagnoses,
                "combined_assessment": combined_assessment,
                "is_urgent": (
                    symptom_data.get("is_urgent", False)
                    or self._image_findings_indicate_urgency(image_analysis)
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
    return ImageProcessor(
        vision_model=VisionModelClient(),
        text_model=TextModelClient(),
        confidence_threshold=IMAGE_CONFIDENCE_THRESHOLD
    )
