import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.config import (
    BODY_AREA_CONDITION_KEYWORDS,
    CLASSIFICATION_MIN_CONFIDENCE,
    CONDITIONS_DB_PATH,
)
from models.text_models import TextModelClient

logger = logging.getLogger(__name__)

_default_classifier: Optional["ConditionClassifier"] = None


class ConditionClassifier:
    """Classify medical conditions from image features using local KB + structured LLM output."""

    def __init__(
        self,
        model_client: Optional[TextModelClient] = None,
        conditions_db_path: Optional[Path] = None,
    ):
        self.model_client = model_client or TextModelClient()
        self.conditions_db_path = conditions_db_path or CONDITIONS_DB_PATH
        self.conditions_db = self._load_conditions_database()

    def _load_conditions_database(self) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        try:
            if self.conditions_db_path.exists():
                with open(self.conditions_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading conditions database: {e}")
            return []

    def _get_body_area_candidates(self, body_area: Optional[str]) -> List[Dict[str, Any]]:
        if not body_area:
            return []

        keywords = BODY_AREA_CONDITION_KEYWORDS.get(
            body_area.lower(),
            [body_area.lower()],
        )
        entries = self.conditions_db if isinstance(self.conditions_db, list) else []
        candidates = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            haystack = f"{entry.get('title', '')} {entry.get('content', '')}".lower()
            if any(keyword in haystack for keyword in keywords):
                candidates.append(entry)

        return candidates[:8]

    def _heuristic_candidates(
        self,
        image_features: Dict[str, Any],
        body_area: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Rule-based preliminary candidates from measurable image features."""
        candidates: List[Dict[str, Any]] = []
        circularity = image_features.get("circularity", 1.0)
        color_diversity = image_features.get("color_diversity", 0.0)
        red_mean = image_features.get("red_mean", 0.0)
        area = image_features.get("area", 0.0)

        if body_area and body_area.lower() == "foot":
            if circularity < 0.45 and red_mean > 110:
                candidates.append({
                    "condition": "Diabetic foot ulcer",
                    "confidence": 0.55,
                    "explanation": "Irregular lesion shape with inflamed tissue tones on the foot.",
                    "source": "heuristic",
                })
            if area > 5000 and color_diversity > 3.0:
                candidates.append({
                    "condition": "Wound infection",
                    "confidence": 0.5,
                    "explanation": "Large irregular region with heterogeneous color patterns.",
                    "source": "heuristic",
                })

        if body_area and body_area.lower() == "skin":
            if circularity < 0.35 and color_diversity > 3.5:
                candidates.append({
                    "condition": "Suspicious skin lesion",
                    "confidence": 0.5,
                    "explanation": "Asymmetric region with varied pigmentation.",
                    "source": "heuristic",
                })

        return candidates

    def classify_condition(
        self,
        image_features: Dict[str, Any],
        body_area: Optional[str] = None,
        min_confidence: float = CLASSIFICATION_MIN_CONFIDENCE,
    ) -> List[Dict[str, Any]]:
        if not image_features:
            return []

        try:
            heuristic_results = self._heuristic_candidates(image_features, body_area)
            local_candidates = self._get_body_area_candidates(body_area)
            feature_desc = self._format_features(image_features)
            prompt = self._create_classification_prompt(
                feature_desc=feature_desc,
                body_area=body_area,
                local_candidates=local_candidates,
                feature_summary=image_features.get("feature_summary", ""),
            )

            response_format = {
                "classifications": [
                    {
                        "condition": "Condition name",
                        "confidence": 0.7,
                        "explanation": "Why the visual features support this condition",
                    }
                ]
            }

            structured = self.model_client.generate_structured_response(
                system_prompt=self._get_system_prompt(),
                prompt=prompt,
                temperature=0.2,
                response_format=response_format,
            )

            llm_classifications = structured.get("classifications", [])
            if not llm_classifications:
                llm_classifications = self._parse_classifications_from_text(
                    structured.get("text", json.dumps(structured, ensure_ascii=False))
                )

            merged = self._merge_classifications(heuristic_results, llm_classifications)
            return [
                item for item in merged
                if item.get("confidence", 0) >= min_confidence
            ][:5]

        except Exception as e:
            logger.error(f"Error classifying conditions: {e}")
            return self._heuristic_candidates(image_features, body_area)

    def _merge_classifications(
        self,
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        for item in primary + secondary:
            name = str(item.get("condition", "")).strip()
            if not name:
                continue
            key = name.lower()
            confidence = float(item.get("confidence", 0))
            if key not in merged or confidence > merged[key].get("confidence", 0):
                merged[key] = {
                    "condition": name,
                    "confidence": confidence,
                    "explanation": item.get("explanation", ""),
                    "source": item.get("source", "model"),
                }

        return sorted(merged.values(), key=lambda x: x.get("confidence", 0), reverse=True)

    def _format_features(self, features: Dict[str, Any]) -> str:
        feature_parts = []

        summary = features.get("feature_summary")
        if summary:
            feature_parts.append(f"Visual summary: {summary}\n")

        if any(k.startswith("texture_") for k in features):
            texture_desc = "Texture characteristics:\n"
            for key, value in features.items():
                if key.startswith("texture_"):
                    texture_desc += f"- {key.replace('texture_', '')}: {float(value):.3f}\n"
            feature_parts.append(texture_desc)

        shape_keys = ["area", "perimeter", "circularity", "aspect_ratio"]
        if any(key in features for key in shape_keys):
            shape_desc = "Shape characteristics:\n"
            for key in shape_keys:
                if key in features:
                    shape_desc += f"- {key}: {float(features[key]):.3f}\n"
            feature_parts.append(shape_desc)

        if any(k.startswith(("red_", "green_", "blue_")) for k in features):
            color_desc = "Color characteristics:\n"
            for color in ["red", "green", "blue"]:
                mean_key = f"{color}_mean"
                std_key = f"{color}_std"
                if mean_key in features:
                    color_desc += (
                        f"- {color} channel: mean={float(features[mean_key]):.3f}, "
                        f"std={float(features.get(std_key, 0)):.3f}\n"
                    )
            feature_parts.append(color_desc)

        return "\n".join(feature_parts)

    def _create_classification_prompt(
        self,
        feature_desc: str,
        body_area: Optional[str],
        local_candidates: List[Dict[str, Any]],
        feature_summary: str,
    ) -> str:
        prompt = (
            "Based on the following image features, identify the most likely medical conditions.\n\n"
            f"{feature_desc}\n"
        )

        if body_area:
            prompt += f"Body area: {body_area}\n"

        if feature_summary:
            prompt += f"\nAutomated visual summary: {feature_summary}\n"

        if local_candidates:
            prompt += "\nRelevant conditions from the medical knowledge base for this body area:\n"
            for entry in local_candidates:
                prompt += f"- {entry.get('title', 'Unknown')}: {entry.get('content', '')[:220]}...\n"

        prompt += (
            "\nReturn up to 5 conditions ranked by likelihood. "
            "Prefer conditions consistent with both the measured features and the body area context. "
            "Use confidence values between 0.0 and 1.0."
        )
        return prompt

    def _get_system_prompt(self) -> str:
        return """
        You are a medical image analysis expert. Analyze the provided quantitative image features
        and identify possible medical conditions. Consider visual characteristics and body area context.
        Base confidence on how strongly the numeric features support each condition.
        Do not invent conditions unrelated to the provided features or body area.
        """

    def _parse_classifications_from_text(self, response: str) -> List[Dict[str, Any]]:
        try:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict) and "classifications" in parsed:
                    return parsed["classifications"]

            json_match = re.search(r"(\[\s*\{.*?\}\s*\])", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Could not parse classifications from text: {e}")
        return []


def create_condition_classifier() -> ConditionClassifier:
    return ConditionClassifier()


def classify_condition(
    image_features: Dict[str, Any],
    body_area: Optional[str] = None,
    min_confidence: float = CLASSIFICATION_MIN_CONFIDENCE,
) -> List[Dict[str, Any]]:
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = create_condition_classifier()
    return _default_classifier.classify_condition(
        image_features=image_features,
        body_area=body_area,
        min_confidence=min_confidence,
    )
