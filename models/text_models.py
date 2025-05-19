import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import re

import ollama  

from core.config import OLLAMA_TEXT_MODEL

logger = logging.getLogger(__name__)

class TextModelClient:
    def __init__(
        self,
        text_model: str = OLLAMA_TEXT_MODEL
    ):
        self.text_model = text_model
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        try:
            # Compose the full prompt
            if system_prompt:
                full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
            else:
                full_prompt = prompt

            # Prepare Ollama options
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            if stop:
                options["stop"] = stop

            # Call Ollama via Python package
            response = ollama.generate(
                model=self.text_model,
                prompt=full_prompt,
                options=options
            )
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_structured_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            if response_format:
                format_prompt = f"\nPlease provide your response in the following JSON format:\n{json.dumps(response_format, indent=2)}\n"
                prompt = prompt + format_prompt
            
            response = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Could not parse response as JSON, returning as text")
                return {"text": response}
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            raise

    def generate_medical_assessment(
        self,
        symptoms: List[str],
        patient_info: Optional[Dict[str, Any]] = None,
        medical_history: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        try:
            prompt_parts = ["Medical assessment for the following symptoms:"]
            for symptom in symptoms:
                prompt_parts.append(f"- {symptom}")
            if patient_info:
                prompt_parts.append("\nPatient Information:")
                for key, value in patient_info.items():
                    prompt_parts.append(f"- {key}: {value}")
            if medical_history:
                prompt_parts.append("\nMedical History:")
                for condition in medical_history:
                    prompt_parts.append(f"- {condition}")
            prompt = "\n".join(prompt_parts)
            response_format = {
                "assessment": "Overall assessment of the case",
                "likely_diagnoses": [
                    {
                        "condition": "Name of condition",
                        "confidence": 0.8,
                        "explanation": "Brief explanation",
                        "recommendations": ["Recommendation 1", "Recommendation 2"]
                    }
                ],
                "urgency_level": "high/medium/low",
                "recommended_next_steps": ["Step 1", "Step 2"],
                "additional_notes": "Any additional important information"
            }
            return self.generate_structured_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                response_format=response_format
            )
        except Exception as e:
            logger.error(f"Error generating medical assessment: {e}")
            raise

    def generate_symptom_questions(
        self,
        symptom: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.5
    ) -> List[Dict[str, Any]]:
        try:
            prompt = f"Generate relevant follow-up questions for the symptom: {symptom}"
            if context:
                prompt += "\nContext:\n"
                for key, value in context.items():
                    prompt += f"- {key}: {value}\n"
            response_format = {
                "questions": [
                    {
                        "question": "The follow-up question",
                        "category": "timing/severity/characteristics/etc",
                        "importance": "high/medium/low",
                        "explanation": "Why this question is important"
                    }
                ]
            }
            response = self.generate_structured_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                response_format=response_format
            )
            return response.get("questions", [])
        except Exception as e:
            logger.error(f"Error generating symptom questions: {e}")
            raise

def create_text_model_client() -> TextModelClient:
    return TextModelClient()
