import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import re

import ollama  

from core.config import OLLAMA_TEXT_MODEL

logger = logging.getLogger(__name__)

class TextModelClient:
    """A client for interacting with text generation models, specifically Ollama models.
    
    This class provides a high-level interface for generating text and structured responses
    using AI language models. It supports various text generation tasks including medical
    assessments, symptom analysis, and general text generation with configurable parameters.
    
    Attributes:
        text_model (str): The name of the text model to use for generation.
    """
    
    def __init__(
        self,
        text_model: str = OLLAMA_TEXT_MODEL
    ):
        """Initialize the TextModelClient.
        
        Args:
            text_model (str): The name of the text model to use. Defaults to OLLAMA_TEXT_MODEL
                from core config.
        """
        self.text_model = text_model
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate text using the configured language model.
        
        This method handles basic text generation with optional system prompts and
        configurable generation parameters. It combines the system prompt and user prompt
        if both are provided.
        
        Args:
            prompt (str): The main prompt/query for text generation.
            system_prompt (Optional[str]): Optional system-level instructions or context.
            temperature (float): Controls randomness in generation (0.0 to 1.0). Defaults to 0.7.
            max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to None.
            stop (Optional[List[str]]): List of strings that will stop generation if encountered.
                Defaults to None.
                
        Returns:
            str: The generated text response.
            
        Raises:
            Exception: If there's an error during text generation.
        """
        try:
            if system_prompt:
                full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
            else:
                full_prompt = prompt

            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            if stop:
                options["stop"] = stop

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
        """Generate a structured (JSON) response from the language model.
        
        This method is designed to generate responses in a specific JSON format.
        It can handle both direct JSON responses and responses wrapped in markdown code blocks.
        
        Args:
            prompt (str): The main prompt/query for text generation.
            system_prompt (Optional[str]): Optional system-level instructions or context.
            temperature (float): Controls randomness in generation (0.0 to 1.0). Defaults to 0.3.
            max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to None.
            response_format (Optional[Dict[str, Any]]): Optional schema defining the expected
                JSON response format.
                
        Returns:
            Dict[str, Any]: The structured response as a dictionary. If JSON parsing fails,
                returns a dictionary with a "text" key containing the raw response.
                
        Raises:
            Exception: If there's an error during response generation or parsing.
        """
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
        """Generate a structured medical assessment based on symptoms and patient information.
        
        This method creates a comprehensive medical assessment including potential diagnoses,
        confidence levels, explanations, and recommendations. It formats the input data
        and uses the language model to generate a structured medical evaluation.
        
        Args:
            symptoms (List[str]): List of reported symptoms.
            patient_info (Optional[Dict[str, Any]]): Optional dictionary containing patient
                demographics and information.
            medical_history (Optional[List[str]]): Optional list of previous medical conditions.
            system_prompt (Optional[str]): Optional system-level instructions for the assessment.
            temperature (float): Controls randomness in generation (0.0 to 1.0). Defaults to 0.3.
                
        Returns:
            Dict[str, Any]: A structured assessment containing:
                - assessment: Overall assessment of the case
                - likely_diagnoses: List of potential diagnoses with confidence levels
                - urgency_level: Urgency classification (high/medium/low)
                - recommended_next_steps: List of recommended actions
                - additional_notes: Any additional important information
                
        Raises:
            Exception: If there's an error during assessment generation.
        """
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
        """Generate relevant follow-up questions for a given symptom.
        
        This method creates a structured set of follow-up questions to gather more
        detailed information about a reported symptom. Questions are categorized and
        include importance levels and explanations.
        
        Args:
            symptom (str): The symptom to generate questions for.
            context (Optional[Dict[str, Any]]): Optional additional context about the symptom
                or patient situation.
            system_prompt (Optional[str]): Optional system-level instructions for question
                generation.
            temperature (float): Controls randomness in generation (0.0 to 1.0). Defaults to 0.5.
                
        Returns:
            List[Dict[str, Any]]: List of question dictionaries, each containing:
                - question: The follow-up question
                - category: Question category (timing/severity/characteristics/etc)
                - importance: Importance level (high/medium/low)
                - explanation: Why this question is important
                
        Raises:
            Exception: If there's an error during question generation.
        """
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
    """Create and return a new TextModelClient instance.
    
    This factory function creates a TextModelClient with default configuration
    from the system settings.
    
    Returns:
        TextModelClient: A configured TextModelClient instance.
    """
    return TextModelClient()
