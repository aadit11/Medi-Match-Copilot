import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import re

import ollama  

from core.config import OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

class VisionModelClient:
    def __init__(
        self,
        vision_model: str = OLLAMA_VISION_MODEL
    ):
        self.vision_model = vision_model
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """Analyze image using the vision model (llama3.2-vision)."""
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            if system_prompt:
                full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
            else:
                full_prompt = prompt

            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            response = ollama.generate(
                model=self.vision_model,  
                prompt=full_prompt,
                images=[str(image_path)],
                options=options
            )
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

    def analyze_medical_image(
        self,
        image_path: Union[str, Path],
        image_type: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Analyze a medical image and return structured assessment.
        
        Args:
            image_path: Path to the medical image
            image_type: Type of medical image (e.g., 'X-ray', 'MRI', 'CT scan')
            context: Additional context about the image or patient
            system_prompt: Optional custom system prompt
            temperature: Model temperature for response generation
            
        Returns:
            Dict containing structured medical image analysis
        """
        try:
            if not system_prompt:
                system_prompt = (
                    "You are a medical imaging specialist. Analyze the provided medical image "
                    "and provide a detailed, structured assessment. Focus on identifying "
                    "abnormalities, their significance, and potential implications. "
                    "Be precise in describing locations and findings."
                )

            prompt_parts = [
                f"Analyze this {image_type} image for medical purposes.",
                "Provide a detailed medical assessment including:",
                "1. Specific findings with their locations",
                "2. Significance of each finding (high/medium/low)",
                "3. Confidence level in each finding (0.0-1.0)",
                "4. Overall assessment",
                "5. Clinical recommendations",
                "6. Any limitations or uncertainties in the analysis"
            ]
            
            if context:
                prompt_parts.append("\nContext:")
                for key, value in context.items():
                    prompt_parts.append(f"- {key}: {value}")
                    
            prompt_parts.append(
                "\nFormat your response as a JSON object with the following structure:\n"
                "{\n"
                '  "findings": [\n'
                "    {\n"
                '      "description": "Detailed description of the finding",\n'
                '      "location": "Specific location in the image",\n'
                '      "significance": "high/medium/low",\n'
                '      "confidence": 0.8,\n'
                '      "notes": "Additional clinical notes"\n'
                "    }\n"
                "  ],\n"
                '  "overall_assessment": "Comprehensive assessment of the image",\n'
                '  "recommendations": ["Specific clinical recommendation 1", "Recommendation 2"],\n'
                '  "limitations": ["Limitation 1", "Limitation 2"]\n'
                "}"
            )
            
            prompt = "\n".join(prompt_parts)
            
            response = self.analyze_image(
                image_path=image_path,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            try:
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                
            
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse response as JSON: {e}")
               
                return {
                    "error": "Could not parse model response as structured data",
                    "raw_response": response,
                    "findings": [],
                    "overall_assessment": "Unable to generate structured assessment",
                    "recommendations": ["Please review the raw response for analysis"],
                    "limitations": ["Response format parsing failed"]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            raise

    def compare_images(
        self,
        image_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        try:
            if not prompt:
                prompt = "Compare these medical images and identify any significant changes or differences."
            response_format = {
                "changes": [
                    {
                        "description": "Description of the change",
                        "location": "Where in the images",
                        "significance": "high/medium/low",
                        "confidence": 0.8,
                        "notes": "Additional notes about the change"
                    }
                ],
                "overall_assessment": "Overall assessment of the changes",
                "recommendations": ["Recommendation 1", "Recommendation 2"]
            }
            analyses = []
            for image_path in image_paths:
                analysis = self.analyze_medical_image(
                    image_path=image_path,
                    image_type="medical",
                    system_prompt=system_prompt,
                    temperature=temperature
                )
                analyses.append(analysis)
            combined_analysis = {
                "changes": [],
                "overall_assessment": f"Comparison of {len(image_paths)} images",
                "recommendations": []
            }
            for i, analysis in enumerate(analyses):
                if isinstance(analysis, dict) and "findings" in analysis:
                    for finding in analysis["findings"]:
                        finding["image_index"] = i
                        combined_analysis["changes"].append(finding)
            return combined_analysis
        except Exception as e:
            logger.error(f"Error comparing images: {e}")
            raise

def create_vision_model_client() -> VisionModelClient:
    return VisionModelClient()
