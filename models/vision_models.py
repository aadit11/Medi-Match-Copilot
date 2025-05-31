import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import re

import ollama  

from core.config import OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

class VisionModelClient:
    """A client for interacting with vision models, particularly for medical image analysis.
    
    This class provides methods to analyze images using vision models, with specialized
    functionality for medical image analysis and comparison. It uses the Ollama API
    for model inference.

    Attributes:
        vision_model (str): The name of the vision model to use for analysis.
    """

    def __init__(
        self,
        vision_model: str = OLLAMA_VISION_MODEL
    ):
        """Initialize the VisionModelClient.

        Args:
            vision_model (str, optional): The name of the vision model to use.
                Defaults to OLLAMA_VISION_MODEL from config.
        """
        self.vision_model = vision_model
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """Analyze an image using the vision model.

        Args:
            image_path (Union[str, Path]): Path to the image file to analyze.
            prompt (str): The prompt to guide the image analysis.
            system_prompt (Optional[str], optional): System prompt to set context.
                Defaults to None.
            temperature (float, optional): Controls randomness in the output.
                Higher values make output more random. Defaults to 0.3.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate.
                Defaults to None.

        Returns:
            str: The model's analysis of the image.

        Raises:
            FileNotFoundError: If the image file does not exist.
            Exception: For other errors during analysis.
        """
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
        
        This method provides a specialized analysis of medical images, returning
        structured data about findings, their significance, and recommendations.

        Args:
            image_path (Union[str, Path]): Path to the medical image.
            image_type (str): Type of medical image (e.g., 'X-ray', 'MRI', 'CT scan').
            context (Optional[Dict[str, Any]], optional): Additional context about
                the image or patient. Defaults to None.
            system_prompt (Optional[str], optional): Custom system prompt to override
                the default medical imaging specialist prompt. Defaults to None.
            temperature (float, optional): Controls randomness in the output.
                Defaults to 0.3.
            
        Returns:
            Dict[str, Any]: A structured dictionary containing:
                - findings: List of identified medical findings
                - overall_assessment: Comprehensive assessment
                - recommendations: List of clinical recommendations
                - limitations: List of analysis limitations
                - error: Error message if analysis failed
        """
        try:
            if not system_prompt:
                system_prompt = (
                    "You are a medical imaging specialist. Analyze the provided medical image "
                    "and provide a detailed, structured assessment. Focus on identifying "
                    "abnormalities, their significance, and potential implications. "
                    "Be precise in describing locations and findings. "
                    "IMPORTANT: Your response must be a valid JSON object following the exact structure specified. "
                    "Do not include any text before or after the JSON object."
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
                "\nIMPORTANT: Your response must be a valid JSON object with this exact structure:\n"
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
            
            parsed_response = None
            json_str = None
            
            json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed_response = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            if not parsed_response:
                json_match = re.search(r"(\{[\s\S]*?\})", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        parsed_response = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            if not parsed_response:
                try:
                    parsed_response = json.loads(response)
                except json.JSONDecodeError:
                    pass
            
            if parsed_response:
                normalized_response = {
                    "findings": [],
                    "overall_assessment": "",
                    "recommendations": [],
                    "limitations": []
                }
                
                if isinstance(parsed_response, dict):
                    if "findings" in parsed_response and isinstance(parsed_response["findings"], list):
                        normalized_response["findings"] = [
                            {
                                "description": str(f.get("description", "")),
                                "location": str(f.get("location", "")),
                                "significance": str(f.get("significance", "medium")).lower(),
                                "confidence": float(f.get("confidence", 0.5)),
                                "notes": str(f.get("notes", ""))
                            }
                            for f in parsed_response["findings"]
                            if isinstance(f, dict)
                        ]
                    
                    normalized_response["overall_assessment"] = str(parsed_response.get("overall_assessment", ""))
                    
                    if "recommendations" in parsed_response and isinstance(parsed_response["recommendations"], list):
                        normalized_response["recommendations"] = [
                            str(r) for r in parsed_response["recommendations"]
                            if isinstance(r, (str, int, float))
                        ]
                    
                    if "limitations" in parsed_response and isinstance(parsed_response["limitations"], list):
                        normalized_response["limitations"] = [
                            str(l) for l in parsed_response["limitations"]
                            if isinstance(l, (str, int, float))
                        ]
                
                return normalized_response
            
            logger.warning("Could not parse model response as structured data")
            return {
                "error": "Could not parse model response as structured data",
                "raw_response": response,
                "findings": [],
                "overall_assessment": "Unable to generate structured assessment",
                "recommendations": ["Please review the raw response for analysis"],
                "limitations": ["Response format parsing failed"],
                "parsing_attempts": {
                    "code_block_json": bool(json_match and json_str),
                    "raw_json": bool(not json_match and json_str),
                    "full_response": bool(not json_str)
                }
            }
                
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return {
                "error": f"Error during image analysis: {str(e)}",
                "findings": [],
                "overall_assessment": "Analysis failed due to an error",
                "recommendations": ["Please try again or contact support"],
                "limitations": ["Technical error occurred during analysis"]
            }

    def compare_images(
        self,
        image_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Compare multiple medical images and identify changes between them.

        This method analyzes multiple images and provides a structured comparison
        of findings across the images, highlighting changes and their significance.

        Args:
            image_paths (List[Union[str, Path]]): List of paths to the images to compare.
            prompt (Optional[str], optional): Custom prompt for the comparison.
                Defaults to a standard medical image comparison prompt.
            system_prompt (Optional[str], optional): Custom system prompt.
                Defaults to None.
            temperature (float, optional): Controls randomness in the output.
                Defaults to 0.3.

        Returns:
            Dict[str, Any]: A structured dictionary containing:
                - changes: List of identified changes between images
                - overall_assessment: Summary of the comparison
                - recommendations: List of recommendations based on the comparison

        Raises:
            Exception: If there's an error during the comparison process.
        """
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

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using the vision model without image input.

        This method uses the same model as image analysis but for pure text generation.
        Useful for generating reports or summaries based on previous analyses.

        Args:
            prompt (str): The prompt to guide text generation.
            system_prompt (Optional[str], optional): System prompt to set context.
                Defaults to None.
            temperature (float, optional): Controls randomness in the output.
                Higher values make output more random. Defaults to 0.3.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate.
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

            response = ollama.generate(
                model=self.vision_model,
                prompt=full_prompt,
                options=options
            )
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

def create_vision_model_client() -> VisionModelClient:
    """Create and return a new instance of VisionModelClient.

    This factory function provides a convenient way to create a VisionModelClient
    instance with default settings.

    Returns:
        VisionModelClient: A new instance of the VisionModelClient class.
    """
    return VisionModelClient()
