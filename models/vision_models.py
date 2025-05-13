import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import base64
import requests

from core.config import OLLAMA_HOST, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

class VisionModelClient:
    def __init__(
        self,
        host: str = OLLAMA_HOST,
        vision_model: str = OLLAMA_VISION_MODEL
    ):
        self.host = host.rstrip('/')
        self.vision_model = vision_model
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_base64
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            })
            
            payload = {
                "model": self.vision_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
        
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
        try:
            prompt_parts = [f"Analyze this {image_type} image for medical purposes."]
            
            if context:
                prompt_parts.append("\nContext:")
                for key, value in context.items():
                    prompt_parts.append(f"- {key}: {value}")
            
            prompt = "\n".join(prompt_parts)
            
            response_format = {
                "findings": [
                    {
                        "description": "Description of the finding",
                        "location": "Where in the image",
                        "significance": "high/medium/low",
                        "confidence": 0.8,
                        "notes": "Additional notes about the finding"
                    }
                ],
                "overall_assessment": "Overall assessment of the image",
                "recommendations": ["Recommendation 1", "Recommendation 2"],
                "limitations": ["Limitation 1", "Limitation 2"]
            }
            
            response = self.analyze_image(
                image_path=image_path,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Could not parse response as JSON, returning as text")
                return {"text": response}
        
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
