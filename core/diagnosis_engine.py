import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from core.config import (
    MAX_DIAGNOSES,
    MIN_CONFIDENCE_THRESHOLD,
    INCLUDE_DETAILED_EXPLANATIONS,
    CONDITIONS_DB_PATH,
    SYMPTOMS_DB_PATH
)
from retrieval.query_engine import QueryEngine, create_query_engine
from utils.medical_validators import validate_symptoms, is_urgent_symptom
from utils.prompt_engineering import create_diagnosis_prompt, format_diagnosis_results
from models.text_models import TextModelClient

logger = logging.getLogger(__name__)

class DiagnosisEngine:
    """A medical diagnosis engine that analyzes symptoms and provides potential diagnoses.
    
    This class combines medical knowledge retrieval, symptom analysis, and AI-powered
    diagnostic assessment to provide comprehensive medical evaluations. It can handle
    both primary and secondary symptoms, patient information, and medical history to
    generate detailed diagnostic reports.
    """
    
    def __init__(
        self,
        query_engine: Optional[QueryEngine] = None,
        model_client: Optional[TextModelClient] = None,
        max_diagnoses: int = MAX_DIAGNOSES,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
        detailed_explanations: bool = INCLUDE_DETAILED_EXPLANATIONS
    ):
        """Initialize the DiagnosisEngine with required components.
        
        Args:
            query_engine: Optional QueryEngine instance for medical knowledge retrieval
            model_client: Optional TextModelClient for AI-powered text generation
            max_diagnoses: Maximum number of diagnoses to return (default from config)
            min_confidence: Minimum confidence threshold for diagnoses (default from config)
            detailed_explanations: Whether to include detailed explanations in results
        """
        
        self.query_engine = query_engine or create_query_engine()
        self.model_client = model_client or TextModelClient()
        self.max_diagnoses = max_diagnoses
        self.min_confidence = min_confidence
        self.detailed_explanations = detailed_explanations
        
        self.symptoms_db = self._load_json_database(SYMPTOMS_DB_PATH)
        self.conditions_db = self._load_json_database(CONDITIONS_DB_PATH)
    
    def _load_json_database(self, path: Path) -> Dict[str, Any]:
        """Load a JSON database file from the specified path.
        
        Args:
            path: Path to the JSON database file
            
        Returns:
            Dict containing the database contents, or empty dict if loading fails
        """
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading database from {path}: {e}")
        
        return {}
    
    def analyze_symptoms(
        self, 
        primary_symptom: str,
        secondary_symptoms: Optional[List[str]] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        medical_history: Optional[List[str]] = None,
        duration_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze symptoms and generate potential diagnoses.
        
        This method processes primary and secondary symptoms, validates them,
        checks for urgent conditions, retrieves relevant medical knowledge,
        and generates a comprehensive diagnostic assessment.
        
        Args:
            primary_symptom: The main symptom to analyze
            secondary_symptoms: Optional list of additional symptoms
            patient_info: Optional dictionary containing patient demographics
            medical_history: Optional list of previous medical conditions
            duration_days: Optional number of days the symptoms have persisted
            
        Returns:
            Dict containing:
                - diagnoses: List of potential diagnoses with confidence levels
                - is_urgent: Boolean indicating if urgent care is needed
                - urgent_symptoms: List of urgent symptoms if any
                - formatted_report: Human-readable diagnostic report
                - Additional metadata and context
        """
        
        logger.info(f"Analyzing symptoms. Primary: {primary_symptom}")
        
        valid_primary, _ = validate_symptoms([primary_symptom])
        if not valid_primary:
            return {"error": "Invalid primary symptom provided"}
        
        primary_symptom = valid_primary[0]
        
        if secondary_symptoms:
            valid_secondary, rejected = validate_symptoms(secondary_symptoms)
            secondary_symptoms = valid_secondary
            
            if rejected:
                logger.warning(f"Rejected {len(rejected)} invalid secondary symptoms")
        else:
            secondary_symptoms = []
        
        all_symptoms = [primary_symptom] + (secondary_symptoms or [])
        is_urgent, urgent_symptoms = is_urgent_symptom(all_symptoms)
        
        if is_urgent:
            logger.warning(f"Urgent symptoms detected: {urgent_symptoms}")
        
        try:
            results = self.query_engine.retrieve_for_diagnosis(
                primary_symptom=primary_symptom,
                secondary_symptoms=secondary_symptoms,
                patient_info=patient_info
            )
            
            knowledge_items = self.query_engine.extract_relevant_knowledge(results)
            context_text = self.query_engine.format_for_diagnosis(results)
            
        except Exception as e:
            logger.error(f"Error retrieving medical knowledge: {e}")
            knowledge_items = []
            context_text = ""
        
        prompt_data = create_diagnosis_prompt(
            primary_symptom=primary_symptom,
            secondary_symptoms=secondary_symptoms,
            patient_info=patient_info,
            medical_history=medical_history,
            duration_days=duration_days,
            relevant_medical_knowledge=knowledge_items
        )
        
        try:
            response = self.model_client.generate_text(
                system_prompt=prompt_data["system"],
                prompt=prompt_data["user"]
            )
        except Exception as e:
            logger.error(f"Error generating diagnostic assessment: {e}")
            return {
                "error": "Failed to generate diagnostic assessment",
                "details": str(e)
            }
        
        diagnoses = self._extract_diagnoses(response)
        
        filtered_diagnoses = [
            d for d in diagnoses 
            if d.get("confidence", 0) >= self.min_confidence
        ]
        
        limited_diagnoses = filtered_diagnoses[:self.max_diagnoses]
        
        result = {
            "primary_symptom": primary_symptom,
            "secondary_symptoms": secondary_symptoms,
            "diagnoses": limited_diagnoses,
            "is_urgent": is_urgent,
            "urgent_symptoms": urgent_symptoms if is_urgent else [],
            "raw_assessment": response if self.detailed_explanations else "",
            "context": context_text if self.detailed_explanations else "",
            "duration_days": duration_days
        }
        
        if patient_info:
            result["patient_info"] = patient_info
        
        result["formatted_report"] = format_diagnosis_results(
            conditions=limited_diagnoses,
            patient_info=patient_info,
            detailed=self.detailed_explanations
        )
        
        return result
    
    def _extract_diagnoses(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured diagnoses from the model's assessment text.
        
        This method parses the AI model's response to extract diagnoses,
        confidence levels, explanations, and recommendations. It handles
        both structured JSON responses and unstructured text.
        
        Args:
            text: The assessment text from the AI model
            
        Returns:
            List of dictionaries containing:
                - name: Diagnosis name
                - confidence: Confidence level (0-1)
                - explanation: Reasoning for the diagnosis
                - recommendations: List of recommended actions
        """
        
        diagnoses = []
        
        try:
            system_prompt = """
            Extract the diagnoses from the medical assessment text.
            For each diagnosis, extract:
            1. The name of the condition
            2. The confidence level (convert percentages to decimal: 80% = 0.8)
            3. The explanation/reasoning
            4. Any recommendations provided
            
            Return the data in this JSON format:
            [
                {
                    "name": "Condition name",
                    "confidence": 0.8,
                    "explanation": "Explanation text",
                    "recommendations": ["Recommendation 1", "Recommendation 2"]
                }
            ]
            """
            
            extraction_result = self.model_client.generate_text(
                system_prompt=system_prompt,
                prompt=f"Extract diagnoses from this assessment:\n\n{text}"
            )
            
            import re
            json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", extraction_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                diagnoses = json.loads(json_str)
            else:
                json_match = re.search(r"(\[\s*\{.*?\}\s*\])", extraction_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    diagnoses = json.loads(json_str)
                else:
                    logger.warning("Could not extract structured diagnoses from model response")
        
        except Exception as e:
            logger.error(f"Error extracting diagnoses: {e}")
            
            lines = text.split('\n')
            current_diagnosis = None
            
            for line in lines:
                line = line.strip()
                
                if ':' in line and ('likely' in line.lower() or '%' in line or 'probability' in line.lower()):
                    parts = line.split(':', 1)
                    name = parts[0].strip().strip('1234567890.-)( ')
                    
                    if len(name) < 3 or name.lower() in ['note', 'recommendation', 'summary']:
                        continue
                    
                    current_diagnosis = {
                        "name": name,
                        "confidence": self._extract_confidence(line),
                        "explanation": "",
                        "recommendations": []
                    }
                    diagnoses.append(current_diagnosis)
                
                elif current_diagnosis and line and not line.startswith('#') and not line.lower().startswith('recommendation'):
                    if current_diagnosis["explanation"]:
                        current_diagnosis["explanation"] += " " + line
                    else:
                        current_diagnosis["explanation"] = line
                
                elif current_diagnosis and line and ('recommend' in line.lower() or line.lower().startswith('- ')):
                    current_diagnosis["recommendations"].append(line.strip('- '))
        
        return diagnoses
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from text using various patterns.
        
        This method looks for confidence indicators in text including:
        - Percentage values (e.g., "80%")
        - Decimal probabilities (e.g., "probability: 0.8")
        - Confidence terms (e.g., "high", "likely", "possible")
        
        Args:
            text: Text containing confidence indicators
            
        Returns:
            Float between 0 and 1 representing the confidence level
        """
        
        percent_match = re.search(r"(\d{1,3})%", text)
        if percent_match:
            return float(percent_match.group(1)) / 100
        
        decimal_match = re.search(r"probability:?\s*([0]?\.[0-9]+)", text, re.IGNORECASE)
        if decimal_match:
            return float(decimal_match.group(1))
        
        confidence_terms = {
            "high": 0.8,
            "moderate": 0.6,
            "likely": 0.7,
            "possible": 0.5,
            "unlikely": 0.3,
            "low": 0.4,
            "very likely": 0.85,
            "most likely": 0.9,
            "almost certain": 0.95,
            "certain": 1.0,
            "suspected": 0.65
        }
        
        for term, value in confidence_terms.items():
            if term in text.lower():
                return value
                
        return 0.5
    
    def get_condition_details(self, condition_name: str) -> Dict[str, Any]:
        """Retrieve detailed information about a medical condition.
        
        This method first checks the local conditions database, then falls back
        to querying the knowledge base if needed. It provides structured
        information about symptoms, causes, treatments, and prognosis.
        
        Args:
            condition_name: Name of the medical condition
            
        Returns:
            Dict containing:
                - name: Condition name
                - information: Detailed description of the condition
                - Additional metadata if available
        """
        
        condition_key = condition_name.lower().strip()
        
        if condition_key in self.conditions_db:
            return self.conditions_db[condition_key]
        
        try:
            results = self.query_engine.retrieve_for_condition(condition_name)
            knowledge_items = self.query_engine.extract_relevant_knowledge(results)
            
            if not knowledge_items:
                return {"name": condition_name, "information": "No detailed information available"}
            
            system_prompt = f"""
            You are a medical information system. Create a structured summary of {condition_name}
            based on the provided information. Include symptoms, causes, treatments, and prognosis
            if that information is available.
            """
            
            user_prompt = f"Information about {condition_name}:\n\n" + "\n\n".join(knowledge_items)
            
            response = self.model_client.generate_text(
                system_prompt=system_prompt,
                prompt=user_prompt
            )
            
            return {
                "name": condition_name,
                "information": response
            }
            
        except Exception as e:
            logger.error(f"Error retrieving condition details for {condition_name}: {e}")
            return {
                "name": condition_name,
                "information": "Error retrieving condition details"
            }
    
    def get_symptom_guidance(self, symptom: str) -> Dict[str, Any]:
        """Generate guidance for exploring a symptom in detail.
        
        This method provides structured questions and guidance for
        healthcare providers to gather more information about a symptom.
        It helps in symptom exploration and differential diagnosis.
        
        Args:
            symptom: The symptom to generate guidance for
            
        Returns:
            Dict containing:
                - symptom: The input symptom
                - exploration_guidance: Structured guidance and questions
                - Additional metadata if available
        """
        
        valid_symptoms, _ = validate_symptoms([symptom])
        if not valid_symptoms:
            return {"error": "Invalid symptom provided"}
        
        symptom = valid_symptoms[0]
        
        symptom_key = symptom.lower().strip()
        
        if symptom_key in self.symptoms_db:
            return self.symptoms_db[symptom_key]
        
        try:
            from utils.prompt_engineering import create_symptom_exploration_prompt
            
            prompt_data = create_symptom_exploration_prompt(symptom)
            
            response = self.model_client.generate_text(
                system_prompt=prompt_data["system"],
                prompt=prompt_data["user"]
            )
            
            return {
                "symptom": symptom,
                "exploration_guidance": response
            }
            
        except Exception as e:
            logger.error(f"Error generating symptom guidance for {symptom}: {e}")
            return {
                "symptom": symptom,
                "error": "Failed to generate symptom guidance"
            }
def create_diagnosis_engine() -> DiagnosisEngine:
    """Create and configure a new DiagnosisEngine instance.
    
    This factory function creates a DiagnosisEngine with default
    configuration from the system settings. It initializes the
    query engine and model client with standard parameters.
    
    Returns:
        A configured DiagnosisEngine instance
    """
    
    query_engine = create_query_engine()
    model_client = TextModelClient()
    
    return DiagnosisEngine(
        query_engine=query_engine,
        model_client=model_client,
        max_diagnoses=MAX_DIAGNOSES,
        min_confidence=MIN_CONFIDENCE_THRESHOLD,
        detailed_explanations=INCLUDE_DETAILED_EXPLANATIONS
    )