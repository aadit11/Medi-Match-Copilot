import logging
import json
import sys
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from core.system_analyzer import SystemAnalyzer, create_system_analyzer
from core.image_processor import ImageProcessor, create_image_processor
from utils.medical_validators import sanitize_patient_data
from retrieval.indexing import create_indexer
from core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_VISION_MODEL,
    MODEL_INIT_TIMEOUT,
    MODEL_RETRY_ATTEMPTS,
    VECTOR_DB_INIT_TIMEOUT,
    get_output_config,
    LOG_FILE,
    get_patient_defaults
)

output_config = get_output_config()

logging.basicConfig(
    level=logging.INFO,
    format=output_config["log_format"],
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_ollama_server() -> bool:
    """Check if Ollama server is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error(f"Ollama server not accessible: {e}")
        return False

def wait_for_ollama_model(model_name: str, timeout: int = MODEL_INIT_TIMEOUT) -> bool:
    """Wait for Ollama model to be available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"Model {model_name} is available")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    logger.error(f"Timeout waiting for model {model_name}")
    return False

def initialize_models() -> bool:
    """Initialize all required models with retries."""
    if not check_ollama_server():
        logger.error("Ollama server is not running. Please start the Ollama server.")
        return False
    
    for attempt in range(MODEL_RETRY_ATTEMPTS):
        logger.info(f"Initializing models (attempt {attempt + 1}/{MODEL_RETRY_ATTEMPTS})...")
        
        if not wait_for_ollama_model(OLLAMA_TEXT_MODEL):
            logger.error(f"Failed to initialize text model {OLLAMA_TEXT_MODEL}")
            if attempt == MODEL_RETRY_ATTEMPTS - 1:
                return False
            continue
        
        if not wait_for_ollama_model(OLLAMA_VISION_MODEL):
            logger.error(f"Failed to initialize vision model {OLLAMA_VISION_MODEL}")
            if attempt == MODEL_RETRY_ATTEMPTS - 1:
                return False
            continue
        
        logger.info("All models initialized successfully")
        return True
    
    return False

def format_assessment_result(result: Dict[str, Any]) -> str:
    """Format the assessment result into a readable text format."""
    output = []
    sections = output_config["assessment_sections"]
    date_format = output_config["assessment_date_format"]
    
    output.append(f"Assessment Date: {datetime.now().strftime(date_format)}\n")
    
    if "patient_info" in result:
        output.append(sections["patient_info"])
        patient = result["patient_info"]
        output.append(f"Name: {patient.get('first_name', '')} {patient.get('last_name', '')}")
        output.append(f"Age: {patient.get('age', 'N/A')}")
        output.append(f"Gender: {patient.get('gender', 'N/A')}")
        output.append(f"Blood Type: {patient.get('blood_type', 'N/A')}")
        output.append("")
    
    symptom_analysis = result.get("symptom_analysis", {})
    
    if "primary_symptom" in symptom_analysis or "secondary_symptoms" in symptom_analysis:
        output.append(sections["symptoms"])
        if "primary_symptom" in symptom_analysis:
            output.append(f"Primary Symptom: {symptom_analysis['primary_symptom']}")
        if symptom_analysis.get("secondary_symptoms"):
            output.append("Secondary Symptoms:")
            for symptom in symptom_analysis["secondary_symptoms"]:
                output.append(f"- {symptom}")
        if symptom_analysis.get("duration_days"):
            output.append(f"Duration: {symptom_analysis['duration_days']} days")
        output.append("")
    
    if "diagnoses" in symptom_analysis:
        output.append(sections["diagnoses"])
        for i, diagnosis in enumerate(symptom_analysis["diagnoses"], 1):
            output.append(f"\n{i}. {diagnosis.get('name', 'Unknown')}")
            output.append(f"   Confidence: {diagnosis.get('confidence', 0)*100:.1f}%")
            if diagnosis.get("explanation"):
                output.append(f"   Explanation: {diagnosis['explanation']}")
            if diagnosis.get("recommendations"):
                output.append("   Recommendations:")
                for rec in diagnosis["recommendations"]:
                    output.append(f"   - {rec}")
        output.append("")
    
    if symptom_analysis.get("is_urgent"):
        output.append("\n⚠️ URGENT WARNING ⚠️")
        if symptom_analysis.get("urgent_symptoms"):
            output.append("Urgent symptoms detected:")
            for symptom in symptom_analysis["urgent_symptoms"]:
                output.append(f"- {symptom}")
        output.append("")
    
    if "image_analysis" in result:
        output.append(sections["image_analysis"])
        image_analysis = result["image_analysis"]
        if image_analysis.get("findings"):
            output.append("Image Findings:")
            for finding in image_analysis["findings"]:
                output.append(f"- {finding}")
        if image_analysis.get("confidence"):
            output.append(f"Analysis Confidence: {image_analysis['confidence']*100:.1f}%")
        output.append("")
    
    output.append("\nThis assessment is generated by an AI system and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers.")
    
    return "\n".join(output)

def save_assessment_to_file(assessment_text: str, patient_name: str) -> str:
    """Save the assessment to a file."""
    timestamp = datetime.now().strftime(output_config["assessment_filename_timestamp_format"])
    filename = output_config["assessment_filename_format"].format(
        patient_name=patient_name.replace(' ', '_'),
        timestamp=timestamp
    )
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(assessment_text)
        logger.info(f"Assessment saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving assessment to file: {e}")
        raise

def initialize_system():
    """Initialize the system components and load necessary data."""
    try:
        logger.info("Starting system initialization...")
        
        if not initialize_models():
            logger.error("Failed to initialize models. Exiting...")
            return None
        
        logger.info("Initializing medical knowledge index...")
        indexer = create_indexer()
        medical_kb_dir = str(Path(__file__).parent / "data" / "medical_kb")
        
        if not Path(medical_kb_dir).exists():
            logger.error(f"Medical knowledge directory not found: {medical_kb_dir}")
            logger.info("Creating empty medical knowledge directory...")
            Path(medical_kb_dir).mkdir(parents=True, exist_ok=True)
            return None
        
        start_time = time.time()
        try:
            doc_count, chunk_count = indexer.index_directory(medical_kb_dir)
            if time.time() - start_time > VECTOR_DB_INIT_TIMEOUT:
                logger.warning("Vector DB initialization took longer than expected")
            
            if doc_count == 0 or chunk_count == 0:
                logger.warning("No documents were indexed. The system may not function properly.")
            else:
                logger.info(f"Successfully indexed {doc_count} medical knowledge documents with {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            logger.warning("Continuing with empty index...")
        
        logger.info("Creating system analyzer...")
        try:
            analyzer = create_system_analyzer()
            logger.info("System initialization completed successfully")
            return analyzer
        except Exception as e:
            logger.error(f"Error creating system analyzer: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Critical error during system initialization: {e}")
        return None

def get_patient_data() -> Dict[str, Any]:
    """Get patient data from configuration defaults."""
    try:
        defaults = get_patient_defaults()
        
        if isinstance(defaults["patient_info"]["date_of_birth"], str):
            try:
                defaults["patient_info"]["date_of_birth"] = datetime.strptime(
                    defaults["patient_info"]["date_of_birth"],
                    "%Y-%m-%d"
                )
            except ValueError as e:
                raise ValueError(f"Invalid date format in patient info: {e}")
        
        return {
            "patient_info": defaults["patient_info"],
            "medical_history": defaults["medical_history"],
            "symptoms": defaults["symptoms"],
            "image_info": defaults["image_info"] if defaults["analysis_params"]["include_image_analysis"] else None,
            "analysis_params": defaults["analysis_params"],
            "additional_notes": defaults["additional_notes"]
        }
    except Exception as e:
        logger.error(f"Error getting patient data: {e}")
        raise

def format_assessment_result(result: Dict[str, Any]) -> str:
    """Format the assessment result into a readable text format."""
    output = []
    sections = output_config["assessment_sections"]
    date_format = output_config["assessment_date_format"]
    
    output.append(f"Assessment Date: {datetime.now().strftime(date_format)}\n")
    
    if "patient_info" in result:
        output.append(sections["patient_info"])
        patient = result["patient_info"]
        output.append(f"Name: {patient.get('first_name', '')} {patient.get('last_name', '')}")
        output.append(f"Age: {patient.get('age', 'N/A')}")
        output.append(f"Gender: {patient.get('gender', 'N/A')}")
        output.append(f"Blood Type: {patient.get('blood_type', 'N/A')}")
        output.append("")
    
    symptom_analysis = result.get("symptom_analysis", {})
    
    output.append(sections["symptoms"])
    if "primary_symptom" in symptom_analysis:
        output.append(f"Primary Symptom: {symptom_analysis['primary_symptom']}")
    if symptom_analysis.get("secondary_symptoms"):
        output.append("Secondary Symptoms:")
        for symptom in symptom_analysis["secondary_symptoms"]:
            output.append(f"- {symptom}")
    if symptom_analysis.get("duration_days"):
        output.append(f"Duration: {symptom_analysis['duration_days']} days")
    output.append("")
    
    if "diagnoses" in symptom_analysis:
        output.append(sections["diagnoses"])
        for i, diagnosis in enumerate(symptom_analysis["diagnoses"], 1):
            output.append(f"\n{i}. {diagnosis.get('name', 'Unknown')}")
            output.append(f"   Confidence: {diagnosis.get('confidence', 0)*100:.1f}%")
            if diagnosis.get("explanation"):
                output.append(f"   Explanation: {diagnosis['explanation']}")
            if diagnosis.get("recommendations"):
                output.append("   Recommendations:")
                for rec in diagnosis["recommendations"]:
                    output.append(f"   - {rec}")
        output.append("")
    
    if symptom_analysis.get("is_urgent"):
        output.append("\n⚠️ URGENT WARNING ⚠️")
        if symptom_analysis.get("urgent_symptoms"):
            output.append("Urgent symptoms detected:")
            for symptom in symptom_analysis["urgent_symptoms"]:
                output.append(f"- {symptom}")
        output.append("")
    
    if "image_analysis" in result:
        output.append(sections["image_analysis"])
        image_analysis = result["image_analysis"]
        
        if "error" in image_analysis:
            output.append(f"Error during image analysis: {image_analysis['error']}")
        else:
            if image_analysis.get("findings"):
                output.append("Image Findings:")
                for finding in image_analysis["findings"]:
                    output.append(f"- {finding}")
            if image_analysis.get("confidence"):
                output.append(f"Analysis Confidence: {image_analysis['confidence']*100:.1f}%")
            if image_analysis.get("detailed_analysis"):
                output.append("\nDetailed Analysis:")
                output.append(image_analysis["detailed_analysis"])
        output.append("")
    
    if "combined_analysis" in result:
        output.append("\nCombined Analysis:")
        combined = result["combined_analysis"]
        if isinstance(combined, str):
            output.append(combined)
        elif isinstance(combined, dict):
            if combined.get("combined_assessment"):
                output.append(combined["combined_assessment"])
        output.append("")
        
    return "\n".join(output) 

def main():
    try:
        logger.info("Starting MediMatch system...")
        
        analyzer = initialize_system()
        if analyzer is None:
            logger.error("Failed to initialize system. Exiting...")
            sys.exit(1)
        
        logger.info("Getting patient data...")
        try:
            patient_data = get_patient_data()
        except Exception as e:
            logger.error(f"Error getting patient data: {e}")
            sys.exit(1)
        
        logger.info("Sanitizing patient information...")
        try:
            sanitized_patient_info = sanitize_patient_data(patient_data["patient_info"])
        except Exception as e:
            logger.error(f"Error sanitizing patient data: {e}")
            sys.exit(1)
        
        logger.info("Starting case analysis...")
        try:
            result = analyzer.analyze_case(
                primary_symptom=patient_data["symptoms"]["primary"],
                secondary_symptoms=patient_data["symptoms"]["secondary"],
                patient_info=sanitized_patient_info,
                medical_history=patient_data["medical_history"],
                duration_days=patient_data["symptoms"]["duration_days"],
                image_path=patient_data["image_info"]["image_path"] if patient_data["image_info"] else None,
                body_area=patient_data["image_info"]["body_area"] if patient_data["image_info"] else None
            )
        except Exception as e:
            logger.error(f"Error during case analysis: {e}")
            sys.exit(1)
        
        result["patient_info"] = sanitized_patient_info
        result["additional_notes"] = patient_data["additional_notes"]
        
        logger.info("Formatting assessment results...")
        try:
            assessment_text = format_assessment_result(result)
            patient_name = f"{sanitized_patient_info['first_name']}_{sanitized_patient_info['last_name']}"
            output_file = save_assessment_to_file(assessment_text, patient_name)
        except Exception as e:
            logger.error(f"Error formatting or saving assessment: {e}")
            sys.exit(1)
        
        print(f"\nAssessment completed and saved to: {output_file}")
        print("\nAssessment Summary:")
        print(output_config["assessment_section_separator"])
        print(assessment_text)
        print(output_config["assessment_section_separator"])
        
    except Exception as e:
        logger.error(f"Critical error in main process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
