"""
Configuration module for MediMatch Copilot.

This module manages all configuration settings for the MediMatch Copilot application,
including model settings, directory paths, timeouts, and default values for various
components. It provides a centralized way to access and manage application settings
through various getter functions.

The configuration is organized into several categories:
- Base directories and paths
- Model settings (Ollama, embeddings)
- Timeout and retry settings
- Vector database settings
- Diagnosis engine settings
- Image analysis settings
- Logging settings
- Medical knowledge settings
- Retrieval settings
- Output formatting settings
- Patient data defaults
- Validation settings

All configuration values can be accessed through the get_config() function or
individual category-specific getter functions.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MEDICAL_KB_DIR = DATA_DIR / "medical_kb"

# Ensure directories exist
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
MEDICAL_KB_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
OLLAMA_BASE_URL = "http://localhost:11434"  
OLLAMA_TEXT_MODEL = "llama3.1:8b"
OLLAMA_VISION_MODEL = "llama3.2-vision:11b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  

# Model initialization settings
MODEL_INIT_TIMEOUT = 30  
MODEL_RETRY_ATTEMPTS = 2  

# Vector database settings
VECTOR_DB_PATH = str(EMBEDDINGS_DIR / "vector_db")
VECTOR_DB_INIT_TIMEOUT = 60  

# Diagnosis engine settings
MAX_DIAGNOSES = 5
MIN_CONFIDENCE_THRESHOLD = 0.3
INCLUDE_DETAILED_EXPLANATIONS = True
DIAGNOSIS_TIMEOUT = 45  # seconds

# Image analysis settings
IMAGE_SIZE = (512, 512)  
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
IMAGE_CONFIDENCE_THRESHOLD = 0.4
IMAGE_ANALYSIS_TIMEOUT = 30  

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = str(BASE_DIR / "medimatch.log")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Medical knowledge settings
SYMPTOMS_DB_PATH = MEDICAL_KB_DIR / "symptoms_database.json"
CONDITIONS_DB_PATH = MEDICAL_KB_DIR / "conditions_database.json"
MEDICATIONS_DB_PATH = MEDICAL_KB_DIR / "medications_database.json"

# Retrieval settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
NUM_RETRIEVAL_RESULTS = 5

# Prompt templates
DIAGNOSIS_SYSTEM_PROMPT = """
You are MediMatch, an AI medical assistant designed to help identify possible conditions
based on patient symptoms and medical data. Analyze the provided information carefully
and suggest possible diagnoses with their likelihood. Always clarify that your suggestions
are not a substitute for professional medical advice.
"""

# Patient Data Defaults
DEFAULT_PATIENT_INFO = {
    "first_name": "John",
    "last_name": "Smith",
    "date_of_birth": "1980-05-15",  
    "gender": "MALE",
    "blood_type": "A_POSITIVE",
    "height": 175.0,
    "weight": 75.0,
    "contact_info": {
        "email": "john.smith@example.com",
        "phone": "123-456-7890",
        "address": "Los Angeles, CA, USA",
        "emergency_contact": {
            "name": "Jane Smith",
            "relationship": "Spouse",
            "phone": "987-654-3210"
        }
    }
}

DEFAULT_MEDICAL_HISTORY = [
    "Hypertension (diagnosed 2015)",
    "Type 2 Diabetes (diagnosed 2018)",
    "Previous appendectomy (2010)"
]

DEFAULT_SYMPTOMS = {
    "primary": "chest pain",
    "secondary": [
        "shortness of breath",
        "fatigue",
        "dizziness"
    ],
    "duration_days": 3
}

DEFAULT_IMAGE_INFO = {
    "image_path": str(DATA_DIR / "xray.png"),  
    "body_area": "chest",
    "image_type": "X-ray",
    "image_description": "Chest X-ray showing anterior-posterior view"
} 

DEFAULT_ANALYSIS_PARAMS = {
    "include_image_analysis": True,
    "detailed_explanation": True,
    "include_treatment_suggestions": True,
    "include_preventive_measures": True
}

DEFAULT_ADDITIONAL_NOTES = """
Patient reports symptoms worsening with physical activity.
Has been taking prescribed medications regularly.
No recent travel history.
No known allergies to medications.
"""

# Validation Settings
MAX_SYMPTOM_DURATION_DAYS = 3650  
MIN_PATIENT_AGE = 0
MAX_PATIENT_AGE = 120
ALLOWED_GENDERS = ["MALE", "FEMALE"]
ALLOWED_BLOOD_TYPES = ["A_POSITIVE", "A_NEGATIVE", "B_POSITIVE", "B_NEGATIVE", 
                       "AB_POSITIVE", "AB_NEGATIVE", "O_POSITIVE", "O_NEGATIVE"]

# Output Formatting
ASSESSMENT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ASSESSMENT_FILENAME_FORMAT = 'assessment_{patient_name}_{timestamp}.txt'
ASSESSMENT_FILENAME_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
ASSESSMENT_SECTION_SEPARATOR = "\n" + "="*80 + "\n"
ASSESSMENT_SECTIONS = {
    "patient_info": "\nPatient Information:",
    "symptoms": "\nReported Symptoms:",
    "diagnoses": "\nPossible Diagnoses:",
    "image_analysis": "\nImage Analysis Results:"
}

# Configuration getter functions
def get_config() -> Dict[str, Any]:
    """Return the complete configuration dictionary."""
    return {
        "model": get_model_config(),
        "retrieval": get_retrieval_config(),
        "timeout": get_timeout_config(),
        "patient_defaults": get_patient_defaults(),
        "validation": get_validation_config(),
        "output": get_output_config()
    }

def get_model_config() -> Dict[str, Any]:
    """Return model-related configuration parameters."""
    return {
        "ollama_base_url": OLLAMA_BASE_URL,
        "text_model": OLLAMA_TEXT_MODEL,
        "vision_model": OLLAMA_VISION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "init_timeout": MODEL_INIT_TIMEOUT,
        "retry_attempts": MODEL_RETRY_ATTEMPTS
    }

def get_retrieval_config() -> Dict[str, Any]:
    """Return retrieval-specific configuration parameters."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_results": NUM_RETRIEVAL_RESULTS,
        "vector_db_path": VECTOR_DB_PATH,
        "init_timeout": VECTOR_DB_INIT_TIMEOUT
    }

def get_timeout_config() -> Dict[str, Any]:
    """Return timeout-related configuration parameters."""
    return {
        "model_init": MODEL_INIT_TIMEOUT,
        "diagnosis": DIAGNOSIS_TIMEOUT,
        "image_analysis": IMAGE_ANALYSIS_TIMEOUT,
        "vector_db_init": VECTOR_DB_INIT_TIMEOUT
    }

def get_patient_defaults() -> Dict[str, Any]:
    """Return default patient data configuration."""
    return {
        "patient_info": DEFAULT_PATIENT_INFO,
        "medical_history": DEFAULT_MEDICAL_HISTORY,
        "symptoms": DEFAULT_SYMPTOMS,
        "image_info": DEFAULT_IMAGE_INFO,
        "analysis_params": DEFAULT_ANALYSIS_PARAMS,
        "additional_notes": DEFAULT_ADDITIONAL_NOTES
    }

def get_validation_config() -> Dict[str, Any]:
    """Return validation-related configuration parameters."""
    return {
        "max_symptom_duration_days": MAX_SYMPTOM_DURATION_DAYS,
        "min_patient_age": MIN_PATIENT_AGE,
        "max_patient_age": MAX_PATIENT_AGE,
        "allowed_genders": ALLOWED_GENDERS,
        "allowed_blood_types": ALLOWED_BLOOD_TYPES
    }

def get_output_config() -> Dict[str, Any]:
    """Return output formatting configuration parameters."""
    return {
        "log_format": LOG_FORMAT,
        "assessment_date_format": ASSESSMENT_DATE_FORMAT,
        "assessment_filename_format": ASSESSMENT_FILENAME_FORMAT,
        "assessment_filename_timestamp_format": ASSESSMENT_FILENAME_TIMESTAMP_FORMAT,
        "assessment_section_separator": ASSESSMENT_SECTION_SEPARATOR,
        "assessment_sections": ASSESSMENT_SECTIONS
    }