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

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TEXT_MODEL = "llama3:latest"
OLLAMA_VISION_MODEL = "llava:latest"

# Vector database settings
VECTOR_DB_PATH = str(EMBEDDINGS_DIR / "vector_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  

# Diagnosis engine settings
MAX_DIAGNOSES = 5
MIN_CONFIDENCE_THRESHOLD = 0.3
INCLUDE_DETAILED_EXPLANATIONS = True

# Image analysis settings
IMAGE_SIZE = (512, 512)  # Width, Height tuple for resizing images
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
IMAGE_CONFIDENCE_THRESHOLD = 0.4

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = str(BASE_DIR / "medimatch.log")

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

def get_config() -> Dict[str, Any]:
    """Return all configuration parameters as a dictionary."""
    return {k: v for k, v in globals().items() if k.isupper()}

def get_model_config() -> Dict[str, Any]:
    """Return model-specific configuration parameters."""
    return {
        "ollama_host": OLLAMA_HOST,
        "text_model": OLLAMA_TEXT_MODEL,
        "vision_model": OLLAMA_VISION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }

def get_retrieval_config() -> Dict[str, Any]:
    """Return retrieval-specific configuration parameters."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_results": NUM_RETRIEVAL_RESULTS,
        "vector_db_path": VECTOR_DB_PATH,
    }