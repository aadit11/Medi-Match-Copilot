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