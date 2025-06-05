# Medi-Match Copilot

Medi-Match Copilot is an advanced medical analysis and diagnosis assistance system that leverages AI to provide comprehensive medical assessments. The system combines text analysis, image processing, and medical knowledge retrieval to assist healthcare professionals in making informed decisions.

## 🌟 Features

- **Intelligent Medical Analysis**: Utilizes advanced AI models for comprehensive medical assessment
- **Image Analysis**: Processes and analyzes medical images using computer vision
- **Medical Knowledge Base**: Maintains an indexed database of medical knowledge for accurate diagnosis
- **Patient Assessment**: Generates detailed patient assessments with symptoms analysis
- **Secure Data Handling**: Implements proper data sanitization and privacy measures
- **Extensible Architecture**: Modular design for easy integration of new features


## 📋 Prerequisites

- Python 3.8 or higher
- Ollama server running locally
- Required Python packages (see `requirements.txt`)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Medi-Match-Copilot.git
   cd Medi-Match-Copilot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Ollama:
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Start the Ollama server
   - Pull required models (specified in configuration)

## ⚙️ Configuration

The system uses a configuration file (`core/config.py`) for various settings:
- Model parameters
- API endpoints
- File paths
- Output formats
- System timeouts


## 🚀 Usage

1. Ensure the Ollama server is running
2. Run the main application:
   ```bash
   python main.py
   ```

The system will:
1. Initialize required models
2. Load and index medical knowledge
3. Process patient data
4. Generate comprehensive medical assessments

## 📝 Output

The system generates detailed assessment reports including:
- Patient information
- Symptom analysis
- Medical history review
- Image analysis results (if applicable)
- Diagnostic suggestions
- Additional medical notes

Reports are saved in a structured text format with timestamps.

