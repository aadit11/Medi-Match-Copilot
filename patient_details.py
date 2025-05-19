from typing import List, Dict, Any, Optional
from datetime import datetime
from schemas.patient import Gender, BloodType

# Patient Information
PATIENT_INFO = {
    "first_name": "John",
    "last_name": "Smith",
    "date_of_birth": datetime(1980, 5, 15),  
    "gender": Gender.MALE,
    "blood_type": BloodType.A_POSITIVE,
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

# Medical History
MEDICAL_HISTORY = [
    "Hypertension (diagnosed 2015)",
    "Type 2 Diabetes (diagnosed 2018)",
    "Previous appendectomy (2010)"
]

# Current Symptoms
PRIMARY_SYMPTOM = "chest pain"
SECONDARY_SYMPTOMS = [
    "shortness of breath",
    "fatigue",
    "dizziness"
]

# Symptom Duration
SYMPTOM_DURATION_DAYS = 3

# Medical Image Information (if available)
IMAGE_INFO = {
    "image_path": "D:/CodeExperimentation/PersonalProjects/Medi-Match_Copilot/Medi-Match-Copilot/data/xray.png",  
    "body_area": "chest",
    "image_type": "X-ray"  
}

# Analysis Parameters
ANALYSIS_PARAMS = {
    "include_image_analysis": True,  
    "detailed_explanation": True,
    "include_treatment_suggestions": True,
    "include_preventive_measures": True
}

# Additional Notes
ADDITIONAL_NOTES = """
Patient reports symptoms worsening with physical activity.
Has been taking prescribed medications regularly.
No recent travel history.
No known allergies to medications.
"""

def get_patient_data() -> Dict[str, Any]:
    """Returns all patient data in a structured format."""
    return {
        "patient_info": PATIENT_INFO,
        "medical_history": MEDICAL_HISTORY,
        "symptoms": {
            "primary": PRIMARY_SYMPTOM,
            "secondary": SECONDARY_SYMPTOMS,
            "duration_days": SYMPTOM_DURATION_DAYS
        },
        "image_info": IMAGE_INFO if ANALYSIS_PARAMS["include_image_analysis"] else None,
        "analysis_params": ANALYSIS_PARAMS,
        "additional_notes": ADDITIONAL_NOTES
    } 