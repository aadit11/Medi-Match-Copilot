import re
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
from core.config import get_validation_config

def validate_symptoms(symptoms: List[str]) -> Tuple[List[str], List[str]]:
   
    valid_symptoms = []
    rejected_symptoms = []
    
    
    for symptom in symptoms:
        
        if not symptom or len(symptom.strip()) < 3:
            rejected_symptoms.append(symptom)
            continue
        
        
        if re.search(r'[<>{}[\]\\]', symptom):
            rejected_symptoms.append(symptom)
            continue
            
        
        cleaned_symptom = symptom.strip().lower()
        valid_symptoms.append(cleaned_symptom)
    
    return valid_symptoms, rejected_symptoms

def validate_patient_age(age: Union[int, str, float]) -> Tuple[Optional[int], str]:
    """Validate patient age."""
    validation_config = get_validation_config()
    min_age = validation_config["min_patient_age"]
    max_age = validation_config["max_patient_age"]
    error_msg = ""
    
    if isinstance(age, str):
        try:
            age = int(float(age.strip()))
        except ValueError:
            return None, "Age must be a number"
    
    if not isinstance(age, (int, float)):
        return None, "Age must be a number"
    
    age = int(age)
    if age < min_age:
        return None, f"Age cannot be less than {min_age}"
    if age > max_age:
        return None, f"Age cannot be greater than {max_age}"
    
    return age, error_msg

def validate_patient_sex(sex: str) -> Tuple[Optional[str], str]:
    """Validate patient gender/sex."""
    validation_config = get_validation_config()
    allowed_genders = validation_config["allowed_genders"]
    error_msg = ""
    
    if not isinstance(sex, str):
        return None, "Gender must be a string"
    
    sex = sex.strip().upper()
    if sex not in allowed_genders:
        return None, f"Gender must be one of: {', '.join(allowed_genders)}"
    
    return sex, error_msg

def validate_duration(duration: Union[int, str, float]) -> Tuple[Optional[float], str]:
    """Validate symptom duration."""
    validation_config = get_validation_config()
    max_duration = validation_config["max_symptom_duration_days"]
    error_msg = ""
    
    if isinstance(duration, str):
        try:
            duration = float(duration.strip())
        except ValueError:
            return None, "Duration must be a number"
    
    if not isinstance(duration, (int, float)):
        return None, "Duration must be a number"
    
    if duration < 0:
        return None, "Duration cannot be negative"
    
    if duration > max_duration:
        return None, f"Duration {duration} days seems unusually long, please verify"
    
    return duration, error_msg

def validate_medical_image(image_path: str) -> Tuple[bool, str]:
   
    path = Path(image_path)
    if not path.exists():
        return False, f"Image file not found: {image_path}"
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dicom', '.dcm']
    if path.suffix.lower() not in valid_extensions:
        return False, f"Unsupported file format: {path.suffix}"
    
    file_size = path.stat().st_size
    
    if file_size < 10 * 1024:  
        return False, "Image file is suspiciously small"
        
    if file_size > 20 * 1024 * 1024:  
        return False, "Image file is too large (max 20MB)"
    
    return True, ""

def sanitize_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    
    sanitized_data = {}
    
    for key, value in patient_data.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
            
        key = key.lower().strip().replace(" ", "_")
        
        if key in ['age', 'patient_age']:
            age, error = validate_patient_age(value)
            if not error:
                sanitized_data['age'] = age
        
        elif key in ['sex', 'gender', 'patient_sex', 'patient_gender']:
            sex, error = validate_patient_sex(value)
            if sex:
                sanitized_data['sex'] = sex
        
        elif key in ['symptoms', 'patient_symptoms']:
            if isinstance(value, list):
                valid_symptoms, _ = validate_symptoms(value)
                if valid_symptoms:
                    sanitized_data['symptoms'] = valid_symptoms
            elif isinstance(value, str):
                symptom_list = [s.strip() for s in value.split(',')]
                valid_symptoms, _ = validate_symptoms(symptom_list)
                if valid_symptoms:
                    sanitized_data['symptoms'] = valid_symptoms
        
        elif key in ['duration', 'symptom_duration']:
            duration, error = validate_duration(value)
            if not error:
                sanitized_data['duration_days'] = duration
        
        elif isinstance(value, str):
            clean_value = re.sub(r'[<>{}[\]\\]', '', value).strip()
            if clean_value:
                sanitized_data[key] = clean_value
        
        elif isinstance(value, (int, float, bool)):
            sanitized_data[key] = value
        
        elif isinstance(value, list):
            sanitized_data[key] = ", ".join(str(v) for v in value)
    
    return sanitized_data

def is_urgent_symptom(symptoms: List[str]) -> Tuple[bool, List[str]]:
   
    urgent_keywords = [
        'severe', 'intense', 'excruciating', 'worst', 
        'chest pain', 'difficulty breathing', 'shortness of breath',
        'stroke', 'heart attack', 'unconscious', 'unresponsive',
        'seizure', 'convulsion', 'paralysis', 'sudden numbness',
        'severe bleeding', 'coughing blood', 'vomiting blood',
        'sudden vision loss', 'sudden severe headache',
        'suicide', 'self-harm', 'overdose'
    ]
    
    urgent_found = []
    for symptom in symptoms:
        if any(keyword in symptom.lower() for keyword in urgent_keywords):
            urgent_found.append(symptom)
    
    return bool(urgent_found), urgent_found