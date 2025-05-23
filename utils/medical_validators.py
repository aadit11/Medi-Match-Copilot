import re
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

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

def validate_patient_age(age: Union[int, str]) -> Tuple[Optional[int], str]:
    
    error_msg = ""
    
    if isinstance(age, str):
        try:
            age = int(age.strip())
        except ValueError:
            return None, "Age must be a number"
    
    if not isinstance(age, int):
        return None, "Age must be a number"
    
    if age < 0:
        return None, "Age cannot be negative"
    
    if age > 120:
        return None, f"Age {age} seems unusually high, please verify"
    
    return age, error_msg

def validate_patient_sex(sex: str) -> Tuple[Optional[str], str]:
 
    if not sex or not isinstance(sex, str):
        return None, "Sex/gender information is missing"
    
    sex = sex.lower().strip()
    
    male_terms = ["m", "male", "man", "boy"]
    female_terms = ["f", "female", "woman", "girl"]
    
    if sex in male_terms:
        return "male", ""
    
    if sex in female_terms:
        return "female", ""
    
    return sex, "Non-standard sex/gender term used"

def validate_duration(duration: Union[int, str, float]) -> Tuple[Optional[float], str]:
   
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
    
    if duration > 365 * 10:  
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
    
    urgent_symptoms = []
    
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for keyword in urgent_keywords:
            if keyword in symptom_lower:
                urgent_symptoms.append(symptom)
                break
    
    return bool(urgent_symptoms), urgent_symptoms