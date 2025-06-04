import re
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

def validate_symptoms(symptoms: List[str]) -> Tuple[List[str], List[str]]:
    """Validate and clean a list of symptoms.
    
    Args:
        symptoms: List of symptom strings to validate
        
    Returns:
        Tuple containing:
            - List of valid, cleaned symptoms
            - List of rejected symptoms that failed validation
    """
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
    """Validate patient age.
    
    Args:
        age: Age as either an integer or string
        
    Returns:
        Tuple containing:
            - Validated age as integer or None if invalid
            - Error message if validation failed, empty string if successful
    """
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
    """Validate patient sex/gender information.
    
    Args:
        sex: Sex/gender string to validate
        
    Returns:
        Tuple containing:
            - Standardized sex string ('male' or 'female') or original string if non-standard
            - Error message if validation failed, empty string if successful
    """
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
    """Validate symptom duration.
    
    Args:
        duration: Duration as number or string (in days)
        
    Returns:
        Tuple containing:
            - Validated duration as float or None if invalid
            - Error message if validation failed, empty string if successful
    """
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
    """Validate medical image file.
    
    Args:
        image_path: Path to the medical image file
        
    Returns:
        Tuple containing:
            - Boolean indicating if image is valid
            - Error message if validation failed, empty string if successful
    """
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

def is_urgent_symptom(symptoms: List[str]) -> Tuple[bool, List[str]]:
    """Check if any symptoms indicate urgent medical attention is needed.
    
    Args:
        symptoms: List of symptoms to check
        
    Returns:
        Tuple containing:
            - Boolean indicating if any urgent symptoms were found
            - List of urgent symptoms found
    """
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

def sanitize_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and validate patient data.
    
    This function processes and validates various patient data fields including:
    - Date of birth (converts to age)
    - Sex/gender
    - Symptoms
    - Duration
    - Other text fields
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        Dictionary containing sanitized and validated patient data
    """
    sanitized_data = {}
    
    for key, value in patient_data.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
            
        key = key.lower().strip().replace(" ", "_")
        
        if key in ['date_of_birth', 'dob', 'birth_date']:
            try:
                if isinstance(value, str):
                    dob = datetime.strptime(value, "%Y-%m-%d")
                elif isinstance(value, datetime):
                    dob = value
                else:
                    continue
                    
                age = (datetime.now() - dob).days // 365
                if 0 <= age <= 120:
                    sanitized_data['age'] = age
            except (ValueError, TypeError):
                continue
        
        elif key in ['sex', 'gender', 'patient_sex', 'patient_gender']:
            sex, error = validate_patient_sex(str(value).upper())
            if sex:
                sanitized_data['gender'] = sex.upper()
        
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
        
        elif isinstance(value, dict):
            sanitized_data[key] = sanitize_patient_data(value)
        
        elif isinstance(value, list):
            sanitized_data[key] = [str(v) for v in value if v is not None]
    
    return sanitized_data 