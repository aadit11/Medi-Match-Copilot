import re
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

from core.config import (
    ALLOWED_BLOOD_TYPES,
    ALLOWED_GENDERS,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_PATIENT_AGE,
    MAX_SYMPTOM_DURATION_DAYS,
    MIN_PATIENT_AGE,
)

_SYMPTOM_UNSAFE_PATTERN = re.compile(r"[<>{}[\]\\]")
_TEXT_UNSAFE_PATTERN = re.compile(r"[<>{}[\]\\]")

_IDENTITY_FIELDS = frozenset({"first_name", "last_name"})
_CLINICAL_FIELDS = frozenset({
    "age", "gender", "blood_type", "height", "weight",
    "first_name", "last_name",
})
_SKIP_PROMPT_FIELDS = frozenset({"contact_info"})

_GENDER_ALIASES = {
    "m": "MALE",
    "male": "MALE",
    "man": "MALE",
    "boy": "MALE",
    "f": "FEMALE",
    "female": "FEMALE",
    "woman": "FEMALE",
    "girl": "FEMALE",
}

_URGENT_PATTERNS = [
    re.compile(r"\bsevere\b", re.I),
    re.compile(r"\bintense\b", re.I),
    re.compile(r"\bexcruciating\b", re.I),
    re.compile(r"\bworst\b", re.I),
    re.compile(r"\bchest pain\b", re.I),
    re.compile(r"\bdifficulty breathing\b", re.I),
    re.compile(r"\bshortness of breath\b", re.I),
    re.compile(r"\bstroke\b", re.I),
    re.compile(r"\bheart attack\b", re.I),
    re.compile(r"\bunconscious\b", re.I),
    re.compile(r"\bunresponsive\b", re.I),
    re.compile(r"\bseizure\b", re.I),
    re.compile(r"\bconvulsion\b", re.I),
    re.compile(r"\bparalysis\b", re.I),
    re.compile(r"\bsudden numbness\b", re.I),
    re.compile(r"\bsevere bleeding\b", re.I),
    re.compile(r"\bcoughing blood\b", re.I),
    re.compile(r"\bvomiting blood\b", re.I),
    re.compile(r"\bsudden vision loss\b", re.I),
    re.compile(r"\bsudden severe headache\b", re.I),
    re.compile(r"\bsuicide\b", re.I),
    re.compile(r"\bself[- ]harm\b", re.I),
    re.compile(r"\boverdose\b", re.I),
    re.compile(r"\bslow[- ]healing wound\b", re.I),
    re.compile(r"\bdiabetic foot\b", re.I),
    re.compile(r"\bfoot ulcer\b", re.I),
    re.compile(r"\bgangrene\b", re.I),
]


def validate_symptoms(symptoms: List[str]) -> Tuple[List[str], List[str]]:
    """Validate and clean a list of symptoms."""
    valid_symptoms = []
    rejected_symptoms = []

    for symptom in symptoms:
        if not symptom or len(symptom.strip()) < 3:
            rejected_symptoms.append(symptom)
            continue

        if _SYMPTOM_UNSAFE_PATTERN.search(symptom):
            rejected_symptoms.append(symptom)
            continue

        valid_symptoms.append(symptom.strip().lower())

    return valid_symptoms, rejected_symptoms


def validate_patient_age(age: Union[int, str]) -> Tuple[Optional[int], str]:
    """Validate patient age."""
    if isinstance(age, str):
        try:
            age = int(age.strip())
        except ValueError:
            return None, "Age must be a number"

    if not isinstance(age, int):
        return None, "Age must be a number"

    if age < MIN_PATIENT_AGE:
        return None, "Age cannot be negative"

    if age > MAX_PATIENT_AGE:
        return None, f"Age {age} seems unusually high, please verify"

    return age, ""


def normalize_gender(value: str) -> Tuple[Optional[str], str]:
    """Normalize gender/sex to config ALLOWED_GENDERS values."""
    if not value or not isinstance(value, str):
        return None, "Sex/gender information is missing"

    normalized = _GENDER_ALIASES.get(value.lower().strip())
    if normalized:
        return normalized, ""

    upper = value.upper().strip()
    if upper in ALLOWED_GENDERS:
        return upper, ""

    return None, "Non-standard sex/gender term used"


def validate_patient_sex(sex: str) -> Tuple[Optional[str], str]:
    """Validate patient sex/gender information (backward-compatible wrapper)."""
    gender, error = normalize_gender(sex)
    if gender:
        return gender.lower(), error
    return None, error


def validate_blood_type(blood_type: str) -> Tuple[Optional[str], str]:
    """Validate blood type against allowed values."""
    if not blood_type or not isinstance(blood_type, str):
        return None, "Blood type is missing"

    normalized = blood_type.upper().strip()
    if normalized in ALLOWED_BLOOD_TYPES:
        return normalized, ""

    return None, f"Unsupported blood type: {blood_type}"


def validate_duration(duration: Union[int, str, float]) -> Tuple[Optional[float], str]:
    """Validate symptom duration in days."""
    if isinstance(duration, str):
        try:
            duration = float(duration.strip())
        except ValueError:
            return None, "Duration must be a number"

    if not isinstance(duration, (int, float)):
        return None, "Duration must be a number"

    if duration < 0:
        return None, "Duration cannot be negative"

    if duration > MAX_SYMPTOM_DURATION_DAYS:
        return None, f"Duration {duration} days seems unusually long, please verify"

    return float(duration), ""


def validate_medical_image(image_path: str) -> Tuple[bool, str]:
    """Validate medical image file."""
    path = Path(image_path)
    if not path.exists():
        return False, f"Image file not found: {image_path}"

    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported file format: {path.suffix}"

    file_size = path.stat().st_size
    if file_size < 10 * 1024:
        return False, "Image file is suspiciously small"
    if file_size > 20 * 1024 * 1024:
        return False, "Image file is too large (max 20MB)"

    return True, ""


def is_urgent_symptom(symptoms: List[str]) -> Tuple[bool, List[str]]:
    """Check if any symptoms indicate urgent medical attention is needed."""
    urgent_symptoms = []

    for symptom in symptoms:
        for pattern in _URGENT_PATTERNS:
            if pattern.search(symptom):
                urgent_symptoms.append(symptom)
                break

    return bool(urgent_symptoms), urgent_symptoms


def sanitize_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and validate patient data for clinical use."""
    sanitized_data: Dict[str, Any] = {}

    for key, value in patient_data.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue

        key = key.lower().strip().replace(" ", "_")

        if key in _SKIP_PROMPT_FIELDS:
            continue

        if key in ("date_of_birth", "dob", "birth_date"):
            try:
                if isinstance(value, str):
                    dob = datetime.strptime(value, "%Y-%m-%d")
                elif isinstance(value, datetime):
                    dob = value
                else:
                    continue

                age = (datetime.now() - dob).days // 365
                age, error = validate_patient_age(age)
                if age is not None and not error:
                    sanitized_data["age"] = age
            except (ValueError, TypeError):
                continue

        elif key in ("sex", "gender", "patient_sex", "patient_gender"):
            gender, _ = normalize_gender(str(value))
            if gender:
                sanitized_data["gender"] = gender

        elif key == "blood_type":
            blood_type, _ = validate_blood_type(str(value))
            if blood_type:
                sanitized_data["blood_type"] = blood_type

        elif key in ("symptoms", "patient_symptoms"):
            if isinstance(value, list):
                valid_symptoms, _ = validate_symptoms(value)
            else:
                valid_symptoms, _ = validate_symptoms(
                    [s.strip() for s in str(value).split(",")]
                )
            if valid_symptoms:
                sanitized_data["symptoms"] = valid_symptoms

        elif key in ("duration", "symptom_duration"):
            duration, error = validate_duration(value)
            if not error:
                sanitized_data["duration_days"] = duration

        elif key in _IDENTITY_FIELDS and isinstance(value, str):
            clean_value = _TEXT_UNSAFE_PATTERN.sub("", value).strip()
            if clean_value:
                sanitized_data[key] = clean_value

        elif key in ("height", "weight") and isinstance(value, (int, float)):
            sanitized_data[key] = float(value)

        elif key == "age":
            age, error = validate_patient_age(value)
            if age is not None and not error:
                sanitized_data["age"] = age

        elif isinstance(value, str) and key in _CLINICAL_FIELDS:
            clean_value = _TEXT_UNSAFE_PATTERN.sub("", value).strip()
            if clean_value:
                sanitized_data[key] = clean_value

        elif isinstance(value, (int, float, bool)) and key in _CLINICAL_FIELDS:
            sanitized_data[key] = value

    return sanitized_data
