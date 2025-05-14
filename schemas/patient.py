from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, EmailStr
from datetime import datetime
from enum import Enum

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNSPECIFIED = "unspecified"

class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"

class MedicalCondition(BaseModel):
    name: str = Field(..., description="Name of the condition")
    diagnosis_date: Optional[datetime] = Field(None, description="When the condition was diagnosed")
    is_active: bool = Field(True, description="Whether the condition is currently active")
    severity: Optional[str] = Field(None, description="Severity of the condition")
    treatment: Optional[str] = Field(None, description="Current or past treatment")
    notes: Optional[str] = Field(None, description="Additional notes about the condition")

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage information")
    frequency: str = Field(..., description="How often the medication is taken")
    start_date: Optional[datetime] = Field(None, description="When the medication was started")
    end_date: Optional[datetime] = Field(None, description="When the medication was stopped (if applicable)")
    is_active: bool = Field(True, description="Whether the medication is currently being taken")
    prescribed_by: Optional[str] = Field(None, description="Who prescribed the medication")
    notes: Optional[str] = Field(None, description="Additional notes about the medication")

class Allergy(BaseModel):
    allergen: str = Field(..., description="The allergen")
    reaction: str = Field(..., description="The reaction to the allergen")
    severity: str = Field(..., description="Severity of the allergy")
    notes: Optional[str] = Field(None, description="Additional notes about the allergy")

class FamilyHistory(BaseModel):
    condition: str = Field(..., description="The medical condition")
    relation: str = Field(..., description="Relation to the patient")
    age_at_onset: Optional[int] = Field(None, description="Age when the condition developed")
    notes: Optional[str] = Field(None, description="Additional notes about the family history")

class PatientContact(BaseModel):
    email: Optional[EmailStr] = Field(None, description="Patient's email address")
    phone: Optional[str] = Field(None, description="Patient's phone number")
    address: Optional[str] = Field(None, description="Patient's address")
    emergency_contact: Optional[Dict[str, str]] = Field(None, description="Emergency contact information")

class Patient(BaseModel):
    id: Optional[str] = Field(None, description="Unique patient identifier")
    first_name: str = Field(..., description="Patient's first name")
    last_name: str = Field(..., description="Patient's last name")
    date_of_birth: datetime = Field(..., description="Patient's date of birth")
    gender: Gender = Field(..., description="Patient's gender")
    age: Optional[int] = Field(None, description="Patient's age")
    blood_type: Optional[BloodType] = Field(None, description="Patient's blood type")
    height: Optional[float] = Field(None, ge=0, description="Patient's height in centimeters")
    weight: Optional[float] = Field(None, ge=0, description="Patient's weight in kilograms")
    contact_info: Optional[PatientContact] = Field(None, description="Patient's contact information")
    medical_conditions: List[MedicalCondition] = Field(default_factory=list, description="List of medical conditions")
    medications: List[Medication] = Field(default_factory=list, description="List of medications")
    allergies: List[Allergy] = Field(default_factory=list, description="List of allergies")
    family_history: List[FamilyHistory] = Field(default_factory=list, description="Family medical history")
    lifestyle_factors: Optional[Dict[str, Any]] = Field(None, description="Lifestyle factors (smoking, exercise, etc.)")
    notes: Optional[str] = Field(None, description="Additional notes about the patient")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('age')
    @classmethod
    def validate_age(cls, v, info):
        if v is not None:
            if 'date_of_birth' in info.data:
                dob = info.data['date_of_birth']
                calculated_age = (datetime.now() - dob).days / 365.25
                if abs(v - calculated_age) > 1:
                    raise ValueError('Age does not match date of birth')
        return v

    @field_validator('height', 'weight')
    @classmethod
    def validate_measurements(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Measurement must be positive')
        return v

class PatientAssessment(BaseModel):
    patient: Patient = Field(..., description="The patient being assessed")
    assessment_date: datetime = Field(default_factory=datetime.now, description="When the assessment was made")
    chief_complaint: str = Field(..., description="The patient's chief complaint")
    vital_signs: Optional[Dict[str, Any]] = Field(None, description="Patient's vital signs")
    assessment_notes: Optional[str] = Field(None, description="Notes from the assessment")
    follow_up_date: Optional[datetime] = Field(None, description="Recommended follow-up date")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
