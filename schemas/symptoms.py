from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum

class SymptomSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class SymptomDuration(BaseModel):
    value: float = Field(..., ge=0, description="Duration value")
    unit: str = Field("days", description="Unit of duration (e.g., 'days', 'hours', 'weeks')")
    is_ongoing: bool = Field(False, description="Whether the symptom is still ongoing")
    start_date: Optional[datetime] = Field(None, description="When the symptom started")
    end_date: Optional[datetime] = Field(None, description="When the symptom ended (if not ongoing)")

    @field_validator('unit')
    @classmethod
    def validate_unit(cls, v):
        valid_units = ['days', 'hours', 'weeks', 'months', 'years']
        if v.lower() not in valid_units:
            raise ValueError(f'Unit must be one of {valid_units}')
        return v.lower()

class SymptomCharacteristic(BaseModel):
    type: str = Field(..., description="Type of characteristic (e.g., 'pain_type', 'location', 'trigger')")
    value: str = Field(..., description="Value of the characteristic")
    description: Optional[str] = Field(None, description="Additional description if needed")

class Symptom(BaseModel):
    name: str = Field(..., description="Name of the symptom")
    severity: SymptomSeverity = Field(..., description="Severity level of the symptom")
    duration: Optional[SymptomDuration] = Field(None, description="Duration information")
    characteristics: List[SymptomCharacteristic] = Field(default_factory=list, description="List of symptom characteristics")
    is_primary: bool = Field(False, description="Whether this is the primary symptom")
    associated_symptoms: List[str] = Field(default_factory=list, description="List of associated symptoms")
    aggravating_factors: List[str] = Field(default_factory=list, description="Factors that make the symptom worse")
    relieving_factors: List[str] = Field(default_factory=list, description="Factors that make the symptom better")
    notes: Optional[str] = Field(None, description="Additional notes about the symptom")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class SymptomAssessment(BaseModel):
    primary_symptom: Symptom = Field(..., description="The primary symptom")
    secondary_symptoms: List[Symptom] = Field(default_factory=list, description="List of secondary symptoms")
    is_urgent: bool = Field(False, description="Whether the symptoms require urgent attention")
    urgent_symptoms: List[str] = Field(default_factory=list, description="List of urgent symptoms if any")
    assessment_date: datetime = Field(default_factory=datetime.now, description="When the assessment was made")
    patient_info: Optional[Dict[str, Any]] = Field(None, description="Patient information used in assessment")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class SymptomGuidance(BaseModel):
    symptom: str = Field(..., description="The symptom being explored")
    exploration_guidance: str = Field(..., description="Guidance text for exploring the symptom")
    follow_up_questions: List[str] = Field(..., description="List of follow-up questions to ask")
    categories: List[str] = Field(..., description="Categories of questions (e.g., 'timing', 'severity', 'characteristics')")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")