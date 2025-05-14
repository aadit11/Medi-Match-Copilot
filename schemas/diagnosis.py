from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class DiagnosisRecommendation(BaseModel):
    """Model for diagnosis recommendations."""
    text: str = Field(..., description="The recommendation text")
    priority: str = Field("medium", description="Priority level (high/medium/low)")
    category: Optional[str] = Field(None, description="Category of recommendation (e.g., 'test', 'referral', 'treatment')")

class Diagnosis(BaseModel):
    name: str = Field(..., description="Name of the condition")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    explanation: Optional[str] = Field(None, description="Explanation for the diagnosis")
    recommendations: List[DiagnosisRecommendation] = Field(default_factory=list, description="List of recommendations")
    source: str = Field(..., description="Source of the diagnosis (e.g., 'symptom_analysis', 'image_analysis')")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class DiagnosticAssessment(BaseModel):
    diagnoses: List[Diagnosis] = Field(..., description="List of potential diagnoses")
    is_urgent: bool = Field(False, description="Whether the case requires urgent attention")
    urgent_symptoms: List[str] = Field(default_factory=list, description="List of urgent symptoms if any")
    raw_assessment: Optional[str] = Field(None, description="Raw assessment text from the model")
    context: Optional[str] = Field(None, description="Context information used for the assessment")
    patient_info: Optional[Dict[str, Any]] = Field(None, description="Patient information used in assessment")
    formatted_report: Optional[str] = Field(None, description="Formatted report of the assessment")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the assessment was made")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class ImageAnalysisFinding(BaseModel):
    description: str = Field(..., description="Description of the finding")
    location: Optional[str] = Field(None, description="Location in the image")
    significance: str = Field(..., description="Significance level (high/medium/low)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    notes: Optional[str] = Field(None, description="Additional notes about the finding")
    urgency_required: bool = Field(False, description="Whether this finding requires urgent attention")

class CombinedAnalysis(BaseModel):
    image_diagnoses: List[Diagnosis] = Field(default_factory=list, description="Diagnoses from image analysis")
    symptom_diagnoses: List[Diagnosis] = Field(default_factory=list, description="Diagnoses from symptom analysis")
    combined_assessment: str = Field(..., description="Integrated assessment combining both analyses")
    is_urgent: bool = Field(False, description="Whether the case requires urgent attention")
    findings: List[ImageAnalysisFinding] = Field(default_factory=list, description="Key findings from image analysis")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
