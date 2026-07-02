from typing import Dict, List, Any, Optional
from core.config import DIAGNOSIS_SYSTEM_PROMPT, MAX_DIAGNOSES, NUM_RETRIEVAL_RESULTS

CLINICAL_PATIENT_FIELDS = ("age", "gender", "blood_type", "height", "weight")
MAX_KNOWLEDGE_ITEM_CHARS = 600
MAX_KNOWLEDGE_ITEMS = NUM_RETRIEVAL_RESULTS

DIAGNOSIS_RESPONSE_SCHEMA = {
    "diagnoses": [
        {
            "name": "Condition name",
            "confidence": 0.8,
            "explanation": "Reasoning citing symptoms, history, and retrieved knowledge",
            "recommendations": ["Recommendation 1", "Recommendation 2"],
        }
    ],
    "overall_assessment": "Summary of the differential diagnosis",
    "urgent_warning_signs": ["Warning sign if any"],
    "missing_information": ["Missing data that would improve accuracy"],
}


def get_diagnosis_response_schema() -> Dict[str, Any]:
    """Return the structured JSON schema expected from the diagnosis model."""
    return DIAGNOSIS_RESPONSE_SCHEMA


def _format_clinical_patient_info(patient_info: Dict[str, Any]) -> str:
    if not patient_info:
        return ""
    lines = []
    for key in CLINICAL_PATIENT_FIELDS:
        if key in patient_info and patient_info[key] is not None:
            lines.append(f"- {key.replace('_', ' ').title()}: {patient_info[key]}")
    return "\n".join(lines)


def _truncate_knowledge_items(items: List[str], max_items: int = MAX_KNOWLEDGE_ITEMS) -> List[str]:
    """Limit knowledge context size to reduce prompt bloat and improve focus."""
    truncated = []
    for item in items[:max_items]:
        text = item.strip()
        if len(text) > MAX_KNOWLEDGE_ITEM_CHARS:
            text = text[:MAX_KNOWLEDGE_ITEM_CHARS].rstrip() + "..."
        if text:
            truncated.append(text)
    return truncated


def create_diagnosis_prompt(
    primary_symptom: str,
    secondary_symptoms: List[str] = None,
    patient_info: Dict[str, Any] = None,
    medical_history: List[str] = None,
    duration_days: Optional[int] = None,
    context_info: Dict[str, Any] = None,
    relevant_medical_knowledge: List[str] = None,
    additional_notes: Optional[str] = None,
    analysis_params: Optional[Dict[str, Any]] = None,
    is_urgent: bool = False,
    urgent_symptoms: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Create a structured prompt for medical diagnosis based on patient information and symptoms."""
    secondary_symptoms = secondary_symptoms or []
    patient_info = patient_info or {}
    medical_history = medical_history or []
    context_info = dict(context_info or {})
    relevant_medical_knowledge = _truncate_knowledge_items(relevant_medical_knowledge or [])
    analysis_params = analysis_params or {}
    urgent_symptoms = urgent_symptoms or []

    if is_urgent:
        context_info["pre_screened_urgent_symptoms"] = urgent_symptoms

    patient_section = ""
    clinical_info = _format_clinical_patient_info(patient_info)
    if clinical_info:
        patient_section = f"## Patient Information\n{clinical_info}\n"

    history_section = ""
    if medical_history:
        history_section = "## Medical History\n"
        for condition in medical_history:
            history_section += f"- {condition}\n"

    symptoms_section = "## Symptoms\n"
    symptoms_section += f"- Primary: {primary_symptom}\n"

    if secondary_symptoms:
        symptoms_section += "- Secondary symptoms:\n"
        for symptom in secondary_symptoms:
            symptoms_section += f"  - {symptom}\n"

    if duration_days is not None:
        symptoms_section += f"- Duration: {duration_days} days\n"

    if is_urgent:
        symptoms_section += "- Pre-screened as potentially urgent\n"

    context_section = ""
    if context_info:
        context_section = "## Additional Context\n"
        for key, value in context_info.items():
            context_section += f"- {key}: {value}\n"

    notes_section = ""
    if additional_notes and additional_notes.strip():
        notes_section = f"## Clinical Notes\n{additional_notes.strip()}\n"

    knowledge_section = ""
    if relevant_medical_knowledge:
        knowledge_section = "## Relevant Medical Knowledge\n"
        for i, knowledge in enumerate(relevant_medical_knowledge, 1):
            knowledge_section += f"{i}. {knowledge}\n\n"

    output_requirements = [
        f"1. Up to {MAX_DIAGNOSES} most likely diagnoses with confidence levels (0.0-1.0)",
        "2. Brief explanation for each potential diagnosis citing the evidence",
        "3. Recommended next steps (tests, specialist referrals)",
        "4. Any urgent warning signs to be aware of",
    ]
    if analysis_params.get("include_treatment_suggestions"):
        output_requirements.append("5. Treatment suggestions where clinically appropriate")
    if analysis_params.get("include_preventive_measures"):
        output_requirements.append("6. Preventive measures relevant to the case")

    user_message = f"""
# Patient Case

{patient_section}
{symptoms_section}
{history_section}
{notes_section}
{context_section}
{knowledge_section}

Based on the above information, provide a differential diagnosis assessment covering:
{chr(10).join(output_requirements)}

Prioritize diagnoses supported by symptoms, medical history, and retrieved knowledge.
Note any important information that might be missing from the assessment.
Return your response as structured JSON matching the requested schema.
"""

    return {
        "system": DIAGNOSIS_SYSTEM_PROMPT,
        "user": user_message.strip(),
    }


def create_image_analysis_prompt(
    image_path: str,
    primary_concern: str = "",
    patient_info: Dict[str, Any] = None,
    relevant_context: str = "",
    body_area: str = "",
    medical_history: Optional[List[str]] = None,
) -> str:
    """Create a structured prompt for medical image analysis."""
    patient_info = patient_info or {}
    medical_history = medical_history or []

    clinical_info = _format_clinical_patient_info(patient_info)
    patient_info_text = f"Patient information:\n{clinical_info}\n" if clinical_info else ""

    history_text = ""
    if medical_history:
        history_text = "Medical history:\n" + "\n".join(f"- {item}" for item in medical_history) + "\n"

    context_text = ""
    if relevant_context:
        context = relevant_context.strip()
        if len(context) > MAX_KNOWLEDGE_ITEM_CHARS:
            context = context[:MAX_KNOWLEDGE_ITEM_CHARS].rstrip() + "..."
        context_text = f"Symptom and diagnostic context: {context}"

    prompt_parts = [
        "You are a medical image analysis expert. Analyze this medical image carefully and provide your assessment.",
        patient_info_text,
        history_text,
        f"Primary concern: {primary_concern}" if primary_concern else "",
        f"Body area: {body_area}" if body_area else "",
        context_text,
        """
Please provide:
1. Description of what you see in the image
2. Possible diagnoses based on visual characteristics
3. Confidence level for each possibility
4. Recommended follow-up diagnostic steps
5. Any concerning features that require urgent attention

Be specific and detailed in your analysis. Indicate if the image quality or angle limits your assessment.
Relate findings to the patient's symptoms and medical history when provided.
""".strip(),
    ]

    return "\n\n".join(part for part in prompt_parts if part).strip()


def create_symptom_exploration_prompt(symptom: str) -> Dict[str, str]:
    """Create a structured prompt for exploring and gathering more information about a symptom."""
    system_message = """
You are MediMatch, an AI medical assistant. Your task is to help healthcare providers
gather more detailed information about a patient's symptoms by suggesting relevant
follow-up questions. Provide a comprehensive but focused set of questions that will
help narrow down potential diagnoses.
"""

    user_message = f"""
The patient has reported "{symptom}" as a symptom.

Please provide a set of focused follow-up questions to gather more information about:
1. The nature and characteristics of this symptom
2. Timing and progression
3. Aggravating and relieving factors
4. Associated symptoms that may be relevant
5. Risk factors or history that might be important

Organize the questions in a logical order, starting with the most important ones.
"""

    return {
        "system": system_message,
        "user": user_message,
    }


def format_diagnosis_results(
    conditions: List[Dict[str, Any]],
    patient_info: Dict[str, Any] = None,
    detailed: bool = True,
) -> str:
    """Format diagnostic assessment results into a readable report."""
    patient_info = patient_info or {}

    report = "# Diagnostic Assessment\n\n"

    clinical_info = _format_clinical_patient_info(patient_info)
    if clinical_info:
        report += "## Patient Information\n"
        report += clinical_info + "\n\n"

    report += "## Potential Diagnoses\n\n"

    for i, condition in enumerate(conditions, 1):
        name = condition.get("name", "Unknown")
        confidence = condition.get("confidence", 0) * 100
        report += f"### {i}. {name} (Confidence: {confidence:.1f}%)\n\n"

        if detailed and condition.get("explanation"):
            report += f"{condition['explanation']}\n\n"

        recommendations = condition.get("recommendations") or []
        if recommendations:
            report += "**Recommendations:**\n"
            for rec in recommendations:
                report += f"- {rec}\n"
            report += "\n"

    report += """
This assessment is generated by an AI system and is not a substitute for professional
medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare
providers.
"""

    return report
