from typing import Dict, List, Any, Optional
from core.config import DIAGNOSIS_SYSTEM_PROMPT

def create_diagnosis_prompt(
    primary_symptom: str,
    secondary_symptoms: List[str] = None,
    patient_info: Dict[str, Any] = None,
    medical_history: List[str] = None,
    duration_days: Optional[int] = None,
    context_info: Dict[str, Any] = None,
    relevant_medical_knowledge: List[str] = None
) -> Dict[str, str]:
    """Create a structured prompt for medical diagnosis based on patient information and symptoms.
    
    This function generates a comprehensive prompt that includes patient information,
    symptoms (both primary and secondary), medical history, and relevant medical knowledge
    to help the AI model generate accurate diagnostic assessments.
    
    Args:
        primary_symptom (str): The main symptom reported by the patient
        secondary_symptoms (List[str], optional): Additional symptoms reported by the patient
        patient_info (Dict[str, Any], optional): Patient demographic and medical information
        medical_history (List[str], optional): List of previous medical conditions
        duration_days (Optional[int], optional): Number of days the symptoms have persisted
        context_info (Dict[str, Any], optional): Additional contextual information
        relevant_medical_knowledge (List[str], optional): List of relevant medical facts
        
    Returns:
        Dict[str, str]: A dictionary containing:
            - system: The system prompt for the AI model
            - user: The formatted user message with patient case details
    """
    secondary_symptoms = secondary_symptoms or []
    patient_info = patient_info or {}
    medical_history = medical_history or []
    context_info = context_info or {}
    relevant_medical_knowledge = relevant_medical_knowledge or []
    
    patient_section = ""
    if patient_info:
        patient_section = "## Patient Information\n"
        for key, value in patient_info.items():
            patient_section += f"- {key}: {value}\n"
    
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
    
    context_section = ""
    if context_info:
        context_section = "## Additional Context\n"
        for key, value in context_info.items():
            context_section += f"- {key}: {value}\n"
    
    knowledge_section = ""
    if relevant_medical_knowledge:
        knowledge_section = "## Relevant Medical Knowledge\n"
        for i, knowledge in enumerate(relevant_medical_knowledge, 1):
            knowledge_section += f"{i}. {knowledge}\n\n"
    
    user_message = f"""
# Patient Case

{patient_section}
{symptoms_section}
{history_section}
{context_section}
{knowledge_section}

Based on the above information, please provide:
1. Most likely diagnoses (maximum 5) with confidence levels
2. Brief explanation for each potential diagnosis
3. Recommended next steps (tests, specialist referrals)
4. Any urgent warning signs to be aware of

Please note any important information that might be missing from the assessment.
"""
    
    return {
        "system": DIAGNOSIS_SYSTEM_PROMPT,
        "user": user_message.strip()
    }

def create_image_analysis_prompt(
    image_path: str,
    primary_concern: str = "",
    patient_info: Dict[str, Any] = None,
    relevant_context: str = "",
    body_area: str = ""
) -> str:
    """Create a structured prompt for medical image analysis.
    
    This function generates a comprehensive prompt for analyzing medical images,
    including patient information, primary concerns, and specific body areas to focus on.
    
    Args:
        image_path (str): Path to the medical image to be analyzed
        primary_concern (str, optional): The main medical concern or symptom
        patient_info (Dict[str, Any], optional): Patient demographic and medical information
        relevant_context (str, optional): Additional context about the image or patient
        body_area (str, optional): Specific body area shown in the image
        
    Returns:
        str: A formatted prompt string for medical image analysis
    """
    patient_info = patient_info or {}
    
    patient_info_text = ""
    if patient_info:
        patient_info_text = "Patient information:\n"
        for key, value in patient_info.items():
            patient_info_text += f"- {key}: {value}\n"
    
    prompt = f"""
You are a medical image analysis expert. Analyze this medical image carefully and provide your assessment.

{patient_info_text}
{"Primary concern: " + primary_concern if primary_concern else ""}
{"Body area: " + body_area if body_area else ""}
{"Additional context: " + relevant_context if relevant_context else ""}

Please provide:
1. Description of what you see in the image
2. Possible diagnoses based on visual characteristics 
3. Confidence level for each possibility
4. Recommended follow-up diagnostic steps
5. Any concerning features that require urgent attention

Be specific and detailed in your analysis. Indicate if the image quality or angle limits your assessment.
"""
    
    return prompt.strip()

def create_symptom_exploration_prompt(symptom: str) -> Dict[str, str]:
    """Create a structured prompt for exploring and gathering more information about a symptom.
    
    This function generates a prompt that helps healthcare providers gather detailed
    information about a patient's symptom through a series of focused questions.
    
    Args:
        symptom (str): The symptom to explore in detail
        
    Returns:
        Dict[str, str]: A dictionary containing:
            - system: The system prompt defining the AI's role
            - user: The formatted user message requesting symptom exploration questions
    """
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
        "user": user_message
    }

def format_diagnosis_results(
    conditions: List[Dict[str, Any]],
    patient_info: Dict[str, Any] = None,
    detailed: bool = True
) -> str:
    """Format diagnostic assessment results into a readable report.
    
    This function takes the raw diagnostic results and formats them into a
    well-structured, human-readable report that includes patient information,
    potential diagnoses with confidence levels, explanations, and recommendations.
    
    Args:
        conditions (List[Dict[str, Any]]): List of potential diagnoses with their details
        patient_info (Dict[str, Any], optional): Patient demographic and medical information
        detailed (bool, optional): Whether to include detailed explanations for each diagnosis
        
    Returns:
        str: A formatted markdown report of the diagnostic assessment
    """
    patient_info = patient_info or {}
    
    report = "# Diagnostic Assessment\n\n"
    
    if patient_info:
        report += "## Patient Information\n"
        for key, value in patient_info.items():
            report += f"- **{key}**: {value}\n"
        report += "\n"
    
    report += "## Potential Diagnoses\n\n"
    
    for i, condition in enumerate(conditions, 1):
        confidence = condition.get("confidence", 0) * 100
        report += f"### {i}. {condition['name']} (Confidence: {confidence:.1f}%)\n\n"
        
        if detailed and "explanation" in condition:
            report += f"{condition['explanation']}\n\n"
        
        if "recommendations" in condition:
            report += "**Recommendations:**\n"
            for rec in condition["recommendations"]:
                report += f"- {rec}\n"
            report += "\n"
    
    report += """
This assessment is generated by an AI system and is not a substitute for professional 
medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare 
providers.
"""
    
    return report