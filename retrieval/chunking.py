import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from core.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text_by_tokens(
    text: str, 
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    token_estimator: Callable[[str], int] = None
) -> List[str]:
    
    if token_estimator is None:
        token_estimator = lambda x: len(x) // 4
    
    if token_estimator(text) <= chunk_size:
        return [text]
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = token_estimator(sentence)
        
        if sentence_size > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
                
            comma_parts = re.split(r'(?<=,)\s+', sentence)
            
            if any(token_estimator(part) > chunk_size for part in comma_parts):
                words = sentence.split()
                temp_chunk = []
                temp_size = 0
                
                for word in words:
                    word_size = token_estimator(word + " ")
                    if temp_size + word_size > chunk_size:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_size = word_size
                    else:
                        temp_chunk.append(word)
                        temp_size += word_size
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            else:
                for part in comma_parts:
                    part_size = token_estimator(part)
                    if current_size + part_size > chunk_size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [part]
                        current_size = part_size
                    else:
                        current_chunk.append(part)
                        current_size += part_size
        else:
            if current_size + sentence_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                
                if chunk_overlap > 0 and current_chunk:
                    overlap_size = 0
                    overlap_chunk = []
                    
                    for s in reversed(current_chunk):
                        s_size = token_estimator(s)
                        if overlap_size + s_size <= chunk_overlap:
                            overlap_chunk.insert(0, s)
                            overlap_size += s_size
                        else:
                            break
                    
                    current_chunk = overlap_chunk
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def split_medical_text_by_sections(text: str, max_chunk_size: int = CHUNK_SIZE) -> List[Dict[str, Any]]:
   
    section_patterns = [
        r"(?:\n|^)#+\s*(History|Background|Introduction|Patient History|Medical History)[\s:]*\n",
        r"(?:\n|^)#+\s*(Symptoms|Clinical Presentation|Presenting Symptoms|Chief Complaint)[\s:]*\n",
        r"(?:\n|^)#+\s*(Diagnosis|Assessment|Clinical Assessment|Differential Diagnosis)[\s:]*\n",
        r"(?:\n|^)#+\s*(Treatment|Management|Therapy|Intervention|Therapeutic Plan)[\s:]*\n",
        r"(?:\n|^)#+\s*(Discussion|Interpretation|Analysis|Clinical Significance)[\s:]*\n",
        r"(?:\n|^)#+\s*(Prognosis|Outcome|Expected Outcome|Follow-up)[\s:]*\n",
        r"(?:\n|^)#+\s*(Recommendation|Conclusion|Summary|Key Points)[\s:]*\n",
        r"(?:\n|^)#+\s*(References|Sources|Citations|Bibliography)[\s:]*\n",
    ]
    
    combined_pattern = "|".join(section_patterns)
    
    sections = re.split(combined_pattern, text)
    headers = re.findall(combined_pattern, text)
    
    if len(sections) > len(headers) + 1:
        processed_sections = [{"text": sections[0], "section": "Introduction"}]
        for i, header in enumerate(headers):
            processed_sections.append({
                "text": sections[i+1], 
                "section": header.strip('# \n:')
            })
    elif len(sections) == len(headers):
        processed_sections = []
        for i, header in enumerate(headers):
            processed_sections.append({
                "text": sections[i], 
                "section": header.strip('# \n:')
            })
    else:
        processed_sections = [{"text": text, "section": "Full Document"}]
    
    result_chunks = []
    for section in processed_sections:
        if len(section["text"]) > max_chunk_size:
            subchunks = split_text_by_tokens(section["text"], max_chunk_size)
            for i, subchunk in enumerate(subchunks):
                result_chunks.append({
                    "text": subchunk,
                    "section": section["section"],
                    "subsection": f"Part {i+1}"
                })
        else:
            result_chunks.append(section)
    
    return result_chunks

def chunk_medical_document(
    document: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    
    chunks = []
    
    doc_id = document.get("id", "unknown")
    doc_title = document.get("title", "Untitled")
    doc_source = document.get("source", "Unknown")
    doc_text = document.get("text", "")
    doc_type = document.get("doc_type", "medical")
    
    if doc_type in ["medical_literature", "research_paper", "clinical_guidelines"]:
        raw_chunks = split_medical_text_by_sections(doc_text, chunk_size)
        
        for i, chunk in enumerate(raw_chunks):
            chunks.append({
                "text": chunk["text"],
                "metadata": {
                    "doc_id": doc_id,
                    "title": doc_title,
                    "source": doc_source,
                    "section": chunk.get("section", "Unknown"),
                    "subsection": chunk.get("subsection", ""),
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_type": doc_type
                }
            })
    else:
        raw_chunks = split_text_by_tokens(doc_text, chunk_size, chunk_overlap)
        
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "title": doc_title,
                    "source": doc_source,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_type": doc_type
                }
            })
    
    return chunks

def preprocess_medical_text(text: str) -> str:
    
    text = re.sub(r'\s+', ' ', text)
    
    medical_abbreviations = {
        r'\bpt\b': 'patient',
        r'\bDx\b': 'diagnosis',
        r'\bHx\b': 'history',
        r'\bTx\b': 'treatment',
        r'\bRx\b': 'prescription',
        r'\bSx\b': 'symptoms',
        r'\bFx\b': 'fracture',
        r'\bCx\b': 'complications',
    }
    
    for abbr, expanded in medical_abbreviations.items():
        text = re.sub(abbr, expanded, text, flags=re.IGNORECASE)
    
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    return text.strip()