import re
from typing import List, Dict, Any
from core.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text_by_tokens(
    text: str, 
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into chunks of approximately chunk_size tokens."""
    token_estimator = lambda x: len(x) // 4
    
    if token_estimator(text) <= chunk_size:
        return [text]
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = token_estimator(sentence)
        
        if current_size + sentence_size > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                if chunk_overlap > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = token_estimator(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunk_medical_document(
    document: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Split a document into chunks with metadata."""
    chunks = []
    
    doc_id = document["id"]
    doc_title = document["title"]
    doc_text = document["text"]
    
    raw_chunks = split_text_by_tokens(doc_text, chunk_size, chunk_overlap)
    
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "doc_id": doc_id,
                "title": doc_title,
                "chunk_id": f"{doc_id}_chunk_{i}"
            }
        })
    
    return chunks

def preprocess_medical_text(text: str) -> str:
    """Clean and normalize medical text."""
    text = re.sub(r'\s+', ' ', text)
    
    medical_abbreviations = {
        r'\bpt\b': 'patient',
        r'\bDx\b': 'diagnosis',
        r'\bHx\b': 'history',
        r'\bTx\b': 'treatment',
        r'\bRx\b': 'prescription',
        r'\bSx\b': 'symptoms',
    }
    
    for abbr, full in medical_abbreviations.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
    
    return text.strip()