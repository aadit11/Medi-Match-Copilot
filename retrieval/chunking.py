import re
from typing import List, Dict, Any
from core.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text_by_tokens(
    text: str, 
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into chunks of approximately chunk_size tokens.
    
    This function splits text into chunks while trying to preserve sentence boundaries.
    It uses a simple token estimation (4 characters per token) and handles overlap
    between chunks to maintain context.
    
    Args:
        text (str): The input text to be split into chunks
        chunk_size (int, optional): Target size for each chunk in tokens. Defaults to CHUNK_SIZE.
        chunk_overlap (int, optional): Number of tokens to overlap between chunks. Defaults to CHUNK_OVERLAP.
    
    Returns:
        List[str]: List of text chunks, each approximately chunk_size tokens long
    """
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
    """Split a medical document into chunks with metadata.
    
    This function takes a medical document and splits its text content into chunks
    while preserving document metadata. Each chunk includes the original document's
    ID, title, and a unique chunk identifier.
    
    Args:
        document (Dict[str, Any]): Dictionary containing document data with keys:
            - id: Unique document identifier
            - title: Document title
            - text: Document content
        chunk_size (int, optional): Target size for each chunk in tokens. Defaults to CHUNK_SIZE.
        chunk_overlap (int, optional): Number of tokens to overlap between chunks. Defaults to CHUNK_OVERLAP.
    
    Returns:
        List[Dict[str, Any]]: List of chunks, each containing:
            - text: The chunk content
            - metadata: Dictionary with doc_id, title, and chunk_id
    """
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
    """Clean and normalize medical text for processing.
    
    This function performs several text normalization steps:
    1. Removes extra whitespace
    2. Expands common medical abbreviations
    3. Standardizes text format
    
    Args:
        text (str): Raw medical text to be preprocessed
    
    Returns:
        str: Cleaned and normalized medical text
    """
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