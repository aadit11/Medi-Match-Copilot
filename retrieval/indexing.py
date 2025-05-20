import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from core.config import (
    EMBEDDINGS_DIR,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from retrieval.chunking import chunk_medical_document, preprocess_medical_text
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        vector_db_path: str = VECTOR_DB_PATH,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        self.vector_store = VectorStore(vector_db_path)
    
    def preprocess_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
       
        if "text" in document:
            document["text"] = preprocess_medical_text(document["text"])
        
        return document
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
            
            return chunks
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def index_document(self, document: Dict[str, Any]) -> int:
        
        try:
            processed_doc = self.preprocess_document(document)
            
            chunks = chunk_medical_document(
                processed_doc,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            embedded_chunks = self.embed_chunks(chunks)
            
            for chunk in embedded_chunks:
                self.vector_store.add_document(
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                    embedding=chunk["embedding"]
                )
            
            logger.info(f"Indexed document {document.get('id', 'unknown')} with {len(chunks)} chunks")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Error indexing document {document.get('id', 'unknown')}: {e}")
            return 0
    
    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> Tuple[int, int]:
        
        total_chunks = 0
        processed_docs = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            for doc in tqdm(batch, desc=f"Indexing batch {i//batch_size + 1}"):
                chunks_indexed = self.index_document(doc)
                
                if chunks_indexed > 0:
                    processed_docs += 1
                    total_chunks += chunks_indexed
            
            self.vector_store.commit()
        
        logger.info(f"Indexed {processed_docs}/{len(documents)} documents with {total_chunks} total chunks")
        return processed_docs, total_chunks
    
    def index_directory(self, directory_path: str, file_extensions: List[str] = ['.txt', '.md', '.json']) -> Tuple[int, int]:
        
        documents = []
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return 0, 0
        
        for ext in file_extensions:
            files = list(path.glob(f"**/*{ext}"))
            
            for file_path in files:
                try:
                    if ext == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and "text" in item:
                                        item["source"] = str(file_path)
                                        item["id"] = f"{file_path.stem}_{id(item)}"
                                        documents.append(item)
                            elif isinstance(content, dict) and "text" in content:
                                content["source"] = str(file_path)
                                content["id"] = file_path.stem
                                documents.append(content)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            document = {
                                "id": file_path.stem,
                                "title": file_path.name,
                                "source": str(file_path),
                                "text": text,
                                "doc_type": "medical"
                            }
                            documents.append(document)
                            
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Found {len(documents)} documents in {directory_path}")
        return self.index_documents(documents)
    
    def reindex_all(self, directory_path: str) -> None:
        
        logger.info("Starting complete reindex")
        
        self.vector_store.clear()
        
        docs_indexed, chunks_indexed = self.index_directory(directory_path)
        
        logger.info(f"Reindex complete: {docs_indexed} documents, {chunks_indexed} chunks")
    
    def get_index_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()

def create_indexer(
    embedding_model: str = EMBEDDING_MODEL,
    vector_db_path: str = VECTOR_DB_PATH
) -> DocumentIndexer:
   
    return DocumentIndexer(
        embedding_model=embedding_model,
        vector_db_path=vector_db_path
    )

def convert_medical_kb_to_documents(kb_dir: Path) -> List[Dict[str, Any]]:
    """Convert medical knowledge base JSON files into indexable documents."""
    documents = []
    
    conditions_path = kb_dir / "conditions_database.json"
    if conditions_path.exists():
        try:
            with open(conditions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for condition_id, condition in data.get("conditions", {}).items():
                    text_parts = [
                        f"Condition: {condition['name']}",
                        f"Description: {condition.get('description', '')}",
                        "\nSymptoms:",
                        *[f"- {symptom}" for symptom in condition.get('symptoms', [])],
                        f"\nDuration: {condition.get('duration', 'N/A')}"
                    ]
                    
                    treatment = condition.get('treatment', {})
                    if treatment:
                        text_parts.append("\nTreatment:")
                        if 'self_care' in treatment:
                            text_parts.append("Self-care measures:")
                            text_parts.extend([f"- {measure}" for measure in treatment['self_care']])
                        if 'medical_treatment' in treatment:
                            text_parts.append("\nMedical treatment:")
                            text_parts.extend([f"- {treatment}" for treatment in treatment['medical_treatment']])
                        if 'lifestyle_changes' in treatment:
                            text_parts.append("\nLifestyle changes:")
                            text_parts.extend([f"- {change}" for change in treatment['lifestyle_changes']])
                        if 'medications' in treatment:
                            text_parts.append("\nMedications:")
                            text_parts.extend([f"- {med}" for med in treatment['medications']])
                    
                    if 'prevention' in condition:
                        text_parts.append("\nPrevention:")
                        text_parts.extend([f"- {prevention}" for prevention in condition['prevention']])
                    
                    if 'complications' in condition:
                        text_parts.append("\nComplications:")
                        text_parts.extend([f"- {complication}" for complication in condition['complications']])
                    
                    if 'risk_factors' in condition:
                        text_parts.append("\nRisk Factors:")
                        text_parts.extend([f"- {factor}" for factor in condition['risk_factors']])
                    
                    if 'urgent_warning_signs' in condition:
                        text_parts.append("\nUrgent Warning Signs:")
                        text_parts.extend([f"- {warning}" for warning in condition['urgent_warning_signs']])
                    
                    if 'diagnosis' in condition:
                        text_parts.append("\nDiagnosis Criteria:")
                        for stage, criteria in condition['diagnosis'].items():
                            text_parts.append(f"- {stage.replace('_', ' ').title()}: {criteria}")
                    
                    documents.append({
                        "id": f"condition_{condition_id}",
                        "title": condition['name'],
                        "text": "\n".join(text_parts),
                        "source": "conditions_database.json",
                        "doc_type": "medical_literature"
                    })
        except Exception as e:
            logger.error(f"Error processing conditions database: {e}")
    
    symptoms_path = kb_dir / "symptoms_database.json"
    if symptoms_path.exists():
        try:
            with open(symptoms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for symptom_id, symptom in data.get("symptoms", {}).items():
                    text_parts = [
                        f"Symptom: {symptom['name']}",
                        f"Description: {symptom.get('description', '')}"
                    ]
                    
                    if 'common_causes' in symptom:
                        text_parts.append("\nCommon Causes:")
                        text_parts.extend([f"- {cause}" for cause in symptom['common_causes']])
                    
                    if 'seek_medical_attention' in symptom:
                        text_parts.append("\nWhen to Seek Medical Attention:")
                        text_parts.extend([f"- {warning}" for warning in symptom['seek_medical_attention']])
                    
                    if 'self_care_measures' in symptom:
                        text_parts.append("\nSelf-Care Measures:")
                        text_parts.extend([f"- {measure}" for measure in symptom['self_care_measures']])
                    
                    documents.append({
                        "id": f"symptom_{symptom_id}",
                        "title": symptom['name'],
                        "text": "\n".join(text_parts),
                        "source": "symptoms_database.json",
                        "doc_type": "medical_literature"
                    })
        except Exception as e:
            logger.error(f"Error processing symptoms database: {e}")
    
    medications_path = kb_dir / "medications_database.json"
    if medications_path.exists():
        try:
            with open(medications_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for med_id, medication in data.get("medications", {}).items():
                    text_parts = [
                        f"Medication: {medication['name']}",
                        f"Description: {medication.get('description', '')}"
                    ]
                    
                    if 'uses' in medication:
                        text_parts.append("\nUses:")
                        text_parts.extend([f"- {use}" for use in medication['uses']])
                    
                    if 'dosage' in medication:
                        text_parts.append(f"\nDosage: {medication['dosage']}")
                    
                    if 'side_effects' in medication:
                        text_parts.append("\nSide Effects:")
                        text_parts.extend([f"- {effect}" for effect in medication['side_effects']])
                    
                    if 'precautions' in medication:
                        text_parts.append("\nPrecautions:")
                        text_parts.extend([f"- {precaution}" for precaution in medication['precautions']])
                    
                    if 'interactions' in medication:
                        text_parts.append("\nInteractions:")
                        text_parts.extend([f"- {interaction}" for interaction in medication['interactions']])
                    
                    documents.append({
                        "id": f"medication_{med_id}",
                        "title": medication['name'],
                        "text": "\n".join(text_parts),
                        "source": "medications_database.json",
                        "doc_type": "medical_literature"
                    })
        except Exception as e:
            logger.error(f"Error processing medications database: {e}")
    
    logger.info(f"Converted {len(documents)} documents from medical knowledge base")
    return documents

def index_directory(directory_path: str) -> Tuple[int, int]:
    """Index all documents in the specified directory."""
    kb_dir = Path(directory_path)
    if not kb_dir.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return 0, 0
    
    indexer = create_indexer()
    
    try:
        documents = convert_medical_kb_to_documents(kb_dir)
        
        if not documents:
            logger.error("No valid documents found to index")
            return 0, 0
        
        doc_count, chunk_count = indexer.index_documents(documents)
        
        if doc_count == 0:
            logger.error("Failed to index any documents")
        else:
            logger.info(f"Successfully indexed {doc_count} documents with {chunk_count} chunks")
        
        return doc_count, chunk_count
    
    except Exception as e:
        logger.error(f"Error indexing directory: {e}")
        return 0, 0

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = str(Path(__file__).parent.parent / "data" / "medical_kb")
    
    doc_count, chunk_count = index_directory(input_dir)
    print(f"Indexed {doc_count} documents with {chunk_count} chunks")
    
    stats = create_indexer().get_index_stats()
    print(f"Index stats: {stats}")