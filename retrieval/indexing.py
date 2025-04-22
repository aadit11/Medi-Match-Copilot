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
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        return self.vector_store.get_stats()

def create_indexer(
    embedding_model: str = EMBEDDING_MODEL,
    vector_db_path: str = VECTOR_DB_PATH
) -> DocumentIndexer:
   
    return DocumentIndexer(
        embedding_model=embedding_model,
        vector_db_path=vector_db_path
    )

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = str(Path(__file__).parent.parent / "data" / "medical_kb")
    
    indexer = create_indexer()
    
    doc_count, chunk_count = indexer.index_directory(input_dir)
    print(f"Indexed {doc_count} documents with {chunk_count} chunks")
    
    stats = indexer.get_index_stats()
    print(f"Index stats: {stats}")