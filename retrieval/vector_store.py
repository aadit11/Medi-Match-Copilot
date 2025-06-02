import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
from datetime import datetime

from core.config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    NUM_RETRIEVAL_RESULTS
)

logger = logging.getLogger(__name__)

class VectorStore:
    """A vector store implementation for efficient similarity search of document embeddings.
    
    This class provides functionality to store, retrieve, and search through document embeddings
    using cosine similarity. It maintains both the embeddings and associated metadata in memory
    and persists them to disk. The store supports adding new documents, searching for similar
    documents, and maintaining statistics about the stored documents.
    
    Attributes:
        db_path (Path): Path to the directory where vector store data is persisted
        embeddings_path (Path): Path to the numpy array file storing embeddings
        metadata_path (Path): Path to the pickle file storing document metadata
        stats_path (Path): Path to the JSON file storing vector store statistics
        embeddings (np.ndarray): In-memory storage of document embeddings
        metadata (List[Dict]): In-memory storage of document metadata
    """
    
    def __init__(self, db_path: str = VECTOR_DB_PATH):
        """Initialize the vector store.
        
        Args:
            db_path (str): Path to the directory where vector store data will be stored.
                          Defaults to VECTOR_DB_PATH from config.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path = self.db_path / "embeddings.npy"
        self.metadata_path = self.db_path / "metadata.pkl"
        self.stats_path = self.db_path / "stats.json"
        
        # Initialize in-memory storage
        self.embeddings = None
        self.metadata = []
        self.load_or_create()
    
    def load_or_create(self) -> None:
        """Load existing vector store data or create a new empty store.
        
        Attempts to load embeddings and metadata from disk. If either file is missing
        or loading fails, creates a new empty vector store.
        """
        if self.embeddings_path.exists() and self.metadata_path.exists():
            try:
                self.embeddings = np.load(str(self.embeddings_path))
                
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded existing vector store with {len(self.metadata)} documents")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self._create_new()
        else:
            self._create_new()
    
    def _create_new(self) -> None:
        """Create a new empty vector store.
        
        Initializes empty embeddings array and metadata list. This is used both for
        creating a new store and clearing an existing one.
        """
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.metadata = []
        logger.info("Created new empty vector store")
    
    def add_document(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        embedding: List[float]
    ) -> bool:
        """Add a new document to the vector store.
        
        Args:
            text (str): The document text content
            metadata (Dict[str, Any]): Document metadata (e.g. doc_id, title, etc.)
            embedding (List[float]): Document embedding vector
            
        Returns:
            bool: True if document was added successfully, False otherwise
        """
        try:
            doc_metadata = {
                "text": text,
                "metadata": metadata
            }
            
            vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
            
            if self.embeddings.size == 0:
                self.embeddings = vector
            else:
                self.embeddings = np.vstack((self.embeddings, vector))
            
            self.metadata.append(doc_metadata)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def commit(self) -> bool:
        """Persist the current state of the vector store to disk.
        
        Saves embeddings, metadata, and statistics to their respective files.
        
        Returns:
            bool: True if commit was successful, False otherwise
        """
        try:
            np.save(str(self.embeddings_path), self.embeddings)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            stats = self.get_stats()
            with open(self.stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Committed vector store with {len(self.metadata)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Error committing vector store: {e}")
            return False
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = NUM_RETRIEVAL_RESULTS,
        filter_fn: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query_embedding (List[float]): The query vector to search with
            k (int): Number of results to return. Defaults to NUM_RETRIEVAL_RESULTS
            filter_fn (Optional[Callable]): Optional function to filter results.
                                          Takes a result dict and returns bool.
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - text: The document text
                - metadata: Document metadata
                - score: Similarity score (cosine similarity)
        """
        if self.embeddings.size == 0 or len(self.metadata) == 0:
            logger.warning("Empty vector store, no results to return")
            return []
        
        try:
            query_vector = np.array(query_embedding, dtype=np.float32)
            
            norm_query = np.linalg.norm(query_vector)
            norm_docs = np.linalg.norm(self.embeddings, axis=1)
            dot_product = np.dot(self.embeddings, query_vector)
            
            similarities = dot_product / (norm_query * norm_docs + 1e-8)
            
            top_indices = np.argsort(similarities)[::-1][:k].tolist()
            
            results = []
            for idx in top_indices:
                doc = self.metadata[idx]
                score = float(similarities[idx])
                
                result = {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": score
                }
                
                if filter_fn is None or filter_fn(result):
                    results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def clear(self) -> bool:
        """Clear all documents from the vector store.
        
        Removes all embeddings and metadata, both from memory and disk.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            self._create_new()
            self.commit()
            logger.info("Vector store cleared")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - document_count: Number of unique documents
                - chunk_count: Total number of chunks
                - last_updated: Timestamp of last update
                - embedding_dimension: Dimension of embedding vectors
        """
        document_ids = set()
        for doc in self.metadata:
            doc_id = doc.get("metadata", {}).get("doc_id")
            if doc_id:
                document_ids.add(doc_id)
        
        stats = {
            "document_count": len(document_ids),
            "chunk_count": len(self.metadata),
            "last_updated": datetime.now().isoformat(),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings.size > 0 else 0
        }
        
        return stats    