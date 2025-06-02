import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from core.config import (
    EMBEDDING_MODEL, 
    VECTOR_DB_PATH, 
    NUM_RETRIEVAL_RESULTS,
    MEDICAL_KB_DIR
)
from retrieval.vector_store import VectorStore
from retrieval.chunking import preprocess_medical_text

logger = logging.getLogger(__name__)

class QueryEngine:
    """A medical knowledge retrieval engine that uses semantic search to find relevant medical information.
    
    This class provides functionality to search through medical knowledge using semantic embeddings
    and vector similarity search. It can handle various types of medical queries including symptoms,
    diagnoses, and conditions.
    """
   
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        vector_db_path: str = VECTOR_DB_PATH
    ):
        """Initialize the QueryEngine with an embedding model and vector database.
        
        Args:
            embedding_model: Name/path of the sentence transformer model to use
            vector_db_path: Path to the vector database containing medical knowledge
        """
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        
        try:
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        self.vector_store = VectorStore(vector_db_path)
    
    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding vector for a query string.
        
        Args:
            query: The text query to embed
            
        Returns:
            List of floats representing the query embedding
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            processed_query = preprocess_medical_text(query)
            embedding = self.model.encode(processed_query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        num_results: int = NUM_RETRIEVAL_RESULTS,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for medical information relevant to a query.
        
        Args:
            query: The search query text
            num_results: Maximum number of results to return
            filters: Optional metadata filters to apply to results
            
        Returns:
            List of dictionaries containing search results with text and metadata
        """
        try:
            query_embedding = self.embed_query(query)
            
            filter_fn = None
            if filters:
                def filter_fn(result: Dict[str, Any]) -> bool:
                    metadata = result.get("metadata", {})
                    return all(metadata.get(k) == v for k, v in filters.items())
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=num_results,
                filter_fn=filter_fn
            )
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            logger.error(f"Error searching for query: {e}")
            return []
    
    def retrieve_for_symptoms(
        self, 
        symptoms: List[str],
        patient_info: Optional[Dict[str, Any]] = None,
        num_results: int = NUM_RETRIEVAL_RESULTS
    ) -> List[Dict[str, Any]]:
        """Search for medical information about specific symptoms.
        
        Args:
            symptoms: List of symptoms to search for
            patient_info: Optional patient information (age, sex) to include in search
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing relevant medical information
        """
        if not symptoms:
            logger.warning("No symptoms provided for retrieval")
            return []
        
        query_parts = ["Medical information about symptoms:"]
        query_parts.extend([f"- {symptom}" for symptom in symptoms])
        
        if patient_info:
            age = patient_info.get("age")
            sex = patient_info.get("sex")
            
            if age is not None:
                query_parts.append(f"For {age} year old patient")
            
            if sex:
                query_parts.append(f"Patient gender: {sex}")
        
        query = "\n".join(query_parts)
        return self.search(query, num_results)
    
    def retrieve_for_diagnosis(
        self, 
        primary_symptom: str,
        secondary_symptoms: Optional[List[str]] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        num_results: int = NUM_RETRIEVAL_RESULTS
    ) -> List[Dict[str, Any]]:
        """Search for diagnostic information based on symptoms.
        
        Args:
            primary_symptom: The main symptom to diagnose
            secondary_symptoms: Optional list of additional symptoms
            patient_info: Optional patient information (age, sex)
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing diagnostic information
        """
        secondary_symptoms = secondary_symptoms or []
        
        query_parts = [f"Diagnosis for primary symptom: {primary_symptom}"]
        
        if secondary_symptoms:
            query_parts.append("Additional symptoms:")
            query_parts.extend([f"- {symptom}" for symptom in secondary_symptoms])
        
        if patient_info:
            age = patient_info.get("age")
            sex = patient_info.get("sex")
            
            patient_desc = []
            if age is not None:
                patient_desc.append(f"{age} year old")
            
            if sex:
                patient_desc.append(sex)
            
            if patient_desc:
                query_parts.append(f"Patient: {' '.join(patient_desc)}")
        
        query = "\n".join(query_parts)
        return self.search(query, num_results)
    
    def retrieve_for_condition(
        self,
        condition: str,
        aspect: Optional[str] = None,
        num_results: int = NUM_RETRIEVAL_RESULTS
    ) -> List[Dict[str, Any]]:
        """Search for information about a specific medical condition.
        
        Args:
            condition: The medical condition to search for
            aspect: Optional specific aspect of the condition to focus on
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing condition information
        """
        if aspect:
            query = f"Information about {aspect} of {condition}"
        else:
            query = f"Medical information about {condition}"
        
        return self.search(query, num_results)
    
    def extract_relevant_knowledge(
        self,
        results: List[Dict[str, Any]],
        max_items: int = 5
    ) -> List[str]:
        """Extract and format the most relevant knowledge from search results.
        
        Args:
            results: List of search result dictionaries
            max_items: Maximum number of knowledge items to extract
            
        Returns:
            List of formatted knowledge strings
        """
        if not results:
            return []
        
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        knowledge_items = []
        
        for result in sorted_results[:max_items]:
            text = result.get("text", "").strip()
            metadata = result.get("metadata", {})
            
            if len(text) < 50:
                continue
            
            section = metadata.get("section", "")
            
            knowledge_item = text
            if section:
                knowledge_item += f"\n(From section: {section})"
            
            knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def format_for_diagnosis(self, results: List[Dict[str, Any]], max_items: int = 3) -> str:
        """Format search results into a structured diagnostic report.
        
        Args:
            results: List of search result dictionaries
            max_items: Maximum number of results to include
            
        Returns:
            Formatted string containing diagnostic information
        """
        if not results:
            return ""
        
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        sorted_results = sorted_results[:max_items]
        
        context_parts = ["# Relevant Medical Knowledge"]
        
        for i, result in enumerate(sorted_results, 1):
            text = result.get("text", "").strip()
            metadata = result.get("metadata", {})
            
            title = metadata.get("title", "Medical Reference")
            section = metadata.get("section", "")
            
            header = f"## {i}. "
            if section:
                header += f"{section} - {title}"
            else:
                header += title
            
            context_parts.append(header)
            context_parts.append(text)
            context_parts.append("")  
        return "\n".join(context_parts)

def create_query_engine() -> QueryEngine:
    """Create and configure a new QueryEngine instance with default settings.
    
    Returns:
        A configured QueryEngine instance
    """
    return QueryEngine(
        embedding_model=EMBEDDING_MODEL,
        vector_db_path=VECTOR_DB_PATH
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = create_query_engine()
    
    test_query = "chest pain with shortness of breath"
    results = engine.search(test_query)
    
    print(f"Found {len(results)} results for '{test_query}'")
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
        print(f"Text: {result['text'][:150]}...")
        metadata = result.get("metadata", {})
        print(f"Source: {metadata.get('title', 'Unknown')}")