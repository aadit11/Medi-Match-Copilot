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
from retrieval.debug_instrumentation import agent_log

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
        """Generate an embedding vector for a query string."""
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Batch-embed multiple queries in a single model call."""
        if not queries:
            return []
        try:
            processed_queries = [preprocess_medical_text(q) for q in queries]
            embeddings = self.model.encode(processed_queries)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating query embeddings: {e}")
            raise

    def search_with_embedding(
        self,
        query_embedding: List[float],
        num_results: int = NUM_RETRIEVAL_RESULTS,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using a precomputed query embedding."""
        try:
            filter_fn = None
            if filters:
                def filter_fn(result: Dict[str, Any]) -> bool:
                    metadata = result.get("metadata", {})
                    return all(metadata.get(k) == v for k, v in filters.items())

            return self.vector_store.search(
                query_embedding=query_embedding,
                k=num_results,
                filter_fn=filter_fn,
            )
        except Exception as e:
            logger.error(f"Error searching with embedding: {e}")
            return []

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
            results = self.search_with_embedding(
                query_embedding=query_embedding,
                num_results=num_results,
                filters=filters,
            )
            
            # region agent log
            agent_log(
                "query_engine.py:search",
                "search completed",
                {
                    "query_preview": query[:80],
                    "num_results": len(results),
                    "top_score": results[0].get("score") if results else None,
                    "store_chunks": len(self.vector_store.metadata),
                },
                "C",
            )
            # endregion
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            logger.error(f"Error searching for query: {e}")
            return []

    def _get_patient_demographics(
        self, patient_info: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[int], Optional[str]]:
        if not patient_info:
            return None, None
        age = patient_info.get("age")
        gender = patient_info.get("gender") or patient_info.get("sex")
        return age, gender

    def _merge_search_results(
        self, result_lists: List[List[Dict[str, Any]]], max_results: int
    ) -> List[Dict[str, Any]]:
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for results in result_lists:
            for result in results:
                metadata = result.get("metadata", {})
                dedupe_key = metadata.get("chunk_id") or result.get("text", "")[:120]
                if dedupe_key and dedupe_key not in seen:
                    seen.add(dedupe_key)
                    merged.append(result)
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:max_results]
    
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
        
        age, gender = self._get_patient_demographics(patient_info)
        if age is not None:
            query_parts.append(f"For {age} year old patient")
        if gender:
            query_parts.append(f"Patient gender: {gender}")
        
        query = "\n".join(query_parts)
        return self.search(query, num_results)
    
    def retrieve_for_diagnosis(
        self, 
        primary_symptom: str,
        secondary_symptoms: Optional[List[str]] = None,
        patient_info: Optional[Dict[str, Any]] = None,
        medical_history: Optional[List[str]] = None,
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
        medical_history = medical_history or []
        age, gender = self._get_patient_demographics(patient_info)

        query_parts = [f"Diagnosis for primary symptom: {primary_symptom}"]
        if secondary_symptoms:
            query_parts.append("Additional symptoms:")
            query_parts.extend([f"- {symptom}" for symptom in secondary_symptoms])
        patient_desc = []
        if age is not None:
            patient_desc.append(f"{age} year old")
        if gender:
            patient_desc.append(str(gender))
        if patient_desc:
            query_parts.append(f"Patient: {' '.join(patient_desc)}")
        if medical_history:
            query_parts.append(f"Medical history: {', '.join(medical_history)}")

        query_specs: List[Tuple[str, int]] = [
            ("\n".join(query_parts), num_results),
        ]
        for symptom in secondary_symptoms[:3]:
            query_specs.append(
                (f"Diagnosis for symptom: {symptom}", max(2, num_results // 3))
            )
        if medical_history:
            history_query = (
                f"Differential diagnosis for patient with {', '.join(medical_history)} "
                f"presenting with {primary_symptom}"
            )
            query_specs.append((history_query, max(2, num_results // 3)))

        queries = [spec[0] for spec in query_specs]
        k_values = [spec[1] for spec in query_specs]
        embeddings = self.embed_queries(queries)

        all_results = [
            self.search_with_embedding(embedding, num_results=k)
            for embedding, k in zip(embeddings, k_values)
        ]

        merged = self._merge_search_results(all_results, num_results)

        # region agent log
        agent_log(
            "query_engine.py:retrieve_for_diagnosis",
            "diagnosis retrieval summary",
            {
                "num_search_calls": len(all_results),
                "raw_result_counts": [len(r) for r in all_results],
                "merged_count": len(merged),
                "top_merged_scores": [r.get("score") for r in merged[:3]],
                "has_patient_age": age is not None,
                "has_patient_gender": bool(gender),
                "secondary_symptom_count": len(secondary_symptoms),
            },
            "A",
        )
        # endregion

        return merged
    
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
        max_items: int = NUM_RETRIEVAL_RESULTS
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
            
            if len(text) < 30:
                continue
            
            section = metadata.get("section", "")
            
            knowledge_item = text
            if section:
                knowledge_item += f"\n(From section: {section})"
            
            knowledge_items.append(knowledge_item)
        
        return knowledge_items
    
    def format_for_diagnosis(self, results: List[Dict[str, Any]], max_items: int = NUM_RETRIEVAL_RESULTS) -> str:
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

_default_query_engine: Optional["QueryEngine"] = None


def create_query_engine() -> QueryEngine:
    """Create or reuse a configured QueryEngine instance."""
    global _default_query_engine
    if _default_query_engine is None:
        _default_query_engine = QueryEngine(
            embedding_model=EMBEDDING_MODEL,
            vector_db_path=VECTOR_DB_PATH,
        )
    return _default_query_engine

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