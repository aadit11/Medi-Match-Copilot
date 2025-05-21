import os
import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
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
        """Preprocess a document, ensuring it has the required fields."""
        required_fields = ["id", "title", "content"]
        
        if not all(field in document for field in required_fields):
            raise ValueError(f"Document must contain fields: {required_fields}")
        
        document["text"] = preprocess_medical_text(document["content"])
        return document
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks."""
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
        """Index a single document."""
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
                    metadata={
                        "doc_id": document["id"],
                        "title": document["title"],
                        "chunk_id": f"{document['id']}_chunk_{len(embedded_chunks)}"
                    },
                    embedding=chunk["embedding"]
                )
            
            logger.info(f"Indexed document {document['id']} with {len(chunks)} chunks")
            return len(chunks)
        
        except Exception as e:
            logger.error(f"Error indexing document {document.get('id', 'unknown')}: {e}")
            return 0
    
    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> Tuple[int, int]:
        """Index multiple documents in batches."""
        total_chunks = 0
        processed_docs = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            for doc in batch:
                chunks_indexed = self.index_document(doc)
                if chunks_indexed > 0:
                    processed_docs += 1
                    total_chunks += chunks_indexed
            
            self.vector_store.commit()
        
        logger.info(f"Indexed {processed_docs}/{len(documents)} documents with {total_chunks} total chunks")
        return processed_docs, total_chunks
    
    def index_directory(self, directory_path: str) -> Tuple[int, int]:
        """Index all JSON files in a directory."""
        documents = []
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return 0, 0
        
        json_files = list(path.glob("**/*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "content" in item:
                                item["id"] = item.get("id", f"{file_path.stem}_{id(item)}")
                                item["title"] = item.get("title", file_path.name)
                                documents.append(item)
                    elif isinstance(content, dict) and "content" in content:
                        content["id"] = content.get("id", file_path.stem)
                        content["title"] = content.get("title", file_path.name)
                        documents.append(content)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Found {len(documents)} documents in {directory_path}")
        return self.index_documents(documents)
    
    def reindex_all(self, directory_path: str) -> None:
        """Clear and rebuild the entire index."""
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
    """Convert medical knowledge base JSON files into indexable documents.
    Dynamically processes any JSON structure into a standardized document format."""
    documents = []
    
    def process_json_content(content: Dict[str, Any], source_file: str, prefix: str = "") -> List[Dict[str, Any]]:
        """Recursively process JSON content into documents."""
        docs = []
        
        if len(content) == 1 and isinstance(next(iter(content.values())), dict):
            category = next(iter(content.keys()))
            items = content[category]
            
            for item_id, item in items.items():
                if not isinstance(item, dict):
                    continue
                    
                title = item.get('name', item.get('title', item_id))
                
                text_parts = [f"{category.title()}: {title}"]
                
                for field, value in item.items():
                    if field in ['name', 'title', 'id']:  
                        continue
                        
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"\n{field.replace('_', ' ').title()}: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"\n{field.replace('_', ' ').title()}:")
                        text_parts.extend([f"- {v}" for v in value])
                    elif isinstance(value, dict):
                        text_parts.append(f"\n{field.replace('_', ' ').title()}:")
                        for subfield, subvalue in value.items():
                            if isinstance(subvalue, list):
                                text_parts.append(f"\n{subfield.replace('_', ' ').title()}:")
                                text_parts.extend([f"- {v}" for v in subvalue])
                            else:
                                text_parts.append(f"- {subfield.replace('_', ' ').title()}: {subvalue}")
                
                docs.append({
                    "id": f"{prefix}{category}_{item_id}",
                    "title": title,
                    "text": "\n".join(text_parts),
                    "source": source_file,
                    "doc_type": "medical_literature",
                    "category": category
                })
        
        return docs
    
    for json_file in kb_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if not isinstance(data, dict):
                    logger.warning(f"Skipping {json_file.name}: not a dictionary")
                    continue
                
                new_docs = process_json_content(data, json_file.name)
                documents.extend(new_docs)
                
                logger.info(f"Processed {len(new_docs)} documents from {json_file.name}")
                
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
    
    logger.info(f"Converted {len(documents)} total documents from medical knowledge base")
    return documents

def index_directory(directory_path: str) -> Tuple[int, int]:
    """Index all documents in the specified directory."""
    kb_dir = Path(directory_path)
    logger.info(f"Looking for documents in: {kb_dir.absolute()}")
    
    if not kb_dir.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return 0, 0
    
    if not kb_dir.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        return 0, 0
    
    files = list(kb_dir.glob("*.json"))
    logger.info(f"Found {len(files)} JSON files: {[f.name for f in files]}")
    
    indexer = create_indexer()
    
    try:
        documents = convert_medical_kb_to_documents(kb_dir)
        
        if not documents:
            logger.error("No valid documents found to index")
            return 0, 0
        
        logger.info(f"Successfully converted {len(documents)} documents")
        doc_count, chunk_count = indexer.index_documents(documents)
        
        if doc_count == 0:
            logger.error("Failed to index any documents")
        else:
            logger.info(f"Successfully indexed {doc_count} documents with {chunk_count} chunks")
        
        return doc_count, chunk_count
    
    except Exception as e:
        logger.error(f"Error indexing directory: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 0, 0

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = str(Path(__file__).parent.parent / "data" / "medical_kb")
    
    logger.info(f"Starting indexing with input directory: {input_dir}")
    doc_count, chunk_count = index_directory(input_dir)
    print(f"Indexed {doc_count} documents with {chunk_count} chunks")
    
    stats = create_indexer().get_index_stats()
    print(f"Index stats: {stats}")