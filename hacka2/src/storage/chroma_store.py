import chromadb
from chromadb.utils import embedding_functions
import os
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Persistent storage for Chroma
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Using default sentence-transformers embedding
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="news_articles",
            embedding_function=self.ef
        )

    def add_article(self, id: str, text: str, metadata: dict):
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[id]
            )
            logger.info(f"Article added to ChromaDB: {id}")
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")

    def query(self, query_text: str, n_results=5):
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return None

vector_store = VectorStore()
