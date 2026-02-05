import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import uuid
import os
from src.llm.client import LLMInterface

class VectorDB:
    """
    Wrapper around ChromaDB for storing and retrieving document chunks.
    """
    
    def __init__(self, collection_name: str = "policies", embedding_function=None):
        """
        Initializes ChromaDB.
        :param embedding_function: Optional instance of a class adhering to Chroma's EmbeddingFunction protocol.
                                   If None, uses default SentenceTransformer (all-MiniLM-L6-v2).
        """
        self.client = chromadb.Client()
        # Delete if exists to start fresh for this hackathon context (optional)
        # try:
        #     self.client.delete_collection(collection_name)
        # except:
        #     pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=embedding_function
        )

    def add_documents(self, documents: List[Dict[str, Any]], embedding_client: LLMInterface = None):
        """
        Embeds and stores documents.
        :param documents: List of dicts {'text': str, 'source': str, 'page': int}
        :param embedding_client: LLMInterface to generate embeddings if not using Chroma's default.
        """
        print(f"Adding {len(documents)} documents to VectorDB...")
        
        ids = [str(uuid.uuid4()) for _ in documents]
        documents_text = [doc['text'] for doc in documents]
        metadatas = [{'source': doc['source'], 'page': doc['page']} for doc in documents]
        
        embeddings = None
        if embedding_client:
            embeddings = [embedding_client.get_embedding(text) for text in documents_text]
            
        self.collection.add(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas,
            embeddings=embeddings # If None, Chroma calculates it if embedding_function was set
        )

    def search(self, query: str, k: int = 3, embedding_client: LLMInterface = None) -> List[Dict[str, Any]]:
        """
        Searches for relevant documents.
        """
        query_embeddings = None
        if embedding_client:
            query_embeddings = [embedding_client.get_embedding(query)]
            
        results = self.collection.query(
            query_embeddings=query_embeddings,
            query_texts=[query] if not query_embeddings else None,
            n_results=k
        )
        
        # Parse results
        parsed_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                parsed_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
                
        return parsed_results
