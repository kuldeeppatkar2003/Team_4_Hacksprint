import sys
import os
import argparse

# Check if .env exists, if not create dummy one to prevent errors if user hasn't set it up
if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write("OPENAI_API_KEY=\nLLM_PROVIDER=mock\n")

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.llm.client import LocalEmbeddingClient, MockLLMClient, OpenAILLMClient, GeminiLLMClient, GroqLLMClient
from src.rag.vector_store import VectorDB
from src.rag.retriever import HybridRetriever
from src.rag.pipeline import RAGPipeline
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description="Helix Corp HR Intelligence Bot")
    parser.add_argument("--query", type=str, help="Direct query to run", default=None)
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--reset-db", action="store_true", help="Reset vector database")
    
    args = parser.parse_args()

    print("Initializing HR Bot... (This may take a moment loading models)")
    
    # Setup LLM based on config (or default to mock if key missing)
    if settings.OPENAI_API_KEY:
        generation_client = OpenAILLMClient(api_key=settings.OPENAI_API_KEY)
        print("Using OpenAI Client.")
    elif settings.GROQ_API_KEY:
        generation_client = GroqLLMClient(api_key=settings.GROQ_API_KEY)
        print("Using Groq Client (Llama3-70b).")
    elif settings.GOOGLE_API_KEY:
        generation_client = GeminiLLMClient(api_key=settings.GOOGLE_API_KEY)
        print("Using Google Gemini Client.")
    else:
        generation_client = MockLLMClient()
        print("Using Mock Client (No API Key found).")
        
    embedding_client = LocalEmbeddingClient()
    
    # Setup Data
    data_loader = DataLoader()
    vector_db = VectorDB(collection_name="helix_policies")
    
    # Optional: Initial Indexing check
    # Check if empty, if so, load PDF
    if args.reset_db or vector_db.collection.count() == 0:
        print("Indexing Documents...")
        pdf_path = os.path.join(settings.DATA_DIR, "Helix_Pro_Policy_v2.pdf")
        if os.path.exists(pdf_path):
            chunks = data_loader.load_policies(pdf_path)
            vector_db.add_documents(chunks, embedding_client)
        else:
            print(f"Warning: Policy PDF not found at {pdf_path}")
    
    retriever = HybridRetriever(data_loader, vector_db, embedding_client)
    pipeline = RAGPipeline(retriever, generation_client)
    
    print("Bot Ready! 🚀")
    
    if args.query:
        print(f"\nUser: {args.query}")
        result = pipeline.process_query(args.query)
        print(f"Bot: {result['response']}")
        print(f"Sources: {result['citations']}")
        
    elif args.interactive:
        print("Enter 'exit' to quit.")
        while True:
            q = input("\nUser: ")
            if q.lower() in ["exit", "quit"]:
                break
            
            result = pipeline.process_query(q)
            print(f"Bot: {result['response']}")
            if result['citations']:
                print(f"Sources: {result['citations']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
