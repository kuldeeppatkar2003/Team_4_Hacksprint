import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.llm.client import LocalEmbeddingClient, GroqLLMClient, OpenAILLMClient, GeminiLLMClient, MockLLMClient
from src.rag.vector_store import VectorDB
from src.rag.retriever import HybridRetriever
from src.rag.pipeline import RAGPipeline
from src.config import settings

def run_benchmark():
    print("Initializing components for Benchmark...")
    
    # LLM Setup
    if settings.GROQ_API_KEY:
        gen_client = GroqLLMClient(api_key=settings.GROQ_API_KEY)
        print("Using Groq.")
    elif settings.OPENAI_API_KEY:
        gen_client = OpenAILLMClient(api_key=settings.OPENAI_API_KEY)
        print("Using OpenAI.")
    elif settings.GOOGLE_API_KEY:
        gen_client = GeminiLLMClient(api_key=settings.GOOGLE_API_KEY)
        print("Using Gemini.")
    else:
        gen_client = MockLLMClient()
        print("Using Mock.")

    embed_client = LocalEmbeddingClient()
    data_loader = DataLoader()
    vector_db = VectorDB(collection_name="helix_policies")
    # Ensure indexed
    if vector_db.collection.count() == 0:
        pdf_path = os.path.join(settings.DATA_DIR, "Helix_Pro_Policy_v2.pdf")
        chunks = data_loader.load_policies(pdf_path)
        vector_db.add_documents(chunks, embed_client)

    retriever = HybridRetriever(data_loader, vector_db, embed_client)
    pipeline = RAGPipeline(retriever, gen_client)

    questions = [
        "I am Gabrielle Davis (EMP1004). Based on the 2026 policy and my specific joining date, exactly how many total days of annual leave am I entitled to this year? Please show your calculation.",
        "I am Allen Robinson (EMP1002). I am feeling unwell today in my Singapore home. If I take only today off as sick leave, do I need to submit a medical certificate when I return, or can I wait until I’ve been out for more than two days?",
        "Looking at the attendance policy, what is the specific penalty for an employee who has 6 instances of missing check-out entries in a single calendar month, and how often does this count reset?",
        "I am Sherri Baker (EMP1015). I have been with Helix Global since early 2018 and work in the Sydney office. Am I eligible to apply for the sabbatical program right now (February 2026)? If so, what is the application process?",
        "I am Thomas Bradley (EMP1010). Since I work in the London office, what is my total leave entitlement including bank holidays, and do I need to submit a formal request for those specific bank holiday dates?"
    ]

    print("\nStarting Benchmark...\n" + "="*50)

    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q}")
        print("-" * 20)
        try:
            result = pipeline.process_query(q)
            print(f"Bot Response:\n{result['response']}")
            print(f"\nSources: {result['citations']}")
        except Exception as e:
            print(f"Error: {e}")
        print("="*50)

if __name__ == "__main__":
    run_benchmark()
