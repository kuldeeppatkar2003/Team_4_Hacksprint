import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.llm.client import LocalEmbeddingClient, MockLLMClient
from src.rag.vector_store import VectorDB
from src.rag.retriever import HybridRetriever
from src.rag.pipeline import RAGPipeline
from src.config import settings

def test_end_to_end():
    print("-- Initializing RAG Pipeline --")
    
    # Init components
    # Using MockLLMClient for generation to avoid API keys, but LocalEmbedding for retrieval validity
    embedding_client = LocalEmbeddingClient()
    generation_client = MockLLMClient() 
    
    vector_db = VectorDB(collection_name="test_helix_policies_e2e")
    data_loader = DataLoader()
    retriever = HybridRetriever(data_loader, vector_db, embedding_client)
    
    # Index data (subset for speed)
    pdf_path = os.path.join(settings.DATA_DIR, "Helix_Pro_Policy_v2.pdf")
    policy_chunks = data_loader.load_policies(pdf_path)
    vector_db.add_documents(policy_chunks[:5], embedding_client)
    
    pipeline = RAGPipeline(retriever, generation_client)
    print("[OK] Pipeline Initialized.")

    # Test 1: Employee Query
    print("\n-- Test 1: Employee Query (EMP1001) --")
    query_emp = "Tell me about EMP1001"
    result_emp = pipeline.process_query(query_emp)
    
    print(f"Intent detected: {result_emp['intent']}")
    if result_emp['intent'] == "EMPLOYEE_INFO":
        print("[Pass] Intent correctly identified.")
    else:
        print(f"[Fail] Expected EMPLOYEE_INFO, got {result_emp['intent']}")
        
    if "Employee Profile" in result_emp['context']:
        print("[Pass] Context contains Employee Profile.")
    else:
        print("[Fail] Context missing profile data.")
        
    print(f"Citations: {result_emp['citations']}")


    # Test 2: Policy Query
    print("\n-- Test 2: Policy Query --")
    query_pol = "What is the policy?"
    result_pol = pipeline.process_query(query_pol)
    
    print(f"Intent detected: {result_pol['intent']}")
    if result_pol['intent'] == "POLICY_SEARCH":
        print("[Pass] Intent correctly identified.")
    else:
        print(f"[Fail] Expected POLICY_SEARCH, got {result_pol['intent']}")
        
    if len(result_pol['citations']) > 0:
        print(f"[Pass] {len(result_pol['citations'])} citations found.")
    else:
        print("[Fail] No citations found.")

if __name__ == "__main__":
    test_end_to_end()
