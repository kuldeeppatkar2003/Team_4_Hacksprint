import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.llm.client import LocalEmbeddingClient
from src.rag.vector_store import VectorDB
from src.rag.retriever import HybridRetriever
from src.config import settings

def test_hybrid_retrieval():
    print("-- Initializing Components --")
    
    # 1. Initialize Clients
    try:
        embedding_client = LocalEmbeddingClient()
        print("[OK] LocalEmbeddingClient initialized.")
    except Exception as e:
        print(f"[FAIL] LocalEmbeddingClient init failed: {e}")
        return

    vector_db = VectorDB(collection_name="test_helix_policies")
    data_loader = DataLoader()
    
    retriever = HybridRetriever(data_loader, vector_db, embedding_client)
    print("[OK] HybridRetriever initialized.")

    # 2. Index Policies (Unstructured)
    print("\n-- Indexing Policies --")
    pdf_path = os.path.join(settings.DATA_DIR, "Helix_Pro_Policy_v2.pdf")
    policy_chunks = data_loader.load_policies(pdf_path)
    
    # Index only first 5 chunks for speed in test
    print(f"Indexing {min(5, len(policy_chunks))} chunks...")
    vector_db.add_documents(policy_chunks[:5], embedding_client)
    print("[OK] Documents added to VectorDB.")

    # 3. Test Semantic Search
    print("\n-- Testing Semantic Search --")
    query = "What is the policy on sick leave?"
    results = retriever.search_policies(query, k=2)
    print(f"Query: '{query}'")
    for res in results:
        print(f" - Match (Dist: {res['distance']:.4f}): {res['text'][:100]}...")
    
    if not results:
        print("[FAIL] No results found for semantic search.")

    # 4. Test Structured Retrieval
    print("\n-- Testing Structured Retrieval --")
    # Get info for a known employee (EMP1001 from previous test)
    emp_id = "EMP1001"
    emp_info = retriever.get_employee_info(emp_id)
    
    if "error" in emp_info:
        print(f"[FAIL] Retrieval for {emp_id}: {emp_info['error']}")
    else:
        print(f"[OK] Retrieved info for {emp_id}:")
        print(f"   Name: {emp_info.get('name')}")
        print(f"   Leaves Taken: {len(emp_info.get('leaves', []))}")
        print(f"   Attendance Records: {emp_info.get('attendance_count')}")

if __name__ == "__main__":
    test_hybrid_retrieval()
