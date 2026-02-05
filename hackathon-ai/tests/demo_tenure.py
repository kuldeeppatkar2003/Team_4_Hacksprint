import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader
from src.llm.client import LocalEmbeddingClient, GroqLLMClient
from src.rag.vector_store import VectorDB
from src.rag.retriever import HybridRetriever
from src.rag.pipeline import RAGPipeline
from src.config import settings

print("=" * 60)
print("MULTI-STEP TENURE CALCULATION DEMO")
print("=" * 60)

# Initialize
gen_client = GroqLLMClient(api_key=settings.GROQ_API_KEY)
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

# Test Query: Gabrielle Davis tenure calculation
query = "I am Gabrielle Davis (EMP1004). Based on the 2026 policy and my specific joining date, exactly how many total days of annual leave am I entitled to this year? Please show your calculation."

print(f"\nQuery: {query}\n")
print("-" * 60)

# Step 1: Show raw tenure calculation
print("\n📊 STEP 1: Tenure Calculation")
print("-" * 60)
tenure_info = retriever.calculate_tenure("EMP1004")
if "error" not in tenure_info:
    print(f"Employee ID: {tenure_info['emp_id']}")
    print(f"Joining Date: {tenure_info['joining_date']}")
    print(f"Current Date: 2026-02-05")
    print(f"Tenure (Days): {tenure_info['tenure_days']} days")
    print(f"Tenure (Years): {tenure_info['tenure_years']} years")
    print(f"Sabbatical Eligible: {'Yes' if tenure_info['sabbatical_eligible'] else 'No'}")
else:
    print(f"Error: {tenure_info['error']}")

# Step 2: Show full RAG response
print("\n🤖 STEP 2: RAG Pipeline Response")
print("-" * 60)
result = pipeline.process_query(query)
print(f"\nAnswer:\n{result['response']}\n")
print(f"Confidence: {result['confidence']['level']} ({result['confidence']['score']}%)")
print(f"Confidence Reasons: {', '.join(result['confidence']['reasons'])}")
print(f"\nSources:")
for cite in result['citations']:
    print(f"  - {cite}")

print("\n" + "=" * 60)
