import streamlit as st
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.llm.client import LocalEmbeddingClient, GroqLLMClient, OpenAILLMClient, GeminiLLMClient, MockLLMClient
from src.rag.vector_store import VectorDB
from src.rag.retriever import HybridRetriever
from src.rag.pipeline import RAGPipeline
from src.config import settings

# Page Config
st.set_page_config(page_title="Helix HR Intelligence", page_icon="🧬", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    }
    .chat-message.user {
        background-color: #2b313e; color: #ffffff;
    }
    .chat-message.bot {
        background-color: #f0f2f6; border: 1px solid #dcdce1;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_pipeline():
    """Initializes the RAG pipeline once and caches it."""
    print("Initializing Pipeline...")
    
    # LLM Setup
    if settings.GROQ_API_KEY:
        gen_client = GroqLLMClient(api_key=settings.GROQ_API_KEY)
        provider = "Groq (Llama-3.3-70b)"
    elif settings.OPENAI_API_KEY:
        gen_client = OpenAILLMClient(api_key=settings.OPENAI_API_KEY)
        provider = "OpenAI (GPT-3.5)"
    elif settings.GOOGLE_API_KEY:
        gen_client = GeminiLLMClient(api_key=settings.GOOGLE_API_KEY)
        provider = "Google Gemini"
    else:
        gen_client = MockLLMClient()
        provider = "Mock Client"

    embed_client = LocalEmbeddingClient()
    data_loader = DataLoader()
    vector_db = VectorDB(collection_name="helix_policies")
    
    # Ensure indexed
    if vector_db.collection.count() == 0:
        pdf_path = os.path.join(settings.DATA_DIR, "Helix_Pro_Policy_v2.pdf")
        if os.path.exists(pdf_path):
            chunks = data_loader.load_policies(pdf_path)
            vector_db.add_documents(chunks, embed_client)
            
    retriever = HybridRetriever(data_loader, vector_db, embed_client)
    pipeline = RAGPipeline(retriever, gen_client)
    
    return pipeline, provider

# Initialize
pipeline, llm_provider = initialize_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dna-helix.png", width=80)
    st.title("Helix Corp")
    st.subheader("HR Intelligence Bot")
    st.markdown("---")
    st.success(f"**Connected LLM:**\n{llm_provider}")
    
    st.markdown("### 🛠 Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
        
    st.markdown("---")
    st.markdown("### 📝 About")
    st.info(
        "This bot uses **RAG (Retrieval-Augmented Generation)** to answer questions based on:\n"
        "- 📄 HR Policy PDF\n"
        "- 📊 Employee Database\n"
        "- 📅 Attendance Logs"
    )

# Main Chat Interface
st.header("🧬 HR Intelligence Assistant")
st.caption("Ask about policies, leave entitlements, or employee details (e.g., 'Tell me about EMP1001')")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            for cite in message["citations"]:
                st.markdown(f"- {cite}")

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            import time
            start_time = time.time()
            
            result = pipeline.process_query(prompt)
            response = result["response"]
            citations = result["citations"]
            confidence = result.get("confidence", {})
            
            elapsed_time = time.time() - start_time
            
            st.markdown(response)
            
            # Confidence Indicator
            if confidence:
                conf_level = confidence.get("level", "Unknown")
                conf_score = confidence.get("score", 0)
                
                if conf_level == "High":
                    st.success(f"🎯 Confidence: {conf_level} ({conf_score}%)")
                elif conf_level == "Medium":
                    st.warning(f"⚠️ Confidence: {conf_level} ({conf_score}%)")
                else:
                    st.error(f"❓ Confidence: {conf_level} ({conf_score}%)")
            
            if citations:
                st.markdown("---")
                st.markdown("**📚 Sources:**")
                for cite in citations:
                    st.markdown(f"- {cite}")
            
            # Performance metric
            st.caption(f"⏱️ Response time: {elapsed_time:.2f}s")
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "citations": citations
    })
