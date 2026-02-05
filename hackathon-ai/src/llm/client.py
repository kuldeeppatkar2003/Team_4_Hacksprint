from abc import ABC, abstractmethod
import os
from typing import List, Optional, Dict, Any

class LLMInterface(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates text from a given prompt."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generates embeddings for a given text."""
        pass

class MockLLMClient(LLMInterface):
    """A mock LLM client for testing and development without API costs."""
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        return f"Mock response to: {prompt[:50]}..."

    def get_embedding(self, text: str) -> List[float]:
        # Return a fixed mock embedding of size 768
        return [0.1] * 768

class OpenAILLMClient(LLMInterface):
    """Client for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

class GeminiLLMClient(LLMInterface):
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        import google.generativeai as genai
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY must be set in environment or passed to constructor.")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating text with Gemini: {e}")
            return ""

    def get_embedding(self, text: str) -> List[float]:
        # Gemini embedding implementation if needed, though we serve retrieval with LocalEmbeddingClient
        # This acts as a fallback or if user wants full Gemini stack
        import google.generativeai as genai
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title="Embedding of text"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error getting embedding with Gemini: {e}")
            return []

class GroqLLMClient(LLMInterface):
    """Client for Groq API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        import groq
        self.client = groq.Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text with Groq: {e}")
            return ""

    def get_embedding(self, text: str) -> List[float]:
        # Groq currently focuses on generation. We use LocalEmbeddingClient for embeddings.
        print("Warning: Groq does not support embeddings natively in this client yet.")
        return []

class LocalEmbeddingClient(LLMInterface):
    """Client for local embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        return "LocalEmbeddingClient does not support text generation."

    def get_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

