import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.client import MockLLMClient, OpenAILLMClient
from src.config import settings

def test_mock_llm():
    print("Testing Mock LLM...")
    client = MockLLMClient()
    response = client.generate_text("Hello, world!")
    print(f"Response: {response}")
    embeddings = client.get_embedding("Hello")
    print(f"Embedding length: {len(embeddings)}")
    assert len(embeddings) == 768
    print("Mock LLM Test Passed!")

if __name__ == "__main__":
    test_mock_llm()
    # Uncomment to test real integration if key is present
    # if settings.OPENAI_API_KEY:
    #     test_openai_llm()
