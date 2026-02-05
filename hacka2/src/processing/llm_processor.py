import google.generativeai as genai
import os
import json
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

MODEL_NAME = "gemini-1.5-flash"  # Standard name

ANALYSIS_PROMPT = """
You are a news analyst. Analyze the following news article and provide a structured JSON response.
Article Title: {title}
Article Content: {content}

The output must be a valid JSON object with the following schema:
{{
  "summary": "A concise 2-3 sentence summary of the article",
  "sentiment_score": float (between -1.0 for very negative and 1.0 for very positive),
  "sentiment_label": "Positive" | "Neutral" | "Negative",
  "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
  "category": "Politics" | "Sports" | "Technology" | "India" | "World" | "Financial",
  "keywords": ["tag1", "tag2", "tag3"]
}}

Return ONLY the JSON object.
"""

class LLMProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel(MODEL_NAME)

    async def analyze_article(self, title: str, content: str) -> Optional[Dict]:
        if not GEMINI_API_KEY:
            logger.error("API Key missing, skipping AI analysis.")
            return None
            
        try:
            prompt = ANALYSIS_PROMPT.format(title=title, content=content)
            # Use generation_config for response_mime_type if supported, 
            # or rely on prompt instruction for Gemini.
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                safety_settings=SAFETY_SETTINGS
            )
            
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            return None

if __name__ == "__main__":
    # Test block
    import asyncio
    processor = LLMProcessor()
    async def test():
        res = await processor.analyze_article("Test Title", "Sample content about market trends.")
        print(res)
    # asyncio.run(test()) # Commented out for now
