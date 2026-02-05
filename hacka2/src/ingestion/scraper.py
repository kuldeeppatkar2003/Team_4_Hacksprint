import feedparser
import time
import hashlib
import logging
import asyncio
from typing import List, Dict
from newspaper import Article
import redis
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Redis setup for deduplication
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RSS_FEEDS = {
    "world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.theguardian.com/world/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
    ],
    "technology": [
        "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms",
        "https://techcrunch.com/feed/",
        "https://www.wired.com/feed/rss",
        "https://www.theverge.com/rss/index.xml"
    ],
    "india": [
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://www.ndtv.com/rss/india"
    ],
    "financial": [
        "https://www.ft.com/?format=rss",
        "https://www.investing.com/rss/news.rss"
    ],
    "sports": [
        "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms",
        "https://www.espn.com/espn/rss/news",
        "https://feeds.bbci.co.uk/sport/rss.xml"
    ],
    "politics": [
        "https://timesofindia.indiatimes.com/rssfeeds/-2128801720.cms",
        "https://www.npr.org/rss/rss.php?id=1014",
        "https://www.politico.com/rss/politicopics.xml"
    ]
}

# Semaphore to limit concurrent processing (avoid rate limits)
MAX_CONCURRENT_SCRAPES = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRAPES)

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def is_duplicate(url: str) -> bool:
    url_hash = get_url_hash(url)
    if r.exists(f"article:{url_hash}"):
        return True
    return False

def mark_as_processed(url: str, ttl: int = 7200):
    url_hash = get_url_hash(url)
    r.setex(f"article:{url_hash}", ttl, "1")

async def fetch_article_content(url: str) -> dict:
    try:
        # Use a timeout and a browser-like user agent to avoid blocks
        article = Article(url, request_timeout=10)
        # newspaper3k doesn't natively support async download, so we run it in a thread
        await asyncio.to_thread(article.download)
        await asyncio.to_thread(article.parse)
        
        # NLP for fallback (newspaper attempts it)
        try:
            await asyncio.to_thread(article.nlp)
        except:
            pass
            
        return {
            "text": article.text,
            "summary": article.summary if hasattr(article, 'summary') and article.summary else article.text[:200] + "...",
            "keywords": article.keywords if hasattr(article, 'keywords') else []
        }
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None

from src.processing.llm_processor import LLMProcessor
from src.storage.database import db
from src.storage.chroma_store import vector_store

# We need to import the manager from main, but to avoid circular imports
# let's assume we'll use a better approach or just a global event bus.
# For simplicity, we'll import it here.
from src.utils.manager import manager

llm = LLMProcessor()

async def process_entry(entry, category):
    async with semaphore:
        link = entry.link
        if is_duplicate(link):
            return

        logger.info(f"Processing: {entry.title}")
        
        # Extract full content
        content_data = await fetch_article_content(link)
        if not content_data or not content_data["text"]: 
            return

        full_text = content_data["text"]

        # AI Analysis
        analysis = await llm.analyze_article(entry.title, full_text)
        
        if not analysis:
            logger.warning(f"AI Analysis failed for: {entry.title}. Using NLP fallback.")
            analysis = {
                "summary": content_data["summary"],
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "key_insights": ["AI analysis unavailable"],
                "category": category.capitalize(),
                "keywords": content_data["keywords"][:5] if content_data["keywords"] else ["news"]
            }
            
        article_data = {
            "title": entry.title,
            "link": link,
            "published": entry.published if hasattr(entry, 'published') else time.ctime(),
            "category": category,
            "full_text": full_text,
            **analysis
        }
        
        try:
            # Store in Mongo/Redis
            await db.save_article(article_data)
            
            # Store in ChromaDB for RAG
            vector_store.add_article(
                get_url_hash(link),
                f"{entry.title}\n{full_text}",
                {"title": entry.title, "link": link, "category": category}
            )
            
            # Broadcast to UI
            await manager.broadcast(json.dumps(article_data))
            
            # Mark as processed in Redis
            mark_as_processed(link)
            logger.info(f"Successfully Persisted: {entry.title}")
        except Exception as e:
            logger.error(f"Failed to save article {entry.title}: {e}")
        
        # Small delay to respect rate limits
        await asyncio.sleep(1)

async def poll_feeds():
    logger.info("Starting High-Throughput RSS Intelligence Scraper...")
    await db.log_event("SCRAPER_START", "High-throughput scraper initialized.")
    
    while True:
        tasks = []
        cycle_start = time.time()
        
        for category, urls in RSS_FEEDS.items():
            if isinstance(urls, str): urls = [urls] # Backward compatibility
            
            for url in urls:
                try:
                    logger.debug(f"Parsing feed: {url}")
                    feed = await asyncio.to_thread(feedparser.parse, url)
                    
                    if feed.bozo:
                        await db.update_source_status(url, "ERROR", str(feed.bozo_exception))
                        logger.error(f"Feed parsing error for {url}: {feed.bozo_exception}")
                        continue
                    
                    # Update status as active
                    await db.update_source_status(url, "ACTIVE")
                    
                    for entry in feed.entries[:10]: # Process top 10 latest per feed per cycle
                        tasks.append(process_entry(entry, category))
                        
                except Exception as e:
                    await db.update_source_status(url, "FAILED", str(e))
                    logger.error(f"Error parsing feed {url}: {e}")
        
        if tasks:
            logger.info(f"Processing {len(tasks)} potential new articles...")
            await asyncio.gather(*tasks)
        
        cycle_duration = time.time() - cycle_start
        await db.log_event("SCRAPER_CYCLE_COMPLETE", f"Processed {len(tasks)} potential articles in {cycle_duration:.2f}s")
        
        logger.info("Scrape cycle complete. Sleeping...")
        await asyncio.sleep(int(os.getenv("POLLING_INTERVAL", 60)))

if __name__ == "__main__":
    asyncio.run(poll_feeds())
