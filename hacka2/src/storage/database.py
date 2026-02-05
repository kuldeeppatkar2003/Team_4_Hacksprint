from motor.motor_asyncio import AsyncIOMotorClient
import redis
import os
import time
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "rss_intelligence")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

class Database:
    def __init__(self):
        self.client = AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.articles = self.db.articles
        self.sources = self.db.sources
        self.logs = self.db.logs
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    async def log_event(self, event_type: str, details: str, level: str = "INFO"):
        """Record a system event in the logs collection."""
        try:
            await self.logs.insert_one({
                "timestamp": time.time(),
                "event_type": event_type,
                "level": level,
                "details": details
            })
        except Exception as e:
            logger.error(f"Failed to write to logs collection: {e}")

    async def update_source_status(self, url: str, status: str, error: str = None):
        """Update the health status of an RSS feed source."""
        try:
            update_data = {
                "last_polled": time.time(),
                "status": status
            }
            if error:
                update_data["last_error"] = error
            
            await self.sources.update_one(
                {"url": url},
                {"$set": update_data},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to update source status for {url}: {e}")

    async def save_article(self, article_data: dict):
        try:
            # Save to MongoDB
            await self.articles.update_one(
                {"link": article_data["link"]},
                {"$set": article_data},
                upsert=True
            )
            
            # Cache top stats in Redis (optional)
            self.redis.lpush("latest_articles", article_data["link"])
            self.redis.ltrim("latest_articles", 0, 49) # Keep latest 50
            
            logger.info(f"Article saved to DB: {article_data['title']}")
        except Exception as e:
            logger.error(f"Error saving article to DB: {e}")

    async def get_latest_articles(self, limit=20):
        cursor = self.articles.find().sort("published", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_insights(self):
        """Aggregate meaningful insights from the latest articles."""
        try:
            # 1. Average sentiment per category
            pipeline = [
                {"$sort": {"published": -1}},
                {"$limit": 100},
                {"$group": {
                    "_id": "$category",
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "count": {"$sum": 1}
                }}
            ]
            category_stats = await self.articles.aggregate(pipeline).to_list(length=10)
            
            # 2. Trending keywords
            keywords_pipeline = [
                {"$unwind": "$keywords"},
                {"$group": {"_id": "$keywords", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            trending_keywords = await self.articles.aggregate(keywords_pipeline).to_list(length=10)
            
            return {
                "category_stats": category_stats,
                "trending_keywords": trending_keywords
            }
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"category_stats": [], "trending_keywords": []}

db = Database()
