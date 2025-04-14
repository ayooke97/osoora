from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class Database:
    client: AsyncIOMotorClient = None
    
    @classmethod
    async def connect_db(cls):
        """Connect to MongoDB"""
        if cls.client is None:
            try:
                mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
                cls.client = AsyncIOMotorClient(
                    mongodb_url,
                    server_api=ServerApi('1')
                )
                # Send a ping to confirm a successful connection
                await cls.client.admin.command('ping')
                print("Connected to MongoDB!")
            except Exception as e:
                print(f"Could not connect to MongoDB: {e}")
                raise e
    
    @classmethod
    async def close_db(cls):
        """Close MongoDB connection"""
        if cls.client is not None:
            await cls.client.close()
            cls.client = None
            print("Closed MongoDB connection")
    
    @classmethod
    def get_db(cls):
        """Get database instance"""
        if cls.client is None:
            raise Exception("Database not connected. Call connect_db() first.")
        return cls.client.osoora

# Database models
class Collections:
    CONVERSATIONS = "conversations"
    MESSAGES = "messages"
    USERS = "users"
