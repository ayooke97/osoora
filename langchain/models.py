from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
    topic: Optional[str] = "New Chat"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    preview: Optional[str] = ""
    user_id: Optional[str] = None

class ConversationManager:
    def __init__(self, db):
        self.db = db
        self.collection = db[Collections.CONVERSATIONS]
    
    async def create_conversation(self, conversation: Conversation):
        """Create a new conversation"""
        result = await self.collection.insert_one(conversation.dict())
        return str(result.inserted_id)
    
    async def get_conversation(self, conversation_id: str):
        """Get a conversation by ID"""
        conversation = await self.collection.find_one({"id": conversation_id})
        return Conversation(**conversation) if conversation else None
    
    async def get_user_conversations(self, user_id: str):
        """Get all conversations for a user"""
        cursor = self.collection.find({"user_id": user_id}).sort("timestamp", -1)
        conversations = []
        async for doc in cursor:
            conversations.append(Conversation(**doc))
        return conversations
    
    async def update_conversation(self, conversation_id: str, update_data: dict):
        """Update a conversation"""
        await self.collection.update_one(
            {"id": conversation_id},
            {"$set": update_data}
        )
    
    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        await self.collection.delete_one({"id": conversation_id})
    
    async def delete_user_conversations(self, user_id: str):
        """Delete all conversations for a user"""
        await self.collection.delete_many({"user_id": user_id})
    
    async def add_message(self, conversation_id: str, message: Message):
        """Add a message to a conversation"""
        await self.collection.update_one(
            {"id": conversation_id},
            {
                "$push": {"messages": message.dict()},
                "$set": {
                    "timestamp": datetime.utcnow(),
                    "preview": message.content
                }
            }
        )
