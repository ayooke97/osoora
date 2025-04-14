import os
from pymongo import MongoClient
from openai import OpenAI
import numpy as np
from datetime import datetime
from langchain_openai import ChatOpenAI

# MongoDB Atlas connection string - replace with your own
MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"

# DashScope API key (for chat functionality)
DASHSCOPE_API_KEY = "sk-f0f088df10a44cd3a5cf172c0ebfaaf6"

# OpenAI API key - replace with your own or use environment variable
OPENAI_API_KEY = "your-openai-api-key"

# Set environment variables for DashScope's OpenAI-compatible API (for chat)
os.environ["OPENAI_API_KEY"] = DASHSCOPE_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Initialize ChatOpenAI with DashScope's compatible API
chat = ChatOpenAI(
    model="qwen-plus",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Initialize OpenAI client for embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_chat_response(prompt):
    """Get a response from the chat model using DashScope."""
    response = chat.invoke(prompt)
    return response.content

def get_embedding(text, model="text-embedding-3-small"):
    """Get embeddings for the provided text using OpenAI's API."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def connect_to_mongodb():
    """Connect to MongoDB Atlas."""
    try:
        # Create a new client and connect to the server
        mongo_client = MongoClient(MONGODB_URI)
        
        # Send a ping to confirm a successful connection
        mongo_client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        
        return mongo_client
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")
        return None

def setup_vector_search(db):
    """Set up vector search index in MongoDB Atlas."""
    try:
        # Check if the collection exists, if not create it
        if "documents" not in db.list_collection_names():
            db.create_collection("documents")
        
        # Create vector search index if it doesn't exist
        existing_indexes = db.documents.list_indexes()
        index_exists = False
        for index in existing_indexes:
            if index.get("name") == "vector_index":
                index_exists = True
                break
        
        if not index_exists:
            # Create a vector search index
            db.documents.create_index(
                [("embedding", "vector")],
                name="vector_index",
                vector_options={
                    "dimensions": 1536,  # for text-embedding-3-small
                    "similarity": "cosine"  # or "euclidean" or "dotProduct"
                }
            )
            print("Vector search index created successfully!")
        else:
            print("Vector search index already exists.")
            
    except Exception as e:
        print(f"Error setting up vector search: {e}")

def insert_documents(db, documents):
    """Insert documents with embeddings into MongoDB Atlas."""
    try:
        # Get embeddings for each document
        for doc in documents:
            embedding = get_embedding(doc["content"])
            if embedding:
                doc["embedding"] = embedding
                doc["timestamp"] = datetime.now()
            else:
                print(f"Skipping document due to embedding failure: {doc['title']}")
        
        # Filter out documents without embeddings
        documents_with_embeddings = [doc for doc in documents if "embedding" in doc]
        
        if documents_with_embeddings:
            # Insert documents into MongoDB
            result = db.documents.insert_many(documents_with_embeddings)
            print(f"Inserted {len(result.inserted_ids)} documents with embeddings.")
        else:
            print("No documents to insert.")
            
    except Exception as e:
        print(f"Error inserting documents: {e}")

def vector_search(db, query_text, limit=3):
    """Perform vector search in MongoDB Atlas."""
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query_text)
        
        if not query_embedding:
            print("Failed to generate embedding for query.")
            return []
        
        # Perform vector search
        results = db.documents.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "title": 1,
                    "content": 1,
                    "category": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ])
        
        return list(results)
        
    except Exception as e:
        print(f"Error performing vector search: {e}")
        return []

def rag_with_mongodb(db, user_query):
    """Implement RAG (Retrieval-Augmented Generation) with MongoDB and DashScope."""
    # Step 1: Retrieve relevant documents using vector search
    relevant_docs = vector_search(db, user_query)
    
    if not relevant_docs:
        # If no relevant documents found, just use the chat model directly
        return get_chat_response(user_query)
    
    # Step 2: Format retrieved documents as context
    context = "I have the following information that might help answer your question:\n\n"
    for i, doc in enumerate(relevant_docs):
        context += f"Document {i+1}: {doc['title']}\n"
        context += f"{doc['content']}\n\n"
    
    # Step 3: Create a prompt with the context and user query
    prompt = f"""
{context}

Based on the information above, please answer the following question:
{user_query}

If the information provided doesn't fully answer the question, please say so and provide what you know.
"""
    
    # Step 4: Generate a response using the chat model
    response = get_chat_response(prompt)
    
    return response

def main():
    """Main function to demonstrate MongoDB Atlas with OpenAI embeddings and DashScope chat."""
    # Connect to MongoDB Atlas
    mongo_client = connect_to_mongodb()
    if not mongo_client:
        print("Please update the MONGODB_URI with your MongoDB Atlas connection string.")
        return
    
    # Get database
    db = mongo_client["vector_db"]
    
    # Setup vector search
    setup_vector_search(db)
    
    # Sample documents
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            "category": "AI"
        },
        {
            "title": "Python Programming",
            "content": "Python is a high-level, interpreted programming language known for its readability and versatility.",
            "category": "Programming"
        },
        {
            "title": "Artificial Intelligence Overview",
            "content": "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
            "category": "AI"
        },
        {
            "title": "Natural Language Processing",
            "content": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
            "category": "AI"
        },
        {
            "title": "Vector Databases",
            "content": "Vector databases store and retrieve high-dimensional vectors efficiently, making them ideal for similarity search operations.",
            "category": "Databases"
        },
        {
            "title": "Embeddings Explained",
            "content": "Embeddings are vector representations of text, images, or other data that capture semantic meaning.",
            "category": "AI"
        },
        {
            "title": "MongoDB Atlas",
            "content": "MongoDB Atlas is a fully-managed cloud database that handles all the complexity of deploying, managing, and healing your deployments on the cloud service provider of your choice.",
            "category": "Databases"
        },
    ]
    
    # Insert documents
    print("Inserting sample documents...")
    insert_documents(db, documents)
    
    # Demonstrate RAG with MongoDB and DashScope
    print("\n--- RAG Demo with MongoDB and DashScope ---")
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "Tell me about vector databases",
        "How does MongoDB Atlas work?",
        "What is the relationship between NLP and AI?"
    ]
    
    for query in queries:
        print(f"\nUser Query: {query}")
        response = rag_with_mongodb(db, query)
        print(f"\nRAG Response:\n{response}")
        print("\n" + "-"*50)
    
    # Close MongoDB connection
    mongo_client.close()

if __name__ == "__main__":
    print("To use this example, please update the following:")
    print("1. MONGODB_URI with your MongoDB Atlas connection string")
    print("2. OPENAI_API_KEY with your OpenAI API key")
    print("\nThen remove this message and uncomment the main() call below")
    # main()
