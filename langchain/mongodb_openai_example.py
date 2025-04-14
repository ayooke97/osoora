import os
import json
from pymongo import MongoClient
from openai import OpenAI
import numpy as np
from datetime import datetime

# MongoDB Atlas connection string - replace with your own
MONGODB_URI = "mongodb+srv://osoora:dbosoora@mdbv-osoorajkt.iu9ed.mongodb.net/?retryWrites=true&w=majority&appName=mdbv-OsooraJKT"

# OpenAI API key - replace with your own or use environment variable
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

def get_embedding(text, model="text-embedding-v3"):
    """Get embeddings for the provided text using OpenAI's API."""
    try:
        response = client.embeddings.create(
            input=text,
            model=model,
            dimensions=1024,
            encoding_format="float"
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

def setup_text_search(db):
    """Set up text search index in MongoDB Atlas."""
    try:
        # Check if the collection exists, if not create it
        if "documents" not in db.list_collection_names():
            db.create_collection("documents")
        
        # Create text search index if it doesn't exist
        existing_indexes = db.documents.list_indexes()
        index_exists = False
        for index in existing_indexes:
            if index.get("name") == "text_search_index":
                index_exists = True
                break
        
        if not index_exists:
            # Create a text search index
            db.documents.create_index(
                [("title", "text"), ("content", "text")],
                name="text_search_index",
                weights={"title": 10, "content": 5}
            )
            print("Text search index created successfully!")
        else:
            print("Text search index already exists.")
            
    except Exception as e:
        print(f"Error setting up text search: {e}")

def insert_documents(db, documents):
    """Insert documents into MongoDB Atlas."""
    try:
        # Add timestamp to each document
        for doc in documents:
            doc["timestamp"] = datetime.now()
            
            # We'll still get embeddings for future use, but won't use them for search yet
            embedding = get_embedding(doc["content"])
            if embedding:
                doc["embedding_data"] = embedding  # Store as regular field, not for vector search
        
        # Insert documents into MongoDB
        result = db.documents.insert_many(documents)
        print(f"Inserted {len(result.inserted_ids)} documents.")
            
    except Exception as e:
        print(f"Error inserting documents: {e}")

def text_search(db, query_text, limit=3):
    """Perform text search in MongoDB Atlas."""
    try:
        # Perform text search
        results = db.documents.find(
            {"$text": {"$search": query_text}},
            {"score": {"$meta": "textScore"}, "title": 1, "content": 1, "category": 1, "timestamp": 1}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        return list(results)
        
    except Exception as e:
        print(f"Error performing text search: {e}")
        return []

def semantic_search_without_vector(db, query_text, limit=3):
    """Perform semantic search without vector search by comparing embeddings manually."""
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query_text)
        
        if not query_embedding:
            print("Failed to generate embedding for query.")
            return []
        
        # Get all documents (this is not efficient for large collections)
        all_docs = list(db.documents.find({"embedding_data": {"$exists": True}}))
        
        # Calculate similarity scores (cosine similarity)
        results_with_scores = []
        for doc in all_docs:
            if "embedding_data" in doc:
                # Calculate cosine similarity
                doc_embedding = doc["embedding_data"]
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                # Create a copy of the document without the embedding (to save memory)
                doc_copy = {
                    "_id": doc["_id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc["category"],
                    "timestamp": doc["timestamp"],
                    "score": similarity
                }
                results_with_scores.append(doc_copy)
        
        # Sort by similarity score and limit results
        results_with_scores.sort(key=lambda x: x["score"], reverse=True)
        return results_with_scores[:limit]
        
    except Exception as e:
        print(f"Error performing semantic search: {e}")
        return []

def main():
    """Main function to demonstrate MongoDB Atlas with OpenAI embeddings."""
    # Connect to MongoDB Atlas
    mongo_client = connect_to_mongodb()
    if not mongo_client:
        return
    
    # Get database
    db = mongo_client["vector_db"]
    
    # Setup text search
    setup_text_search(db)
    
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
        {
            "title": "Web Development Fundamentals",
            "content": "Web development involves building and maintaining websites, covering aspects from frontend design to backend server configuration.",
            "category": "Programming"
        },
        {
            "title": "JavaScript Frameworks",
            "content": "Modern JavaScript frameworks like React, Vue, and Angular help developers build complex, interactive web applications with reusable components.",
            "category": "Programming"
        },
        {
            "title": "Data Science Overview",
            "content": "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data.",
            "category": "Data"
        },
        {
            "title": "Cloud Computing Services",
            "content": "Cloud computing provides on-demand access to computing resources, including servers, storage, databases, networking, and software over the internet.",
            "category": "Infrastructure"
        },
        {
            "title": "Cybersecurity Best Practices",
            "content": "Implementing strong passwords, multi-factor authentication, regular updates, and employee training are essential cybersecurity practices for organizations.",
            "category": "Security"
        }
    ]
    
    # Insert documents
    insert_documents(db, documents)
    
    # Perform text search
    print("\nText Search for: 'artificial intelligence'")
    text_results = text_search(db, "artificial intelligence")
    for i, result in enumerate(text_results):
        print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   {result['content'][:100]}...")
    
    # Perform semantic search
    print("\nSemantic Search for: 'What is artificial intelligence?'")
    semantic_results = semantic_search_without_vector(db, "What is artificial intelligence?")
    for i, result in enumerate(semantic_results):
        print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   {result['content'][:100]}...")
    
    # Close MongoDB connection
    mongo_client.close()

def insert_custom_documents():
    """Function to only insert documents without running the full main function."""
    # Connect to MongoDB Atlas
    mongo_client = connect_to_mongodb()
    if not mongo_client:
        return
    
    # Get database
    db = mongo_client["vector_db"]
    
    # Setup text search
    setup_text_search(db)
    
    # Custom documents to insert
    custom_documents = [
        {
            "title": "Blockchain Technology",
            "content": "Blockchain is a distributed ledger technology that enables secure, transparent, and immutable record-keeping without requiring a central authority.",
            "category": "Technology"
        },
        {
            "title": "Quantum Computing",
            "content": "Quantum computing uses quantum mechanics principles to process information, potentially solving certain problems exponentially faster than classical computers.",
            "category": "Computing"
        },
        {
            "title": "Sustainable Energy Solutions",
            "content": "Renewable energy sources like solar, wind, and hydroelectric power offer sustainable alternatives to fossil fuels, reducing carbon emissions and environmental impact.",
            "category": "Environment"
        },
        {
            "title": "Digital Marketing Strategies",
            "content": "Effective digital marketing combines SEO, content marketing, social media, email campaigns, and analytics to reach target audiences and drive conversions.",
            "category": "Marketing"
        },
        {
            "title": "Mobile App Development",
            "content": "Mobile app development involves creating software applications that run on mobile devices, requiring knowledge of platform-specific languages and user experience design.",
            "category": "Programming"
        }
    ]
    
    # Insert documents
    insert_documents(db, custom_documents)
    print(f"Inserted {len(custom_documents)} custom documents into the database.")
    
    # Close connection
    mongo_client.close()

if __name__ == "__main__":
    # Uncomment the line you want to run
    main()
    # insert_custom_documents()
