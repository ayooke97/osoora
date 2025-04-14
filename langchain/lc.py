import os
# from dotenv import load_dotenv
import env_config
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from openai import OpenAI

# Set environment variables for OpenAI-compatible API
os.environ["OPENAI_API_KEY"] = "sk-f0f088df10a44cd3a5cf172c0ebfaaf6"
os.environ["OPENAI_BASE_URL"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Initialize ChatOpenAI with DashScope's compatible API
chat = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

# Example usage
def get_chat_response(prompt):
    """Get a response from the chat model."""
    response = chat.invoke(prompt)
    return response.content

def get_embedding(text):
    """Get embeddings for the provided text using DashScope's compatible API."""
    # Use OpenAI client with DashScope's compatible API
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"]
    )
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-v3"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Test the models
if __name__ == "__main__":
    # Test chat model
    print("Testing Chat Model:")
    user_prompt = "Who are you?"
    response = get_chat_response(user_prompt)
    print(response)
    print("\n" + "-"*50 + "\n")
    
    # Test embeddings model
    print("Testing Embeddings Model:")
    sample_text = "This is a sample text to generate embeddings for."
    print(f"Generating embedding for: '{sample_text}'")
    
    embedding_vector = get_embedding(sample_text)
    
    if embedding_vector:
        # Display information about the embedding vector
        print("\nEmbedding Vector Information:")
        print(f"Type: {type(embedding_vector)}")
        print(f"Length: {len(embedding_vector)}")
        
        # Convert to numpy array for easier manipulation
        embedding_array = np.array(embedding_vector)
        
        # Display a subset of the embedding vector (first 10 elements)
        print("\nFirst 10 elements of the embedding vector:")
        print(embedding_array[:10])
        
        # Display some statistics about the embedding vector
        print("\nEmbedding Statistics:")
        print(f"Mean: {embedding_array.mean()}")
        print(f"Min: {embedding_array.min()}")
        print(f"Max: {embedding_array.max()}")
        print(f"Standard Deviation: {embedding_array.std()}")

        print(OllamaEmbeddings(model="llama3.1").embed_query(sample_text))
    else:
        print("Failed to generate embedding.")