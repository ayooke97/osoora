import os
import env_config
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
# Set the API key from env_config
os.environ["OPENAI_API_KEY"] = env_config.DASHSCOPE_API_KEY

# Initialize the OpenAI client with DashScope's compatible API
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def get_embedding(text, model="text-embedding-v3"):
    """Get embeddings for the provided text using the specified model."""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        # Extract the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

if __name__ == "__main__":
    # Sample text to embed
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
        print(len(embedding_array))
        
        # Display some statistics about the embedding vector
        print("\nEmbedding Statistics:")
        print(f"Mean: {embedding_array.mean()}")
        print(f"Min: {embedding_array.min()}")
        print(f"Max: {embedding_array.max()}")
        print(f"Standard Deviation: {embedding_array.std()}")

        
    else:
        print("Failed to generate embedding.")
