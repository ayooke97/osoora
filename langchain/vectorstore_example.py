import os
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from typing import List, Optional

# Set environment variables for OpenAI-compatible API
os.environ["OPENAI_API_KEY"] = "sk-f0f088df10a44cd3a5cf172c0ebfaaf6"
os.environ["OPENAI_BASE_URL"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Create a custom embeddings class that uses DashScope's OpenAI-compatible API
class DashScopeEmbeddings(Embeddings):
    """DashScope embeddings using OpenAI-compatible API."""
    
    def __init__(
        self,
        api_key: str = os.environ["OPENAI_API_KEY"],
        base_url: str = os.environ["OPENAI_BASE_URL"],
        model: str = "text-embedding-v3",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using DashScope."""
        embeddings = []
        # Process in batches to avoid rate limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch: {e}")
                # Return empty embeddings for failed batch
                embeddings.extend([[0.0] * 1024] * len(batch))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using DashScope."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Return empty embedding on failure
            return [0.0] * 1024

# Initialize our custom embeddings class
embeddings = DashScopeEmbeddings()

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Python is a high-level, interpreted programming language known for its readability and versatility.",
    "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
    "Vector databases store and retrieve high-dimensional vectors efficiently, making them ideal for similarity search operations.",
    "Embeddings are vector representations of text, images, or other data that capture semantic meaning.",
    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
    "LangChain is a framework for developing applications powered by language models.",
]

# Convert to LangChain Document objects
doc_objects = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) for i, doc in enumerate(documents)]

# Create a text splitter (optional, for larger documents)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(doc_objects)

print(f"Processing {len(docs)} documents...")

# Create a FAISS vector store from the documents
vectorstore = FAISS.from_documents(docs, embeddings)

# Save the vector store to disk (optional)
vectorstore.save_local("faiss_index")

print("Vector store created and saved successfully!")

# Example: Perform a similarity search
query = "What is artificial intelligence?"
print(f"\nSearching for: '{query}'")

# Get similar documents
similar_docs = vectorstore.similarity_search(query, k=3)

print("\nTop 3 most similar documents:")
for i, doc in enumerate(similar_docs):
    print(f"{i+1}. Source: {doc.metadata['source']}")
    print(f"   Content: {doc.page_content}")
    print()

# Example: Perform a similarity search with scores
query = "programming languages"
print(f"\nSearching for: '{query}' (with similarity scores)")

# Get similar documents with scores
similar_docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

print("\nTop 3 most similar documents (with scores):")
for i, (doc, score) in enumerate(similar_docs_with_scores):
    print(f"{i+1}. Source: {doc.metadata['source']} (Score: {score})")
    print(f"   Content: {doc.page_content}")
    print()

# Example: Load the vector store from disk
print("\nLoading vector store from disk...")
loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)

# Verify it works
query = "vector databases"
print(f"\nSearching loaded vector store for: '{query}'")
similar_docs = loaded_vectorstore.similarity_search(query, k=2)

print("\nTop 2 most similar documents from loaded vector store:")
for i, doc in enumerate(similar_docs):
    print(f"{i+1}. Source: {doc.metadata['source']}")
    print(f"   Content: {doc.page_content}")
    print()
