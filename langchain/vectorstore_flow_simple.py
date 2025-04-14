"""
Simple Vector Store Flow Inspection Script

This script demonstrates the basic flow of vector store operations
without relying on complex LangChain components.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
import colorama
from colorama import Fore, Style

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

# ----- Helper Functions -----

def print_separator(color=Fore.BLUE, char="=", length=80):
    """Print a separator line with the specified color."""
    print(f"{color}{char * length}{Style.RESET_ALL}")

def print_header(text, color=Fore.MAGENTA, char="=", length=80):
    """Print a header with the specified text and color."""
    print_separator(color, char, length)
    padding = (length - len(text)) // 2
    print(f"{color}{' ' * padding}{text}{' ' * (length - len(text) - padding)}{Style.RESET_ALL}")
    print_separator(color, char, length)

def print_step(step_num, description):
    """Print a step header."""
    print_separator(Fore.BLUE)
    print(f"{Fore.CYAN}[Step {step_num}] {description}{Style.RESET_ALL}")

# ----- Mock Vector Store Implementation -----

class SimpleDocument:
    """A simple document class with content and metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(content={self.content[:30]}..., metadata={self.metadata})"

class SimpleEmbedding:
    """A simple embedding class that creates random vectors."""
    
    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
    
    def embed_text(self, text: str) -> List[float]:
        """Create a simple deterministic embedding based on text length."""
        # This is a very simple mock embedding - not for real use
        np.random.seed(sum(ord(c) for c in text))
        return list(np.random.rand(self.dimensions))

class SimpleVectorStore:
    """A simple vector store implementation."""
    
    def __init__(self, embedding_function):
        self.documents = []
        self.embeddings = []
        self.embedding_function = embedding_function
    
    def add_documents(self, documents: List[SimpleDocument]):
        """Add documents to the vector store."""
        for doc in documents:
            self.documents.append(doc)
            self.embeddings.append(self.embedding_function.embed_text(doc.content))
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 3, filter_dict: Dict = None):
        """Search for similar documents."""
        if not self.documents:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_function.embed_text(query)
        
        # Calculate similarities (dot product)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Apply filter if provided
            if filter_dict:
                skip = False
                for key, value in filter_dict.items():
                    if key not in self.documents[i].metadata or self.documents[i].metadata[key] != value:
                        skip = True
                        break
                if skip:
                    similarities.append(-1)  # Will be filtered out by the top-k
                    continue
            
            # Simple dot product similarity
            similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            similarities.append(similarity)
        
        # Get top k indices
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
        
        # Return top k documents
        return [self.documents[i] for i in top_indices if similarities[i] > 0]

# ----- Main Flow -----

def main():
    """Main function to demonstrate vector store flow."""
    try:
        # Clear the console for better readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_header("Vector Store Flow Inspection (Simple Version)")
        
        # Step 1: Initialize embedding function
        print_step(1, "Initializing embedding function")
        embedding_function = SimpleEmbedding(dimensions=128)
        print(f"{Fore.GREEN}✓ Embedding function initialized with dimension={embedding_function.dimensions}{Style.RESET_ALL}")
        
        # Step 2: Create sample documents
        print_step(2, "Creating sample documents")
        
        # Create a set of documents for testing
        person_docs = [
            SimpleDocument(
                content="Michael Brown is a project manager with experience in agile methodologies.",
                metadata={"type": "person", "source": "employee_database", "id": "michael_brown"}
            ),
            SimpleDocument(
                content="Sarah Johnson is a data scientist specializing in machine learning and AI applications.",
                metadata={"type": "person", "source": "employee_database", "id": "sarah_johnson"}
            )
        ]
        
        company_docs = [
            SimpleDocument(
                content="Acme Corp is a technology company founded in 2005, specializing in cloud solutions.",
                metadata={"type": "company", "source": "company_database", "id": "acme_corp"}
            ),
            SimpleDocument(
                content="TechGiant Inc. is a global leader in AI and machine learning technologies.",
                metadata={"type": "company", "source": "company_database", "id": "techgiant"}
            )
        ]
        
        all_docs = person_docs + company_docs
        print(f"{Fore.GREEN}✓ Created {len(all_docs)} sample documents ({len(person_docs)} person, {len(company_docs)} company){Style.RESET_ALL}")
        
        # Step 3: Create vector store
        print_step(3, "Creating vector store")
        vector_store = SimpleVectorStore(embedding_function)
        added = vector_store.add_documents(all_docs)
        print(f"{Fore.GREEN}✓ Vector store created with {added} documents{Style.RESET_ALL}")
        
        # Step 4: Basic search
        print_step(4, "Testing basic search")
        query = "Who is Michael Brown?"
        print(f"{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
        
        results = vector_store.similarity_search(query, k=2)
        print(f"{Fore.GREEN}✓ Found {len(results)} matching documents{Style.RESET_ALL}")
        
        for i, doc in enumerate(results):
            print(f"  {Fore.YELLOW}Document #{i+1}:{Style.RESET_ALL} {doc.content}")
            print(f"    {Fore.CYAN}Type:{Style.RESET_ALL} {doc.metadata['type']}, {Fore.CYAN}ID:{Style.RESET_ALL} {doc.metadata['id']}")
        
        # Step 5: Filtered search
        print_step(5, "Testing filtered search")
        query = "Tell me about technology companies"
        print(f"{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
        
        # Search with company filter
        results = vector_store.similarity_search(query, k=2, filter_dict={"type": "company"})
        print(f"{Fore.GREEN}✓ Found {len(results)} matching company documents{Style.RESET_ALL}")
        
        for i, doc in enumerate(results):
            print(f"  {Fore.YELLOW}Document #{i+1}:{Style.RESET_ALL} {doc.content}")
            print(f"    {Fore.CYAN}Type:{Style.RESET_ALL} {doc.metadata['type']}, {Fore.CYAN}ID:{Style.RESET_ALL} {doc.metadata['id']}")
        
        # Step 6: Multi-query demonstration
        print_step(6, "Testing multiple queries")
        
        queries = [
            "Who is a data scientist?",
            "Tell me about Acme Corp",
            "What companies work with AI?"
        ]
        
        for i, query in enumerate(queries):
            print(f"\n{Fore.YELLOW}Query #{i+1}:{Style.RESET_ALL} {query}")
            results = vector_store.similarity_search(query, k=2)
            print(f"{Fore.GREEN}✓ Found {len(results)} matching documents{Style.RESET_ALL}")
            
            for j, doc in enumerate(results):
                print(f"  {Fore.CYAN}Document #{j+1}:{Style.RESET_ALL} {doc.content}")
                print(f"    {Fore.MAGENTA}Type:{Style.RESET_ALL} {doc.metadata['type']}, {Fore.MAGENTA}ID:{Style.RESET_ALL} {doc.metadata['id']}")
        
        print_separator(Fore.GREEN, "=", 80)
        print(f"{Fore.GREEN}Vector Store Flow Inspection Complete!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Error during flow inspection: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
