import json
import os
import numpy as np
import time
import datetime
import webbrowser
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style
import html

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

load_dotenv()

# Output files
OUTPUT_JSON = "vectorstore_results.json"
OUTPUT_HTML = "vectorstore_results.html"
OUTPUT_LOG = "vectorstore_output.json"
VECTOR_STORE_PATH = "faiss_index"  # Directory to save FAISS index

# Initialize output log file
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    f.write("[\n")  # Start JSON array

output_entries = 0

def log_json_message(status: str, message: str, additional_data: Dict = None):
    """Helper function to log messages in JSON format to file and print colored output."""
    global output_entries
    
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "message": message
    }
    
    if additional_data:
        output.update(additional_data)
    
    # Print a colored progress message to console with clear formatting
    if status == "info":
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {message}")
    elif status == "success":
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {message}")
    elif status == "error":
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")
    elif status == "result":
        print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} {message}")
    else:
        print(f"[{status.upper()}] {message}")
    
    # Write to output file
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        if output_entries > 0:
            f.write(",\n")
        f.write(json.dumps(output, indent=2, ensure_ascii=False))
        output_entries += 1

def print_separator(color=Fore.BLUE, char="=", length=80):
    """Print a separator line with the specified color."""
    print(f"{color}{char * length}{Style.RESET_ALL}")

def print_header(text, color=Fore.MAGENTA, char="=", length=80):
    """Print a header with the specified text and color."""
    print_separator(color, char, length)
    padding = (length - len(text)) // 2
    print(f"{color}{' ' * padding}{text}{' ' * (length - len(text) - padding)}{Style.RESET_ALL}")
    print_separator(color, char, length)

def print_answer(query, answer, color=Fore.GREEN):
    """Print the answer with prominent formatting."""
    print("\n") # Add extra spacing
    print_separator(Fore.CYAN, "=", 80)
    print(f"{Fore.YELLOW}Question:{Style.RESET_ALL} {query}")
    print(f"{Fore.GREEN}Answer:{Style.RESET_ALL} {answer}")
    print_separator(Fore.CYAN, "=", 80)
    print("\n") # Add extra spacing

def print_document(doc, index):
    """Print a document with proper formatting."""
    print(f"{Fore.YELLOW}Document #{index}:{Style.RESET_ALL}")
    print(f"{doc['content']}")
    print(f"  {Fore.CYAN}Type:{Style.RESET_ALL} {doc['metadata']['type']}, " +
          f"{Fore.CYAN}Source:{Style.RESET_ALL} {doc['metadata']['source']}, " +
          f"{Fore.CYAN}ID:{Style.RESET_ALL} {doc['metadata']['id']}\n")

def open_html_in_browser(html_file):
    """Open the HTML file in the default web browser."""
    try:
        html_path = os.path.abspath(html_file)
        webbrowser.open(f'file:///{html_path}', new=2)
        return True
    except Exception as e:
        print(f"{Fore.RED}Error opening HTML file: {e}{Style.RESET_ALL}")
        return False

# ----- Simplified Document and Vector Store Implementation -----

class Document:
    """A simple document class with content and metadata."""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(page_content={self.page_content[:30]}..., metadata={self.metadata})"

class SimpleEmbedding:
    """A simple embedding class that creates deterministic vectors."""
    
    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Create a simple deterministic embedding based on text."""
        # This is a simplified mock embedding - not for real use
        np.random.seed(sum(ord(c) for c in text))
        return list(np.random.rand(self.dimensions))

class FAISS:
    """A simplified FAISS-like vector store implementation."""
    
    @classmethod
    def from_documents(cls, documents: List[Document], embedding):
        """Create a vector store from documents."""
        store = cls(embedding)
        store.add_documents(documents)
        return store
    
    def __init__(self, embedding):
        self.documents = []
        self.embeddings = []
        self.embedding = embedding
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding.embed_documents(texts)
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 3, filter_dict: Dict = None):
        """Search for similar documents."""
        if not self.documents:
            return []
        
        # Create query embedding
        query_embedding = self.embedding.embed_query(query)
        
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
    
    def as_retriever(self, search_kwargs=None):
        """Create a retriever from this vector store."""
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return VectorStoreRetriever(vectorstore=self, search_kwargs=search_kwargs)
    
    def save_local(self, folder_path):
        """Mock saving the vector store."""
        os.makedirs(folder_path, exist_ok=True)
        return True
    
    @classmethod
    def load_local(cls, folder_path, embeddings):
        """Mock loading the vector store."""
        if not os.path.exists(folder_path):
            return None
        return cls(embeddings)

class VectorStoreRetriever:
    """A simple retriever that wraps a vector store."""
    
    def __init__(self, vectorstore, search_kwargs=None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 3}
    
    def invoke(self, query):
        """Retrieve documents for a query."""
        return self.vectorstore.similarity_search(query, **self.search_kwargs)

# ----- Core Functions -----

def query_vectorstore(retriever, query_text: str):
    """
    Query the vector store and generate an answer based on retrieved documents.
    
    Args:
        retriever: The retriever to use
        query_text: The query text
    
    Returns:
        Dict: The answer and sources
    """
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(query_text)
        
        if not docs:
            return {
                "query": query_text,
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
        
        # In a real implementation, this would use an LLM to generate an answer
        # For this simplified version, we'll just use the first document's content
        answer = docs[0].page_content
        
        # Format sources
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "query": query_text,
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        return {
            "query": query_text,
            "error": str(e)
        }

def validate_document_synonyms(text: str):
    """
    Extract document-related synonyms from the text.
    Returns a dictionary of document types and their synonyms.
    """
    # In a real implementation, this would use an LLM
    # For this simplified version, we'll use a rule-based approach
    
    synonyms = {
        "person": [],
        "company": []
    }
    
    # Simple rule-based synonym extraction
    person_synonyms = ["employee", "staff", "personnel", "individual", "worker"]
    company_synonyms = ["organization", "firm", "business", "corporation", "enterprise"]
    
    for synonym in person_synonyms:
        if synonym.lower() in text.lower():
            synonyms["person"].append(synonym)
    
    for synonym in company_synonyms:
        if synonym.lower() in text.lower():
            synonyms["company"].append(synonym)
    
    return {"synonyms": synonyms}

def print_synonym_validation(query, synonyms):
    """Print the synonym validation results in a readable format."""
    print(f"\n{Fore.CYAN}Synonym Validation for:{Style.RESET_ALL} {query}")
    print(f"{Fore.YELLOW}Extracted Synonyms:{Style.RESET_ALL}")
    
    for doc_type, terms in synonyms["synonyms"].items():
        if terms:
            print(f"  {Fore.MAGENTA}{doc_type.capitalize()}:{Style.RESET_ALL} {', '.join(terms)}")
    
    print()  # Add spacing

def filter_by_metadata(vstore, metadata_filter):
    """Create a retriever that filters by metadata."""
    retriever = vstore.as_retriever()
    # In this simplified version, we'll modify the search_kwargs
    retriever.search_kwargs["filter_dict"] = metadata_filter
    return retriever

def format_retrieved_docs(docs):
    """Format retrieved documents as JSON."""
    formatted_docs = []
    for doc in docs:
        formatted_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return formatted_docs

def paraphrase_text(text: str, style: str = "standard"):
    """
    Generate alternative phrasings of the input text.
    
    Args:
        text: The text to paraphrase
        style: The style of paraphrasing to use
    
    Returns:
        str: The paraphrased text
    """
    # In a real implementation, this would use an LLM
    # For this simplified version, we'll use predefined transformations
    
    if style == "formal":
        # Make more formal by adding some formal phrases
        return f"It is noted that {text.lower()}"
    
    elif style == "simple":
        # Simplify by shortening
        words = text.split()
        if len(words) > 5:
            return " ".join(words[:5]) + "..."
        return text
    
    elif style == "creative":
        # Add some creative flair
        return f"Interestingly, {text} - quite remarkable!"
    
    else:  # standard
        # Minor rewording
        return text.replace("the", "a").replace("is", "appears to be")

def paraphrase_documents(docs: List[Document], style: str = "standard"):
    """
    Paraphrase a list of documents while preserving their metadata.
    
    Args:
        docs: List of Document objects to paraphrase
        style: The style of paraphrasing to use
    
    Returns:
        List[Document]: The paraphrased documents
    """
    paraphrased_docs = []
    for doc in docs:
        paraphrased_content = paraphrase_text(doc.page_content, style)
        paraphrased_docs.append(Document(
            page_content=paraphrased_content,
            metadata=doc.metadata.copy()  # Copy metadata to preserve the original
        ))
    return paraphrased_docs

def print_paraphrased_document(original_doc, paraphrased_doc, index):
    """Print original and paraphrased document with proper formatting."""
    print(f"{Fore.YELLOW}Document #{index}:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Original:{Style.RESET_ALL} {original_doc.page_content}")
    print(f"{Fore.MAGENTA}Paraphrased:{Style.RESET_ALL} {paraphrased_doc.page_content}")
    print(f"  {Fore.BLUE}Type:{Style.RESET_ALL} {original_doc.metadata['type']}, " +
          f"{Fore.BLUE}ID:{Style.RESET_ALL} {original_doc.metadata['id']}\n")

def paraphrase_query(query_text: str, style: str = "standard", min_similarity: float = 0.7):
    """
    Paraphrase a user query to potentially improve retrieval results.
    
    Args:
        query_text: The original query text
        style: The style of paraphrasing to use
        min_similarity: Minimum similarity threshold between original and paraphrased query
    
    Returns:
        str: The paraphrased query
    """
    if style == "expanded":
        # Expand the query with additional context
        return f"{query_text} Please provide detailed information."
    
    elif style == "precise":
        # Make the query more specific and focused
        return f"Specifically regarding {query_text}"
    
    else:  # standard
        # Simple rewording
        return paraphrase_text(query_text, "standard")

def compare_retrieval_results(retriever, original_query: str, paraphrase_style: str = "expanded"):
    """
    Compare retrieval results between original and paraphrased queries.
    
    Args:
        retriever: The retriever to use
        original_query: The original query text
        paraphrase_style: The style to use for paraphrasing
    
    Returns:
        Dict: Comparison results including both sets of retrieved documents and answers
    """
    # Get results for original query
    original_results = query_vectorstore(retriever, original_query)
    
    # Paraphrase the query
    paraphrased_query = paraphrase_query(original_query, paraphrase_style)
    
    # Get results for paraphrased query
    paraphrased_results = query_vectorstore(retriever, paraphrased_query)
    
    # Calculate overlap in retrieved documents
    original_doc_ids = [doc["metadata"]["id"] for doc in original_results.get("sources", [])]
    paraphrased_doc_ids = [doc["metadata"]["id"] for doc in paraphrased_results.get("sources", [])]
    
    overlap_ids = set(original_doc_ids).intersection(set(paraphrased_doc_ids))
    
    # Check if paraphrased query has an answer
    has_paraphrased_answer = "answer" in paraphrased_results and paraphrased_results["answer"] != "I don't have enough information to answer this question."
    
    # Check if we need to fall back to original query results
    used_fallback = False
    if not has_paraphrased_answer and "answer" in original_results:
        paraphrased_results = original_results
        used_fallback = True
    
    return {
        "original_query": original_query,
        "paraphrased_query": paraphrased_query,
        "original_results": original_results,
        "paraphrased_results": paraphrased_results,
        "document_overlap": {
            "count": len(overlap_ids),
            "ids": list(overlap_ids)
        },
        "used_fallback": used_fallback
    }

def print_comparison_results(comparison_results: Dict[str, Any]):
    """Print comparison results in a readable format."""
    print(f"\n{Fore.CYAN}Comparison Results:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Original Query:{Style.RESET_ALL} {comparison_results['original_query']}")
    print(f"{Fore.YELLOW}Paraphrased Query:{Style.RESET_ALL} {comparison_results['paraphrased_query']}")
    
    print(f"\n{Fore.MAGENTA}Original Answer:{Style.RESET_ALL} {comparison_results['original_results'].get('answer', 'No answer available')}")
    print(f"{Fore.MAGENTA}Paraphrased Answer:{Style.RESET_ALL} {comparison_results['paraphrased_results'].get('answer', 'No answer available')}")
    
    print(f"\n{Fore.GREEN}Document Overlap:{Style.RESET_ALL} {comparison_results['document_overlap']['count']} documents")
    
    used_fallback = comparison_results.get("used_fallback", False)
    if used_fallback:
        print(f"{Fore.RED}Used Fallback:{Style.RESET_ALL} Yes (paraphrased query didn't yield an answer)")
    
    print("\n" + "="*80)

# Flow inspection main method
if __name__ == "__main__":
    try:
        # Clear the console for better readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_header("Vector Store Flow Inspection")
        
        # Step 1: Initialize embeddings
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 1] Initializing embeddings...{Style.RESET_ALL}")
        embeddings = SimpleEmbedding()
        print(f"{Fore.GREEN}✓ Embeddings initialized: {type(embeddings).__name__}{Style.RESET_ALL}")
        
        # Step 2: Create sample documents
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 2] Creating sample documents...{Style.RESET_ALL}")
        
        # Create a smaller set of documents for testing
        person_docs = [
            Document(
                page_content="Michael Brown is a project manager with experience in agile methodologies.",
                metadata={"type": "person", "source": "employee_database", "id": "michael_brown"}
            ),
            Document(
                page_content="Sarah Johnson is a data scientist specializing in machine learning and AI applications.",
                metadata={"type": "person", "source": "employee_database", "id": "sarah_johnson"}
            )
        ]
        
        company_docs = [
            Document(
                page_content="Acme Corp is a technology company founded in 2005, specializing in cloud solutions.",
                metadata={"type": "company", "source": "company_database", "id": "acme_corp"}
            ),
            Document(
                page_content="TechGiant Inc. is a global leader in AI and machine learning technologies.",
                metadata={"type": "company", "source": "company_database", "id": "techgiant"}
            )
        ]
        
        all_docs = person_docs + company_docs
        print(f"{Fore.GREEN}✓ Created {len(all_docs)} sample documents ({len(person_docs)} person, {len(company_docs)} company){Style.RESET_ALL}")
        
        # Step 3: Create vector store
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 3] Creating vector store...{Style.RESET_ALL}")
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        print(f"{Fore.GREEN}✓ Vector store created with {len(all_docs)} documents{Style.RESET_ALL}")
        
        # Step 4: Create retrievers
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 4] Creating retrievers...{Style.RESET_ALL}")
        general_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        person_retriever = filter_by_metadata(vectorstore, {"type": "person"})
        company_retriever = filter_by_metadata(vectorstore, {"type": "company"})
        print(f"{Fore.GREEN}✓ Created general, person, and company retrievers{Style.RESET_ALL}")
        
        # Step 5: Test basic query
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 5] Testing basic query...{Style.RESET_ALL}")
        query = "Who is Michael Brown?"
        print(f"{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
        
        # Step 5.1: Retrieve documents
        print(f"{Fore.MAGENTA}5.1 Retrieving documents...{Style.RESET_ALL}")
        docs = general_retriever.invoke(query)
        print(f"{Fore.GREEN}✓ Retrieved {len(docs)} documents{Style.RESET_ALL}")
        for i, doc in enumerate(docs):
            print(f"  {Fore.YELLOW}Document #{i+1}:{Style.RESET_ALL} {doc.page_content[:50]}...")
            print(f"    {Fore.CYAN}Type:{Style.RESET_ALL} {doc.metadata['type']}, {Fore.CYAN}ID:{Style.RESET_ALL} {doc.metadata['id']}")
        
        # Step 5.2: Generate answer
        print(f"{Fore.MAGENTA}5.2 Generating answer...{Style.RESET_ALL}")
        result = query_vectorstore(general_retriever, query)
        print_answer(query, result["answer"])
        
        # Step 6: Test filtered query
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 6] Testing filtered query (company only)...{Style.RESET_ALL}")
        query = "Tell me about Acme Corp"
        print(f"{Fore.YELLOW}Query:{Style.RESET_ALL} {query}")
        
        # Step 6.1: Retrieve documents with filter
        print(f"{Fore.MAGENTA}6.1 Retrieving documents with company filter...{Style.RESET_ALL}")
        docs = company_retriever.invoke(query)
        print(f"{Fore.GREEN}✓ Retrieved {len(docs)} documents{Style.RESET_ALL}")
        for i, doc in enumerate(docs):
            print(f"  {Fore.YELLOW}Document #{i+1}:{Style.RESET_ALL} {doc.page_content[:50]}...")
            print(f"    {Fore.CYAN}Type:{Style.RESET_ALL} {doc.metadata['type']}, {Fore.CYAN}ID:{Style.RESET_ALL} {doc.metadata['id']}")
        
        # Step 6.2: Generate answer
        print(f"{Fore.MAGENTA}6.2 Generating answer...{Style.RESET_ALL}")
        result = query_vectorstore(company_retriever, query)
        print_answer(query, result["answer"])
        
        # Step 7: Test synonym extraction
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 7] Testing synonym extraction...{Style.RESET_ALL}")
        text = "Find employees and organizations related to AI"
        print(f"{Fore.YELLOW}Text:{Style.RESET_ALL} {text}")
        
        synonyms = validate_document_synonyms(text)
        print(f"{Fore.GREEN}✓ Extracted synonyms:{Style.RESET_ALL}")
        print_synonym_validation(text, synonyms)
        
        # Step 8: Test paraphrasing
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 8] Testing paraphrasing...{Style.RESET_ALL}")
        original_text = "The quick brown fox jumps over the lazy dog"
        print(f"{Fore.YELLOW}Original text:{Style.RESET_ALL} {original_text}")
        
        styles = ["standard", "formal", "creative"]
        for style in styles:
            paraphrased = paraphrase_text(original_text, style)
            print(f"{Fore.MAGENTA}{style.capitalize()} style:{Style.RESET_ALL} {paraphrased}")
        
        # Step 9: Test document paraphrasing
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 9] Testing document paraphrasing...{Style.RESET_ALL}")
        
        # Paraphrase the first document of each type
        print(f"{Fore.YELLOW}Paraphrasing sample documents...{Style.RESET_ALL}")
        
        for i, doc in enumerate([person_docs[0], company_docs[0]]):
            paraphrased_doc = Document(
                page_content=paraphrase_text(doc.page_content, "formal"),
                metadata=doc.metadata.copy()
            )
            print_paraphrased_document(doc, paraphrased_doc, i+1)
        
        # Step 10: Test query paraphrasing
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 10] Testing query paraphrasing...{Style.RESET_ALL}")
        query = "Who is the data scientist?"
        print(f"{Fore.YELLOW}Original query:{Style.RESET_ALL} {query}")
        
        styles = ["standard", "expanded", "precise"]
        for style in styles:
            paraphrased_query = paraphrase_query(query, style)
            print(f"{Fore.MAGENTA}{style.capitalize()} style:{Style.RESET_ALL} {paraphrased_query}")
        
        # Step 11: Compare retrieval results
        print_separator(Fore.BLUE)
        print(f"{Fore.CYAN}[Step 11] Comparing retrieval results...{Style.RESET_ALL}")
        comparison = compare_retrieval_results(general_retriever, query, "expanded")
        print_comparison_results(comparison)
        
        print_separator(Fore.GREEN, "=", 80)
        print(f"{Fore.GREEN}Vector Store Flow Inspection Complete!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Error during flow inspection: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close the JSON array in the output log
        with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n]")
