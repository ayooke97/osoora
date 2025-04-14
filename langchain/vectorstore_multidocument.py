import json
import os
import numpy as np
import time
import datetime
import webbrowser
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import colorama
from colorama import Fore, Style
import html

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)

load_dotenv()

# Output files
OUTPUT_JSON = "vectorstore_multidoc_results.json"
OUTPUT_HTML = "vectorstore_multidoc_results.html"
OUTPUT_LOG = "vectorstore_multidoc_output.json"
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

# Custom Embeddings class that implements LangChain's Embeddings interface
class DashScopeEmbeddings(Embeddings):
    """DashScope embeddings using OpenAI-compatible API."""
    
    def __init__(
        self,
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY"),
        base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using DashScope."""
        embeddings = []
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                log_json_message("error", f"Error embedding batch: {e}")
                # Return empty embeddings for failed batch
                embeddings.extend([[0.0] * self.dimensions] * len(batch))
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using DashScope."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            log_json_message("error", f"Error embedding query: {e}")
            # Return empty embedding on failure
            return [0.0] * self.dimensions

def generate_html_output(data):
    """Generate a pretty HTML visualization of the JSON data."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Document Vector Store Results</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                color: #3498db;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .query-container {{
                margin-bottom: 30px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                background-color: #f9f9f9;
                border-radius: 0 8px 8px 0;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .query {{
                font-size: 1.2em;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 10px;
                padding: 5px 0;
                border-bottom: 1px solid #e0e0e0;
            }}
            .answer {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 3px solid #2ecc71;
            }}
            .documents {{
                margin-top: 15px;
            }}
            .document {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 3px solid #e74c3c;
                transition: all 0.2s ease;
            }}
            .document:hover {{
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }}
            .metadata {{
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 5px;
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }}
            .metadata span {{
                margin-right: 10px;
                background-color: #ecf0f1;
                padding: 3px 8px;
                border-radius: 3px;
                display: inline-block;
            }}
            .timestamp {{
                text-align: center;
                font-size: 0.8em;
                color: #95a5a6;
                margin-top: 30px;
            }}
            .filtered-section {{
                border-top: 2px dashed #3498db;
                margin-top: 30px;
                padding-top: 20px;
            }}
            .filter-badge {{
                display: inline-block;
                background-color: #9b59b6;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                margin-left: 10px;
                font-size: 0.8em;
            }}
            .sources {{
                margin-top: 10px;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .source-item {{
                background-color: #f0f0f0;
                padding: 8px;
                margin-bottom: 5px;
                border-radius: 4px;
                border-left: 2px solid #3498db;
            }}
            .nav {{
                position: sticky;
                top: 0;
                background-color: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                z-index: 100;
            }}
            .nav-links {{
                display: flex;
                justify-content: center;
                gap: 15px;
            }}
            .nav-link {{
                color: #3498db;
                text-decoration: none;
                padding: 5px 10px;
                border-radius: 4px;
                transition: background-color 0.2s;
            }}
            .nav-link:hover {{
                background-color: #f0f7ff;
            }}
            .section {{
                margin-top: 40px;
                padding-top: 10px;
            }}
            .doc-type {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 0.8em;
                color: white;
                margin-right: 5px;
            }}
            .doc-type-person {{
                background-color: #3498db;
            }}
            .doc-type-company {{
                background-color: #e74c3c;
            }}
            .doc-type-report {{
                background-color: #2ecc71;
            }}
            .doc-type-event {{
                background-color: #f39c12;
            }}
            .doc-type-article {{
                background-color: #9b59b6;
            }}
            .doc-type-default {{
                background-color: #95a5a6;
            }}
            @media (max-width: 768px) {{
                .container {{
                    padding: 10px;
                }}
                .metadata {{
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Multi-Document Vector Store Results</h1>
            
            <div class="nav">
                <div class="nav-links">
                    <a href="#general-queries" class="nav-link">General Queries</a>
                    <a href="#document-types" class="nav-link">Document Types</a>
                    <a href="#cross-document" class="nav-link">Cross-Document Queries</a>
                </div>
            </div>
            
            <div id="general-queries" class="section">
                <h2>General Queries</h2>
    """
    
    # Add general queries
    for query_result in data["general_queries"]:
        query = query_result["query"]
        answer = query_result["answer"].get("answer", "No answer generated")
        sources = query_result["answer"].get("sources", [])
        documents = query_result["retrieved_documents"]
        
        html_content += f"""
            <div class="query-container">
                <div class="query">Q: {html.escape(query)}</div>
                <div class="answer">
                    <strong>Answer:</strong> {html.escape(answer)}
                </div>
                
                <div class="documents">
                    <h3>Retrieved Documents ({len(documents)})</h3>
        """
        
        for doc in documents:
            doc_type = doc["metadata"]["type"]
            doc_type_class = f"doc-type-{doc_type}" if doc_type in ["person", "company", "report", "event", "article"] else "doc-type-default"
            
            html_content += f"""
                    <div class="document">
                        <div>
                            <span class="doc-type {doc_type_class}">{html.escape(doc_type)}</span>
                            {html.escape(doc["content"])}
                        </div>
                        <div class="metadata">
                            <span>Type: {html.escape(doc["metadata"]["type"])}</span>
                            <span>Source: {html.escape(doc["metadata"]["source"])}</span>
                            <span>ID: {html.escape(doc["metadata"]["id"])}</span>
                        </div>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    # Add document type section
    html_content += f"""
            <div id="document-types" class="section">
                <h2>Document Types</h2>
                <div class="query-container">
                    <p>This vector store contains the following document types:</p>
                    <ul>
    """
    
    doc_types = data.get("document_types", {})
    for doc_type, count in doc_types.items():
        html_content += f"""
                        <li><strong>{html.escape(doc_type)}</strong>: {count} documents</li>
        """
    
    html_content += """
                    </ul>
                </div>
            </div>
    """
    
    # Add cross-document section
    html_content += f"""
            <div id="cross-document" class="section">
                <h2>Cross-Document Queries</h2>
    """
    
    for query_result in data.get("cross_document_queries", []):
        query = query_result["query"]
        answer = query_result["answer"].get("answer", "No answer generated")
        documents = query_result["retrieved_documents"]
        synonyms = query_result.get("synonym_validation", {})
        
        html_content += f"""
                <div class="query-container">
                    <div class="query">Q: {html.escape(query)}</div>
                    <div class="answer">
                        <strong>Answer:</strong> {html.escape(answer)}
                    </div>
                    
                    <div class="documents">
                        <h3>Retrieved Documents from Multiple Sources ({len(documents)})</h3>
        """
        
        for doc in documents:
            doc_type = doc["metadata"]["type"]
            doc_type_class = f"doc-type-{doc_type}" if doc_type in ["person", "company", "report", "event", "article"] else "doc-type-default"
            
            html_content += f"""
                        <div class="document">
                            <div>
                                <span class="doc-type {doc_type_class}">{html.escape(doc_type)}</span>
                                {html.escape(doc["content"])}
                            </div>
                            <div class="metadata">
                                <span>Type: {html.escape(doc["metadata"]["type"])}</span>
                                <span>Source: {html.escape(doc["metadata"]["source"])}</span>
                                <span>ID: {html.escape(doc["metadata"]["id"])}</span>
                            </div>
                        </div>
            """
        
        html_content += f"""
                    </div>
                    
                    <div class="synonym-validation">
                        <h3>Synonym Validation</h3>
                        <ul>
        """
        
        for base_type, syn_list in synonyms.items():
            html_content += f"""
                            <li><strong>{html.escape(base_type)}</strong>: {', '.join([html.escape(s) for s in syn_list])}</li>
            """
        
        html_content += """
                        </ul>
                    </div>
                </div>
        """
    
    html_content += f"""
        </div>
        
        <div class="timestamp">
            Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        
        <script>
            // Add smooth scrolling for navigation links
            document.querySelectorAll('.nav-link').forEach(link => {{
                link.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const targetId = this.getAttribute('href');
                    const targetElement = document.querySelector(targetId);
                    window.scrollTo({{
                        top: targetElement.offsetTop - 20,
                        behavior: 'smooth'
                    }});
                }});
            }});
        </script>
    </div>
    </body>
    </html>
    """
    
    return html_content

def query_vectorstore(retriever, query_text: str) -> Dict[str, Any]:
    """Query the vector store and generate an answer using RAG with JSON output."""
    # First, validate and extract document-related synonyms
    log_json_message("info", "Validating document-related synonyms in the query...")
    synonyms = validate_document_synonyms(query_text)
    
    # Create a prompt template for retrieval that includes metadata and synonym awareness
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Include relevant information from the document metadata if available.
    
    When referring to document types, be aware of these synonym relationships:
    {synonyms}
    
    Your response MUST be in valid JSON format with the following structure:
    {{
      "answer": "your detailed answer here",
      "sources": [
        {{
          "content": "source document content",
          "metadata": {{
            "source": "source name",
            "type": "document type",
            "id": "document id"
          }}
        }}
      ]
    }}

    Context:
    {context}

    Question: {input}

    JSON Response:
    """)
    
    # Format synonyms for inclusion in the prompt
    synonym_text = ""
    for base_type, syn_list in synonyms.items():
        if syn_list:
            synonym_text += f"- '{base_type}' may be referred to as: {', '.join([f'\"{s}\"' for s in syn_list])}\n"
    
    if not synonym_text:
        synonym_text = "No specific document type synonyms identified in this query."
    
    # Create a retrieval chain using the LCEL pattern
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough(), "synonyms": lambda _: synonym_text}
        | prompt
        | llm
        | JsonOutputParser()
    )
    
    # Execute the chain
    try:
        answer = rag_chain.invoke(query_text)
        return answer
    except Exception as e:
        log_json_message("error", f"Error generating answer: {str(e)}")
        return {"error": str(e)}

def validate_document_synonyms(text: str) -> Dict[str, List[str]]:
    """
    Use a secondary LLM to validate and extract document-related synonyms from the text.
    Returns a dictionary of document types and their synonyms.
    """
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following text and identify any terms that could be synonyms for document types.
    Focus specifically on words related to documents, records, files, or information sources.
    
    For example:
    - "paper", "article", "publication" are synonyms for "document"
    - "financial statement", "earnings report" are synonyms for "report"
    - "profile", "biography", "resume" are synonyms for "person document"
    - "organization", "firm", "enterprise" are synonyms for "company document"
    
    Return your analysis as a valid JSON object with the following structure:
    {{
      "document_types": [
        {{
          "base_type": "document type name",
          "synonyms": ["synonym1", "synonym2", "..."]
        }}
      ]
    }}
    
    Text to analyze: {input_text}
    
    JSON Response:
    """)
    
    try:
        chain = prompt | validation_llm | JsonOutputParser()
        result = chain.invoke({"input_text": text})
        
        # Convert to simplified format for easier lookup
        synonym_map = {}
        for item in result.get("document_types", []):
            base_type = item.get("base_type", "")
            if base_type:
                synonym_map[base_type] = item.get("synonyms", [])
        
        return synonym_map
    except Exception as e:
        log_json_message("error", f"Error validating synonyms: {str(e)}")
        return {}

def print_synonym_validation(query, synonyms):
    """Print the synonym validation results in a readable format."""
    print(f"\n{Fore.CYAN}Synonym Validation for Query:{Style.RESET_ALL} {query}")
    
    if not synonyms:
        print(f"{Fore.YELLOW}No document-related synonyms detected.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Detected Document Type Synonyms:{Style.RESET_ALL}")
    for base_type, syn_list in synonyms.items():
        print(f"  {Fore.CYAN}{base_type}:{Style.RESET_ALL} {', '.join(syn_list)}")
    print()

def filter_by_metadata(vstore, metadata_filter):
    """Create a retriever that filters by metadata."""
    return vstore.as_retriever(
        search_kwargs={"k": 3, "filter": metadata_filter}
    )

def format_retrieved_docs(docs):
    """Format retrieved documents as JSON."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        formatted_docs.append({
            "index": i+1,
            "content": doc.page_content,
            "metadata": {
                "type": doc.metadata.get("type", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "id": doc.metadata.get("id", "unknown")
            }
        })
    return formatted_docs

def print_pretty_json(data):
    """Print JSON data with color formatting."""
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Add color to keys and values
    lines = formatted.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            print(f"{Fore.CYAN}{key}{Style.RESET_ALL}:{Fore.YELLOW}{value}{Style.RESET_ALL}")
        else:
            print(line)

def save_vectorstore(vectorstore, directory=VECTOR_STORE_PATH):
    """Save FAISS vector store to disk.
    
    Args:
        vectorstore: The FAISS vectorstore to save
        directory: Directory to save the vectorstore to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save the vectorstore
        vectorstore.save_local(directory)
        log_json_message("success", f"Vector store saved to {directory}")
        return True
    except Exception as e:
        log_json_message("error", f"Error saving vector store: {e}")
        return False

def load_vectorstore(directory=VECTOR_STORE_PATH, embeddings=None):
    """Load FAISS vector store from disk.
    
    Args:
        directory: Directory to load the vectorstore from
        embeddings: Embeddings instance to use with the loaded vectorstore
    
    Returns:
        FAISS: The loaded vectorstore or None if loading failed
    """
    try:
        if not os.path.exists(directory):
            log_json_message("error", f"Vector store directory {directory} does not exist")
            return None
            
        # Load the vectorstore
        vectorstore = FAISS.load_local(directory, embeddings)
        log_json_message("success", f"Vector store loaded from {directory}")
        return vectorstore
    except Exception as e:
        log_json_message("error", f"Error loading vector store: {e}")
        return None

# Initialize LLMs
llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0,
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# Secondary LLM for synonym validation
validation_llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0.2,
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"]
)

if __name__ == "__main__":
    try:
        # Clear the console for better readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_header("Multi-Document Vector Store Demo")
        
        # Initialize embeddings
        log_json_message("info", "Initializing embeddings...")
        embeddings = DashScopeEmbeddings()
        
        # Create sample documents with different types
        log_json_message("info", "Creating sample documents...")
        
        # Person documents
        person_docs = [
            Document(
                page_content="John Smith is a software engineer with 10 years of experience in Python and JavaScript.",
                metadata={"type": "person", "source": "employee_database", "id": "p001"}
            ),
            Document(
                page_content="Sarah Johnson is a data scientist specializing in machine learning and AI applications.",
                metadata={"type": "person", "source": "employee_database", "id": "p002"}
            ),
            Document(
                page_content="Michael Brown is a project manager with experience in agile methodologies.",
                metadata={"type": "person", "source": "employee_database", "id": "p003"}
            )
        ]
        
        # Company documents
        company_docs = [
            Document(
                page_content="Acme Corp is a technology company founded in 2005, specializing in cloud solutions.",
                metadata={"type": "company", "source": "company_database", "id": "c001"}
            ),
            Document(
                page_content="TechGiant Inc. is a global leader in AI and machine learning technologies.",
                metadata={"type": "company", "source": "company_database", "id": "c002"}
            ),
            Document(
                page_content="DataSolutions Ltd provides data analytics and business intelligence services.",
                metadata={"type": "company", "source": "company_database", "id": "c003"}
            )
        ]
        
        # Report documents
        report_docs = [
            Document(
                page_content="Q1 2023 Financial Report: Revenue increased by 15% compared to previous quarter.",
                metadata={"type": "report", "source": "financial_reports", "id": "r001"}
            ),
            Document(
                page_content="Annual Market Analysis: The technology sector grew by 8% in 2023, with AI applications leading the growth.",
                metadata={"type": "report", "source": "market_analysis", "id": "r002"}
            ),
            Document(
                page_content="Project Status Report: The cloud migration project is 75% complete and on schedule.",
                metadata={"type": "report", "source": "project_reports", "id": "r003"}
            )
        ]
        
        # Article documents
        article_docs = [
            Document(
                page_content="New Advances in AI: Researchers have developed a more efficient neural network architecture.",
                metadata={"type": "article", "source": "tech_news", "id": "a001"}
            ),
            Document(
                page_content="The Future of Remote Work: Studies show that hybrid work models increase productivity.",
                metadata={"type": "article", "source": "business_journal", "id": "a002"}
            ),
            Document(
                page_content="Cybersecurity Challenges in 2023: Ransomware attacks have increased by 30% in the first half of the year.",
                metadata={"type": "article", "source": "security_digest", "id": "a003"}
            )
        ]
        
        # Combine all documents
        all_docs = person_docs + company_docs + report_docs + article_docs
        
        # Create vector store
        log_json_message("info", f"Creating vector store with {len(all_docs)} documents...")
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        
        # Save vector store to disk
        log_json_message("info", "Saving vector store to disk...")
        save_vectorstore(vectorstore)
        
        # Create retrievers for different document types
        log_json_message("info", "Creating retrievers for different document types...")
        general_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        person_retriever = filter_by_metadata(vectorstore, {"type": "person"})
        company_retriever = filter_by_metadata(vectorstore, {"type": "company"})
        report_retriever = filter_by_metadata(vectorstore, {"type": "report"})
        article_retriever = filter_by_metadata(vectorstore, {"type": "article"})
        
        # Initialize results dictionary
        all_results = {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_types": {
                "person": len(person_docs),
                "company": len(company_docs),
                "report": len(report_docs),
                "article": len(article_docs)
            },
            "general_queries": [],
            "cross_document_queries": []
        }
        
        # Define general queries
        general_queries = [
            "Who is Michael Brown?",
            "what is the most hype industry in 2023?",
            "What were the financial results in Q1 2023?"
        ]
        
        # Process general queries
        print_header("General Queries")
        for query in general_queries:
            print(f"\n{Fore.YELLOW}Processing query:{Style.RESET_ALL} {query}")
            
            # Create query result structure
            query_result = {
                "query": query,
                "retrieved_documents": [],
                "answer": {}
            }
            
            # Retrieve documents
            log_json_message("info", f"Retrieving documents for: {query}")
            docs = general_retriever.invoke(query)
            query_result["retrieved_documents"] = format_retrieved_docs(docs)
            
            # Print retrieved documents in a pretty format
            print(f"\n{Fore.GREEN}Retrieved Documents:{Style.RESET_ALL}")
            for j, doc in enumerate(query_result["retrieved_documents"]):
                print_document(doc, j+1)
            
            # Get answer
            log_json_message("info", f"Generating answer for: {query}")
            answer = query_vectorstore(general_retriever, query)
            query_result["answer"] = answer
            
            # Print answer in a pretty format
            if "answer" in answer:
                print_answer(query, answer["answer"])
            else:
                print(f"\n{Fore.RED}Error generating answer:{Style.RESET_ALL} {answer.get('error', 'Unknown error')}\n")
            
            # Add a small delay to ensure console output is properly displayed
            time.sleep(0.5)
            
            all_results["general_queries"].append(query_result)
        
        # Define cross-document queries
        cross_doc_queries = [
            "What tech firms are mentioned and which personnel work at these organizations?",
            "Summarize the latest market analysis and financial statements from the reports.",
            "What are the main cybersecurity challenges and technological developments mentioned in the articles?"
        ]
        
        # Process cross-document queries
        print_header("Cross-Document Queries")
        for query in cross_doc_queries:
            print(f"\n{Fore.YELLOW}Processing cross-document query:{Style.RESET_ALL} {query}")
            
            # Validate synonyms first and display results
            synonyms = validate_document_synonyms(query)
            print_synonym_validation(query, synonyms)
            
            # Create query result structure
            query_result = {
                "query": query,
                "retrieved_documents": [],
                "answer": {},
                "synonym_validation": synonyms
            }
            
            # Retrieve documents from multiple sources
            log_json_message("info", f"Retrieving documents for cross-document query: {query}")
            
            # Get documents from different retrievers
            person_docs = person_retriever.invoke(query)
            company_docs = company_retriever.invoke(query)
            report_docs = report_retriever.invoke(query)
            article_docs = article_retriever.invoke(query)
            
            # Combine and sort by relevance (simplified approach)
            all_retrieved_docs = person_docs + company_docs + report_docs + article_docs
            query_result["retrieved_documents"] = format_retrieved_docs(all_retrieved_docs)
            
            # Print retrieved documents in a pretty format
            print(f"\n{Fore.GREEN}Retrieved Documents from Multiple Sources:{Style.RESET_ALL}")
            for j, doc in enumerate(query_result["retrieved_documents"]):
                print_document(doc, j+1)
            
            # Get answer using all documents
            log_json_message("info", f"Generating answer for cross-document query: {query}")
            answer = query_vectorstore(general_retriever, query)
            query_result["answer"] = answer
            
            # Print answer in a pretty format
            if "answer" in answer:
                print_answer(query, answer["answer"])
            else:
                print(f"\n{Fore.RED}Error generating answer:{Style.RESET_ALL} {answer.get('error', 'Unknown error')}\n")
            
            # Add a small delay to ensure console output is properly displayed
            time.sleep(0.5)
            
            all_results["cross_document_queries"].append(query_result)
        
        # Save results to JSON file
        log_json_message("info", f"Saving results to {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log_json_message("success", f"Results saved to {OUTPUT_JSON}")
        
        # Generate HTML output
        log_json_message("info", f"Generating HTML output to {OUTPUT_HTML}...")
        html_content = generate_html_output(all_results)
        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(html_content)
        log_json_message("success", f"HTML output saved to {OUTPUT_HTML}")
        
        # Open HTML in browser
        log_json_message("info", "Opening HTML in browser...")
        open_html_in_browser(OUTPUT_HTML)
        
        # Close the JSON array in the output log
        with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n]")
        
        print_header("Multi-Document Vector Store Demo Completed", Fore.GREEN)
        
    except Exception as e:
        log_json_message("error", f"An error occurred: {str(e)}")
        # Close the JSON array in the output log
        with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n]")
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
