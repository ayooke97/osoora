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
OUTPUT_JSON = "vectorstore_results.json"
OUTPUT_HTML = "vectorstore_results.html"
OUTPUT_LOG = "vectorstore_output.json"

# Initialize output log file
with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
    f.write("[\n")  # Start JSON array

output_entries = 0

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
        batch_size = 16
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

def generate_html_output(data):
    """Generate a pretty HTML visualization of the JSON data."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vector Store Query Results</title>
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
            <h1>Vector Store Query Results</h1>
            
            <div class="nav">
                <div class="nav-links">
                    <a href="#general-queries" class="nav-link">General Queries</a>
                    <a href="#filtered-query" class="nav-link">Filtered Query</a>
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
            html_content += f"""
                    <div class="document">
                        <div>{html.escape(doc["content"])}</div>
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
    
    # Add filtered query section
    filtered = data["filtered_query"]
    html_content += f"""
            <div id="filtered-query" class="filtered-section section">
                <h2>Filtered Query</h2>
                <div class="query-container">
                    <div class="query">
                        Q: {html.escape(filtered["query"])}
                        <span class="filter-badge">Filter: {html.escape(str(filtered["filter"]))}</span>
                    </div>
                    <div class="answer">
                        <strong>Answer:</strong> {html.escape(filtered["answer"].get("answer", "No answer generated"))}
                    </div>
                    
                    <div class="documents">
                        <h3>Retrieved Documents ({len(filtered["retrieved_documents"])})</h3>
    """
    
    for doc in filtered["retrieved_documents"]:
        html_content += f"""
                    <div class="document">
                        <div>{html.escape(doc["content"])}</div>
                        <div class="metadata">
                            <span>Type: {html.escape(doc["metadata"]["type"])}</span>
                            <span>Source: {html.escape(doc["metadata"]["source"])}</span>
                            <span>ID: {html.escape(doc["metadata"]["id"])}</span>
                        </div>
                    </div>
        """
    
    html_content += f"""
                </div>
            </div>
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

def open_html_in_browser(html_file):
    """Open the HTML file in the default web browser."""
    try:
        html_path = os.path.abspath(html_file)
        webbrowser.open(f'file:///{html_path}', new=2)
        return True
    except Exception as e:
        print(f"{Fore.RED}Error opening HTML file: {e}{Style.RESET_ALL}")
        return False

# Initialize the custom embeddings class
embeddings = DashScopeEmbeddings()

# Initialize ChatOpenAI for LLM interactions
llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

# Example usage with different document types
def create_sample_vectorstore():
    """Create and save a sample vector store with different document types."""
    # Create Document objects with different types and metadata
    documents = [
        Document(
            page_content="Elonmusk adalah seorang pengusaha teknologi dan pendiri beberapa perusahaan terkenal.",
            metadata={"source": "biography", "type": "person", "id": "elon_musk_bio"}
        ),
        Document(
            page_content="Tesla adalah perusahaan mobil listrik yang didirikan oleh Elon Musk. Perusahaan ini terkenal dengan inovasi teknologi dan desain yang futuristik.",
            metadata={"source": "company_profile", "type": "company", "id": "tesla_profile"}
        ),
        Document(
            page_content="SpaceX adalah perusahaan eksplorasi luar angkasa yang didirikan oleh Elon Musk dengan tujuan mengurangi biaya transportasi luar angkasa dan memungkinkan kolonisasi Mars.",
            metadata={"source": "company_profile", "type": "company", "id": "spacex_profile"}
        ),
        Document(
            page_content="Laporan keuangan Tesla menunjukkan peningkatan pendapatan sebesar 30% pada kuartal terakhir, didorong oleh penjualan Model Y yang kuat.",
            metadata={"source": "financial_report", "type": "report", "id": "tesla_q4_2024"}
        ),
        Document(
            page_content="Peluncuran roket Starship oleh SpaceX mencapai orbit dengan sukses pada misi terakhir, menandai tonggak penting dalam program luar angkasa perusahaan.",
            metadata={"source": "news", "type": "event", "id": "spacex_launch_2024"}
        )
    ]
    
    # Create vector store from documents
    vstore = FAISS.from_documents(documents, embeddings)
    retriever = vstore.as_retriever()
    
    # Save the vector store locally
    vstore.save_local("faiss_store_json")
    
    log_json_message(
        "success", 
        f"Added {len(documents)} documents to the vector store.",
        {"documents_added": len(documents)}
    )
    
    return vstore, retriever

def query_vectorstore(retriever, query_text: str) -> Dict[str, Any]:
    """Query the vector store and generate an answer using RAG with JSON output."""
    # Create a prompt template for retrieval that includes metadata
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Include relevant information from the document metadata if available.
    
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
    
    # Create a retrieval chain using the LCEL pattern
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
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

def print_document(doc, index):
    """Print a document with proper formatting."""
    print(f"{Fore.YELLOW}Document #{index}:{Style.RESET_ALL}")
    print(f"{doc['content']}")
    print(f"  {Fore.CYAN}Type:{Style.RESET_ALL} {doc['metadata']['type']}, " +
          f"{Fore.CYAN}Source:{Style.RESET_ALL} {doc['metadata']['source']}, " +
          f"{Fore.CYAN}ID:{Style.RESET_ALL} {doc['metadata']['id']}\n")

if __name__ == "__main__":
    try:
        # Clear the console for better readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print_header("Vector Store JSON Demo")
        
        # Create and save a sample vector store
        log_json_message("info", "Creating sample vector store...")
        vstore, retriever = create_sample_vectorstore()
        
        # Test multiple queries to demonstrate retrieval across different documents
        queries = [
            "Siapa Elon Musk?",
            "Apa itu Tesla?",
            "Apa tujuan SpaceX?"
        ]
        
        all_results = []
        for i, query in enumerate(queries):
            print()  # Add blank line for better readability
            print_header(f"Query #{i+1}: {query}", Fore.BLUE)
            
            log_json_message("info", f"Processing query: {query}")
            
            query_result = {
                "query": query,
                "retrieved_documents": [],
                "answer": {}
            }
            
            # Retrieve documents for debugging
            log_json_message("info", f"Retrieving documents for: {query}")
            retrieved_docs = retriever.invoke(query)
            query_result["retrieved_documents"] = format_retrieved_docs(retrieved_docs)
            
            # Print retrieved documents in a pretty format
            print(f"\n{Fore.GREEN}Retrieved Documents:{Style.RESET_ALL}")
            for j, doc in enumerate(query_result["retrieved_documents"]):
                print_document(doc, j+1)
            
            # Get answer
            log_json_message("info", f"Generating answer for: {query}")
            answer = query_vectorstore(retriever, query)
            query_result["answer"] = answer
            
            # Print answer in a pretty format
            if "answer" in answer:
                print_answer(query, answer["answer"])
            else:
                print(f"\n{Fore.RED}Error generating answer:{Style.RESET_ALL} {answer.get('error', 'Unknown error')}\n")
            
            # Add a small delay to ensure console output is properly displayed
            time.sleep(0.5)
            
            all_results.append(query_result)
            
            # Log individual result
            log_json_message("result", f"Result for query: {query}", {"data": query_result})
        
        # Demonstrate metadata filtering
        print()  # Add blank line for better readability
        print_header("Filtered Query (Company Documents Only)", Fore.MAGENTA)
        
        log_json_message("info", "Filtering by metadata (only company documents)")
        
        # Create a filtered retriever
        company_retriever = filter_by_metadata(vstore, {"type": "company"})
        query = "Berikan informasi tentang perusahaan Elon Musk"
        
        filtered_result = {
            "query": query,
            "filter": {"type": "company"},
            "retrieved_documents": [],
            "answer": {}
        }
        
        print(f"{Fore.BLUE}Query:{Style.RESET_ALL} {query}")
        print(f"{Fore.BLUE}Filter:{Style.RESET_ALL} type=company\n")
        
        # Retrieve only company documents
        log_json_message("info", f"Retrieving company documents for: {query}")
        filtered_docs = company_retriever.invoke(query)
        filtered_result["retrieved_documents"] = format_retrieved_docs(filtered_docs)
        
        # Print retrieved documents in a pretty format
        print(f"\n{Fore.GREEN}Retrieved Company Documents:{Style.RESET_ALL}")
        for i, doc in enumerate(filtered_result["retrieved_documents"]):
            print_document(doc, i+1)
        
        # Get answer using filtered documents
        log_json_message("info", f"Generating answer for filtered query: {query}")
        answer = query_vectorstore(company_retriever, query)
        filtered_result["answer"] = answer
        
        # Print answer in a pretty format
        if "answer" in answer:
            print_answer(query, answer["answer"])
        else:
            print(f"\n{Fore.RED}Error generating answer:{Style.RESET_ALL} {answer.get('error', 'Unknown error')}\n")
        
        # Add a small delay to ensure console output is properly displayed
        time.sleep(0.5)
        
        # Log filtered result
        log_json_message("result", "Result for filtered query", {"data": filtered_result})
        
        # Compile final results
        final_output = {
            "status": "success",
            "general_queries": all_results,
            "filtered_query": filtered_result
        }
        
        # Save results to JSON file
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # Generate HTML visualization
        html_content = generate_html_output(final_output)
        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        log_json_message("success", f"Results saved to {OUTPUT_JSON}")
        log_json_message("success", f"HTML visualization saved to {OUTPUT_HTML}")
        
        # Close the output file JSON array
        with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n]")
            
        log_json_message("info", f"All JSON output saved to {OUTPUT_LOG}")
        
        print_header("Processing Complete", Fore.GREEN)
        print(f"{Fore.CYAN}JSON results:{Style.RESET_ALL} {OUTPUT_JSON}")
        print(f"{Fore.CYAN}HTML visualization:{Style.RESET_ALL} {OUTPUT_HTML}")
        print(f"{Fore.CYAN}Log file:{Style.RESET_ALL} {OUTPUT_LOG}\n")
        
        # Open HTML file in browser
        print(f"{Fore.GREEN}Opening HTML visualization in browser...{Style.RESET_ALL}")
        if open_html_in_browser(OUTPUT_HTML):
            print(f"{Fore.GREEN}HTML file opened successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Could not open HTML file automatically. Please open it manually.{Style.RESET_ALL}")
        
    except Exception as e:
        # Log error
        log_json_message("error", str(e))
        
        # Close the output file JSON array even on error
        with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
            f.write("\n]")
