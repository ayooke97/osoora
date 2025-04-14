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
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .answer {{
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
                margin-bottom: 20px;
            }}
            .document {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                border-left: 3px solid #e74c3c;
            }}
            .metadata {{
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .highlight {{
                background-color: #ffffcc;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .query {{
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 10px;
            }}
            .comparison-container {{
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }}
            .comparison-column {{
                flex: 1;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .fallback-notice {{
                background-color: #fff3cd;
                color: #856404;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #ffeeba;
            }}
            .similarity-score {{
                background-color: #d4edda;
                color: #155724;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 10px;
            }}
            .similarity-low {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .tab {{
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 8px 8px 0 0;
            }}
            .tab button {{
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }}
            .tab button:hover {{
                background-color: #ddd;
            }}
            .tab button.active {{
                background-color: #3498db;
                color: white;
            }}
            .tabcontent {{
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 8px 8px;
                animation: fadeEffect 1s;
                background-color: white;
            }}
            @keyframes fadeEffect {{
                from {{opacity: 0;}}
                to {{opacity: 1;}}
            }}
        </style>
    </head>
    <body>
        <h1>Vector Store Query Results</h1>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'QueryResults')">Query Results</button>
            <button class="tablinks" onclick="openTab(event, 'Paraphrasing')">Paraphrasing Examples</button>
            <button class="tablinks" onclick="openTab(event, 'Comparison')">Query Comparison</button>
        </div>
        
        <div id="QueryResults" class="tabcontent" style="display: block;">
            <h2>Query Results</h2>
    """
    
    # Add query results
    if "query_results" in data:
        for i, result in enumerate(data["query_results"]):
            query = result.get("query", "Unknown query")
            answer = result.get("answer", {}).get("answer", "No answer available")
            has_answer = result.get("answer", {}).get("has_answer", True)
            
            html_content += f"""
            <div class="section">
                <div class="query">Query {i+1}: {query}</div>
            """
            
            if has_answer:
                html_content += f"""
                <div class="answer">{answer}</div>
                """
            else:
                html_content += f"""
                <div class="answer" style="border-left: 4px solid #e74c3c;">
                    <strong>No Answer Available:</strong> {answer}
                </div>
                """
            
            html_content += "<h3>Retrieved Documents</h3>"
            
            documents = result.get("documents", [])
            if documents:
                for j, doc in enumerate(documents):
                    content = doc.get("content", "No content available")
                    metadata = doc.get("metadata", {})
                    doc_type = metadata.get("type", "unknown")
                    source = metadata.get("source", "unknown")
                    doc_id = metadata.get("id", "unknown")
                    
                    html_content += f"""
                    <div class="document">
                        <div>{content}</div>
                        <div class="metadata">
                            Type: {doc_type}, Source: {source}, ID: {doc_id}
                        </div>
                    </div>
                    """
            else:
                html_content += "<p>No documents retrieved</p>"
            
            html_content += "</div>"
    
    html_content += """
        </div>
        
        <div id="Paraphrasing" class="tabcontent">
            <h2>Paraphrasing Examples</h2>
    """
    
    # Add paraphrasing examples
    if "paraphrasing_examples" in data:
        for i, example in enumerate(data["paraphrasing_examples"]):
            original_text = example.get("original_text", "")
            paraphrased_text = example.get("paraphrased_text", "")
            style = example.get("style", "standard")
            similarity = example.get("similarity", 0.0)
            
            similarity_class = "similarity-low" if similarity < 0.8 else ""
            
            html_content += f"""
            <div class="section">
                <h3>Example {i+1}: {style.capitalize()} Style</h3>
                <div class="similarity-score {similarity_class}">Similarity Score: {similarity:.2f}</div>
                <h4>Original Text:</h4>
                <div class="document">{original_text}</div>
                <h4>Paraphrased Text:</h4>
                <div class="document">{paraphrased_text}</div>
            </div>
            """
    
    html_content += """
        </div>
        
        <div id="Comparison" class="tabcontent">
            <h2>Query Comparison Results</h2>
    """
    
    # Add comparison results
    if "comparison_results" in data:
        for i, comparison in enumerate(data["comparison_results"]):
            original_query = comparison.get("original_query", "")
            paraphrased_query = comparison.get("paraphrased_query", "")
            style = comparison.get("paraphrase_style", "standard")
            used_fallback = comparison.get("used_fallback", False)
            
            original_answer = comparison.get("original_results", {}).get("answer", {}).get("answer", "No answer available")
            original_has_answer = comparison.get("original_results", {}).get("answer", {}).get("has_answer", True)
            
            paraphrased_answer = comparison.get("paraphrased_results", {}).get("answer", {}).get("answer", "No answer available")
            paraphrased_has_answer = comparison.get("paraphrased_results", {}).get("answer", {}).get("has_answer", True)
            
            html_content += f"""
            <div class="section">
                <h3>Comparison {i+1}</h3>
                <div class="query">Original Query: {original_query}</div>
                <div class="query">Paraphrased Query ({style} style): {paraphrased_query}</div>
                
                <div class="comparison-container">
                    <div class="comparison-column">
                        <h4>Original Query Results</h4>
            """
            
            if original_has_answer:
                html_content += f"""
                        <div class="answer">{original_answer}</div>
                """
            else:
                html_content += f"""
                        <div class="answer" style="border-left: 4px solid #e74c3c;">
                            <strong>No Answer Available:</strong> {original_answer}
                        </div>
                """
            
            html_content += """
                    </div>
                    <div class="comparison-column">
                        <h4>Paraphrased Query Results</h4>
            """
            
            if used_fallback:
                html_content += f"""
                        <div class="fallback-notice">
                            <strong>Note:</strong> Fallback to original query results was used because the paraphrased query didn't yield relevant results.
                        </div>
                """
            
            if paraphrased_has_answer:
                html_content += f"""
                        <div class="answer">{paraphrased_answer}</div>
                """
            else:
                html_content += f"""
                        <div class="answer" style="border-left: 4px solid #e74c3c;">
                            <strong>No Answer Available:</strong> {paraphrased_answer}
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
            </div>
            """
    
    html_content += """
        </div>
        
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        </script>
    </body>
    </html>
    """
    
    return html_content

def query_vectorstore(retriever, query_text: str) -> Dict[str, Any]:
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
                "has_answer": False,
                "answer": "I don't have enough information to answer this question",
                "sources": []
            }
        
        # Create a prompt for generating an answer
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question based on the provided context.
        If the context doesn't contain enough information to fully answer the question, 
        provide a partial answer based on what you know from the context, and clearly indicate 
        what aspects you're uncertain about or what additional information would be needed.
        
        Always try to provide some relevant information, even if it's not a complete answer.
        Only say "I don't have enough information" if there is absolutely nothing relevant in the context.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """)
        
        # Format the context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate an answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query_text})
        
        # Check if the answer indicates insufficient information
        has_answer = "I don't have enough information" not in answer
        
        # Format the result
        result = {
            "has_answer": has_answer,
            "answer": answer,
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        }
        
        return result
    except Exception as e:
        log_json_message("error", f"Error querying vector store: {str(e)}")
        return {
            "has_answer": False,
            "error": str(e),
            "sources": []
        }

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
        # Use the validation LLM to get synonyms
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

def paraphrase_text(text: str, style: str = "standard") -> str:
    """
    Use the paraphrase LLM to generate alternative phrasings of the input text.
    
    Args:
        text: The text to paraphrase
        style: The style of paraphrasing to use
    
    Returns:
        str: The paraphrased text
    """
    style_descriptions = {
        "standard": "Maintain the same level of formality and tone as the original.",
        "formal": "Use more formal language and academic tone.",
        "simple": "Simplify the language for better readability and understanding.",
        "creative": "Use more expressive and engaging language.",
        "expanded": "Expand the query with related terms that might improve search results.",
        "precise": "Make the query more specific and focused on the core information need."
    }
    
    style_desc = style_descriptions.get(style.lower(), style_descriptions["standard"])
    
    prompt = ChatPromptTemplate.from_template("""
    Paraphrase the following text while preserving its original meaning.
    
    Style instructions: {style_desc}
    
    Original text:
    {input_text}
    
    Paraphrased version:
    """)
    
    try:
        chain = prompt | paraphrase_llm | StrOutputParser()
        result = chain.invoke({
            "input_text": text,
            "style_desc": style_desc
        })
        log_json_message("success", f"Text paraphrased in {style} style")
        return result
    except Exception as e:
        log_json_message("error", f"Error paraphrasing text: {str(e)}")
        return text  # Return original text if paraphrasing fails

def paraphrase_documents(docs: List[Document], style: str = "standard") -> List[Document]:
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
        # Create a new document with paraphrased content but same metadata
        paraphrased_doc = Document(
            page_content=paraphrased_content,
            metadata=doc.metadata
        )
        paraphrased_docs.append(paraphrased_doc)
    
    return paraphrased_docs

def print_paraphrased_document(original_doc, paraphrased_doc, index):
    """Print original and paraphrased document with proper formatting."""
    print(f"{Fore.YELLOW}Document #{index}:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Original:{Style.RESET_ALL}")
    print(f"{original_doc.page_content}")
    print(f"{Fore.GREEN}Paraphrased:{Style.RESET_ALL}")
    print(f"{paraphrased_doc.page_content}")
    print(f"  {Fore.CYAN}Type:{Style.RESET_ALL} {original_doc.metadata.get('type', 'unknown')}, " +
          f"{Fore.CYAN}Source:{Style.RESET_ALL} {original_doc.metadata.get('source', 'unknown')}, " +
          f"{Fore.CYAN}ID:{Style.RESET_ALL} {original_doc.metadata.get('id', 'unknown')}\n")

def paraphrase_query(query_text: str, style: str = "standard", min_similarity: float = 0.7) -> str:
    """
    Paraphrase a user query to potentially improve retrieval results.
    
    Args:
        query_text: The original query text
        style: The style of paraphrasing to use
        min_similarity: Minimum similarity threshold between original and paraphrased query
    
    Returns:
        str: The paraphrased query
    """
    style_descriptions = {
        "standard": "Maintain the same level of formality and tone as the original.",
        "formal": "Use more formal language and academic tone.",
        "simple": "Simplify the language for better readability and understanding.",
        "creative": "Use more expressive and engaging language.",
        "expanded": "Expand the query with related terms that might improve search results.",
        "precise": "Make the query more specific and focused on the core information need."
    }
    
    style_desc = style_descriptions.get(style.lower(), style_descriptions["standard"])
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert at reformulating search queries to improve information retrieval results.
    Paraphrase the following query in a way that might help retrieve more relevant information.
    
    GUIDELINES:
    1. Preserve the core meaning and intent of the original query
    2. Maintain all key entities, concepts, and important terms
    3. You can add related terms or synonyms that might help with retrieval
    4. Feel free to restructure the query if it helps clarify the information need
    5. The goal is to maximize the chance of retrieving relevant information
    
    Style instructions: {style_desc}
    
    Original query:
    {input_text}
    
    Paraphrased query:
    """)
    
    try:
        chain = prompt | paraphrase_llm | StrOutputParser()
        result = chain.invoke({
            "input_text": query_text,
            "style_desc": style_desc
        })
        
        # Check similarity between original and paraphrased query
        similarity = calculate_query_similarity(query_text, result)
        
        # If similarity is below threshold, use original query
        if similarity < min_similarity:
            log_json_message("warning", f"Paraphrased query similarity ({similarity:.4f}) below threshold ({min_similarity}). Using original query.")
            return query_text
            
        log_json_message("success", f"Query paraphrased in {style} style with similarity {similarity:.4f}")
        return result
    except Exception as e:
        log_json_message("error", f"Error paraphrasing query: {str(e)}")
        return query_text  # Return original query if paraphrasing fails

def compare_retrieval_results(retriever, original_query: str, paraphrase_style: str = "expanded") -> Dict[str, Any]:
    """
    Compare retrieval results between original and paraphrased queries.
    
    Args:
        retriever: The retriever to use
        original_query: The original query text
        paraphrase_style: The style to use for paraphrasing
    
    Returns:
        Dict: Comparison results including both sets of retrieved documents and answers
    """
    log_json_message("info", f"Comparing retrieval results for original and paraphrased query ({paraphrase_style} style)")
    
    # Get paraphrased query
    paraphrased_query = paraphrase_query(original_query, paraphrase_style)
    
    # Retrieve documents for original query
    log_json_message("info", f"Retrieving documents for original query: {original_query}")
    original_docs = retriever.invoke(original_query)
    original_formatted_docs = format_retrieved_docs(original_docs)
    
    # Get answer for original query
    log_json_message("info", f"Generating answer for original query")
    original_answer = query_vectorstore(retriever, original_query)
    
    # Retrieve documents for paraphrased query
    log_json_message("info", f"Retrieving documents for paraphrased query: {paraphrased_query}")
    paraphrased_docs = retriever.invoke(paraphrased_query)
    paraphrased_formatted_docs = format_retrieved_docs(paraphrased_docs)
    
    # Get answer for paraphrased query
    log_json_message("info", f"Generating answer for paraphrased query")
    paraphrased_answer = query_vectorstore(retriever, paraphrased_query)
    
    # Check if paraphrased query has an answer
    has_paraphrased_answer = paraphrased_answer.get("has_answer", True)
    
    # If paraphrased query doesn't have an answer but original does, use original as fallback
    if not has_paraphrased_answer and original_answer.get("has_answer", False):
        log_json_message("info", "Paraphrased query didn't yield results, using original query results as fallback")
        paraphrased_answer = {
            "has_answer": True,
            "answer": f"[FALLBACK TO ORIGINAL QUERY] {original_answer.get('answer', '')}",
            "sources": original_answer.get("sources", []),
            "used_fallback": True
        }
    
    # Format results
    results = {
        "original_query": original_query,
        "paraphrased_query": paraphrased_query,
        "paraphrase_style": paraphrase_style,
        "original_results": {
            "retrieved_documents": original_formatted_docs,
            "answer": original_answer
        },
        "paraphrased_results": {
            "retrieved_documents": paraphrased_formatted_docs,
            "answer": paraphrased_answer
        },
        "used_fallback": paraphrased_answer.get("used_fallback", False)
    }
    
    return results

def print_comparison_results(comparison_results: Dict[str, Any]):
    """Print comparison results in a readable format."""
    original_query = comparison_results["original_query"]
    paraphrased_query = comparison_results["paraphrased_query"]
    paraphrase_style = comparison_results["paraphrase_style"]
    
    print("\n" + "-"*80)
    print(f"Comparison Results for Original vs. Paraphrased Query ({paraphrase_style} style)")
    print("-"*80)
    
    # Print queries
    print(f"\n{Fore.CYAN}Original Query:{Style.RESET_ALL} {original_query}")
    print(f"{Fore.CYAN}Paraphrased Query:{Style.RESET_ALL} {paraphrased_query}")
    
    # Check if fallback was used
    if comparison_results.get('used_fallback', False):
        print(f"\n{Fore.YELLOW}FALLBACK USED:{Style.RESET_ALL} The paraphrased query didn't yield results, using original query results as fallback.")
    
    # Print original query results
    print(f"\n{Fore.GREEN}Original Query Results:{Style.RESET_ALL}")
    if comparison_results['original_results']['answer'].get('has_answer', False):
        print(f"\n{Fore.WHITE}{comparison_results['original_results']['answer'].get('answer', 'No answer generated')}{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.RED}No Answer Available:{Style.RESET_ALL} {comparison_results['original_results']['answer'].get('answer', 'I don\'t have enough information to answer this question')}\n")
    
    # Print paraphrased query results
    print(f"\n{Fore.GREEN}Paraphrased Query Results:{Style.RESET_ALL}")
    if comparison_results['paraphrased_results']['answer'].get('has_answer', False):
        print(f"\n{Fore.WHITE}{comparison_results['paraphrased_results']['answer'].get('answer', 'No answer generated')}{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.RED}No Answer Available:{Style.RESET_ALL} {comparison_results['paraphrased_results']['answer'].get('answer', 'I don\'t have enough information to answer this question')}\n")

def calculate_query_similarity(query1: str, query2: str) -> float:
    """
    Calculate semantic similarity between two queries using embeddings.
    
    Args:
        query1: First query
        query2: Second query
        
    Returns:
        float: Similarity score between 0 and 1
    """
    try:
        # Get embeddings for both queries
        embedding1 = embeddings.embed_query(query1)
        embedding2 = embeddings.embed_query(query2)
        
        # Convert to numpy arrays for calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        log_json_message("info", f"Query similarity: {similarity:.4f}")
        return float(similarity)
    except Exception as e:
        log_json_message("error", f"Error calculating query similarity: {str(e)}")
        return 0.0  # Default to 0 on error

def test_no_answer_fallback(retriever):
    """
    Test the fallback mechanism when a paraphrased query doesn't yield an answer
    but the original query does.
    
    Args:
        retriever: The retriever to use for testing
    """
    print("\n" + "="*80)
    print("TESTING NO ANSWER FALLBACK MECHANISM".center(80))
    print("="*80)
    
    # Use a query that will likely have an answer
    original_query = "What were the financial results in Q1 2023?"
    
    # Create a deliberately poor paraphrase that might not yield results
    poor_paraphrase = "What were the accounting figures for some random period?"
    
    # First, test with the original query
    print(f"\nOriginal Query: \"{original_query}\"")
    original_answer = query_vectorstore(retriever, original_query)
    has_original_answer = original_answer.get("has_answer", False)
    
    print(f"Has Answer: {has_original_answer}")
    print(f"Answer: {original_answer.get('answer', 'No answer generated')}")
    
    # Now test with the poor paraphrase
    print(f"\nPoor Paraphrase: \"{poor_paraphrase}\"")
    paraphrase_answer = query_vectorstore(retriever, poor_paraphrase)
    has_paraphrase_answer = paraphrase_answer.get("has_answer", False)
    
    print(f"Has Answer: {has_paraphrase_answer}")
    print(f"Answer: {paraphrase_answer.get('answer', 'No answer generated')}")
    
    # Now test the comparison with fallback
    print("\nTesting comparison with fallback mechanism:")
    comparison = compare_retrieval_results(retriever, original_query)
    
    used_fallback = comparison.get("used_fallback", False)
    print(f"Used Fallback: {used_fallback}")
    
    if used_fallback:
        print(f"Fallback Answer: {comparison['paraphrased_results']['answer'].get('answer', 'No fallback answer')}")
    
    print("\n" + "="*80)

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

# Paraphrase LLM with higher temperature for creative rephrasing
paraphrase_llm = ChatOpenAI(
    model="qwen2.5-72b-instruct",
    temperature=0.7,
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
        
        # Demonstrate paraphrasing functionality
        print_header("Paraphrasing Demonstration")
        log_json_message("info", "Demonstrating paraphrasing functionality with different styles...")
        
        # Select a document to paraphrase
        demo_doc = article_docs[0]
        print(f"\n{Fore.CYAN}Original Document:{Style.RESET_ALL}")
        print(f"{demo_doc.page_content}")
        print()
        
        # Paraphrase in different styles
        styles = ["standard", "formal", "simple", "creative"]
        for i, style in enumerate(styles):
            print(f"{Fore.YELLOW}Paraphrasing in {style.upper()} style:{Style.RESET_ALL}")
            paraphrased = paraphrase_text(demo_doc.page_content, style)
            print(f"{paraphrased}\n")
        
        # Paraphrase a set of documents
        print(f"\n{Fore.CYAN}Paraphrasing a set of documents (creative style):{Style.RESET_ALL}")
        paraphrased_articles = paraphrase_documents(article_docs, "creative")
        
        # Display original and paraphrased versions
        for i, (orig, para) in enumerate(zip(article_docs, paraphrased_articles)):
            print_paraphrased_document(orig, para, i+1)
        
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
            "cross_document_queries": [],
            "comparison_results": []
        }
        
        # Define general queries
        general_queries = [
            "Who is Michael Brown?",
            "what is the most hype industry in 2023?",
            "What were the financial results in Q1 2023?"
        ]
        
        # Demonstrate query paraphrasing
        print_header("Query Paraphrasing Demonstration")
        log_json_message("info", "Demonstrating query paraphrasing with different styles...")
        
        for query in general_queries:
            print(f"\n{Fore.CYAN}Original Query:{Style.RESET_ALL} {query}")
            
            # Paraphrase query in different styles
            paraphrase_styles = ["standard", "expanded", "precise"]
            for style in paraphrase_styles:
                paraphrased_query = paraphrase_query(query, style)
                print(f"{Fore.YELLOW}{style.capitalize()} style:{Style.RESET_ALL} {paraphrased_query}")
            
            print()  # Add spacing between queries
        
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
        
        # Compare retrieval results for original and paraphrased queries
        print_header("Comparison of Retrieval Results")
        for query in general_queries:
            comparison_results = compare_retrieval_results(general_retriever, query)
            all_results["comparison_results"].append(comparison_results)
            print_comparison_results(comparison_results)
        
        # Add a specific test for the "No Answer Available" scenario
        print_header("No Answer Fallback Test")
        print("\nTesting fallback mechanism with a query that might not have an answer when paraphrased...")
        
        # Create a query that likely has an answer
        test_query = "What were the financial results in Q1 2023?"
        
        # Create a deliberately poor paraphrase by modifying the compare_retrieval_results function temporarily
        original_paraphrase_func = paraphrase_query
        
        # Override paraphrase_query function to return a deliberately poor paraphrase
        def poor_paraphrase(query_text, style="standard", min_similarity=0.8):
            if "financial results" in query_text and "Q1 2023" in query_text:
                return "What were the accounting figures for some random period?"
            return original_paraphrase_func(query_text, style, min_similarity)
        
        # Temporarily replace the paraphrase function
        globals()["paraphrase_query"] = poor_paraphrase
        
        # Run comparison with the poor paraphrase
        fallback_test_results = compare_retrieval_results(general_retriever, test_query, "standard")
        print_comparison_results(fallback_test_results)
        
        # Restore original paraphrase function
        globals()["paraphrase_query"] = original_paraphrase_func
        
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
