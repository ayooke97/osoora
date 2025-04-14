import os
import sys
import time
import datetime
import webbrowser
import json
import re
import traceback
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from colorama import Fore, Style, init
import html
import requests
from bs4 import BeautifulSoup, SoupStrainer
from urllib.parse import urljoin
from openai import OpenAI

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# Initialize colorama
init()

# Load environment variables
load_dotenv()

# Helper functions for logging
def log_message(level: str, message: str):
    """Log a message with a timestamp and color."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_colors = {
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED
    }
    color = level_colors.get(level.lower(), Fore.WHITE)
    print(f"{color}[{timestamp}] [{level.upper()}] {message}{Style.RESET_ALL}")

def log_json_message(level: str, message: str):
    """Log a message in JSON format."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level.upper(),
        "message": message
    }
    log_message(level, message)

class DashScopeEmbeddings(Embeddings):
    """DashScope embeddings using OpenAI-compatible API."""
    
    def __init__(
        self,
        api_key: Optional[str] = os.environ.get("OPENAI_API_KEY"),
        base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        model: str = "text-embedding-v3",
        dimensions: int = 1024,
    ):
        try:
            # Try with standard parameters
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except TypeError as e:
            if "proxies" in str(e):
                # If error is about proxies, try with a different approach
                log_json_message("info", "Detected proxies issue in embeddings, using alternative initialization")
                import httpx
                # Create a custom httpx client without proxies
                http_client = httpx.Client()
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=http_client
                )
            else:
                # Re-raise if it's a different error
                raise
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

class SimpleOpenAIWrapper:
    """Simple wrapper for OpenAI API to handle proxies issues."""
    
    def __init__(self, client, model, temperature=0):
        self.client = client
        self.model = model
        self.temperature = temperature
    
    def invoke(self, input_data):
        try:
            # Handle different input types (string or dict with template)
            if isinstance(input_data, str):
                messages = [{"role": "user", "content": input_data}]
            else:
                messages = [{"role": "user", "content": input_data}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            log_json_message("error", f"Error invoking OpenAI: {e}")
            return f"Error: {str(e)}"

class LegalWebScraper:
    """Class for scraping legal websites and retrieving information."""
    
    def __init__(self):
        """Initialize the scraper with an embeddings model."""
        try:
            # Initialize OpenAI API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            # Initialize OpenAI client
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            try:
                # Try with standard parameters
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            except TypeError as e:
                if "proxies" in str(e):
                    # If error is about proxies, try with a different approach
                    log_json_message("info", "Detected proxies issue in client, using alternative initialization")
                    import httpx
                    # Create a custom httpx client without proxies
                    http_client = httpx.Client()
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        http_client=http_client
                    )
                else:
                    # Re-raise if it's a different error
                    raise
                
            # Initialize embeddings with DashScopeEmbeddings to avoid proxies issue
            self.embeddings = DashScopeEmbeddings(
                api_key=api_key,
                model="text-embedding-v3"
            )
            
            # Initialize LLM with custom wrapper
            self.llm = SimpleOpenAIWrapper(
                client=self.client,
                model="qwen2.5-72b-instruct",  # Use a specific model version
                temperature=0
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            self.vectorstore = None
            log_json_message("info", "LegalWebScraper initialized successfully")
        except Exception as e:
            log_json_message("error", f"Error initializing LegalWebScraper: {str(e)}")
            raise
        
    def scrape_hukumonline(self, query: str, max_pages: int = 5) -> List[Document]:
        """
        Scrape hukumonline.com for legal information based on a query.
        
        Args:
            query: The search query
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of Document objects
        """
        log_json_message("info", f"Scraping hukumonline.com for query: {query}")
        
        # Format the search URL
        search_url = f"https://www.hukumonline.com/search/a/?q={query.replace(' ', '+')}"
        
        try:
            # Set up headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.hukumonline.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
            
            # Get the search results page
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find search result links
            search_results = soup.select('.search-result-item a')
            if not search_results:
                # Try alternative selector
                search_results = soup.select('article h3 a')
                if not search_results:
                    # Try more generic selectors
                    search_results = soup.select('a[href*="/hukum/"]') or soup.select('a[href*="/berita/"]')
            
            # Extract URLs (up to max_pages)
            urls = []
            for i, result in enumerate(search_results):
                if i >= max_pages:
                    break
                    
                href = result.get('href')
                if href and href.startswith('http'):
                    urls.append(href)
                elif href:
                    # Handle relative URLs
                    urls.append(urljoin('https://www.hukumonline.com', href))
            
            log_json_message("info", f"Found {len(urls)} URLs to scrape from hukumonline.com")
            
            # Use WebBaseLoader to load the content from each URL
            documents = []
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    loader.requests_kwargs = {
                        'timeout': 15,
                        'headers': headers
                    }
                    
                    # For hukumonline.com, don't use bs_kwargs to avoid parsing issues
                    # Just load the whole page and process it
                    
                    docs = loader.load()
                    
                    # Add source information to metadata
                    for doc in docs:
                        doc.metadata["source"] = url
                        doc.metadata["website"] = "hukumonline.com"
                    
                    documents.extend(docs)
                    log_json_message("success", f"Scraped {url}")
                except Exception as e:
                    log_json_message("error", f"Error scraping {url}: {str(e)}")
            
            return documents
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                log_json_message("error", f"Access forbidden (403) to hukumonline.com. The website may be blocking scrapers.")
                # Try an alternative approach or provide guidance
                return [Document(
                    page_content="The website hukumonline.com is currently blocking automated access. You may need to manually visit the website.",
                    metadata={"source": search_url, "website": "hukumonline.com", "error": "403 Forbidden"}
                )]
            else:
                log_json_message("error", f"HTTP error scraping hukumonline.com: {str(e)}")
                return []
        except Exception as e:
            log_json_message("error", f"Error scraping hukumonline.com: {str(e)}")
            return []
    
    def scrape_peraturan_bpk(self, query: str, max_pages: int = 10) -> List[Document]:
        """
        Scrape peraturan.bpk.go.id for legal information based on a query.
        
        Args:
            query: The search query
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of Document objects
        """
        log_json_message("info", f"Scraping peraturan.bpk.go.id for query: {query}")
        
        # Format the search URL
        search_url = f"https://peraturan.bpk.go.id/Search?text={query.replace(' ', '+')}"
        
        try:
            # Set up headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://peraturan.bpk.go.id/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            }
            
            # Get the search results page
            log_json_message("info", f"Requesting search URL: {search_url}")
            response = requests.get(search_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find search result links - try multiple selector patterns
            search_results = []
            
            # Try different selector patterns
            selectors = [
                'a[href*="/Details/"]',
                '.search-result a',
                '.result-item a',
                'table.table a[href*="/Details/"]',
                '.table-responsive a[href*="/Details/"]',
                'td a[href*="/Details/"]'
            ]
            
            for selector in selectors:
                results = soup.select(selector)
                if results:
                    search_results.extend(results)
                    log_json_message("info", f"Found {len(results)} results with selector: {selector}")
            
            # Remove duplicates by href
            unique_results = {}
            for result in search_results:
                href = result.get('href')
                if href and "/Details/" in href:
                    unique_results[href] = result
            
            search_results = list(unique_results.values())
            
            # Extract URLs (up to max_pages)
            urls = []
            for i, result in enumerate(search_results):
                if i >= max_pages:
                    break
                    
                href = result.get('href')
                if href and href.startswith('http'):
                    urls.append(href)
                elif href:
                    # Handle relative URLs
                    urls.append(urljoin('https://peraturan.bpk.go.id', href))
            
            log_json_message("info", f"Found {len(urls)} unique URLs to scrape from peraturan.bpk.go.id")
            
            # Use custom scraping for each detail page
            documents = []
            for url in urls:
                try:
                    log_json_message("info", f"Scraping document: {url}")
                    
                    # Get the detail page
                    detail_response = requests.get(url, headers=headers, timeout=30)
                    detail_response.raise_for_status()
                    
                    detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                    
                    # Extract document metadata
                    title_elem = detail_soup.select_one('h2') or detail_soup.select_one('h1') or detail_soup.select_one('title')
                    title = title_elem.text.strip() if title_elem else "Unknown Title"
                    
                    # Try to extract document number and type
                    doc_number = ""
                    doc_type = ""
                    
                    # Look for document metadata in various formats
                    metadata_elems = detail_soup.select('table.table tr') or detail_soup.select('.row .col-md-6')
                    
                    for elem in metadata_elems:
                        text = elem.text.strip().lower()
                        if "nomor" in text:
                            doc_number = text.split("nomor")[-1].strip()
                        elif "jenis" in text:
                            doc_type = text.split("jenis")[-1].strip()
                    
                    # Extract content - try different selectors for the main content
                    content_selectors = [
                        '.container .row .col-md-12',
                        '#main-content',
                        '.detail-content',
                        '.container .row',
                        'body'  # Fallback to entire body if specific content can't be found
                    ]
                    
                    content = ""
                    for selector in content_selectors:
                        content_elem = detail_soup.select_one(selector)
                        if content_elem:
                            content = content_elem.text.strip()
                            if len(content) > 500:  # If we found substantial content, use it
                                break
                    
                    # Create document with metadata
                    metadata = {
                        "source": url,
                        "website": "peraturan.bpk.go.id",
                        "title": title,
                        "doc_number": doc_number,
                        "doc_type": doc_type,
                        "query": query
                    }
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                    
                    log_json_message("success", f"Successfully scraped document: {title}")
                except Exception as e:
                    log_json_message("error", f"Error scraping {url}: {str(e)}")
            
            return documents
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                log_json_message("error", f"Access forbidden (403) to peraturan.bpk.go.id. The website may be blocking scrapers.")
                # Try an alternative approach or provide guidance
                return [Document(
                    page_content="The website peraturan.bpk.go.id is currently blocking automated access. You may need to manually visit the website.",
                    metadata={"source": search_url, "website": "peraturan.bpk.go.id", "error": "403 Forbidden"}
                )]
            else:
                log_json_message("error", f"HTTP error scraping peraturan.bpk.go.id: {str(e)}")
                return []
        except Exception as e:
            log_json_message("error", f"Error scraping peraturan.bpk.go.id: {str(e)}")
            return []
    
    def scrape_peraturan_bpk_only(self, query: str) -> bool:
        """
        Scrape only peraturan.bpk.go.id and create a vector store from the results.
        
        Args:
            query: The search query
            
        Returns:
            True if successful, False otherwise
        """
        try:
            log_json_message("info", f"Scraping peraturan.bpk.go.id for query: {query}")
            
            # Scrape peraturan.bpk.go.id
            peraturan_bpk_docs = self.scrape_peraturan_bpk(query, max_pages=15)
            
            if not peraturan_bpk_docs:
                log_json_message("warning", f"No documents found for query: {query}")
                return False
                
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(peraturan_bpk_docs)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            log_json_message("success", f"Created vector store with {len(split_docs)} document chunks from peraturan.bpk.go.id")
            return True
            
        except Exception as e:
            log_json_message("error", f"Error creating vector store: {str(e)}")
            return False
    
    def scrape_hukumonline_only(self, query: str) -> bool:
        """
        Scrape only hukumonline.com and create a vector store from the results.
        
        Args:
            query: The search query
            
        Returns:
            True if successful, False otherwise
        """
        try:
            log_json_message("info", f"Scraping hukumonline.com for query: {query}")
            
            # Scrape hukumonline.com
            hukumonline_docs = self.scrape_hukumonline(query, max_pages=15)
            
            if not hukumonline_docs:
                log_json_message("warning", f"No documents found for query: {query}")
                return False
                
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(hukumonline_docs)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            log_json_message("success", f"Created vector store with {len(split_docs)} document chunks from hukumonline.com")
            return True
            
        except Exception as e:
            log_json_message("error", f"Error creating vector store: {str(e)}")
            return False
    
    def scrape_and_create_vectorstore(self, query: str) -> bool:
        """
        Scrape both websites and create a vector store from the results.
        
        Args:
            query: The search query
            
        Returns:
            True if successful, False otherwise
        """
        try:
            log_json_message("info", f"Scraping websites for query: {query}")
            
            # Scrape both websites
            hukumonline_docs = self.scrape_hukumonline(query)
            peraturan_bpk_docs = self.scrape_peraturan_bpk(query)
            
            # Combine documents
            all_docs = hukumonline_docs + peraturan_bpk_docs
            
            if not all_docs:
                log_json_message("warning", f"No documents found for query: {query}")
                return False
                
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(all_docs)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            log_json_message("success", f"Created vector store with {len(split_docs)} document chunks")
            return True
            
        except Exception as e:
            log_json_message("error", f"Error creating vector store: {str(e)}")
            return False
    
    def query_vectorstore(
        self,
        query: str,
        num_docs: int = 5
    ) -> Dict[str, Any]:
        """
        Query the vector store with a question and return the answer with source documents.
        
        Args:
            query: The question to ask
            num_docs: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        log_json_message("info", f"Querying vector store with: {query}")
        
        if not self.vectorstore:
            log_json_message("error", "Vector store not initialized. Run scrape_and_create_vectorstore first.")
            return {"answer": "Error: Vector store not initialized", "documents": []}
        
        try:
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(query, k=num_docs)
            
            # Format documents for context
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Create prompt template
            prompt_template = f"""
            You are a legal assistant specialized in Indonesian law. Answer the question based only on the provided context.
            If you don't know the answer, say that you don't know and avoid making up information.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """
            
            # Generate answer using the LLM
            answer = self.llm.invoke(prompt_template)
            
            # Return answer and source documents
            return {
                "answer": answer,
                "documents": docs
            }
            
        except Exception as e:
            log_json_message("error", f"Error querying vector store: {str(e)}")
            return {"answer": f"Error: {str(e)}", "documents": []}

    def generate_html_report(self, query: str, result: Dict[str, Any]) -> str:
        """
        Generate an HTML report of the query results.
        
        Args:
            query: The original query
            result: The result dictionary from query_vectorstore
            
        Returns:
            The path to the generated HTML file
        """
        try:
            # Extract data from result
            answer = result.get("answer", "No answer available")
            if isinstance(answer, dict) and "error" in answer:
                answer = answer["error"]
            
            documents = result.get("documents", [])
            
            # Create a timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a filename
            safe_query = re.sub(r'[^\w\s]', '', query)[:30].strip().replace(' ', '_')
            filename = f"legal_report_{safe_query}_{timestamp}.html"
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Legal Information Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    header {{
                        background-color: #2c3e50;
                        color: white;
                        padding: 20px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    header h1 {{
                        color: white;
                    }}
                    .query {{
                        font-size: 1.2em;
                        font-weight: bold;
                        margin-bottom: 20px;
                        padding: 10px;
                        background-color: #f8f9fa;
                        border-left: 5px solid #2c3e50;
                    }}
                    .answer {{
                        margin-bottom: 30px;
                        padding: 20px;
                        background-color: #f1f8e9;
                        border-radius: 5px;
                        border-left: 5px solid #7cb342;
                    }}
                    .documents {{
                        margin-top: 30px;
                    }}
                    .document {{
                        margin-bottom: 20px;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                        border: 1px solid #ddd;
                    }}
                    .document-title {{
                        font-weight: bold;
                        color: #2c3e50;
                        margin-bottom: 10px;
                    }}
                    .document-source {{
                        color: #666;
                        font-size: 0.9em;
                        margin-bottom: 10px;
                    }}
                    .document-content {{
                        margin-top: 10px;
                        padding: 10px;
                        background-color: white;
                        border: 1px solid #eee;
                        border-radius: 3px;
                        max-height: 300px;
                        overflow-y: auto;
                    }}
                    .hukumonline {{
                        border-left: 5px solid #2196F3;
                    }}
                    .peraturan-bpk {{
                        border-left: 5px solid #FF9800;
                    }}
                    footer {{
                        margin-top: 50px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 0.9em;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <header>
                    <h1>Legal Information Report</h1>
                    <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </header>
                
                <div class="query">
                    <h2>Query</h2>
                    <p>{html.escape(query)}</p>
                </div>
                
                <div class="answer">
                    <h2>Answer</h2>
                    <div>{html.escape(str(answer))}</div>
                </div>
                
                <div class="documents">
                    <h3>Retrieved {len(documents)} relevant documents</h3>
            """
            
            for i, doc in enumerate(documents):
                metadata = doc.metadata
                source = metadata.get("source", "Unknown")
                website = metadata.get("website", "Unknown")
                title = metadata.get("title", "Unknown Title")
                
                website_class = ""
                if "hukumonline" in website:
                    website_class = "hukumonline"
                elif "peraturan.bpk" in website:
                    website_class = "peraturan-bpk"
                
                html_content += f"""
                    <div class="document {website_class}">
                        <div class="document-title">Document {i+1}: {html.escape(title)}</div>
                        <div class="document-source">Source: <a href="{html.escape(source)}" target="_blank">{html.escape(source)}</a> ({html.escape(website)})</div>
                        <div class="document-content">{html.escape(doc.page_content)}</div>
                    </div>
                """
            
            html_content += """
                </div>
                
                <footer>
                    <p>This report was generated using LangChain and OpenAI. The information provided is based on the retrieved documents and should be verified with official sources.</p>
                </footer>
            </body>
            </html>
            """
            
            # Write the HTML file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            log_json_message("success", f"HTML report generated: {filename}")
            return filename
            
        except Exception as e:
            log_json_message("error", f"Error generating HTML report: {str(e)}")
            return f"Error generating report: {str(e)}"

def main():
    """Main function to run the legal web scraper."""
    print(f"{Fore.CYAN}===== Legal Web Scraper =====\n{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}This tool retrieves legal information from hukumonline.com and peraturan.bpk.go.id{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}It uses LangChain for document processing and retrieval{Style.RESET_ALL}\n")
    
    try:
        # Initialize the scraper
        scraper = LegalWebScraper()
        
        while True:
            # Get user query
            query = input(f"{Fore.GREEN}Enter your legal query (or 'exit' to quit): {Style.RESET_ALL}")
            if query.lower() == 'exit':
                break
            
            # Ask which site to scrape
            site_choice = input(f"{Fore.GREEN}Scrape from: (1) Both sites, (2) peraturan.bpk.go.id only, (3) hukumonline.com only: {Style.RESET_ALL}")
            
            # Scrape and index
            print(f"{Fore.YELLOW}Scraping and indexing relevant legal documents...{Style.RESET_ALL}")
            
            success = False
            if site_choice == '2':
                success = scraper.scrape_peraturan_bpk_only(query)
            elif site_choice == '3':
                success = scraper.scrape_hukumonline_only(query)
            else:
                success = scraper.scrape_and_create_vectorstore(query)
                
            if success:
                # Query the vector store
                print(f"{Fore.YELLOW}Generating answer based on retrieved documents...{Style.RESET_ALL}")
                result = scraper.query_vectorstore(query)
                
                # Print the answer
                answer = result.get("answer", "No answer available")
                documents = result.get("documents", [])
                
                print(f"\n{Fore.CYAN}Answer:{Style.RESET_ALL}")
                print(f"{answer}\n")
                
                print(f"{Fore.CYAN}Retrieved {len(documents)} relevant documents{Style.RESET_ALL}")
                for i, doc in enumerate(documents[:2]):  # Show only first 2 documents in console
                    source = doc.metadata.get("source", "Unknown")
                    website = doc.metadata.get("website", "Unknown")
                    title = doc.metadata.get("title", "Unknown Title")
                    print(f"{Fore.YELLOW}Document {i+1}:{Style.RESET_ALL} {title} - {source} ({website})")
                
                # Generate HTML report
                generate_report = input(f"{Fore.GREEN}Generate HTML report? (y/n): {Style.RESET_ALL}")
                if generate_report.lower() == 'y':
                    filename = scraper.generate_html_report(query, result)
                    
                    # Open the report in browser
                    open_report = input(f"{Fore.GREEN}Open report in browser? (y/n): {Style.RESET_ALL}")
                    if open_report.lower() == 'y':
                        webbrowser.open(f"file://{os.path.abspath(filename)}")
            else:
                print(f"{Fore.RED}Failed to create vector store. Please try again.{Style.RESET_ALL}")
            
            print()
    except Exception as e:
        print(f"{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
