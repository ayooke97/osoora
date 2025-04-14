"""
BPK Legal Document Scraper

This script focuses specifically on scraping legal documents from peraturan.bpk.go.id
based on user queries, with enhanced Indonesian language processing capabilities.
"""

import os
import re
import json
import time
import random
import requests
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama
init()

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("legal_workflow.log"),
            logging.StreamHandler()
        ]
    )
# Create logger instance
logger = logging.getLogger("legal_workflow")

# Try importing LangChain Document class
try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    # Define a simple LangChainDocument class if langchain is not available
    class LangChainDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Document class for compatibility with both BPKScraper and LangChain
class Document:
    def __init__(self, content="", metadata=None, page_content=None):
        """
        Initialize a Document object.
        
        Args:
            content (str): The document content (for backward compatibility)
            metadata (dict): The document metadata
            page_content (str): The document content (LangChain format)
        """
        # Support both content and page_content for compatibility
        self.page_content = page_content if page_content is not None else content
        self.content = self.page_content  # For backward compatibility
        self.metadata = metadata or {}
    
    def to_langchain_document(self):
        """Convert to LangChain Document format."""
        return LangChainDocument(page_content=self.content, metadata=self.metadata)
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a Document from a dictionary."""
        if not isinstance(data, dict):
            return data
        if "page_content" in data:
            return cls(page_content=data["page_content"], metadata=data.get("metadata", {}))
        elif "content" in data:
            return cls(content=data["content"], metadata=data.get("metadata", {}))
        return data

import sys
import tempfile
import warnings
import traceback
from dotenv import load_dotenv
import webbrowser
import httpx
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Global variables to track available features
HAS_SASTRAWI = True
HAS_OPENAI = True
HAS_INDOBERT = True
HAS_PYPDF = True

# Try importing optional dependencies
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    HAS_SASTRAWI = True
except ImportError:
    HAS_SASTRAWI = False
    print(f"{Fore.YELLOW}Sastrawi not found. Indonesian stemming will be disabled.{Style.RESET_ALL}")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print(f"{Fore.YELLOW}OpenAI not found. Legal language conversion will be disabled.{Style.RESET_ALL}")

# Only try to import torch and transformers if they're available
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_INDOBERT = True
except ImportError:
    print(f"{Fore.YELLOW}Torch or transformers not found. IndoBERT embeddings will be disabled.{Style.RESET_ALL}")

# Add PyPDF2 for PDF processing
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    print(f"{Fore.YELLOW}PyPDF2 not found. PDF extraction will be disabled.{Style.RESET_ALL}")
    HAS_PYPDF2 = False

class IndoBERTEmbeddings:
    """
    Class for generating embeddings using IndoBERT model.
    """
    
    def __init__(self):
        """
        Initialize the IndoBERT embeddings model.
        """
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        # Set environment variables to suppress warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        try:
            # Load IndoBERT tokenizer and model
            model_name = "indolem/indobert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            print(f"\n{Fore.GREEN}IndoBERT model loaded successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error loading IndoBERT model: {str(e)}{Style.RESET_ALL}")
            raise
    
    def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embeddings
        """
        import torch
        
        embeddings = []
        
        # Process each text
        for text in texts:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean of last hidden states as embedding
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            embeddings.append(embedding.tolist())
        
        return embeddings

class IndonesianLegalProcessor:
    """Process Indonesian text with legal context."""
    
    def __init__(self):
        """Initialize the Indonesian legal processor."""
        try:
            # Initialize Sastrawi stemmer if available
            self.has_stemmer = False
            if HAS_SASTRAWI:
                try:
                    factory = StemmerFactory()
                    self.stemmer = factory.create_stemmer()
                    self.has_stemmer = True
                    print(f"\n{Fore.GREEN}Sastrawi stemmer initialized successfully{Style.RESET_ALL}")
                except Exception as e:
                    print(f"\n{Fore.YELLOW}Could not initialize Sastrawi stemmer: {str(e)}{Style.RESET_ALL}")
            
            # Initialize OpenAI client if API key is available
            self.has_openai = False
            if HAS_OPENAI:
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                if api_key:
                    try:
                        # Try with standard parameters
                        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
                        self.has_openai = True
                        print(f"\n{Fore.GREEN}OpenAI client initialized successfully with base URL: {base_url}{Style.RESET_ALL}")
                    except TypeError as e:
                        if "proxies" in str(e):
                            # If error is about proxies, try with a different approach
                            print(f"\n{Fore.BLUE}Detected proxies issue, using alternative initialization{Style.RESET_ALL}")
                            import httpx
                            # Create a custom httpx client without proxies
                            http_client = httpx.Client()
                            try:
                                self.openai_client = OpenAI(
                                    api_key=api_key,
                                    base_url=base_url,
                                    http_client=http_client
                                )
                                self.has_openai = True
                                print(f"\n{Fore.GREEN}OpenAI client initialized successfully with custom HTTP client{Style.RESET_ALL}")
                            except Exception as inner_e:
                                print(f"\n{Fore.YELLOW}Could not initialize OpenAI client with custom HTTP client: {str(inner_e)}{Style.RESET_ALL}")
                        else:
                            # Re-raise if it's a different error
                            print(f"\n{Fore.YELLOW}Could not initialize OpenAI client: {str(e)}{Style.RESET_ALL}")
            
            # Legal terms dictionary (can be expanded)
            self.legal_terms = {
                "hak": ["hak", "hak asasi"],
                "ulayat": ["ulayat", "hak ulayat", "tanah ulayat", "tanah adat"],
                "tanah": ["tanah", "pertanahan", "agraria"],
                "adat": ["adat", "hukum adat", "masyarakat adat"],
                "hukum": ["hukum", "peraturan", "undang-undang"],
                "undang": ["undang-undang", "peraturan"],
                "peraturan": ["peraturan", "regulasi"],
                "pemerintah": ["pemerintah", "pemerintahan"],
                "keputusan": ["keputusan", "ketetapan"],
                "menteri": ["menteri", "kementerian"],
                "presiden": ["presiden", "kepresidenan"],
                "agraria": ["agraria", "pertanahan"],
                "pertanahan": ["pertanahan", "tanah"],
                "masyarakat": ["masyarakat", "komunitas"],
                "hutan": ["hutan", "kehutanan"],
                "wilayah": ["wilayah", "area", "kawasan"],
                "daerah": ["daerah", "area", "wilayah"],
                "provinsi": ["provinsi", "daerah"],
                "kabupaten": ["kabupaten", "daerah"],
                "kota": ["kota", "perkotaan"]
            }
            
            print(f"\n{Fore.GREEN}Indonesian Legal Processor initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error initializing Indonesian Legal Processor: {str(e)}{Style.RESET_ALL}")
            raise
    
    def stem_text(self, text):
        """Stem Indonesian text using Sastrawi."""
        if not self.has_stemmer:
            return text
            
        try:
            return self.stemmer.stem(text)
        except Exception as e:
            print(f"\n{Fore.RED}Error stemming text: {str(e)}{Style.RESET_ALL}")
            return text
    
    def enhance_query_with_legal_terms(self, query):
        """Enhance query with related legal terms."""
        try:
            # Process the query (stem if available)
            processed_query = self.stem_text(query.lower()) if self.has_stemmer else query.lower()
            
            # Find matching legal terms
            additional_terms = set()
            for term in processed_query.split():
                if term in self.legal_terms:
                    # Add related terms
                    for related_term in self.legal_terms[term]:
                        additional_terms.add(related_term)
            
            # Add relevant legal terms to the query
            enhanced_query = query
            for term in additional_terms:
                if term.lower() not in query.lower():
                    enhanced_query += f" {term}"
            
            return enhanced_query
        except Exception as e:
            print(f"\n{Fore.RED}Error enhancing query: {str(e)}{Style.RESET_ALL}")
            return query
    
    def convert_to_legal_language(self, query):
        """Convert user query to formal legal language using OpenAI if available."""
        if not self.has_openai:
            return self.enhance_query_with_legal_terms(query)
            
        try:
            print(f"\n{Fore.BLUE}Converting query to legal language with AI...{Style.RESET_ALL}")
            
            # Use DashScope's Qwen model for Indonesian legal language conversion
            model = "qwen2.5-72b-instruct"  # DashScope's Qwen model
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Anda adalah asisten hukum Indonesia yang ahli. Ubah pertanyaan pengguna menjadi bahasa hukum formal dalam Bahasa Indonesia. Gunakan terminologi hukum yang tepat dan relevan dengan konteks pertanyaan. Jangan menambahkan informasi baru atau mengubah maksud pertanyaan."},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            legal_query = response.choices[0].message.content.strip()
            print(f"\n{Fore.GREEN}Converted query to legal language: {legal_query}{Style.RESET_ALL}")
            return legal_query
        except Exception as e:
            print(f"\n{Fore.YELLOW}Error converting to legal language with AI: {str(e)}{Style.RESET_ALL}")
            # Fall back to rule-based enhancement
            return self.enhance_query_with_legal_terms(query)

class BPKScraper:
    """Class for scraping peraturan.bpk.go.id website."""
    
    def __init__(self, openai_api_key=None, openai_base_url=None):
        """
        Initialize the BPK legal document scraper.
        
        Args:
            openai_api_key (str, optional): OpenAI API key for LLM integration
            openai_base_url (str, optional): OpenAI API base URL
        """
        global HAS_SASTRAWI, HAS_OPENAI, HAS_INDOBERT, HAS_PYPDF
        
        print("\nInitializing BPK Scraper...")
        
        # Initialize attributes
        self.openai_wrapper = None
        self.stemmer = None
        self.embeddings = None
        self.has_embeddings = False
        
        # Try to initialize OpenAI wrapper
        if HAS_OPENAI:
            try:
                # Load API key from environment if not provided
                if not openai_api_key:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    
                if not openai_base_url:
                    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                
                if openai_api_key:
                    try:
                        # Initialize OpenAI client
                        from openai import OpenAI
                        import httpx
                        
                        # Try with custom httpx client
                        try:
                            # Create a custom httpx client with timeout settings
                            http_client = httpx.Client(timeout=60.0)
                            
                            client = OpenAI(
                                api_key=openai_api_key,
                                base_url=openai_base_url,
                                http_client=http_client
                            )
                            print("[INFO] OpenAI client initialized successfully with custom HTTP client")
                        except Exception as e:
                            print(f"[INFO] Detected proxies issue, trying alternative initialization")
                            # Try without custom client
                            client = OpenAI(
                                api_key=openai_api_key,
                                base_url=openai_base_url
                            )
                        
                        # Initialize the wrapper with the client
                        try:
                            from final3 import SimpleOpenAIWrapper
                            self.openai_wrapper = SimpleOpenAIWrapper(
                                client=client,
                                model="qwen2.5-72b-instruct",  # Use the same model as in final3.py
                                temperature=0
                            )
                            print("OpenAI API initialized successfully")
                        except ImportError as ie:
                            print(f"Could not import SimpleOpenAIWrapper: {str(ie)}")
                            # If SimpleOpenAIWrapper can't be imported, create a basic wrapper
                            class BasicOpenAIWrapper:
                                def __init__(self, client, model="qwen2.5-72b-instruct", temperature=0):
                                    self.client = client
                                    self.model = model
                                    self.temperature = temperature
                                
                                def invoke(self, input_data):
                                    try:
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
                                        print(f"Error invoking OpenAI: {e}")
                                        return f"Error: {str(e)}"
                            
                            self.openai_wrapper = BasicOpenAIWrapper(
                                client=client,
                                model="qwen2.5-72b-instruct",
                                temperature=0
                            )
                            print("Basic OpenAI wrapper initialized successfully")
                    except Exception as e:
                        print(f"Error initializing OpenAI API: {str(e)}")
                        HAS_OPENAI = False
                else:
                    print("OpenAI API key not found")
                    HAS_OPENAI = False
            except Exception as e:
                print(f"Error setting up OpenAI: {str(e)}")
                HAS_OPENAI = False
        
        # Try to initialize Sastrawi stemmer
        if HAS_SASTRAWI:
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
                print("Sastrawi stemmer initialized successfully")
            except Exception as e:
                print(f"Error initializing Sastrawi stemmer: {str(e)}")
                HAS_SASTRAWI = False
                self.stemmer = None
        
        # Try to initialize IndoBERT embeddings
        if HAS_INDOBERT:
            try:
                # Initialize IndoBERT embeddings
                try:
                    self.embeddings = IndoBERTEmbeddings()
                    self.has_embeddings = True
                except Exception as e:
                    print(f"[INFO] Detected proxies issue in embeddings, using alternative initialization")
                    # Try alternative initialization
                    self.embeddings = IndoBERTEmbeddings()
                    self.has_embeddings = True
                
                print("IndoBERT model initialized successfully")
            except Exception as e:
                print(f"Error initializing IndoBERT model: {str(e)}")
                HAS_INDOBERT = False
                self.embeddings = None
                self.has_embeddings = False
        
        # Check PyPDF2 availability
        if HAS_PYPDF:
            try:
                import PyPDF2
                print("PyPDF2 initialized successfully")
            except Exception as e:
                print(f"Error initializing PyPDF2: {str(e)}")
                HAS_PYPDF = False
        
        print("\n" + "=" * 80)
        print("BPK Scraper initialized successfully")
        print("=" * 80)

    def preprocess_query(self, query):
        """
        Preprocess and enhance the user query.
        
        Args:
            query (str): The original user query
            
        Returns:
            str: The enhanced query
        """
        try:
            print(f"\n{Fore.BLUE}Preprocessing query: {query}{Style.RESET_ALL}")
            
            # Use stemming if Sastrawi is available
            if HAS_SASTRAWI and self.stemmer:
                # Stem the query
                stemmed_words = []
                for word in query.split():
                    if len(word) > 3:  # Only stem words longer than 3 characters
                        stemmed_word = self.stemmer.stem(word)
                        stemmed_words.append(stemmed_word)
                    else:
                        stemmed_words.append(word)
                
                stemmed_query = " ".join(stemmed_words)
                print(f"\n{Fore.GREEN}Stemmed query: {stemmed_query}{Style.RESET_ALL}")
                
                # Combine original and stemmed query for better results
                enhanced_query = f"{query} {stemmed_query}"
            else:
                enhanced_query = query
            
            # Use OpenAI to enhance the query with legal terminology if available
            if HAS_OPENAI and self.openai_wrapper:
                try:
                    prompt = f"""
                    As a legal expert in Indonesian law, enhance this query to include proper legal terminology and relevant legal concepts:
                    
                    Query: {query}
                    
                    Enhanced query:
                    """
                    
                    response = self.openai_wrapper.invoke(prompt)
                    
                    # Extract the enhanced query from the response
                    enhanced_query = response.strip()
                    
                    print(f"\n{Fore.GREEN}Enhanced query with legal terminology: {enhanced_query}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"\n{Fore.YELLOW}Error enhancing query with OpenAI: {str(e)}{Style.RESET_ALL}")
            
            return enhanced_query
            
        except Exception as e:
            print(f"\n{Fore.RED}Error preprocessing query: {str(e)}{Style.RESET_ALL}")
            return query
    
    def download_and_extract_pdf(self, pdf_url, headers=None):
        """
        Download a PDF file and extract its text content.
        
        Args:
            pdf_url (str): URL of the PDF file
            headers (dict): HTTP headers for the request
            
        Returns:
            tuple: (content, metadata) - The extracted text and metadata from the PDF
        """
        if not HAS_PYPDF2:
            print(f"\n{Fore.YELLOW}PyPDF2 is not available. Cannot extract PDF content.{Style.RESET_ALL}")
            return None, None
            
        if not headers:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
        try:
            print(f"\n{Fore.BLUE}Downloading PDF from {pdf_url}{Style.RESET_ALL}")
            
            # Download the PDF
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file to save the PDF
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf.write(response.content)
                temp_pdf_path = temp_pdf.name
                
            try:
                # Extract text from the PDF
                print(f"\n{Fore.BLUE}Extracting text from PDF{Style.RESET_ALL}")
                pdf_reader = PdfReader(temp_pdf_path)
                
                # Extract metadata
                metadata = {
                    "source": pdf_url,
                    "title": os.path.basename(pdf_url),
                    "pages": len(pdf_reader.pages),
                    "type": "pdf"
                }
                
                # Extract PDF info dictionary if available
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        if key.startswith('/'):
                            clean_key = key[1:].lower()
                            if isinstance(value, str):
                                metadata[clean_key] = value
                
                # Extract text content
                content = ""
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- Page {i+1} ---\n"
                        content += page_text
                
                print(f"\n{Fore.GREEN}Successfully extracted {len(pdf_reader.pages)} pages from PDF{Style.RESET_ALL}")
                return content, metadata
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    print(f"\n{Fore.YELLOW}Could not delete temporary PDF file: {str(e)}{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"\n{Fore.RED}Error processing PDF: {str(e)}{Style.RESET_ALL}")
            return None, None
    
    def find_pdf_links(self, url, headers=None):
        """
        Find PDF links on a given BPK website page.
        
        Args:
            url (str): URL of the page to search for PDF links
            headers (dict): HTTP headers for the request
            
        Returns:
            list: List of PDF URLs found on the page
        """
        if not headers:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
        pdf_links = []
        
        try:
            print(f"\n{Fore.BLUE}Searching for PDF links on {url}{Style.RESET_ALL}")
            
            # Get the page content
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Check if the link is a PDF
                if href.lower().endswith('.pdf'):
                    # Make sure the URL is absolute
                    if not href.startswith('http'):
                        href = urljoin(url, href)
                    
                    pdf_links.append({
                        'url': href,
                        'text': link.get_text().strip() or os.path.basename(href),
                        'source_page': url
                    })
            
            print(f"\n{Fore.GREEN}Found {len(pdf_links)} PDF links on {url}{Style.RESET_ALL}")
            return pdf_links
            
        except Exception as e:
            print(f"\n{Fore.RED}Error finding PDF links: {str(e)}{Style.RESET_ALL}")
            return []
    
    def scrape_peraturan_bpk(self, query: str, max_pages: int = 10) -> List[Document]:
        """
        Scrape peraturan.bpk.go.id for legal information based on a query.
        
        Args:
            query (str): The search query
            max_pages (int): Maximum number of search result pages to process
            
        Returns:
            List[Document]: List of document objects with content and metadata
        """
        print(f"\n{Fore.BLUE}Searching for legal information related to: {query}{Style.RESET_ALL}")
        
        documents = []
        
        try:
            # Process the query with legal language conversion if available
            processed_query = self.preprocess_query(query)
            
            # If the query was enhanced, log it
            if processed_query != query:
                print(f"\n{Fore.BLUE}Enhanced query: {processed_query}{Style.RESET_ALL}")
                
            # Prepare the search URL
            base_url = "https://peraturan.bpk.go.id"
            search_url = f"{base_url}/Search"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://peraturan.bpk.go.id/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Process search result pages
            for page in range(1, max_pages + 1):
                try:
                    print(f"\n{Fore.BLUE}Searching page {page}{Style.RESET_ALL}")
                    
                    # Construct the page URL with parameters
                    params = {
                        'keywords': processed_query,
                        'page': page if page > 1 else None
                    }
                    
                    # Remove None values
                    params = {k: v for k, v in params.items() if v is not None}
                    
                    # Make the request with retries
                    session = requests.Session()
                    retries = Retry(total=3, backoff_factor=0.5)
                    session.mount('https://', HTTPAdapter(max_retries=retries))
                    
                    response = session.get(search_url, params=params, headers=headers, timeout=15)
                    response.raise_for_status()
                    
                    # Parse the HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Save the HTML for debugging
                    with open(f"debug_search_page_{page}.html", "w", encoding="utf-8") as f:
                        f.write(str(soup))
                    print(f"\n{Fore.BLUE}Saved HTML to debug_search_page_{page}.html for analysis{Style.RESET_ALL}")
                    
                    # Find all search result items - try multiple selectors for the current website structure
                    result_items = []
                    
                    # Try different selectors for result items based on current website structure
                    for selector in ['.card', '.card-body', '.search-result', '.search-result-item', '.row .col-md-12']:
                        items = soup.select(selector)
                        if items:
                            print(f"\n{Fore.GREEN}Found {len(items)} results using selector: {selector}{Style.RESET_ALL}")
                            result_items = items
                            break
                    
                    if not result_items:
                        # Last resort: look for any links that might be results
                        links = soup.select('a[href*="/Home/Detail/"]')
                        if links:
                            print(f"\n{Fore.GREEN}Found {len(links)} results by searching for detail links{Style.RESET_ALL}")
                            
                            # Create simple result items from links
                            result_items = []
                            for link in links:
                                # Create a simple wrapper div for each link
                                div = soup.new_tag('div')
                                div.append(link)
                                result_items.append(div)
                    
                    if not result_items:
                        print(f"\n{Fore.YELLOW}No results found on page {page}{Style.RESET_ALL}")
                        break
                    
                    print(f"\n{Fore.GREEN}Found {len(result_items)} results on page {page}{Style.RESET_ALL}")
                    
                    # Process each result item
                    for i, item in enumerate(result_items):
                        try:
                            print(f"\n{Fore.BLUE}Processing result item {i+1}/{len(result_items)}{Style.RESET_ALL}")
                            
                            # Extract title and link - try multiple approaches for the current website structure
                            title_element = None
                            
                            # Try different selectors to find the title and link
                            for selector in [
                                'h3.fw-bold.text-gray-800.mb-5 a', 
                                'h3 a', 
                                '.fw-bold.text-gray-800 a',
                                'a[href*="/Home/Detail/"]',
                                'a'
                            ]:
                                candidates = item.select(selector)
                                for candidate in candidates:
                                    href = candidate.get('href', '')
                                    if href and ('/Home/Detail/' in href or '/Details/' in href):
                                        title_element = candidate
                                        print(f"Found title using selector: {selector}")
                                        break
                                if title_element:
                                    break
                            
                            if not title_element:
                                print(f"\n{Fore.YELLOW}Could not find title element, skipping item{Style.RESET_ALL}")
                                continue
                                
                            title = title_element.text.strip()
                            link = title_element.get('href')
                            if not link:
                                print(f"\n{Fore.YELLOW}No link found, skipping item{Style.RESET_ALL}")
                                continue
                                
                            # Make the link absolute
                            link = urljoin('https://peraturan.bpk.go.id/', link)
                            
                            print(f"\n{Fore.GREEN}Found document: {title}{Style.RESET_ALL}")
                            print(f"Link: {link}")
                            
                            # Extract metadata where available
                            doc_type = "Unknown Type"
                            date = "Unknown Date"
                            preview = ""
                            
                            # Try to extract metadata - updated selectors for current website structure
                            try:
                                # Try different selectors for metadata
                                meta_elements = item.select('.text-gray-600 span, .text-muted span, small, .card-text small') or item.select('.search-result-item-meta span')
                                if meta_elements and len(meta_elements) > 0:
                                    doc_type = meta_elements[0].text.strip()
                                if meta_elements and len(meta_elements) > 1:
                                    date = meta_elements[1].text.strip()
                                    
                                # Try different selectors for preview
                                preview_element = item.select_one('.card-text:not(:has(small))') or item.select_one('.search-result-item-preview')
                                if preview_element:
                                    preview = preview_element.text.strip()
                            except Exception as meta_error:
                                print(f"\n{Fore.YELLOW}Error extracting metadata: {str(meta_error)}{Style.RESET_ALL}")
                            
                            # Retrieve the full document content
                            try:
                                print(f"\n{Fore.BLUE}Retrieving document content from: {link}{Style.RESET_ALL}")
                                
                                # Use a session with retries
                                session = requests.Session()
                                retries = Retry(total=3, backoff_factor=0.5)
                                session.mount('https://', HTTPAdapter(max_retries=retries))
                                
                                doc_response = session.get(link, headers=headers, timeout=15)
                                doc_response.raise_for_status()
                                
                                # Save the HTML for debugging
                                with open(f"debug_document_{i}.html", "w", encoding="utf-8") as f:
                                    f.write(doc_response.text)
                                print(f"\n{Fore.BLUE}Saved document HTML to debug_document_{i}.html{Style.RESET_ALL}")
                                
                                doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
                                
                                # Extract the main content - try multiple approaches
                                content = ""
                                
                                # Approach 1: Try specific selectors
                                for selector in ['.card-body', 'main .container', '.document-content', '.content', 'article', '#mainContent', '.detail-content']:
                                    content_element = doc_soup.select_one(selector)
                                    if content_element and len(content_element.get_text(strip=True)) > 100:
                                        content = content_element.get_text(separator='\n', strip=True)
                                        print(f"\n{Fore.GREEN}Found content using selector: {selector} ({len(content)} chars){Style.RESET_ALL}")
                                        break
                                
                                # Approach 2: If no content yet, try to extract from paragraphs
                                if not content or len(content) < 200:
                                    paragraphs = doc_soup.select('p') or doc_soup.select('.card-text') or doc_soup.select('div > div')
                                    if paragraphs:
                                        content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                                
                                # Approach 3: If still no content, try to get any text from the page
                                if not content or len(content) < 200:
                                    # Get all text from the body, excluding scripts and styles
                                    for script in doc_soup(["script", "style"]):
                                        script.extract()
                                    content = doc_soup.body.get_text(separator='\n', strip=True)
                                
                                if content and len(content) > 200:
                                    # Create a Document object
                                    document = Document(
                                        content=content,
                                        page_content=content,
                                        metadata={
                                            'title': title,
                                            'source': link,
                                            'type': doc_type,
                                            'date': date,
                                            'preview': preview,
                                            'page': page
                                        }
                                    )
                                    
                                    documents.append(document)
                                    print(f"\n{Fore.GREEN}Added document: {title} ({len(content)} chars){Style.RESET_ALL}")
                                else:
                                    print(f"\n{Fore.YELLOW}Could not extract sufficient content for: {title}{Style.RESET_ALL}")
                                    
                                    # Try to find PDF links
                                    pdf_links = self.find_pdf_links(link, headers)
                                    
                                    if pdf_links:
                                        print(f"\n{Fore.GREEN}Found {len(pdf_links)} PDF links for: {title}{Style.RESET_ALL}")
                                        
                                        # Extract content from the first PDF
                                        try:
                                            pdf_content, pdf_metadata = self.download_and_extract_pdf(pdf_links[0], headers)
                                            
                                            if pdf_content:
                                                # Create a Document object for the PDF
                                                pdf_document = Document(
                                                    content=pdf_content,
                                                    page_content=pdf_content,
                                                    metadata={
                                                        'title': title,
                                                        'source': pdf_links[0],
                                                        'type': f"{doc_type} (PDF)",
                                                        'date': date,
                                                        'preview': preview,
                                                        'pdf_metadata': pdf_metadata,
                                                        'page': page
                                                    }
                                                )
                                                
                                                documents.append(pdf_document)
                                                print(f"\n{Fore.GREEN}Added PDF document: {title} ({len(pdf_content)} chars){Style.RESET_ALL}")
                                        except Exception as pdf_error:
                                            print(f"\n{Fore.YELLOW}Error extracting PDF content: {str(pdf_error)}{Style.RESET_ALL}")
                            except Exception as doc_error:
                                print(f"\n{Fore.YELLOW}Error retrieving document content: {str(doc_error)}{Style.RESET_ALL}")
                                traceback.print_exc()
                        except Exception as item_error:
                            print(f"\n{Fore.YELLOW}Error processing result item: {str(item_error)}{Style.RESET_ALL}")
                            traceback.print_exc()
                    
                    # Check if there are more pages - updated selectors for pagination
                    next_page = soup.select_one('.pagination .next:not(.disabled)') or soup.select_one('.pagination .page-item:not(.active):not(.disabled) .page-link')
                    if not next_page:
                        print(f"\n{Fore.BLUE}No more pages available{Style.RESET_ALL}")
                        break
                        
                except Exception as page_error:
                    print(f"\n{Fore.RED}Error scraping page {page}: {str(page_error)}{Style.RESET_ALL}")
            
            print(f"\n{Fore.GREEN}Scraped {len(documents)} documents from peraturan.bpk.go.id{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}Error scraping peraturan.bpk.go.id: {str(e)}{Style.RESET_ALL}")
            traceback.print_exc()
        
        return documents
    
    def search_pdf_documents(self, query, max_pages=5):
        """
        Search for PDF documents on peraturan.bpk.go.id based on a query.
        
        Args:
            query (str): The search query
            max_pages (int): Maximum number of search result pages to process
            
        Returns:
            List[Document]: List of document objects with PDF content and metadata
        """
        print(f"\n{Fore.BLUE}Searching for PDF documents related to: {query}{Style.RESET_ALL}")
        
        if not HAS_PYPDF2:
            print(f"\n{Fore.YELLOW}PyPDF2 is not available. Cannot extract PDF content.{Style.RESET_ALL}")
            return []
            
        try:
            # Process the query with legal language conversion if available
            processed_query = self.preprocess_query(query)
            
            # If the query was enhanced, log it
            if processed_query != query:
                print(f"\n{Fore.BLUE}Enhanced query for PDF search: {processed_query}{Style.RESET_ALL}")
                
            # Prepare the search URL
            base_url = "https://peraturan.bpk.go.id"
            search_url = f"{base_url}/Search/Results?query={processed_query.replace(' ', '+')}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            documents = []
            pdf_urls_processed = set()  # Track processed PDFs to avoid duplicates
            
            # Process search result pages
            for page_num in range(1, max_pages + 1):
                page_url = f"{search_url}&page={page_num}"
                
                # Find PDF links on the search results page
                pdf_links = self.find_pdf_links(page_url, headers=headers)
                
                # Process each PDF link
                for pdf_link in pdf_links:
                    pdf_url = pdf_link['url']
                    
                    # Skip if already processed
                    if pdf_url in pdf_urls_processed:
                        continue
                        
                    pdf_urls_processed.add(pdf_url)
                    
                    # Download and extract the PDF
                    content, metadata = self.download_and_extract_pdf(pdf_url, headers=headers)
                    
                    if content and metadata:
                        # Add additional metadata
                        metadata.update({
                            'query': query,
                            'enhanced_query': processed_query,
                            'link_text': pdf_link['text'],
                            'source_page': pdf_link['source_page']
                        })
                        
                        # Create document
                        doc = Document(content, metadata)
                        documents.append(doc)
                        
                        print(f"\n{Fore.GREEN}Added PDF document: {metadata.get('title', 'Untitled')}{Style.RESET_ALL}")
            
            # Sort documents by relevance if embeddings are available
            if self.has_embeddings and documents:
                print(f"\n{Fore.BLUE}Sorting PDF documents by relevance using IndoBERT embeddings{Style.RESET_ALL}")
                
                # Generate query embedding
                query_embedding = self.embeddings.get_embeddings([processed_query])[0]
                
                # Generate document embeddings and calculate similarity
                for doc in documents:
                    doc_embedding = self.embeddings.get_embeddings([doc.content[:1000]])[0]  # Use first 1000 chars for efficiency
                    similarity = self.calculate_similarity(query_embedding, doc_embedding)
                    doc.metadata['relevance_score'] = similarity
                
                # Sort by relevance score
                documents.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
                print(f"\n{Fore.GREEN}PDF documents sorted by relevance using IndoBERT embeddings{Style.RESET_ALL}")
            
            return documents
            
        except Exception as e:
            print(f"\n{Fore.RED}Error searching for PDF documents: {str(e)}{Style.RESET_ALL}")
            return []
    
    def search(self, query, max_pages=5, max_results=10):
        """
        Search for legal documents based on the query.
        
        Args:
            query (str): The user's query
            max_pages (int): Maximum number of pages to scrape
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Document]: List of document objects with content and metadata
        """
        print(f"\n{Fore.BLUE}Searching for documents related to: {query}{Style.RESET_ALL}")
        
        try:
            # Process the query with legal language conversion if available
            processed_query = self.preprocess_query(query)
            
            # If the query was enhanced, log it
            if processed_query != query:
                print(f"\n{Fore.BLUE}Enhanced query: {processed_query}{Style.RESET_ALL}")
            
            # Scrape documents from peraturan.bpk.go.id
            documents = self.scrape_peraturan_bpk(processed_query, max_pages=max_pages)
            
            # Search for PDF documents if PyPDF2 is available
            if HAS_PYPDF2:
                print(f"\n{Fore.BLUE}Searching for PDF documents...{Style.RESET_ALL}")
                pdf_documents = self.search_pdf_documents(processed_query, max_pages=max_pages)
                
                if pdf_documents:
                    print(f"\n{Fore.GREEN}Found {len(pdf_documents)} PDF documents{Style.RESET_ALL}")
                    documents.extend(pdf_documents)
            
            # Sort all documents by relevance if embeddings are available
            if self.has_embeddings and documents:
                print(f"\n{Fore.BLUE}Sorting all documents by relevance using IndoBERT embeddings{Style.RESET_ALL}")
                
                # Generate query embedding
                query_embedding = self.embeddings.get_embeddings([processed_query])[0]
                
                # Calculate similarity for documents without a relevance score
                for doc in documents:
                    if 'relevance_score' not in doc.metadata:
                        doc_text = doc.content[:1000]  # Use first 1000 chars for efficiency
                        doc_embedding = self.embeddings.get_embeddings([doc_text])[0]
                        similarity = self.calculate_similarity(query_embedding, doc_embedding)
                        doc.metadata['relevance_score'] = similarity
                
                # Sort by relevance score
                documents.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
                print(f"\n{Fore.GREEN}All documents sorted by relevance using IndoBERT embeddings{Style.RESET_ALL}")
            
            # Limit results
            if max_results and len(documents) > max_results:
                documents = documents[:max_results]
            
            if documents:
                print(f"\n{Fore.GREEN}Found {len(documents)} relevant documents in total{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}No documents found for query: {query}{Style.RESET_ALL}")
                
            return documents
            
        except Exception as e:
            print(f"\n{Fore.RED}Error searching for documents: {str(e)}{Style.RESET_ALL}")
            return []

    def generate_html_report(self, query: str, documents: List[Document], response: str) -> str:
        """
        Generate an HTML report of the scraped documents.
        
        Args:
            query: The original query
            documents: List of Document objects
            response: The response from the LLM
            
        Returns:
            The path to the generated HTML file
        """
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate a safe filename from the query
            safe_query = re.sub(r'[^\w\s-]', '', query)
            safe_query = re.sub(r'[\s-]+', '_', safe_query)
            
            # Create the filename
            filename = f"bpk_report_{safe_query}_{timestamp}.html"
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>BPK Legal Document Report: {query}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: #fff;
                        padding: 20px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    header {{
                        background-color: #005A9C;
                        color: white;
                        padding: 20px;
                        text-align: center;
                        margin-bottom: 20px;
                    }}
                    h1 {{
                        margin: 0;
                        font-size: 24px;
                    }}
                    h2 {{
                        color: #005A9C;
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 10px;
                        margin-top: 30px;
                    }}
                    .query-info {{
                        background-color: #f5f5f5;
                        padding: 15px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .document {{
                        margin-bottom: 30px;
                        padding: 15px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        border-left: 5px solid #005A9C;
                    }}
                    .document-header {{
                        margin-bottom: 10px;
                    }}
                    .document-title {{
                        font-weight: bold;
                        font-size: 18px;
                        color: #005A9C;
                    }}
                    .document-meta {{
                        color: #666;
                        font-size: 14px;
                        margin: 5px 0;
                    }}
                    .document-content {{
                        max-height: 300px;
                        overflow-y: auto;
                        padding: 10px;
                        background-color: #fff;
                        border: 1px solid #ddd;
                        border-radius: 3px;
                        margin-top: 10px;
                    }}
                    .document-content pre {{
                        white-space: pre-wrap;
                        font-family: monospace;
                        margin: 0;
                    }}
                    .response {{
                        background-color: #e6f7ff;
                        padding: 20px;
                        border-radius: 5px;
                        margin-bottom: 30px;
                        border-left: 5px solid #1890ff;
                    }}
                    footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        color: #666;
                        font-size: 14px;
                    }}
                    .relevance-score {{
                        display: inline-block;
                        padding: 3px 8px;
                        background-color: #005A9C;
                        color: white;
                        border-radius: 12px;
                        font-size: 12px;
                        margin-left: 10px;
                    }}
                    .pdf-badge {{
                        display: inline-block;
                        padding: 3px 8px;
                        background-color: #d9534f;
                        color: white;
                        border-radius: 12px;
                        font-size: 12px;
                        margin-left: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <h1>BPK Legal Document Report</h1>
                    </header>
                    
                    <div class="query-info">
                        <h2>Query Information</h2>
                        <p><strong>Original Query:</strong> {query}</p>
                        <p><strong>Search Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p><strong>Documents Found:</strong> {len(documents)}</p>
                    </div>
                    
                    <div class="response">
                        <h2>Response</h2>
                        <p>{response.replace('\n', '<br>')}</p>
                    </div>
                    
                    <h2>Retrieved Documents</h2>
            """
            
            # Add each document to the HTML
            for i, doc in enumerate(documents):
                title = doc.metadata.get('title', 'Untitled Document')
                source = doc.metadata.get('source', 'Unknown Source')
                doc_type = doc.metadata.get('type', 'html')
                
                # Get document type indicator
                doc_type_badge = ""
                if doc_type == 'pdf':
                    doc_type_badge = '<span class="pdf-badge">PDF</span>'
                
                # Get relevance score if available
                relevance_badge = ""
                if 'relevance_score' in doc.metadata:
                    score = doc.metadata['relevance_score']
                    relevance_badge = f'<span class="relevance-score">Relevance: {score:.2f}</span>'
                
                # Get page number if available
                page_badge = ""
                if 'page' in doc.metadata:
                    page = doc.metadata['page']
                    page_badge = f'<span class="page-badge">Page {page}</span>'
                
                # Format content based on type
                if doc_type == 'pdf':
                    # For PDF content, preserve formatting
                    content_html = f"<pre>{doc.content}</pre>"
                else:
                    # For HTML content, preserve HTML formatting
                    content_html = doc.content
                
                # Add document to HTML
                html_content += f"""
                    <div class="document">
                        <div class="document-header">
                            <div class="document-title">{i+1}. {title} {doc_type_badge} {relevance_badge} {page_badge}</div>
                            <div class="document-meta"><strong>Source:</strong> <a href="{source}" target="_blank">{source}</a></div>
                """
                
                # Add document type specific metadata
                if doc_type == 'pdf':
                    html_content += f"""
                            <div class="document-meta"><strong>Pages:</strong> {doc.metadata.get('pages', 'Unknown')}</div>
                    """
                    if 'author' in doc.metadata:
                        html_content += f"""
                            <div class="document-meta"><strong>Author:</strong> {doc.metadata.get('author', 'Unknown')}</div>
                        """
                else:
                    html_content += f"""
                            <div class="document-meta"><strong>Document Number:</strong> {doc.metadata.get('doc_number', 'Unknown')}</div>
                            <div class="document-meta"><strong>Document Type:</strong> {doc.metadata.get('doc_type', 'Unknown')}</div>
                    """
                
                # Add content preview
                html_content += f"""
                        </div>
                        <div class="document-content">
                            {content_html}
                        </div>
                    </div>
                """
            
            # Close HTML tags
            html_content += """
                </div>
                
                <footer>
                    <p>This report was generated using the BPK Legal Document Scraper with Sastrawi and IndoBERT. The information provided is based on the retrieved documents and should be verified with official sources.</p>
                </footer>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\n{Fore.GREEN}Report saved to {filename}{Style.RESET_ALL}")
            return filename
            
        except Exception as e:
            print(f"\n{Fore.RED}Error generating HTML report: {str(e)}{Style.RESET_ALL}")
            return f"Error generating report: {str(e)}"
    
    def process_query_with_llm(self, query, user_preferences=None):
        """
        Process a user query through a legal LLM workflow:
        1. Extract keywords from query using LLM
        2. Use keywords to retrieve relevant documents by scraping BPK website
        3. Paraphrase and summarize the answer based on user preferences
        
        Args:
            query (str): The user's original query
            user_preferences (dict): Optional user preferences for response formatting
                - verbosity: 'concise', 'detailed', or 'comprehensive'
                - format: 'simple', 'legal', or 'technical'
                - citations: True/False whether to include citations
                - max_results: Maximum number of documents to retrieve (default: 5)
                
        Returns:
            dict: A dictionary containing the extracted keywords, retrieved documents, and paraphrased response
        """
        try:
            # Default user preferences if none provided
            if user_preferences is None:
                user_preferences = {
                    'verbosity': 'detailed',
                    'format': 'simple',
                    'citations': True,
                    'max_results': 5
                }
            
            # Step 1: Extract keywords from query using LLM
            keywords = []
            
            if HAS_OPENAI and self.openai_wrapper:
                try:
                    print(f"\n{Fore.BLUE}Extracting keywords for web search...{Style.RESET_ALL}")
                    
                    prompt = f"""
                    As a legal expert in Indonesian law, extract the most important keywords from this query that would be effective for searching on a legal document website.
                    
                    Original query: {query}
                    
                    Extract 3-5 specific keywords or phrases that are most relevant for searching legal documents. Focus on legal terminology, document types, or specific regulations.
                    
                    Format your response as a comma-separated list of keywords only, without any additional text.
                    """
                    
                    try:
                        llm_response = self.openai_wrapper.invoke(prompt)
                        
                        # Parse the keywords
                        keywords = [kw.strip() for kw in llm_response.split(',') if kw.strip()]
                        
                        print(f"\n{Fore.GREEN}Extracted keywords for web search: {', '.join(keywords)}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"\n{Fore.YELLOW}Error extracting keywords with LLM: {str(e)}{Style.RESET_ALL}")
                        # Fallback: use simple keyword extraction
                        keywords = [w for w in query.split() if len(w) > 3]
                        print(f"\n{Fore.YELLOW}Using fallback keywords: {', '.join(keywords)}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"\n{Fore.YELLOW}Error with OpenAI wrapper: {str(e)}{Style.RESET_ALL}")
                    # Fallback: use simple keyword extraction
                    keywords = [w for w in query.split() if len(w) > 3]
                    print(f"\n{Fore.YELLOW}Using fallback keywords: {', '.join(keywords)}{Style.RESET_ALL}")
            else:
                # Fallback if no LLM is available
                keywords = [w for w in query.split() if len(w) > 3]
                print(f"\n{Fore.YELLOW}No LLM available. Using simple keyword extraction: {', '.join(keywords)}{Style.RESET_ALL}")
            
            # If no keywords were extracted, use the original query
            if not keywords:
                keywords = [query]
                print(f"\n{Fore.YELLOW}No keywords extracted. Using original query as keyword.{Style.RESET_ALL}")
                
            # Step 2: Use keywords to retrieve relevant documents by scraping BPK website
            print(f"\n{Fore.BLUE}Retrieving documents from BPK using keywords...{Style.RESET_ALL}")
            
            max_results = user_preferences.get('max_results', 5)
            documents = []
            
            # Try each keyword for searching
            for keyword in keywords[:3]:  # Limit to top 3 keywords to avoid too many requests
                try:
                    print(f"\n{Fore.BLUE}Scraping peraturan.bpk.go.id for keyword: {keyword}{Style.RESET_ALL}")
                    keyword_docs = self.scrape_peraturan_bpk(keyword, max_pages=2)
                    
                    if keyword_docs:
                        # Filter out duplicates
                        new_docs = []
                        existing_urls = [doc.metadata.get('source', '') for doc in documents]
                        
                        for doc in keyword_docs:
                            if doc.metadata.get('source', '') not in existing_urls:
                                new_docs.append(doc)
                        
                        if new_docs:
                            print(f"\n{Fore.GREEN}Found {len(new_docs)} new documents for keyword: {keyword}{Style.RESET_ALL}")
                            documents.extend(new_docs)
                            
                            # Update existing URLs
                            existing_urls = [doc.metadata.get('source', '') for doc in documents]
                            
                            # Break if we've reached max_results
                            if len(documents) >= max_results:
                                break
                except Exception as e:
                    print(f"\n{Fore.YELLOW}Error searching for keyword {keyword}: {str(e)}{Style.RESET_ALL}")
                    continue  # Continue with next keyword even if this one fails
            
            # Limit to max_results
            if len(documents) > max_results:
                documents = documents[:max_results]
                
            print(f"\n{Fore.GREEN}Total documents retrieved: {len(documents)}{Style.RESET_ALL}")
            
            if not documents:
                print(f"\n{Fore.YELLOW}No documents found for the keywords.{Style.RESET_ALL}")
                return {
                    'original_query': query,
                    'keywords': keywords,
                    'documents': [],
                    'response': "I couldn't find any relevant legal documents for your query."
                }
                
            # Step 3: Generate a paraphrased response based on user preferences
            print(f"\n{Fore.BLUE}Generating response based on retrieved documents...{Style.RESET_ALL}")
            
            # Prepare document content for the LLM
            document_summaries = []
            try:
                for i, doc in enumerate(documents):  # Include all retrieved documents
                    try:
                        title = doc.metadata.get('title', 'Untitled Document')
                        source = doc.metadata.get('source', 'Unknown Source')
                        doc_type = doc.metadata.get('type', 'Unknown Type')
                        
                        # Create a summary of the document content (first 1000 chars)
                        content = doc.content if hasattr(doc, 'content') and doc.content else "No content available"
                        content_preview = content[:1000] + "..." if len(content) > 1000 else content
                        
                        document_summaries.append(f"Document {i+1}: {title}\nType: {doc_type}\nSource: {source}\nPreview: {content_preview}\n")
                    except Exception as e:
                        print(f"\n{Fore.YELLOW}Error processing document {i+1}: {str(e)}{Style.RESET_ALL}")
                        document_summaries.append(f"Document {i+1}: Error processing document: {str(e)}\n")
            except Exception as e:
                print(f"\n{Fore.YELLOW}Error preparing document summaries: {str(e)}{Style.RESET_ALL}")
                document_summaries = ["Error preparing document summaries. Using simplified approach."]
            
            # Generate response with LLM
            response = ""
            if HAS_OPENAI and self.openai_wrapper:
                try:
                    # Determine verbosity level
                    verbosity_instruction = ""
                    if user_preferences.get('verbosity') == 'concise':
                        verbosity_instruction = "Keep your response concise and to the point, focusing only on the most relevant information."
                    elif user_preferences.get('verbosity') == 'detailed':
                        verbosity_instruction = "Provide a detailed response that covers the main points from the documents."
                    elif user_preferences.get('verbosity') == 'comprehensive':
                        verbosity_instruction = "Provide a comprehensive response that thoroughly analyzes all relevant information from the documents."
                    else:
                        verbosity_instruction = "Provide a detailed response that covers the main points from the documents."
                    
                    # Determine format style
                    format_instruction = ""
                    if user_preferences.get('format') == 'simple':
                        format_instruction = "Use simple, everyday language that a non-legal expert can understand."
                    elif user_preferences.get('format') == 'legal':
                        format_instruction = "Use proper legal terminology and formatting appropriate for legal professionals."
                    elif user_preferences.get('format') == 'technical':
                        format_instruction = "Use technical language and provide specific details about legal mechanisms and procedures."
                    else:
                        format_instruction = "Use simple, everyday language that a non-legal expert can understand."
                    
                    # Determine citation style
                    citation_instruction = ""
                    if user_preferences.get('citations', True):
                        citation_instruction = "Include citations to specific documents and sections when making claims."
                    else:
                        citation_instruction = "Do not include formal citations in your response."
                    
                    prompt = f"""
                    As a legal expert in Indonesian law, answer the following query based on the provided legal documents from BPK (Badan Pemeriksa Keuangan).
                    
                    Original query: {query}
                    Search keywords used: {', '.join(keywords)}
                    
                    Retrieved documents:
                    {"".join(document_summaries)}
                    
                    Instructions:
                    1. {verbosity_instruction}
                    2. {format_instruction}
                    3. {citation_instruction}
                    4. Focus on directly answering the query based on the legal documents provided.
                    5. If the documents don't contain sufficient information to answer the query, acknowledge this limitation.
                    6. Include specific references to BPK regulations and documents where relevant.
                    
                    Your response:
                    """
                    
                    try:
                        response = self.openai_wrapper.invoke(prompt)
                        print(f"\n{Fore.GREEN}Generated response based on user preferences{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"\n{Fore.YELLOW}Error generating response with LLM: {str(e)}{Style.RESET_ALL}")
                        # Fallback response generation
                        response = self._generate_fallback_response(query, documents)
                        
                except Exception as e:
                    print(f"\n{Fore.YELLOW}Error preparing LLM prompt: {str(e)}{Style.RESET_ALL}")
                    # Fallback response generation
                    response = self._generate_fallback_response(query, documents)
            else:
                # Fallback if no LLM is available
                response = self._generate_fallback_response(query, documents)
                
            # Return the results
            return {
                'original_query': query,
                'keywords': keywords,
                'documents': documents,
                'response': response
            }
            
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Process interrupted by user.{Style.RESET_ALL}")
            return {
                'original_query': query,
                'keywords': keywords if 'keywords' in locals() else [],
                'documents': documents if 'documents' in locals() else [],
                'response': "The process was interrupted by the user before completion."
            }
        except Exception as e:
            print(f"\n{Fore.RED}Unexpected error in process_query_with_llm: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return {
                'original_query': query,
                'keywords': keywords if 'keywords' in locals() else [],
                'documents': documents if 'documents' in locals() else [],
                'response': f"An error occurred while processing your query: {str(e)}"
            }
            
    def _generate_fallback_response(self, query, documents):
        """
        Generate a fallback response when LLM processing fails.
        
        Args:
            query (str): The original query
            documents (list): List of retrieved documents
            
        Returns:
            str: A simple response based on the documents
        """
        try:
            # Create a simple response based on document titles and content previews
            response = f"Based on the retrieved documents from BPK, the query about '{query}' relates to the following legal information:\n\n"
            
            for i, doc in enumerate(documents[:3]):  # Limit to first 3 documents for brevity
                try:
                    title = doc.metadata.get('title', 'Untitled Document')
                    content = doc.content if hasattr(doc, 'content') and doc.content else "No content available"
                    preview = content[:200] + "..." if len(content) > 200 else content
                    response += f"- {title}: {preview}\n\n"
                except Exception:
                    response += f"- Document {i+1}: Unable to extract content\n\n"
                    
            return response
        except Exception as e:
            return f"Unable to generate a response due to an error: {str(e)}"
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays if they aren't already
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure the result is between 0 and 1
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"\n{Fore.RED}Error calculating similarity: {str(e)}{Style.RESET_ALL}")
            return 0

class PeraturanScraper:
    """
    Scraper for peraturan.go.id to retrieve legal documents.
    """
    
    def __init__(self, openai_api_key=None, openai_base_url=None):
        """
        Initialize the peraturan.go.id legal document scraper.
        
        Args:
            openai_api_key (str): OpenAI API key for language processing
            openai_base_url (str): Base URL for OpenAI API
        """
        self.openai_wrapper = None
        
        try:
            # Load API key from environment if not provided
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                
            if not openai_base_url:
                openai_base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                
            if openai_api_key:
                try:
                    # Create custom httpx client with timeout settings
                    import httpx
                    http_client = httpx.Client(
                        timeout=httpx.Timeout(30.0, connect=10.0)
                    )
                    
                    # Initialize OpenAI client with custom HTTP client
                    client = OpenAI(
                        api_key=openai_api_key,
                        base_url=openai_base_url,
                        http_client=http_client
                    )
                    print("[INFO] OpenAI client initialized successfully with custom HTTP client")
                    
                    # Initialize OpenAI wrapper
                    try:
                        from final3 import SimpleOpenAIWrapper
                        self.openai_wrapper = SimpleOpenAIWrapper(
                            client=client,
                            model="qwen2.5-72b-instruct",  # Use the same model as in final3.py
                            temperature=0
                        )
                        print("OpenAI API initialized successfully")
                    except ImportError as ie:
                        print(f"Could not import SimpleOpenAIWrapper: {str(ie)}")
                        # If SimpleOpenAIWrapper can't be imported, create a basic wrapper
                        class BasicOpenAIWrapper:
                            def __init__(self, client, model="qwen2.5-72b-instruct", temperature=0):
                                self.client = client
                                self.model = model
                                self.temperature = temperature
                            
                            def invoke(self, input_data):
                                try:
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
                                    print(f"Error invoking OpenAI: {e}")
                                    return f"Error: {str(e)}"
                        
                        self.openai_wrapper = BasicOpenAIWrapper(
                            client=client,
                            model="qwen2.5-72b-instruct",
                            temperature=0
                        )
                        print("Basic OpenAI wrapper initialized successfully")
                except Exception as e:
                    print(f"Error initializing OpenAI client: {str(e)}")
        except Exception as e:
            print(f"Error in PeraturanScraper initialization: {str(e)}")
        
        # Initialize Sastrawi stemmer if available
        try:
            if HAS_SASTRAWI:
                self.stemmer_factory = StemmerFactory()
                self.stemmer = self.stemmer_factory.create_stemmer()
                print("Sastrawi stemmer initialized successfully")
            else:
                self.stemmer = None
        except Exception as e:
            print(f"Error initializing Sastrawi stemmer: {str(e)}")
            self.stemmer = None
        
        # Initialize IndoBERT embeddings if available
        try:
            self.embeddings = IndoBERTEmbeddings()
            print("IndoBERT model initialized successfully")
        except Exception as e:
            print(f"Error initializing IndoBERT embeddings: {str(e)}")
            self.embeddings = None
        
        # Check if PyPDF2 is available
        if HAS_PYPDF2:
            print("PyPDF2 initialized successfully")
        
        print("\n" + "="*80)
        print("Peraturan Scraper initialized successfully")
        print("="*80)

    def extract_pdf_content(self, pdf_url, headers=None, title="PDF Document"):
        """
        Extract content from a PDF file
        
        Args:
            pdf_url (str): URL of the PDF file
            headers (dict, optional): HTTP headers for the request
            title (str, optional): Title of the document
            
        Returns:
            tuple: (content, metadata) or (None, None) if extraction fails
        """
        try:
            import requests
            import tempfile
            import os
            import PyPDF2
            
            if not headers:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/pdf',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://peraturan.go.id/',
                }
            
            # Download the PDF
            logger.info(f"Downloading PDF from {pdf_url}")
            pdf_response = requests.get(pdf_url, headers=headers, timeout=30)
            pdf_response.raise_for_status()
            
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_response.content)
                temp_path = temp_file.name
            
            logger.info(f"PDF saved to temporary file: {temp_path}")
            
            # Extract text from PDF
            pdf_content = ""
            pdf_metadata = {}
            
            try:
                with open(temp_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract metadata if available
                    if pdf_reader.metadata:
                        for key, value in pdf_reader.metadata.items():
                            if key.startswith('/'):
                                key = key[1:]  # Remove leading slash
                            pdf_metadata[key] = str(value)
                    
                    # Extract text from all pages
                    num_pages = len(pdf_reader.pages)
                    logger.info(f"PDF has {num_pages} pages")
                    
                    for page_num in range(num_pages):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                pdf_content += page_text + "\n\n"
                        except Exception as page_error:
                            logger.error(f"Error extracting text from page {page_num}: {str(page_error)}")
                
                # Create a preview of the content
                preview = pdf_content[:500] + "..." if len(pdf_content) > 500 else pdf_content
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                    logger.info("Temporary PDF file deleted")
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up temporary file: {str(cleanup_error)}")
                
                if pdf_content and len(pdf_content) > 200:
                    logger.info(f"Successfully extracted {len(pdf_content)} characters from PDF")
                    return pdf_content, {
                        'title': title,
                        'source': pdf_url,
                        'type': "PDF Document",
                        'date': pdf_metadata.get('ModDate', "Unknown Date"),
                        'preview': preview,
                        'pdf_metadata': pdf_metadata
                    }
                else:
                    logger.warning(f"Extracted content too short or empty: {len(pdf_content)} chars")
                    return None, None
            except Exception as pdf_extract_error:
                logger.error(f"Error extracting PDF content: {str(pdf_extract_error)}")
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                return None, None
        except ImportError as import_error:
            logger.error(f"Required module not available for PDF extraction: {str(import_error)}")
            return None, None
        except Exception as e:
            logger.error(f"Error in extract_pdf_content: {str(e)}")
            return None, None

    def scrape_peraturan_go_id(self, query: str, max_pages: int = 10) -> List[Document]:
        """
        Scrape peraturan.go.id for legal information based on a query.
        
        Args:
            query (str): The search query
            max_pages (int): Maximum number of search result pages to process
            
        Returns:
            List[Document]: List of document objects with content and metadata
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            import tempfile
            import os
            
            documents = []
            
            # Set up headers to avoid bot detection
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://peraturan.go.id/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Process search pages
            for page in range(1, max_pages + 1):
                try:
                    logger.info(f"Scraping page {page} of {max_pages}")
                    
                    # Construct search URL for peraturan.go.id using the correct format
                    search_url = f"https://peraturan.go.id/cariglobal"
                    params = {
                        'PeraturanSearch[idglobal]': query,
                        'page': page
                    }
                    
                    # Make request
                    response = requests.get(search_url, params=params, headers=headers, timeout=15)
                    response.raise_for_status()
                    
                    # Save debug copy of the search page
                    with open(f"search_page_{page}.html", "w", encoding="utf-8") as f:
                        f.write(response.text)
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for PDF links directly first
                    pdf_links = soup.select('a[href$=".pdf"]')
                    if pdf_links:
                        logger.info(f"Found {len(pdf_links)} PDF links directly")
                        
                        # Process each PDF link
                        for pdf_link in pdf_links:
                            try:
                                pdf_url = pdf_link.get('href')
                                if not pdf_url:
                                    continue
                                    
                                # Make the URL absolute
                                pdf_url = urljoin('https://peraturan.go.id/', pdf_url)
                                
                                # Get the title from the link text or parent element
                                title = pdf_link.text.strip()
                                if not title or len(title) < 5:
                                    # Try to get title from parent
                                    parent = pdf_link.parent
                                    if parent:
                                        title = parent.text.strip()
                                
                                if not title or len(title) < 5:
                                    title = f"PDF Document from {pdf_url.split('/')[-1]}"
                                
                                logger.info(f"Found PDF: {title}")
                                logger.info(f"PDF URL: {pdf_url}")
                                
                                # Use the extract_pdf_content method to handle PDF extraction
                                pdf_content, pdf_metadata = self.extract_pdf_content(pdf_url, headers=headers, title=title)
                                
                                if pdf_content:
                                    # Create a Document object for the PDF
                                    pdf_document = Document(
                                        content=pdf_content,
                                        page_content=pdf_content,
                                        metadata={
                                            **pdf_metadata,
                                            'page': page
                                        }
                                    )
                                    
                                    documents.append(pdf_document)
                                    logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
                            except Exception as pdf_error:
                                logger.error(f"Error processing PDF link: {str(pdf_error)}")
                    
                    # Find result items - using selectors for peraturan.go.id
                    result_items = soup.select('.item, .hasil-pencarian .col-md-12, .row .col-md-12')
                    
                    if not result_items:
                        logger.warning(f"No results found on page {page}")
                        
                        # Try alternative approach - look for links directly
                        links = soup.select('a[href*="/peraturan/"]')
                        if links:
                            logger.info(f"Found {len(links)} results by searching for detail links")
                            
                            # Create simple result items from links
                            result_items = []
                            for link in links:
                                parent = link.parent
                                if parent:
                                    result_items.append(parent)
                        else:
                            continue
                    
                    logger.info(f"Found {len(result_items)} results on page {page}")
                    
                    # Process each result
                    for i, item in enumerate(result_items):
                        try:
                            logger.info(f"Processing result {i+1}/{len(result_items)}")
                            
                            # Check for PDF links in this item first
                            pdf_links = item.select('a[href$=".pdf"]')
                            if pdf_links:
                                logger.info(f"Found {len(pdf_links)} PDF links in this item")
                                # Process the first PDF link
                                pdf_url = pdf_links[0].get('href')
                                if pdf_url:
                                    # Make the URL absolute
                                    pdf_url = urljoin(link, pdf_url)
                                    logger.info(f"PDF URL: {pdf_url}")
                                    
                                    # Use the extract_pdf_content method to handle PDF extraction
                                    pdf_content, pdf_metadata = self.extract_pdf_content(pdf_url, headers=headers, title=title)
                                    
                                    if pdf_content:
                                        # Create a Document object for the PDF
                                        pdf_document = Document(
                                            content=pdf_content,
                                            page_content=pdf_content,
                                            metadata={
                                                **pdf_metadata,
                                                'page': page
                                            }
                                        )
                                        
                                        documents.append(pdf_document)
                                        logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
                            else:
                                # Find the title and link - updated selectors for peraturan.go.id
                                title_element = None
                                
                                # Try different selectors to find the title and link
                                for selector in [
                                    'h3 a', 
                                    '.title a',
                                    '.result-title a',
                                    'a[href*="/peraturan/"]',
                                    'a'
                                ]:
                                    candidates = item.select(selector)
                                    for candidate in candidates:
                                        href = candidate.get('href', '')
                                        if href and '/peraturan/' in href:
                                            title_element = candidate
                                            break
                                    if title_element:
                                        break
                                
                                if not title_element:
                                    logger.warning("Could not find title element, skipping item")
                                    continue
                                
                                # Extract title and link
                                title = title_element.text.strip()
                                link = title_element.get('href')
                                
                                if not link:
                                    logger.warning("No link found, skipping item")
                                    continue
                                
                                # Make the link absolute
                                link = urljoin('https://peraturan.go.id/', link)
                                
                                logger.info(f"Found document: {title}")
                                logger.info(f"Link: {link}")
                                
                                # Extract metadata
                                doc_type = "Unknown Type"
                                date = "Unknown Date"
                                
                                # Try to extract metadata
                                meta_elements = item.select('.meta, .date, .info, small, .jenis, .tanggal')
                                if meta_elements and len(meta_elements) > 0:
                                    meta_text = meta_elements[0].text.strip()
                                    # Try to extract document type and date from metadata
                                    if "," in meta_text:
                                        parts = meta_text.split(",", 1)
                                        doc_type = parts[0].strip()
                                        date = parts[1].strip()
                                    else:
                                        doc_type = meta_text
                                
                                # Retrieve the document content
                                try:
                                    doc_response = requests.get(link, headers=headers, timeout=15)
                                    doc_response.raise_for_status()
                                    
                                    # Save debug copy of the document
                                    with open(f"debug_document_{i+1}.html", "w", encoding="utf-8") as f:
                                        f.write(doc_response.text)
                                    
                                    doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
                                    
                                    # Extract the main content - try multiple approaches
                                    content = ""
                                    
                                    # Try specific selectors for peraturan.go.id
                                    for selector in [
                                        '.document-content', 
                                        '.content', 
                                        'article', 
                                        'main', 
                                        '.detail-content',
                                        '#content',
                                        '.col-md-12',
                                        '.container'
                                    ]:
                                        content_element = doc_soup.select_one(selector)
                                        if content_element and len(content_element.get_text(strip=True)) > 100:
                                            content = content_element.get_text(separator='\n', strip=True)
                                            logger.info(f"Found content using selector: {selector} ({len(content)} chars)")
                                            break
                                    
                                    # If no content yet, try paragraphs
                                    if not content or len(content) < 200:
                                        paragraphs = doc_soup.select('p')
                                        if paragraphs:
                                            content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                                    
                                    # If still no content, try any text
                                    if not content or len(content) < 200:
                                        # Get all text from the body, excluding scripts and styles
                                        for script in doc_soup(["script", "style"]):
                                            script.extract()
                                        content = doc_soup.body.get_text(separator='\n', strip=True)
                                    
                                    if content and len(content) > 200:
                                        # Create document with both content and page_content
                                        document = Document(
                                            content=content,
                                            page_content=content,
                                            metadata={
                                                'title': title,
                                                'source': link,
                                                'type': doc_type,
                                                'date': date,
                                                'page': page
                                            }
                                        )
                                        
                                        documents.append(document)
                                        logger.info(f"Added document: {title} ({len(content)} chars)")
                                except Exception as doc_error:
                                    logger.error(f"Error retrieving document content: {str(doc_error)}")
                        except Exception as item_error:
                            logger.error(f"Error processing result item: {str(item_error)}")
                except Exception as page_error:
                    logger.error(f"Error scraping page {page}: {str(page_error)}")
            
            logger.info(f"Scraped {len(documents)} documents from peraturan.go.id")
            return documents
        except Exception as e:
            logger.error(f"Error in scrape_peraturan_go_id: {str(e)}")
            return []

def main():
    """
    Main function to run the BPK legal document scraper.
    """
    init()  # Initialize colorama
    print(f"\n{Fore.CYAN}===== BPK Legal Document Scraper ====={Style.RESET_ALL}")
    
    # Load environment variables
    load_dotenv()
    
    # Suppress warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    warnings.filterwarnings("ignore")
    
    # Create the scraper
    scraper = BPKScraper()
    
    while True:
        print(f"\n{Fore.CYAN}Enter your legal query (or 'exit' to quit):{Style.RESET_ALL}")
        query = input("> ")
        
        if query.lower() == 'exit' or query.lower() == 'quit':
            print(f"\n{Fore.CYAN}Thank you for using the BPK Legal Document Scraper. Goodbye!{Style.RESET_ALL}")
            break
            
        if not query.strip():
            print(f"\n{Fore.YELLOW}Please enter a valid query.{Style.RESET_ALL}")
            continue
            
        # Ask for user preferences
        print(f"\n{Fore.CYAN}Select response verbosity:{Style.RESET_ALL}")
        print(f"1. Concise (brief summary)")
        print(f"2. Detailed (default)")
        print(f"3. Comprehensive (thorough analysis)")
        verbosity_choice = input("> ")
        
        verbosity = "detailed"  # default
        if verbosity_choice == "1":
            verbosity = "concise"
        elif verbosity_choice == "3":
            verbosity = "comprehensive"
            
        print(f"\n{Fore.CYAN}Select response format:{Style.RESET_ALL}")
        print(f"1. Simple language (for non-experts)")
        print(f"2. Legal terminology (for legal professionals)")
        print(f"3. Technical (detailed legal mechanisms)")
        format_choice = input("> ")
        
        format_style = "simple"  # default
        if format_choice == "2":
            format_style = "legal"
        elif format_choice == "3":
            format_style = "technical"
            
        print(f"\n{Fore.CYAN}Include citations?{Style.RESET_ALL}")
        print(f"1. Yes (default)")
        print(f"2. No")
        citations_choice = input("> ")
        
        include_citations = True  # default
        if citations_choice == "2":
            include_citations = False
            
        # Set user preferences
        user_preferences = {
            'verbosity': verbosity,
            'format': format_style,
            'citations': include_citations
        }
        
        # Process the query with LLM and get response
        result = scraper.process_query_with_llm(query, user_preferences)
        
        print(f"\n{Fore.GREEN}=== Response ==={Style.RESET_ALL}")
        print(f"\n{result['response']}")
        
        # Ask if user wants to save the results
        print(f"\n{Fore.CYAN}Would you like to save these results? (y/n){Style.RESET_ALL}")
        save_choice = input("> ")
        
        if save_choice.lower() == 'y':
            # Generate a report with the documents
            report_path = scraper.generate_html_report(query, result['documents'], result['response'])
            print(f"\n{Fore.GREEN}Report saved to: {report_path}{Style.RESET_ALL}")
            
if __name__ == "__main__":
    main()
