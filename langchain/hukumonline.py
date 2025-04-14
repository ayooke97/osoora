"""
Hukumonline Scraper

This module provides functionality to scrape legal information from hukumonline.com
using LangChain and DashScope integration for advanced processing.
"""

import os
import re
import json
import time
import random
import requests
import logging
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urljoin, quote_plus
from bs4 import BeautifulSoup
from datetime import datetime
from colorama import Fore, Style, init
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Initialize colorama
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hukumonline_scraper")

# Import LangChain components
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document as LangChainDocument
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.embeddings import DashScopeEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import DashScope
    from langchain.chat_models import ChatDashScope
    from langchain.chains.summarize import load_summarize_chain
    HAS_LANGCHAIN = True
except ImportError:
    logger.warning(f"{Fore.YELLOW}LangChain not found. Some features will be disabled.{Style.RESET_ALL}")
    HAS_LANGCHAIN = False
    # Define a simple LangChainDocument class if langchain is not available
    class LangChainDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Import environment configuration
from env_config import DASHSCOPE_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID

# Try to import Selenium for headless browser scraping
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except ImportError:
    logger.warning(f"{Fore.YELLOW}Selenium not found. Headless browser scraping will be disabled.{Style.RESET_ALL}")
    HAS_SELENIUM = False

# Document class for compatibility
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


class HukumonlineScraper:
    """
    Class for scraping hukumonline.com website and processing legal information
    using LangChain with DashScope integration.
    """
    
    def __init__(self, dashscope_api_key=None, dashscope_base_url=None):
        """
        Initialize the Hukumonline legal document scraper.
        
        Args:
            dashscope_api_key (str, optional): DashScope API key
            dashscope_base_url (str, optional): DashScope API base URL
        """
        print("\nInitializing Hukumonline Scraper...")
        
        # Initialize attributes
        self.llm = None
        self.embeddings = None
        self.has_langchain = HAS_LANGCHAIN
        
        # Try to initialize DashScope
        if HAS_LANGCHAIN:
            try:
                # Load API key from environment if not provided
                if not dashscope_api_key:
                    dashscope_api_key = DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
                
                if not dashscope_base_url:
                    dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com")
                
                if dashscope_api_key:
                    # Initialize DashScope LLM
                    self.llm = ChatDashScope(
                        model="qwen2.5-72b-instruct",
                        dashscope_api_key=dashscope_api_key,
                        dashscope_api_base=dashscope_base_url,
                        temperature=0
                    )
                    
                    # Initialize DashScope Embeddings
                    self.embeddings = DashScopeEmbeddings(
                        model="text-embedding-v3",
                        dashscope_api_key=dashscope_api_key,
                        dashscope_api_base=dashscope_base_url
                    )
                    
                    print(f"{Fore.GREEN}DashScope LLM and Embeddings initialized successfully{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}DashScope API key not found{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error initializing DashScope: {str(e)}{Style.RESET_ALL}")
                logger.error(f"Error initializing DashScope: {str(e)}")
        
        # Initialize session with retry capability
        self.session = self._create_session_with_retry()
        
        print("\n" + "=" * 80)
        print("Hukumonline Scraper initialized successfully")
        print("=" * 80)
    
    def _create_session_with_retry(self):
        """
        Create a requests session with retry capability.
        
        Returns:
            requests.Session: Session with retry capability
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def search(self, query: str, max_results: int = 10) -> List[Document]:
        """
        Search for legal information on hukumonline.com.
        
        This method first tries to search using Google CSE, and if that fails or returns no results,
        it falls back to directly scraping the hukumonline.com search page.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to retrieve
            
        Returns:
            List[Document]: List of document objects with content and metadata
        """
        logger.info(f"Searching for: {query}")
        
        # First, try to use Google CSE for search
        try:
            logger.info("Attempting to search using Google CSE...")
            cse_documents = self.google_cse_search(query, num_results=max_results)
            
            if cse_documents:
                logger.info(f"Found {len(cse_documents)} documents via Google CSE")
                return cse_documents
            else:
                logger.info("No results from Google CSE, falling back to direct scraping")
        except Exception as e:
            logger.error(f"Error using Google CSE: {str(e)}")
            logger.info("Falling back to direct scraping")
        
        # If Google CSE failed or returned no results, fall back to direct scraping
        documents = []
        
        try:
            # Format the search URL
            search_url = f"https://www.hukumonline.com/search/a/c/q/{quote_plus(query)}/"
            logger.info(f"Searching at URL: {search_url}")
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            # Make the request
            response = self.session.get(search_url)
            if response.status_code != 200:
                logger.error(f"Failed to retrieve search results: {response.status_code}")
                # Try using headless browser for search if direct request fails
                if HAS_SELENIUM:
                    logger.info("Attempting to search using headless browser...")
                    return self._search_with_headless_browser(query, max_results)
                return documents
                
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result items
            result_items = soup.select('.search-result-item, .artikel-list-item, .list-artikel')
            
            if not result_items:
                logger.warning("No search results found or unexpected page structure")
                # Try using headless browser for search if no results found
                if HAS_SELENIUM:
                    logger.info("Attempting to search using headless browser...")
                    return self._search_with_headless_browser(query, max_results)
                return documents
                
            # Process each search result
            for i, item in enumerate(result_items):
                if i >= max_results:
                    break
                    
                try:
                    # Find the title and link
                    title_elem = item.select_one('h2 a, h3 a, .title a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.text.strip()
                    relative_url = title_elem.get('href', '')
                    
                    # Make the URL absolute
                    url = urljoin('https://www.hukumonline.com/', relative_url)
                    
                    logger.info(f"Found result: {title} ({url})")
                    
                    # Extract article content
                    article_content, article_metadata = self._extract_article_content(url)
                    
                    # If regular extraction fails, try headless browser
                    if not article_content and HAS_SELENIUM:
                        logger.info(f"Regular extraction failed for {url}, trying headless browser...")
                        article_content, article_metadata = self._extract_article_content_headless(url)
                    
                    if article_content:
                        # Create a document object
                        doc = Document(
                            content=article_content,
                            metadata={
                                "title": title,
                                "url": url,
                                "source": "hukumonline",
                                "query": query,
                                **article_metadata
                            }
                        )
                        
                        documents.append(doc)
                        logger.info(f"Added document: {title} ({len(article_content)} chars)")
                    else:
                        logger.warning(f"Could not extract content from {url}")
                        
                except Exception as item_error:
                    logger.error(f"Error processing search result: {str(item_error)}")
                    continue
                    
            return documents
            
        except Exception as e:
            logger.error(f"Error searching hukumonline: {str(e)}")
            # Try using headless browser as a last resort
            if HAS_SELENIUM:
                logger.info("Attempting to search using headless browser as last resort...")
                return self._search_with_headless_browser(query, max_results)
            return documents
    
    def _search_with_headless_browser(self, query: str, max_results: int = 10) -> List[Document]:
        """
        Search hukumonline.com using a headless browser.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to retrieve
            
        Returns:
            List[Document]: List of document objects with content and metadata
        """
        if not HAS_SELENIUM:
            logger.error("Selenium is not installed. Cannot use headless browser search.")
            return []
            
        driver = None
        documents = []
        
        try:
            logger.info(f"Searching with headless browser for: {query}")
            
            # Set up Chrome options for headless browsing
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Initialize the Chrome driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            # Navigate to the search URL
            search_url = f"https://www.hukumonline.com/search/a/c/q/{quote_plus(query)}/"
            driver.get(search_url)
            
            # Wait for the search results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Allow time for JavaScript to render content
            time.sleep(3)
            
            # Find search result items
            result_items = driver.find_elements(By.CSS_SELECTOR, '.search-result-item, .artikel-list-item, .list-artikel, .article-item')
            
            if not result_items:
                # Try alternative selectors
                result_items = driver.find_elements(By.CSS_SELECTOR, 'article, .article, .post, .search-item')
            
            if not result_items:
                logger.warning("No search results found with headless browser")
                return documents
                
            # Process each search result
            for i, item in enumerate(result_items):
                if i >= max_results:
                    break
                    
                try:
                    # Find the title and link
                    try:
                        title_elem = item.find_element(By.CSS_SELECTOR, 'h2 a, h3 a, .title a, a')
                        title = title_elem.text.strip()
                        url = title_elem.get_attribute('href')
                    except NoSuchElementException:
                        continue
                    
                    if not url or not title:
                        continue
                        
                    logger.info(f"Found result with headless browser: {title} ({url})")
                    
                    # Extract article content using headless browser
                    article_content, article_metadata = self._extract_article_content_headless(url)
                    
                    if article_content:
                        # Create a document object
                        doc = Document(
                            content=article_content,
                            metadata={
                                "title": title,
                                "url": url,
                                "source": "hukumonline_headless",
                                "query": query,
                                **article_metadata
                            }
                        )
                        
                        documents.append(doc)
                        logger.info(f"Added document with headless browser: {title} ({len(article_content)} chars)")
                    else:
                        logger.warning(f"Could not extract content from {url} with headless browser")
                        
                except Exception as item_error:
                    logger.error(f"Error processing search result with headless browser: {str(item_error)}")
                    continue
                    
            return documents
            
        except Exception as e:
            logger.error(f"Error in headless browser search: {str(e)}")
            return documents
        finally:
            # Clean up the driver
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def _extract_article_content(self, article_url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content and metadata from a hukumonline article.
        
        Args:
            article_url (str): URL of the article
            
        Returns:
            tuple: (content, metadata) or (None, None) if extraction fails
        """
        try:
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(
                article_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to retrieve article: {response.status_code}")
                return "", {}
                
            # Parse article
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract metadata
            metadata = {}
            
            # Get publication date
            date_element = soup.select_one("div.date")
            if date_element:
                metadata["date"] = date_element.text.strip()
            
            # Get author
            author_element = soup.select_one("div.author")
            if author_element:
                metadata["author"] = author_element.text.strip()
            
            # Get categories/tags
            tags = []
            tag_elements = soup.select("div.tags a")
            for tag in tag_elements:
                tags.append(tag.text.strip())
            
            if tags:
                metadata["tags"] = tags
            
            # Extract article content
            content_element = soup.select_one("div.content-text")
            if not content_element:
                return "", metadata
            
            # Remove unwanted elements
            for unwanted in content_element.select("script, style, iframe, div.ads"):
                unwanted.decompose()
            
            # Get clean text
            content = content_element.get_text(separator="\n").strip()
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting article content: {str(e)}")
            return "", {}
    
    def _extract_article_content_headless(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract article content using a headless browser to handle JavaScript-rendered content.
        
        Args:
            url (str): URL of the article to scrape
            
        Returns:
            tuple: (content, metadata) or ("", {}) if extraction fails
        """
        if not HAS_SELENIUM:
            logger.error("Selenium is not installed. Cannot use headless browser scraping.")
            return "", {}
            
        driver = None
        try:
            logger.info(f"Extracting content from {url} using headless browser")
            
            # Set up Chrome options for headless browsing
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Initialize the Chrome driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            # Navigate to the URL
            driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Allow time for JavaScript to render content
            time.sleep(3)
            
            # Extract metadata
            metadata = {
                "url": url,
                "source": "hukumonline_headless",
                "extracted_date": datetime.now().isoformat()
            }
            
            # Try to get the title
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, "h1.title, .article-title, .post-title")
                metadata["title"] = title_element.text.strip()
            except NoSuchElementException:
                try:
                    metadata["title"] = driver.title
                except:
                    metadata["title"] = "Untitled"
            
            # Try to get the publication date
            try:
                date_element = driver.find_element(By.CSS_SELECTOR, ".date, .post-date, .article-date, time")
                metadata["publication_date"] = date_element.text.strip()
            except NoSuchElementException:
                pass
            
            # Try to get the author
            try:
                author_element = driver.find_element(By.CSS_SELECTOR, ".author, .post-author, .article-author")
                metadata["author"] = author_element.text.strip()
            except NoSuchElementException:
                pass
            
            # Extract the main content
            content = ""
            try:
                # Try different selectors for the main content
                content_selectors = [
                    ".article-content", 
                    ".post-content", 
                    ".entry-content", 
                    "article", 
                    ".content-detail",
                    "#content-area"
                ]
                
                for selector in content_selectors:
                    try:
                        content_element = driver.find_element(By.CSS_SELECTOR, selector)
                        content = content_element.text.strip()
                        if content:
                            break
                    except NoSuchElementException:
                        continue
                
                # If no content found with selectors, get the body content
                if not content:
                    body_element = driver.find_element(By.TAG_NAME, "body")
                    
                    # Remove unwanted elements
                    for unwanted in driver.find_elements(By.CSS_SELECTOR, "nav, header, footer, .sidebar, .comments, script, style"):
                        driver.execute_script("arguments[0].remove();", unwanted)
                    
                    content = body_element.text.strip()
            except Exception as e:
                logger.error(f"Error extracting content: {str(e)}")
                content = ""
            
            # Clean up the content
            if content:
                # Remove excessive whitespace
                content = re.sub(r'\s+', ' ', content)
                # Remove any script content that might have been captured
                content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
                # Remove any style content
                content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
            
            return content, metadata
            
        except TimeoutException:
            logger.error(f"Timeout while loading {url}")
            return "", {}
        except Exception as e:
            logger.error(f"Error in headless browser scraping: {str(e)}")
            return "", {}
        finally:
            # Clean up the driver
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def extract_legal_references(self, text: str) -> List[str]:
        """
        Extract legal references from text.
        
        Args:
            text (str): Text to extract references from
            
        Returns:
            List[str]: List of legal references
        """
        references = []
        
        # Define patterns for different types of legal references
        patterns = [
            r"UU (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",  # UU No. 1/2020 or UU Nomor 1 Tahun 2020
            r"PP (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",  # PP No. 1/2020 or PP Nomor 1 Tahun 2020
            r"Peraturan Pemerintah (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",
            r"Perpres (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",
            r"Peraturan Presiden (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",
            r"Permen[a-zA-Z]+ (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",
            r"Peraturan Menteri [a-zA-Z]+ (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",
            r"Perma (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?",
            r"Peraturan Mahkamah Agung (?:No\.|Nomor) \d+(?:/\d+)? (?:Tahun \d{4})?"
        ]
        
        # Find all matches for each pattern
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                reference = match.group(0)
                if reference not in references:
                    references.append(reference)
        
        return references
    
    def summarize_document(self, document: Document) -> str:
        """
        Summarize a legal document using LangChain and DashScope.
        
        Args:
            document (Document): Document to summarize
            
        Returns:
            str: Summarized content
        """
        if not self.has_langchain or not self.llm:
            logger.warning("LangChain or LLM not available for summarization")
            return document.content[:500] + "..."  # Return truncated content as fallback
        
        try:
            # Convert to LangChain document format
            lc_doc = document.to_langchain_document()
            
            # Create text splitter for long documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200
            )
            
            # Split document if it's too long
            if len(lc_doc.page_content) > 4000:
                docs = text_splitter.split_documents([lc_doc])
            else:
                docs = [lc_doc]
            
            # Define summarization prompts
            map_prompt_template = """
            Berikut adalah bagian dari dokumen hukum:
            {text}
            
            Berikan ringkasan penting dari bagian ini, fokus pada:
            1. Ketentuan hukum utama
            2. Definisi penting
            3. Kewajiban atau hak yang diatur
            4. Sanksi atau konsekuensi yang disebutkan
            
            Ringkasan:
            """
            
            combine_prompt_template = """
            Berikut adalah kumpulan ringkasan dari berbagai bagian dokumen hukum:
            {text}
            
            Berikan ringkasan komprehensif yang menggabungkan informasi-informasi tersebut.
            Fokus pada:
            1. Tujuan utama dari dokumen hukum
            2. Ketentuan-ketentuan penting
            3. Implikasi praktis
            4. Hubungan dengan peraturan lain (jika disebutkan)
            
            Ringkasan Komprehensif:
            """
            
            # Create prompts
            map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
            combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
            
            # Create and run the summarization chain
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=False
            )
            
            # Run the chain
            summary = chain.run(docs)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            # Return a portion of the original content as fallback
            return document.content[:500] + "..."
    
    def process_legal_query(self, query: str, max_documents: int = 5) -> Dict[str, Any]:
        """
        Process a legal query using LangChain and DashScope.
        
        Args:
            query (str): User's legal query
            max_documents (int): Maximum number of documents to retrieve
            
        Returns:
            Dict[str, Any]: Results including documents and answer
        """
        try:
            # Search for relevant documents
            documents = self.search(query, max_results=max_documents)
            
            # Extract legal references from all documents
            all_references = []
            for doc in documents:
                references = self.extract_legal_references(doc.content)
                all_references.extend(references)
            
            # Remove duplicates
            all_references = list(set(all_references))
            
            # Generate answer using LangChain if available
            if self.has_langchain and self.llm and self.embeddings and documents:
                try:
                    # Convert documents to LangChain format
                    lc_docs = [doc.to_langchain_document() for doc in documents]
                    
                    # Create vector store
                    vectorstore = FAISS.from_documents(lc_docs, self.embeddings)
                    
                    # Create retrieval QA chain
                    qa_prompt_template = """
                    Kamu adalah asisten hukum yang ahli dalam hukum Indonesia. Gunakan konteks berikut untuk menjawab pertanyaan.
                    
                    Konteks:
                    {context}
                    
                    Pertanyaan: {question}
                    
                    Berikan jawaban yang komprehensif, akurat, dan berdasarkan konteks yang diberikan. Jika jawabannya tidak ada dalam konteks, katakan bahwa kamu tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut.
                    
                    Jawaban:
                    """
                    
                    qa_prompt = PromptTemplate(
                        template=qa_prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(),
                        chain_type_kwargs={"prompt": qa_prompt}
                    )
                    
                    # Run the chain
                    answer = qa_chain.run(query)
                    
                except Exception as e:
                    logger.error(f"Error in QA chain: {str(e)}")
                    # Fallback to simple summarization
                    answer = self._generate_fallback_response(query, documents)
            else:
                # Fallback to simple response
                answer = self._generate_fallback_response(query, documents)
            
            # Prepare result
            result = {
                "query": query,
                "documents": [doc.to_dict() for doc in documents],
                "legal_references": all_references,
                "answer": answer
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing legal query: {str(e)}")
            return {
                "query": query,
                "documents": [],
                "legal_references": [],
                "answer": f"Error processing query: {str(e)}"
            }
    
    def _generate_fallback_response(self, query: str, documents: List[Document]) -> str:
        """
        Generate a fallback response when LangChain processing fails.
        
        Args:
            query (str): The original query
            documents (list): List of retrieved documents
            
        Returns:
            str: A simple response based on the documents
        """
        if not documents:
            return "Maaf, tidak ditemukan dokumen yang relevan dengan pertanyaan Anda."
        
        # Get the first document as the most relevant
        main_doc = documents[0]
        
        # Create a simple response
        response = f"Berdasarkan pencarian untuk '{query}', ditemukan informasi berikut:\n\n"
        response += f"Dari artikel '{main_doc.metadata.get('title', 'Untitled')}':\n"
        
        # Add a snippet from the document
        content_preview = main_doc.content[:500] + "..." if len(main_doc.content) > 500 else main_doc.content
        response += content_preview + "\n\n"
        
        # Add references to other documents
        if len(documents) > 1:
            response += "Sumber informasi lainnya:\n"
            for i, doc in enumerate(documents[1:], 1):
                response += f"{i}. {doc.metadata.get('title', 'Untitled')}\n"
        
        # Add disclaimer
        response += "\nUntuk informasi lebih lengkap, silakan kunjungi sumber aslinya di hukumonline.com."
        
        return response
    
    def _extract_pdf_content(self, pdf_url: str, title: str = "PDF Document") -> Tuple[str, Dict[str, Any]]:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_url (str): URL of the PDF file
            title (str, optional): Title of the document
            
        Returns:
            tuple: (content, metadata) or (None, None) if extraction fails
        """
        try:
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 2))
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Download PDF file to temporary location
            response = self.session.get(pdf_url, headers=headers, stream=True)
            if response.status_code != 200:
                logger.warning(f"Failed to download PDF: {response.status_code}")
                return "", {}
                
            # Create a temporary file to store the PDF
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_path = temp_file.name
                # Write PDF content to file
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        temp_file.write(chunk)
            
            # Extract text from PDF
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(temp_path)
                
                # Get document info
                info = reader.metadata
                print(f"{Fore.CYAN}PDF Metadata: {info}{Style.RESET_ALL}")
                
                # Extract text from each page
                text = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                # Create metadata
                metadata = {
                    "title": title,
                    "source": "pdf",
                    "url": pdf_url,
                    "pages": len(reader.pages),
                    "extracted_date": datetime.now().isoformat()
                }
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return text, metadata
                
            except Exception as e:
                logger.error(f"Error extracting PDF content: {str(e)}")
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return "", {}
                
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return "", {}
    
    def google_cse_search(self, query: str, num_results: int = 5) -> List[Document]:
        """
        Search for legal information using Google Custom Search Engine (CSE).
        
        Args:
            query (str): The search query
            num_results (int): Maximum number of results to retrieve
            
        Returns:
            List[Document]: List of document objects with content and metadata
        """
        try:
            if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
                logger.error("Google API Key or CSE ID not configured")
                return []
                
            logger.info(f"Searching Google CSE for: {query}")
            
            # Construct the Google CSE API URL
            base_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query + " site:hukumonline.com",  # Restrict to hukumonline.com
                "num": num_results
            }
            
            # Make the API request
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Check if we have search results
            if "items" not in response.json():
                logger.warning("No search results found")
                return []
                
            # Process search results
            documents = []
            
            for item in response.json()["items"]:
                try:
                    title = item.get("title", "Untitled")
                    url = item.get("link", "")
                    snippet = item.get("snippet", "")
                    
                    logger.info(f"Processing search result: {title} ({url})")
                    
                    # Skip if no URL
                    if not url:
                        continue
                        
                    # Check if it's a PDF
                    if url.lower().endswith(".pdf"):
                        # Extract PDF content
                        pdf_content, pdf_metadata = self._extract_pdf_content(url, title=title)
                        
                        if pdf_content:
                            pdf_document = Document(
                                content=pdf_content,
                                metadata={
                                    "title": title,
                                    "url": url,
                                    "source": "google_cse_pdf",
                                    "snippet": snippet,
                                    "query": query,
                                    **pdf_metadata
                                }
                            )
                            documents.append(pdf_document)
                            logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
                    else:
                        # Scrape HTML content
                        article_content, article_metadata = self._extract_article_content(url)
                        
                        if article_content:
                            html_document = Document(
                                content=article_content,
                                metadata={
                                    "title": title,
                                    "url": url,
                                    "source": "google_cse_html",
                                    "snippet": snippet,
                                    "query": query,
                                    **article_metadata
                                }
                            )
                            
                            documents.append(html_document)
                            logger.info(f"Added HTML document: {title} ({len(article_content)} chars)")
                        else:
                            logger.warning(f"Could not extract content from {url}")
                except Exception as item_error:
                    logger.error(f"Error processing search result: {str(item_error)}")
                    
            return documents
        except Exception as e:
            logger.error(f"Error in Google CSE search: {str(e)}")
            return []
    
    def extract_legal_basis_pdfs(self, article_url: str) -> List[Dict[str, Any]]:
        """
        Extract PDFs from the "Dasar Hukum" section of a Hukumonline article.
        
        Args:
            article_url (str): URL of the article to scrape
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing PDF information
        """
        pdf_documents = []
        
        try:
            logger.info(f"Extracting legal basis PDFs from {article_url}")
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            # Make the request
            response = self.session.get(article_url)
            if response.status_code != 200:
                logger.error(f"Failed to retrieve article: {response.status_code}")
                return pdf_documents
                
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for "Dasar Hukum" section
            dasar_hukum_section = None
            
            # Try different approaches to find the "Dasar Hukum" section
            
            # 1. Look for headings that contain "Dasar Hukum"
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if 'Dasar Hukum' in heading.text:
                    dasar_hukum_section = heading
                    logger.info(f"Found 'Dasar Hukum' in heading: {heading.name}")
                    break
            
            # 2. Look for strong or b tags containing "Dasar Hukum"
            if not dasar_hukum_section:
                for tag in soup.find_all(['strong', 'b']):
                    if 'Dasar Hukum' in tag.text:
                        dasar_hukum_section = tag
                        logger.info(f"Found 'Dasar Hukum' in tag: {tag.name}")
                        break
            
            # 3. Look for divs or sections with class or id containing "dasar-hukum"
            if not dasar_hukum_section:
                for tag in soup.find_all(class_=lambda c: c and 'dasar' in c.lower()):
                    dasar_hukum_section = tag
                    logger.info(f"Found element with 'dasar' in class: {tag.name}")
                    break
            
            # If we found the section, look for links in it or after it
            if dasar_hukum_section:
                logger.info("Found 'Dasar Hukum' section")
                
                # Get the parent element to search within
                parent = dasar_hukum_section.parent
                
                # Look for links in the parent element
                links = parent.find_all('a')
                
                # If no links found in parent, try to find links in the next sibling elements
                if not links:
                    sibling = dasar_hukum_section.find_next_sibling()
                    while sibling and not links and len(links) == 0:
                        links = sibling.find_all('a')
                        sibling = sibling.find_next_sibling()
                
                # If still no links, look in the entire article content after the dasar_hukum_section
                if not links:
                    # Find the index of the dasar_hukum_section in its parent's children
                    parent_children = list(parent.children)
                    try:
                        section_index = parent_children.index(dasar_hukum_section)
                        # Get all elements after the section
                        for elem in parent_children[section_index:]:
                            if hasattr(elem, 'find_all'):
                                links.extend(elem.find_all('a'))
                    except ValueError:
                        pass
                
                # Process each link
                for link in links:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Make the URL absolute
                    url = urljoin(article_url, href)
                    
                    # Check if it's a PDF
                    if url.lower().endswith('.pdf'):
                        title = link.text.strip() or "PDF Document"
                        logger.info(f"Found PDF link: {title} ({url})")
                        
                        # Extract PDF content
                        pdf_content, pdf_metadata = self._extract_pdf_content(url, title=title)
                        
                        if pdf_content:
                            pdf_document = {
                                "title": title,
                                "url": url,
                                "content": pdf_content,
                                "metadata": {
                                    "source": "dasar_hukum_pdf",
                                    "article_url": article_url,
                                    **pdf_metadata
                                }
                            }
                            
                            pdf_documents.append(pdf_document)
                            logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
            
            # If we didn't find any PDFs in the "Dasar Hukum" section, look for any PDFs in the article
            if not pdf_documents:
                logger.info("No PDFs found in 'Dasar Hukum' section, looking for any PDFs in the article")
                
                # Find all links in the article
                links = soup.find_all('a')
                
                # Process each link
                for link in links:
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Make the URL absolute
                    url = urljoin(article_url, href)
                    
                    # Check if it's a PDF or points to a peraturan.go.id URL
                    if url.lower().endswith('.pdf') or 'peraturan.go.id' in url.lower():
                        title = link.text.strip() or "Legal Document"
                        logger.info(f"Found document link: {title} ({url})")
                        
                        # If it's a PDF, extract content
                        if url.lower().endswith('.pdf'):
                            pdf_content, pdf_metadata = self._extract_pdf_content(url, title=title)
                            
                            if pdf_content:
                                pdf_document = {
                                    "title": title,
                                    "url": url,
                                    "content": pdf_content,
                                    "metadata": {
                                        "source": "article_pdf",
                                        "article_url": article_url,
                                        **pdf_metadata
                                    }
                                }
                                
                                pdf_documents.append(pdf_document)
                                logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
            
            return pdf_documents
            
        except Exception as e:
            logger.error(f"Error extracting legal basis PDFs: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pdf_documents


# Specialized functions for scraping PDFs from hukumonline.com
def find_hukumonline_pdf_links(url):
    """
    Find PDF links on hukumonline.com pages, specifically looking for links with class="css-140jb81".
    When these links are found, follow them to get the actual PDF URL.
    
    Args:
        url (str): URL of the hukumonline.com page to search
        
    Returns:
        list: List of dictionaries containing URL and text of PDF links
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        print(f"{Fore.CYAN}Searching for PDF links on hukumonline.com: {url}{Style.RESET_ALL}")
        
        # Headers for the request
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Send request to the URL
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links with class="css-140jb81" which are likely PDF links
        pdf_links = []
        pdf_class_links = soup.find_all('a', class_='css-140jb81')
        
        if pdf_class_links:
            print(f"{Fore.GREEN}Found {len(pdf_class_links)} links with class='css-140jb81'{Style.RESET_ALL}")
            for a in pdf_class_links:
                link_url = a['href']
                link_text = a.get_text().strip()
                
                # Make the URL absolute
                absolute_url = urljoin(url, link_url)
                print(f"{Fore.CYAN}Following link: {absolute_url}{Style.RESET_ALL}")
                
                # Follow the link to get the actual PDF URL (simulating a click)
                try:
                    follow_response = requests.get(absolute_url, headers=headers, timeout=30)
                    follow_response.raise_for_status()
                    
                    # Parse the resulting page
                    follow_soup = BeautifulSoup(follow_response.text, 'html.parser')
                    
                    # Look for PDF download links on the page
                    pdf_found = False
                    
                    # Method 1: Look for download buttons
                    download_buttons = follow_soup.find_all('a', class_='btn-download')
                    if download_buttons:
                        for button in download_buttons:
                            button_url = button.get('href')
                            if button_url and (button_url.lower().endswith('.pdf') or 'download' in button_url.lower()):
                                pdf_url = urljoin(absolute_url, button_url)
                                print(f"{Fore.GREEN}Found PDF download button: {pdf_url}{Style.RESET_ALL}")
                                pdf_links.append({
                                    'url': pdf_url,
                                    'text': link_text or pdf_url,
                                    'source': 'hukumonline',
                                    'original_link': absolute_url
                                })
                                pdf_found = True
                                break
                    
                    # Method 2: Look for iframe with PDF source
                    if not pdf_found:
                        iframes = follow_soup.find_all('iframe')
                        for iframe in iframes:
                            iframe_src = iframe.get('src')
                            if iframe_src and (iframe_src.lower().endswith('.pdf') or 'pdf' in iframe_src.lower()):
                                pdf_url = urljoin(absolute_url, iframe_src)
                                print(f"{Fore.GREEN}Found PDF in iframe: {pdf_url}{Style.RESET_ALL}")
                                pdf_links.append({
                                    'url': pdf_url,
                                    'text': link_text or pdf_url,
                                    'source': 'hukumonline',
                                    'original_link': absolute_url
                                })
                                pdf_found = True
                                break
                    
                    # Method 3: Look for any link that might be a PDF
                    if not pdf_found:
                        for link in follow_soup.find_all('a', href=True):
                            link_href = link.get('href')
                            if link_href and (link_href.lower().endswith('.pdf') or 'download' in link_href.lower()):
                                pdf_url = urljoin(absolute_url, link_href)
                                print(f"{Fore.GREEN}Found potential PDF link: {pdf_url}{Style.RESET_ALL}")
                                pdf_links.append({
                                    'url': pdf_url,
                                    'text': link_text or pdf_url,
                                    'source': 'hukumonline',
                                    'original_link': absolute_url
                                })
                                pdf_found = True
                                break
                    
                    # If no PDF link found, add the original link
                    if not pdf_found:
                        print(f"{Fore.YELLOW}No PDF link found on page, using original link{Style.RESET_ALL}")
                        pdf_links.append({
                            'url': absolute_url,
                            'text': link_text or absolute_url,
                            'source': 'hukumonline',
                            'original_link': absolute_url
                        })
                
                except Exception as e:
                    print(f"{Fore.RED}Error following link {absolute_url}: {str(e)}{Style.RESET_ALL}")
                    # Add the original link if we couldn't follow it
                    pdf_links.append({
                        'url': absolute_url,
                        'text': link_text or absolute_url,
                        'source': 'hukumonline',
                        'original_link': absolute_url
                    })
        else:
            print(f"{Fore.YELLOW}No links with class='css-140jb81' found{Style.RESET_ALL}")
            
            # If no specific class links found, try to find any links that might be PDFs
            for a in soup.find_all('a', href=True):
                link_url = a['href']
                if link_url.lower().endswith('.pdf'):
                    link_text = a.get_text().strip()
                    absolute_url = urljoin(url, link_url)
                    pdf_links.append({
                        'url': absolute_url,
                        'text': link_text or absolute_url,
                        'source': 'hukumonline',
                        'original_link': None
                    })
        
        print(f"{Fore.GREEN}Found {len(pdf_links)} potential PDF links{Style.RESET_ALL}")
        return pdf_links
        
    except Exception as e:
        print(f"{Fore.RED}Error finding PDF links on hukumonline.com: {str(e)}{Style.RESET_ALL}")
        return []

def download_hukumonline_pdf(pdf_url, original_link=None):
    """
    Download a PDF file from hukumonline.com.
    
    Args:
        pdf_url (str): URL of the PDF to download
        original_link (str, optional): Original link where the PDF was found
        
    Returns:
        bytes: The PDF content as bytes or None if download failed
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        print(f"{Fore.CYAN}Downloading PDF from hukumonline.com: {pdf_url}{Style.RESET_ALL}")
        
        # Set up headers with referer if original link is provided
        headers = {
            'User-Agent': user_agent,
            'Accept': 'application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        if original_link:
            headers['Referer'] = original_link
        else:
            headers['Referer'] = 'https://www.hukumonline.com/'
        
        # Create httpx client with custom settings
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers=headers
        ) as client:
            # Send GET request to download the PDF
            response = client.get(pdf_url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Check if the content is a PDF
                content_type = response.headers.get('Content-Type', '')
                content_length = int(response.headers.get('Content-Length', '0') or '0')
                
                if 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf'):
                    print(f"{Fore.GREEN}Successfully downloaded PDF ({len(response.content)} bytes){Style.RESET_ALL}")
                    return response.content
                elif content_length > 0:
                    # Check if the content might be a PDF despite the content type
                    content = response.content
                    if content.startswith(b'%PDF-'):
                        print(f"{Fore.GREEN}Content appears to be PDF despite Content-Type: {content_type}{Style.RESET_ALL}")
                        return content
                    else:
                        print(f"{Fore.YELLOW}Downloaded content is not a PDF (Content-Type: {content_type}){Style.RESET_ALL}")
                        # Try to return content anyway, might be a PDF with wrong content type
                        return content
                else:
                    print(f"{Fore.YELLOW}Downloaded content is not a PDF (Content-Type: {content_type}){Style.RESET_ALL}")
                    # Try to return content anyway, might be a PDF with wrong content type
                    return response.content
            else:
                print(f"{Fore.RED}Failed to download PDF: HTTP {response.status_code}{Style.RESET_ALL}")
                return None
                
    except Exception as e:
        print(f"{Fore.RED}Error downloading PDF from hukumonline.com: {str(e)}{Style.RESET_ALL}")
        return None

def search_hukumonline(query, max_results=5):
    """
    Search for content on hukumonline.com using a query.
    
    Args:
        query (str): The search query
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        
    Returns:
        list: List of dictionaries containing search results with title, url, and snippet
    """
    try:
        # Format the query for URL
        encoded_query = urllib.parse.quote(query)
        
        # Create the search URL
        search_url = f"https://www.hukumonline.com/search/a/c/q/{encoded_query}"
        
        print(f"{Fore.CYAN}Searching hukumonline.com with URL: {search_url}{Style.RESET_ALL}")
        
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        # Headers for the request
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Send request to the search URL
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find search results
        results = []
        
        # Look for search result containers
        search_results = soup.find_all('div', class_='search-result-item')
        
        if search_results:
            print(f"{Fore.GREEN}Found {len(search_results)} search results{Style.RESET_ALL}")
            
            for result in search_results[:max_results]:
                try:
                    # Extract title and URL
                    title_elem = result.find('h3')
                    if title_elem:
                        title = title_elem.get_text().strip()
                        url_elem = title_elem.find('a')
                        if url_elem and 'href' in url_elem.attrs:
                            url = url_elem['href']
                            # Make the URL absolute
                            url = urljoin(search_url, url)
                        else:
                            url = ""
                    else:
                        title = "Untitled"
                        url = ""
                    
                    # Extract snippet
                    snippet_elem = result.find('div', class_='search-result-snippet')
                    snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                    
                    # Add to results
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
                except Exception as e:
                    print(f"{Fore.RED}Error extracting search result: {str(e)}{Style.RESET_ALL}")
        else:
            # Alternative search result format
            search_results = soup.find_all('div', class_='article-item')
            
            if search_results:
                print(f"{Fore.GREEN}Found {len(search_results)} article items{Style.RESET_ALL}")
                
                for result in search_results[:max_results]:
                    try:
                        # Extract title and URL
                        title_elem = result.find('h3', class_='article-title')
                        if title_elem:
                            title = title_elem.get_text().strip()
                            url_elem = title_elem.find('a')
                            if url_elem and 'href' in url_elem.attrs:
                                url = url_elem['href']
                                # Make the URL absolute
                                url = urljoin(search_url, url)
                            else:
                                url = ""
                        else:
                            title = "Untitled"
                            url = ""
                        
                        # Extract snippet
                        snippet_elem = result.find('div', class_='article-snippet')
                        snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                        
                        # Add to results
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                    except Exception as e:
                        print(f"{Fore.RED}Error extracting article item: {str(e)}{Style.RESET_ALL}")
            else:
                # Try one more format - pusat data format
                search_results = soup.find_all('div', class_='css-1dbjc4n')
                
                if search_results:
                    print(f"{Fore.GREEN}Found {len(search_results)} pusat data items{Style.RESET_ALL}")
                    
                    for result in search_results[:max_results]:
                        try:
                            # Extract title and URL
                            title_elem = result.find('div', class_='css-901oao')
                            if title_elem:
                                title = title_elem.get_text().strip()
                                url_elem = result.find('a')
                                if url_elem and 'href' in url_elem.attrs:
                                    url = url_elem['href']
                                    # Make the URL absolute
                                    url = urljoin(search_url, url)
                                else:
                                    url = ""
                            else:
                                title = "Untitled"
                                url = ""
                            
                            # Extract snippet (may not be available)
                            snippet = ""
                            
                            # Add to results
                            results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet
                            })
                        except Exception as e:
                            print(f"{Fore.RED}Error extracting pusat data item: {str(e)}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No search results found with standard parsing{Style.RESET_ALL}")
                    
                    # If all else fails, look for any links that might be relevant
                    links = soup.find_all('a')
                    relevant_links = []
                    
                    for link in links:
                        href = link.get('href', '')
                        text = link.get_text().strip()
                        
                        # Check if the link might be a search result
                        if (query.lower() in text.lower() or 
                            'pusatdata' in href or 
                            'detail' in href or 
                            'berita' in href):
                            
                            # Make the URL absolute
                            url = urljoin(search_url, href)
                            
                            # Add to relevant links
                            relevant_links.append({
                                'title': text or "Untitled",
                                'url': url,
                                'snippet': ""
                            })
                    
                    # Take the top results
                    results.extend(relevant_links[:max_results])
                    
                    if relevant_links:
                        print(f"{Fore.GREEN}Found {len(relevant_links)} potentially relevant links{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}No relevant links found{Style.RESET_ALL}")
                        
                        # Print part of the HTML to help debug
                        print(f"{Fore.YELLOW}HTML snippet for debugging:{Style.RESET_ALL}")
                        print(soup.prettify()[:1000])  # Print first 1000 chars of HTML
        
        # Filter out any results with empty URLs
        results = [r for r in results if r['url']]
        
        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()
        
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        print(f"{Fore.GREEN}Returning {len(unique_results)} unique search results{Style.RESET_ALL}")
        return unique_results
        
    except Exception as e:
        print(f"{Fore.RED}Error searching hukumonline.com: {str(e)}{Style.RESET_ALL}")
        return []

def test_hukumonline_pdf_scraping(url=None, save_pdf=False, output_dir=None):
    """
    Test function for scraping PDFs from hukumonline.com.
    
    Args:
        url (str, optional): URL of the hukumonline.com page to scrape. If None, a default URL will be used.
        save_pdf (bool, optional): Whether to save the downloaded PDF to a file.
        output_dir (str, optional): Directory to save the PDF to. If None, the current directory will be used.
        
    Returns:
        dict: Dictionary containing the results of the test
    """
    try:
        if not url:
            # Default URL to test
            url = "https://www.hukumonline.com/pusatdata/detail/lt5f9a1e86a069c/undang-undang-nomor-11-tahun-2020"
        
        print(f"{Fore.CYAN}Testing PDF scraping for URL: {url}{Style.RESET_ALL}")
        
        # Create output directory if it doesn't exist
        if save_pdf and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find PDF links
        start_time = time.time()
        pdf_links = find_hukumonline_pdf_links(url)
        find_time = time.time() - start_time
        
        if not pdf_links:
            print(f"{Fore.RED}No PDF links found. Test failed.{Style.RESET_ALL}")
            return {
                'success': False,
                'error': 'No PDF links found',
                'url': url,
                'find_time': find_time,
                'links_found': 0
            }
        
        print(f"{Fore.GREEN}Found {len(pdf_links)} PDF links in {find_time:.2f} seconds{Style.RESET_ALL}")
        
        # Process each PDF link
        results = []
        for i, link in enumerate(pdf_links, 1):
            print(f"\n{Fore.YELLOW}Processing PDF link {i}/{len(pdf_links)}: {link['text']}{Style.RESET_ALL}")
            print(f"URL: {link['url']}")
            
            # Download the PDF
            start_time = time.time()
            pdf_content = download_hukumonline_pdf(link['url'], original_link=link.get('original_link'))
            download_time = time.time() - start_time
            
            if not pdf_content:
                print(f"{Fore.RED}Failed to download PDF. Skipping.{Style.RESET_ALL}")
                results.append({
                    'success': False,
                    'error': 'Failed to download PDF',
                    'url': link['url'],
                    'text': link['text'],
                    'download_time': download_time
                })
                continue
            
            # Extract text from the PDF
            start_time = time.time()
            try:
                # Create a PDF reader object
                pdf_file = io.BytesIO(pdf_content)
                pdf_reader = PdfReader(pdf_file)
                
                # Get document info
                info = pdf_reader.metadata
                print(f"{Fore.CYAN}PDF Metadata: {info}{Style.RESET_ALL}")
                
                # Extract text from each page
                text = ""
                num_pages = len(pdf_reader.pages)
                print(f"{Fore.CYAN}Extracting text from PDF ({num_pages} pages){Style.RESET_ALL}")
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"--- Page {i+1} ---\n{page_text}\n\n"
                        
                        # Print progress for large PDFs
                        if i % 10 == 0 and i > 0:
                            print(f"{Fore.CYAN}Processed {i}/{num_pages} pages{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error extracting text from page {i+1}: {str(e)}{Style.RESET_ALL}")
                
                extract_time = time.time() - start_time
                print(f"{Fore.GREEN}Successfully extracted {len(text)} characters from PDF in {extract_time:.2f} seconds{Style.RESET_ALL}")
                
                # Save the PDF to a file if requested
                if save_pdf:
                    try:
                        # Generate a filename from the link text or URL
                        if link['text'] and link['text'] != link['url']:
                            filename = re.sub(r'[^\w\s-]', '', link['text'])
                            filename = re.sub(r'[\s-]+', '_', filename)
                        else:
                            filename = os.path.basename(link['url'])
                            if not filename.lower().endswith('.pdf'):
                                filename += '.pdf'
                        
                        # Add output directory if provided
                        if output_dir:
                            filepath = os.path.join(output_dir, filename)
                        else:
                            filepath = filename
                        
                        # Write the PDF content to a file
                        with open(filepath, 'wb') as f:
                            f.write(pdf_content)
                        
                        print(f"{Fore.GREEN}Saved PDF to: {filepath}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}Error saving PDF to file: {str(e)}{Style.RESET_ALL}")
                
                # Add result to list
                results.append({
                    'success': True,
                    'url': link['url'],
                    'text': link['text'],
                    'download_time': download_time,
                    'extract_time': extract_time,
                    'num_pages': num_pages,
                    'text_length': len(text),
                    'metadata': str(info),
                    'text_preview': text[:500] + ('...' if len(text) > 500 else '')
                })
                
            except Exception as e:
                print(f"{Fore.RED}Error extracting text from PDF: {str(e)}{Style.RESET_ALL}")
                results.append({
                    'success': False,
                    'error': f'Error extracting text from PDF: {str(e)}',
                    'url': link['url'],
                    'text': link['text'],
                    'download_time': download_time
                })
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{Fore.GREEN}Test completed. Successfully processed {successful}/{len(results)} PDFs.{Style.RESET_ALL}")
        
        return {
            'success': successful > 0,
            'url': url,
            'find_time': find_time,
            'links_found': len(pdf_links),
            'links_processed': len(results),
            'successful_links': successful,
            'results': results
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error testing PDF scraping: {str(e)}{Style.RESET_ALL}")
        return {
            'success': False,
            'error': f'Error testing PDF scraping: {str(e)}',
            'url': url
        }

# Add this block to allow direct testing when the script is run
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Hukumonline.com PDF scraper')
    parser.add_argument('url', nargs='?', help='URL of the hukumonline.com page to scrape')
    parser.add_argument('--save', action='store_true', help='Save downloaded PDFs to files')
    parser.add_argument('--output-dir', default='./pdf_downloads', help='Directory to save PDFs to')
    parser.add_argument('--search', help='Search query for hukumonline.com')
    
    args = parser.parse_args()
    
    if args.search:
        print(f"{Fore.CYAN}Searching hukumonline.com for: {args.search}{Style.RESET_ALL}")
        results = search_hukumonline(args.search)
        if results:
            print(f"{Fore.GREEN}Found {len(results)} results:{Style.RESET_ALL}")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   Snippet: {result['snippet'][:100]}...")
                print()
            
            # Ask if user wants to scrape PDFs from the first result
            if args.url is None and len(results) > 0:
                args.url = results[0]['url']
                print(f"{Fore.YELLOW}Using first result URL for PDF scraping: {args.url}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}No search results found.{Style.RESET_ALL}")
            if args.url is None:
                sys.exit(1)
    
    if args.url:
        # Test PDF scraping
        test_results = test_hukumonline_pdf_scraping(args.url, save_pdf=args.save, output_dir=args.output_dir)
        
        # Print a summary of the results
        if test_results['success']:
            print(f"\n{Fore.GREEN}PDF scraping test successful!{Style.RESET_ALL}")
            print(f"Found {test_results['links_found']} PDF links")
            print(f"Successfully processed {test_results['successful_links']}/{test_results['links_processed']} PDFs")
            
            # Print a preview of the first successful result
            successful_results = [r for r in test_results['results'] if r['success']]
            if successful_results:
                first_result = successful_results[0]
                print(f"\n{Fore.CYAN}Preview of first successful result:{Style.RESET_ALL}")
                print(f"Title: {first_result['text']}")
                print(f"URL: {first_result['url']}")
                print(f"Pages: {first_result['num_pages']}")
                print(f"Text length: {first_result['text_length']} characters")
                print(f"\nText preview:\n{first_result['text_preview']}")
        else:
            print(f"\n{Fore.RED}PDF scraping test failed: {test_results.get('error', 'Unknown error')}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No URL provided. Please provide a URL to scrape or a search query.{Style.RESET_ALL}")
        parser.print_help()