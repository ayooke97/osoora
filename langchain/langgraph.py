"""
Legal Information Retrieval Workflow

This script implements a workflow to process user input for legal information retrieval.
The workflow includes:
1. Converting user prompts into legal language
2. Checking for existing answers in a FAISS vectorstore
3. Web scraping from peraturan.go.id when needed
4. Returning the relevant legal information
"""

import os
import sys
import json
import logging
import traceback
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from typing import Dict, List, Optional, Any, Tuple, Union, TypedDict
from datetime import datetime
from bpk_scraper import BPKScraper, PeraturanScraper, Document
import langchain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import log functions
from legal_scraper import log_message, log_json_message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("legal_workflow")

# Try to import LangChain Document class
try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    # Define a simple Document class if langchain is not available
    class LangChainDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

# Use Document from bpk_scraper if available, otherwise use LangChainDocument
try:
    from bpk_scraper import Document
except ImportError:
    Document = LangChainDocument

# Define state schema
class WorkflowState(TypedDict):
    """State for the legal workflow."""
    original_query: str
    legal_query: Optional[str]
    keywords: Optional[List[str]]
    vectorstore_result: Optional[Dict[str, Any]]
    scraping_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]

# Initialize PeraturanScraper
def initialize_scraper() -> Optional[PeraturanScraper]:
    """Initialize the peraturan.go.id scraper."""
    try:
        # Initialize scraper
        print("\nInitializing Peraturan Scraper...")
        scraper = PeraturanScraper(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        )
        return scraper
    except Exception as e:
        logger.error(f"Error initializing scraper: {str(e)}")
        return None

# Custom scraping function to bypass PeraturanScraper issues
def scrape_peraturan_go_id_direct(query: str, max_pages: int = 1) -> List[Document]:
    """
    Direct implementation of peraturan.go.id scraping for cases where the scraper fails.
    This is a fallback method.
    
    Args:
        query (str): The search query
        max_pages (int, optional): Maximum number of pages to scrape. Defaults to 1.
        
    Returns:
        List[Document]: List of scraped documents
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
                # Construct search URL for peraturan.go.id using the correct format
                search_url = f"https://peraturan.go.id/cariglobal"
                params = {
                    'PeraturanSearch[idglobal]': query,
                    'page': page
                }
                
                # Make request
                response = requests.get(search_url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                
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
                            print(f"PDF URL: {pdf_url}")
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
                            
                            # Use the extract_pdf_content function to handle PDF extraction
                            pdf_content, pdf_metadata = extract_pdf_content(pdf_url, headers=headers, title=title)
                            
                            if pdf_content:
                                # Create a Document object for the PDF
                                pdf_document = Document(
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
                            # Process in the PDF extraction code above
                            continue
                        
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
                            
                            doc_soup = BeautifulSoup(doc_response.content, 'html.parser')
                            
                            # Check for PDF links in the detail page
                            pdf_links = doc_soup.select('a[href$=".pdf"]')
                            if pdf_links:
                                logger.info(f"Found {len(pdf_links)} PDF links in detail page")
                                # Process the first PDF link
                                pdf_url = pdf_links[0].get('href')
                                if pdf_url:
                                    # Make the URL absolute
                                    pdf_url = urljoin(link, pdf_url)
                                    logger.info(f"PDF URL: {pdf_url}")
                                    
                                    # Use the extract_pdf_content function to handle PDF extraction
                                    pdf_content, pdf_metadata = extract_pdf_content(pdf_url, headers=headers, title=title)
                                    
                                    if pdf_content:
                                        # Create a Document object for the PDF
                                        pdf_document = Document(
                                            page_content=pdf_content,
                                            metadata={
                                                **pdf_metadata,
                                                'page': page
                                            }
                                        )
                                        
                                        documents.append(pdf_document)
                                        logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
                            else:
                                # Extract content - try multiple approaches
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
                                            'date': date
                                        }
                                    )
                                    
                                    documents.append(document)
                                    logger.info(f"Added document: {title} ({len(content)} chars)")
                                else:
                                    logger.warning(f"Could not extract sufficient content for: {title}")
                        except Exception as doc_error:
                            logger.error(f"Error retrieving document content: {str(doc_error)}")
                    except Exception as item_error:
                        logger.error(f"Error processing result item: {str(item_error)}")
            except Exception as page_error:
                logger.error(f"Error scraping page {page}: {str(page_error)}")
        
        logger.info(f"Scraped {len(documents)} documents from peraturan.go.id")
        return documents
    except Exception as e:
        logger.error(f"Error in direct scraping: {str(e)}")
        return []

def extract_pdf_content(pdf_url, headers=None, title="PDF Document"):
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
        # Check if PyPDF2 is available
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.error("PyPDF2 is required for PDF extraction")
            return None, None
        
        # Download the PDF
        try:
            logger.info(f"Downloading PDF from {pdf_url}")
            
            # Set up headers if not provided
            if not headers:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                }
            
            # Download the PDF to a temporary file
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            logger.info(f"PDF saved to temporary file: {temp_path}")
            
            # Extract text from the PDF
            try:
                # Open the PDF
                pdf_reader = PdfReader(temp_path)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                pdf_content = ""
                metadata = {
                    'source': pdf_url,
                    'title': title,
                    'num_pages': num_pages
                }
                
                logger.info(f"PDF has {num_pages} pages")
                
                # Extract text from each page
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            pdf_content += page_text + "\n\n"
                    except Exception as page_error:
                        logger.error(f"Error extracting text from page {page_num}: {str(page_error)}")
                
                # Clean up the temporary file
                try:
                    import os
                    os.unlink(temp_path)
                    logger.info("Temporary PDF file deleted")
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up temporary file: {str(cleanup_error)}")
                
                # Check if we got any content
                if pdf_content and len(pdf_content) > 100:
                    logger.info(f"Successfully extracted {len(pdf_content)} characters from PDF")
                    
                    # Add metadata
                    metadata.update({
                        'source_type': 'pdf',
                        'content_length': len(pdf_content),
                        'extraction_method': 'PyPDF2'
                    })
                    
                    return pdf_content, metadata
                else:
                    logger.warning(f"Extracted content too short or empty: {len(pdf_content)} chars")
                    return None, None
            except Exception as pdf_extract_error:
                logger.error(f"Error extracting PDF content: {str(pdf_extract_error)}")
                return None, None
        except Exception as download_error:
            logger.error(f"Error downloading PDF: {str(download_error)}")
            return None, None
    except ImportError as import_error:
        logger.error(f"Required module not available for PDF extraction: {str(import_error)}")
        return None, None
    except Exception as e:
        logger.error(f"Error in extract_pdf_content: {str(e)}")
        return None, None

def convert_documents(documents: List[Any]) -> List[Any]:
    """
    Convert documents to ensure compatibility with LangChain.
    
    This function handles the conversion between our custom Document class
    and LangChain's Document class, ensuring proper serialization.
    
    Args:
        documents (List[Any]): List of documents to convert
        
    Returns:
        List[Any]: List of converted documents compatible with LangChain
    """
    try:
        # If no documents, return empty list
        if not documents:
            return []
        
        # Initialize list for converted documents
        converted_docs = []
        
        # Process each document
        for doc in documents:
            # Check if document is already a dict
            if isinstance(doc, dict):
                # If it has page_content, it's already in the right format
                if "page_content" in doc:
                    converted_docs.append(doc)
                # If it has content but not page_content, convert it
                elif "content" in doc:
                    converted_docs.append({
                        "page_content": doc["content"],
                        "metadata": doc.get("metadata", {})
                    })
                # Otherwise, just add it as is
                else:
                    converted_docs.append(doc)
            
            # Check if document is a Document object from bpk_scraper
            elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                # Convert to dict format
                converted_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Check if document is a LangChain Document
            elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                # Convert to dict format
                converted_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Handle string documents (create with empty metadata)
            elif isinstance(doc, str):
                converted_docs.append({
                    "page_content": doc,
                    "metadata": {}
                })
            
            # Handle other types
            else:
                # Try to convert to string
                try:
                    converted_docs.append({
                        "page_content": str(doc),
                        "metadata": {}
                    })
                except:
                    # Skip if can't convert
                    continue
        
        logger.info(f"Converted {len(converted_docs)} documents for LangChain compatibility")
        return converted_docs
    except Exception as e:
        logger.error(f"Error converting documents: {str(e)}")
        # Return original documents if conversion fails
        return documents

def save_to_vectorstore(state: WorkflowState) -> WorkflowState:
    """
    Save the scraped documents to the FAISS vectorstore for future use.
    
    Args:
        state (WorkflowState): The current state of the workflow.
        
    Returns:
        WorkflowState: The updated state with vectorstore information.
    """
    try:
        # Check if we have documents to save
        if not state.get("scraping_result") or not state["scraping_result"].get("success"):
            return state
        
        documents = state["scraping_result"].get("documents", [])
        if not documents:
            logger.warning("No documents to save to vectorstore")
            return state
        
        # Get the query for naming the vectorstore
        query = state.get("legal_query") or state.get("original_query", "unknown")
        query_slug = query.lower().replace(" ", "_")[:30]
        
        # Initialize embeddings
        logger.info(f"Saving {len(documents)} documents to vectorstore")
        
        # Get embeddings
        embeddings = initialize_embeddings()
        if not embeddings:
            logger.error("Failed to initialize embeddings")
            return state
        
        try:
            # Convert documents to LangChain format if needed
            langchain_docs = []
            for doc in documents:
                if isinstance(doc, dict) and "page_content" in doc:
                    langchain_docs.append(
                        LangchainDocument(
                            page_content=doc["page_content"],
                            metadata=doc.get("metadata", {})
                        )
                    )
                elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    langchain_docs.append(doc)
            
            logger.info(f"Converted {len(langchain_docs)} documents for vectorstore")
            
            # Create FAISS index
            vectorstore_dir = "faiss_store"
            os.makedirs(vectorstore_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create vectorstore name
            vectorstore_name = f"{query_slug}_{timestamp}"
            vectorstore_path = os.path.join(vectorstore_dir, vectorstore_name)
            
            # Create and save the vectorstore
            db = FAISS.from_documents(langchain_docs, embeddings)
            db.save_local(vectorstore_path)
            
            logger.info(f"Saved {len(langchain_docs)} documents to vectorstore at {vectorstore_path}")
            
            # Update state with vectorstore information
            return {
                **state,
                "vectorstore_info": {
                    "path": vectorstore_path,
                    "document_count": len(langchain_docs),
                    "query": query
                }
            }
        except Exception as conversion_error:
            logger.error(f"Error converting documents: {str(conversion_error)}")
            # Continue with original state
            return state
    except Exception as e:
        logger.error(f"Error saving to vectorstore: {str(e)}")
        return state

# Node 1: Convert user query to legal language
def convert_to_legal_language(state: WorkflowState) -> WorkflowState:
    """Convert the user's query into legal language using an LLM."""
    logger.info(f"Converting query to legal language: {state['original_query']}")
    
    try:
        # Initialize PeraturanScraper (which has LLM capabilities)
        scraper = initialize_scraper()
        
        # Use the LLM to convert the query
        if scraper.openai_wrapper:
            prompt = f"""
            You are a legal expert specializing in Indonesian law. Convert the following user query into formal legal language 
            and extract specific keywords that would be useful for searching legal documents.
            
            User query: {state['original_query']}
            
            Respond in the following JSON format:
            {{
                "legal_query": "the formal legal version of the query",
                "keywords": ["keyword1", "keyword2", "keyword3"]
            }}
            """
            
            # Get response from LLM
            response = scraper.openai_wrapper.invoke(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                legal_query = result.get("legal_query", state['original_query'])
                keywords = result.get("keywords", [])
                
                logger.info(f"Converted query: {legal_query}")
                logger.info(f"Extracted keywords: {keywords}")
                
                # Update state
                return {
                    **state,
                    "legal_query": legal_query,
                    "keywords": keywords
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, extract information using string manipulation
                logger.warning(f"Failed to parse JSON response: {response}")
                
                # Try to extract legal query
                if "legal_query" in response:
                    legal_query = response.split("legal_query")[1].split('"')[2].strip()
                else:
                    legal_query = state['original_query']
                
                # Try to extract keywords
                keywords = []
                if "keywords" in response:
                    keywords_text = response.split("keywords")[1]
                    if "[" in keywords_text and "]" in keywords_text:
                        keywords_list = keywords_text.split("[")[1].split("]")[0]
                        keywords = [k.strip().strip('"\'') for k in keywords_list.split(",")]
                
                # Update state
                return {
                    **state,
                    "legal_query": legal_query,
                    "keywords": keywords
                }
        else:
            # If LLM is not available, use the original query
            logger.warning("LLM not available, using original query")
            return {
                **state,
                "legal_query": state['original_query'],
                "keywords": []
            }
    except Exception as e:
        logger.error(f"Error converting query to legal language: {str(e)}")
        # Return original query with error
        return {
            **state,
            "legal_query": state['original_query'],
            "keywords": [],
            "error": f"Error converting query: {str(e)}"
        }

# Node 2: Check FAISS vectorstore
def check_vectorstore(state: WorkflowState) -> WorkflowState:
    """Check if a relevant answer exists in the FAISS vectorstore."""
    logger.info(f"Checking vectorstore for: {state['legal_query']}")
    
    try:
        # Initialize embeddings
        embeddings = initialize_embeddings()
        
        # Check if a FAISS index exists
        faiss_index_path = "legal_faiss_index"
        if os.path.exists(faiss_index_path):
            # Load existing vectorstore
            vectorstore = FAISS.load_local(faiss_index_path, embeddings)
            
            # Search for relevant documents
            query = state['legal_query']
            docs = vectorstore.similarity_search(query, k=5)
            
            if docs:
                # Create context from documents
                context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                
                # Initialize PeraturanScraper (for LLM)
                scraper = initialize_scraper()
                
                # Generate answer using LLM
                if scraper.openai_wrapper:
                    prompt = f"""
                    You are a legal expert specializing in Indonesian law. Answer the following question based only on the provided context.
                    If the context doesn't contain enough information to answer the question, say so clearly.
                    
                    Context:
                    {context}
                    
                    Question: {query}
                    
                    Answer:
                    """
                    
                    answer = scraper.openai_wrapper.invoke(prompt)
                    
                    # Check if answer is relevant
                    if "tidak cukup informasi" not in answer.lower() and "tidak dapat menjawab" not in answer.lower():
                        # Return answer and end workflow
                        logger.info("Found relevant answer in vectorstore")
                        return {
                            **state,
                            "vectorstore_result": {
                                "answer": answer,
                                "documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
                            },
                            "final_answer": answer
                        }
            
            # If no relevant answer found, continue to web scraping
            logger.info("No relevant answer found in vectorstore, continuing to web scraping")
            return {
                **state,
                "vectorstore_result": None
            }
        else:
            # If no index exists, continue to web scraping
            logger.info("No FAISS index found, continuing to web scraping")
            return {
                **state,
                "vectorstore_result": None
            }
    except Exception as e:
        logger.error(f"Error checking vectorstore: {str(e)}")
        # Continue to web scraping with error
        return {
            **state,
            "vectorstore_result": None,
            "error": f"Error checking vectorstore: {str(e)}"
        }

# Node 3: Web scraping
def scrape_web(state: WorkflowState) -> WorkflowState:
    """
    Scrape legal information based on the user query.
    
    Args:
        state (WorkflowState): The current state of the workflow.
        
    Returns:
        WorkflowState: The updated state with scraped documents.
    """
    try:
        query = state.get("legal_query", "")
        if not query:
            logger.warning("No query provided for scraping")
            return {**state, "scraping_result": {"success": False, "error": "No query provided for scraping"}}
        
        logger.info(f"Scraping web for query: {query}")
        
        # Initialize documents list
        documents = []
        error_messages = []
        
        # Try Google CSE first
        try:
            logger.info("Attempting to search with Google CSE")
            google_docs = google_cse_search_and_scrape(query, num_results=5)
            
            if google_docs and len(google_docs) > 0:
                logger.info(f"Successfully retrieved {len(google_docs)} documents with Google CSE")
                documents.extend(google_docs)
            else:
                logger.warning("No documents found with Google CSE")
                error_messages.append("No documents found with Google CSE")
        except Exception as e:
            logger.error(f"Error using Google CSE: {str(e)}")
            error_messages.append(f"Error using Google CSE: {str(e)}")
        
        # Try PeraturanScraper next
        try:
            logger.info("Attempting to scrape with PeraturanScraper")
            peraturan_scraper = initialize_scraper()
            peraturan_docs = peraturan_scraper.scrape_peraturan_go_id(query, max_pages=10)
            print(peraturan_docs)
            if peraturan_docs and len(peraturan_docs) > 0:
                logger.info(f"Successfully scraped {len(peraturan_docs)} documents with PeraturanScraper")
                documents.extend(peraturan_docs)
            else:
                logger.warning("No documents found with PeraturanScraper")
                error_messages.append("No documents found with PeraturanScraper")
        except Exception as e:
            logger.error(f"Error using PeraturanScraper: {str(e)}")
            error_messages.append(f"Error using PeraturanScraper: {str(e)}")
        
        # If PeraturanScraper didn't yield results, try direct scraping
        if not documents:
            try:
                logger.info("Attempting direct scraping of peraturan.go.id")
                direct_docs = scrape_peraturan_go_id_direct(query, max_pages=3)
                
                if direct_docs and len(direct_docs) > 0:
                    logger.info(f"Successfully scraped {len(direct_docs)} documents with direct scraping")
                    documents.extend(direct_docs)
                else:
                    logger.warning("No documents found with direct scraping")
                    error_messages.append("No documents found with direct scraping")
            except Exception as e:
                logger.error(f"Error using direct scraping: {str(e)}")
                error_messages.append(f"Error using direct scraping: {str(e)}")
        
        # If still no documents, try BPKScraper as fallback
        if not documents:
            try:
                logger.info("Attempting to scrape with BPKScraper as fallback")
                bpk_scraper = BPKScraper(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                )
                bpk_docs = bpk_scraper.scrape_peraturan_bpk(query, max_pages=3)
                
                if bpk_docs and len(bpk_docs) > 0:
                    logger.info(f"Successfully scraped {len(bpk_docs)} documents with BPKScraper")
                    documents.extend(bpk_docs)
                else:
                    logger.warning("No documents found with BPKScraper")
                    error_messages.append("No documents found with BPKScraper")
            except Exception as e:
                logger.error(f"Error using BPKScraper: {str(e)}")
                error_messages.append(f"Error using BPKScraper: {str(e)}")
        
        # If we have documents, return them
        if documents:
            # Convert documents to ensure compatibility with LangChain
            converted_docs = convert_documents(documents)
            logger.info(f"Returning {len(converted_docs)} documents")
            return {**state, "scraping_result": {"success": True, "documents": converted_docs}}
        else:
            # If all methods failed, return error
            error_message = "Failed to retrieve documents from all sources: " + "; ".join(error_messages)
            logger.error(error_message)
            return {**state, "scraping_result": {"success": False, "error": error_message}}
    except Exception as e:
        logger.error(f"Error in scrape_web: {str(e)}")
        return {**state, "scraping_result": {"success": False, "error": f"Error in scrape_web: {str(e)}"}}

# Try to import environment variables from env_config
try:
    from env_config import GOOGLE_API_KEY, GOOGLE_CSE_ID
except ImportError:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

def google_cse_search_and_scrape(query: str, num_results: int = 5) -> List[Document]:
    """
    Search for information using Google Custom Search Engine (CSE),
    open the most relevant result, and scrape the related information.
    
    Args:
        query (str): The search query
        num_results (int, optional): Number of results to retrieve. Defaults to 5.
        
    Returns:
        List[Document]: List of scraped documents
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
            "q": query,
            "num": num_results
        }
        
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        # Check if we have search results
        if "items" not in search_results:
            logger.warning("No search results found")
            return []
            
        # Process search results
        documents = []
        
        for item in search_results["items"]:
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
                    pdf_content, pdf_metadata = extract_pdf_content(url, title=title)
                    
                    if pdf_content:
                        pdf_document = Document(
                            page_content=pdf_content,
                            metadata={
                                "title": title,
                                "url": url,
                                "source": "google_cse_pdf",
                                **pdf_metadata
                            }
                        )
                        documents.append(pdf_document)
                        logger.info(f"Added PDF document: {title} ({len(pdf_content)} chars)")
                else:
                    # Scrape HTML content
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Referer': 'https://www.google.com/',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    
                    # Make request to the webpage
                    page_response = requests.get(url, headers=headers, timeout=15)
                    page_response.raise_for_status()
                    
                    # Parse HTML
                    soup = BeautifulSoup(page_response.content, 'html.parser')
                    
                    # Extract main content
                    # First, try to find article or main content
                    main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
                    
                    # If no specific content container found, use body
                    if not main_content:
                        main_content = soup.find('body')
                        
                    # Extract text content
                    if main_content:
                        # Remove script and style elements
                        for script in main_content(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                            
                        # Get text
                        text_content = main_content.get_text(separator="\n", strip=True)
                        
                        # Clean up text (remove excessive newlines, etc.)
                        import re
                        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
                        
                        # Create document
                        html_document = Document(
                            page_content=text_content,
                            metadata={
                                "title": title,
                                "url": url,
                                "snippet": snippet,
                                "source": "google_cse_html"
                            }
                        )
                        
                        documents.append(html_document)
                        logger.info(f"Added HTML document: {title} ({len(text_content)} chars)")
                    else:
                        logger.warning(f"Could not extract content from {url}")
            except Exception as item_error:
                logger.error(f"Error processing search result: {str(item_error)}")
                
        return documents
    except Exception as e:
        logger.error(f"Error in Google CSE search: {str(e)}")
        return []

# Initialize embeddings
def initialize_embeddings():
    """Initialize embeddings for vectorstore."""
    try:
        # Try to import DashScopeEmbeddings
        from legal_scraper import DashScopeEmbeddings
        
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
        
        # Initialize embeddings
        embeddings = DashScopeEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model="text-embedding-v3"
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        # Fallback to OpenAI embeddings
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                model="text-embedding-3-small"
            )
        except Exception as fallback_error:
            logger.error(f"Error initializing fallback embeddings: {str(fallback_error)}")
            raise

def extract_legal_references(text: str) -> List[str]:
    """
    Extract legal document references from text.
    
    Args:
        text (str): The text to extract references from
        
    Returns:
        List[str]: List of legal document references
    """
    # Define patterns for different types of legal references
    patterns = [
        # UU (Undang-Undang) pattern
        r'(?:UU|Undang-Undang|Undang Undang)(?:\s+No[mor]?\.?\s*|\s+)(\d+)(?:\s+Tahun\s+|\/)(\d{4})',
        # PP (Peraturan Pemerintah) pattern
        r'(?:PP|Peraturan Pemerintah)(?:\s+No[mor]?\.?\s*|\s+)(\d+)(?:\s+Tahun\s+|\/)(\d{4})',
        # Perpres (Peraturan Presiden) pattern
        r'(?:Perpres|Peraturan Presiden)(?:\s+No[mor]?\.?\s*|\s+)(\d+)(?:\s+Tahun\s+|\/)(\d{4})',
        # Permen (Peraturan Menteri) pattern
        r'(?:Permen\w*|Peraturan Menteri\s+\w+)(?:\s+No[mor]?\.?\s*|\s+)(\d+)(?:\s+Tahun\s+|\/)(\d{4})',
        # Perda (Peraturan Daerah) pattern
        r'(?:Perda|Peraturan Daerah)(?:\s+No[mor]?\.?\s*|\s+)(\d+)(?:\s+Tahun\s+|\/)(\d{4})'
    ]
    
    references = []
    
    # Extract references using patterns
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                number = match.group(1)
                year = match.group(2)
                
                # Determine the type of regulation
                if 'UU' in match.group(0) or 'Undang-Undang' in match.group(0) or 'Undang Undang' in match.group(0):
                    ref_type = 'UU'
                elif 'PP' in match.group(0) or 'Peraturan Pemerintah' in match.group(0):
                    ref_type = 'PP'
                elif 'Perpres' in match.group(0) or 'Peraturan Presiden' in match.group(0):
                    ref_type = 'Perpres'
                elif 'Permen' in match.group(0) or 'Peraturan Menteri' in match.group(0):
                    ref_type = 'Permen'
                elif 'Perda' in match.group(0) or 'Peraturan Daerah' in match.group(0):
                    ref_type = 'Perda'
                else:
                    ref_type = 'Peraturan'
                
                reference = f"{ref_type} No. {number} Tahun {year}"
                if reference not in references:
                    references.append(reference)
            except Exception as e:
                logger.error(f"Error extracting reference: {str(e)}")
    
    return references

def retrieve_legal_documents(references: List[str]) -> List[Document]:
    """
    Retrieve legal documents based on references.
    
    Args:
        references (List[str]): List of legal document references
        
    Returns:
        List[Document]: List of retrieved legal documents
    """
    documents = []
    
    for reference in references:
        try:
            logger.info(f"Retrieving legal document: {reference}")
            
            # Try PeraturanScraper first
            try:
                peraturan_scraper = initialize_scraper()
                peraturan_docs = peraturan_scraper.scrape_peraturan_go_id(reference, max_pages=3)
                
                if peraturan_docs and len(peraturan_docs) > 0:
                    logger.info(f"Successfully retrieved {len(peraturan_docs)} documents for {reference} with PeraturanScraper")
                    documents.extend(peraturan_docs)
                    continue  # Skip to next reference if successful
            except Exception as e:
                logger.error(f"Error using PeraturanScraper for {reference}: {str(e)}")
            
            # If PeraturanScraper fails, try direct scraping
            try:
                direct_docs = scrape_peraturan_go_id_direct(reference, max_pages=2)
                
                if direct_docs and len(direct_docs) > 0:
                    logger.info(f"Successfully retrieved {len(direct_docs)} documents for {reference} with direct scraping")
                    documents.extend(direct_docs)
                    continue  # Skip to next reference if successful
            except Exception as e:
                logger.error(f"Error using direct scraping for {reference}: {str(e)}")
            
            # If direct scraping fails, try BPKScraper
            try:
                bpk_scraper = BPKScraper(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                )
                bpk_docs = bpk_scraper.scrape_peraturan_bpk(reference, max_pages=2)
                
                if bpk_docs and len(bpk_docs) > 0:
                    logger.info(f"Successfully retrieved {len(bpk_docs)} documents for {reference} with BPKScraper")
                    documents.extend(bpk_docs)
                else:
                    logger.warning(f"No documents found for {reference} with BPKScraper")
            except Exception as e:
                logger.error(f"Error using BPKScraper for {reference}: {str(e)}")
        except Exception as e:
            logger.error(f"Error retrieving legal document for {reference}: {str(e)}")
    
    return documents

def summarize_documents(documents: List[Any], query: str = None) -> str:
    """
    Summarize a list of documents.
    
    Args:
        documents (List[Any]): List of documents to summarize
        query (str, optional): The original query for context
        
    Returns:
        str: Summary of the documents
    """
    try:
        if not documents:
            return "No documents to summarize."
        
        # Prepare documents for summarization
        docs_content = []
        for doc in documents:
            if isinstance(doc, dict):
                content = doc.get('page_content', '')
                metadata = doc.get('metadata', {})
            else:
                content = getattr(doc, 'page_content', '')
                metadata = getattr(doc, 'metadata', {})
            
            # Add metadata to content
            title = metadata.get('title', 'Untitled Document')
            source = metadata.get('source', 'Unknown Source')
            url = metadata.get('url', '')
            
            # Truncate content if too long
            if len(content) > 5000:
                content = content[:5000] + "... [content truncated]"
            
            doc_info = f"Document: {title}\nSource: {source}\n"
            if url:
                doc_info += f"URL: {url}\n"
            doc_info += f"Content:\n{content}\n\n"
            
            docs_content.append(doc_info)
        
        # Combine document contents
        combined_content = "\n".join(docs_content)
        
        # Check for legal references in the documents
        all_text = " ".join([getattr(doc, 'page_content', '') if not isinstance(doc, dict) 
                            else doc.get('page_content', '') for doc in documents])
        legal_references = extract_legal_references(all_text)
        
        # If legal references found, retrieve and add those documents
        if legal_references:
            logger.info(f"Found legal references: {legal_references}")
            legal_docs = retrieve_legal_documents(legal_references)
            
            if legal_docs:
                logger.info(f"Retrieved {len(legal_docs)} legal documents")
                
                # Add legal documents to content
                legal_docs_content = []
                for doc in legal_docs:
                    if isinstance(doc, dict):
                        content = doc.get('page_content', '')
                        metadata = doc.get('metadata', {})
                    else:
                        content = getattr(doc, 'page_content', '')
                        metadata = getattr(doc, 'metadata', {})
                    
                    # Add metadata to content
                    title = metadata.get('title', 'Untitled Legal Document')
                    source = metadata.get('source', 'Legal Document')
                    
                    # Truncate content if too long
                    if len(content) > 5000:
                        content = content[:5000] + "... [content truncated]"
                    
                    doc_info = f"Legal Document: {title}\nSource: {source}\n"
                    doc_info += f"Content:\n{content}\n\n"
                    
                    legal_docs_content.append(doc_info)
                
                combined_content += "\n\nLEGAL DOCUMENTS:\n" + "\n".join(legal_docs_content)
        
        # Create summary prompt
        summary_prompt = """
        You are a legal expert assistant. Your task is to summarize the following documents and provide a comprehensive answer.
        
        QUERY: {query}
        
        DOCUMENTS:
        {documents}
        
        Please provide:
        1. A summary of each document (keep it concise)
        2. Key legal points and references
        3. A comprehensive answer to the query based on the documents
        4. Any relevant legal advice or considerations
        
        FORMAT YOUR RESPONSE IN MARKDOWN.
        """
        
        # Try using OpenAI first
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            
            # Initialize LLM
            llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model="gpt-3.5-turbo"
            )
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template(summary_prompt)
            
            # Generate summary
            summary_chain = prompt_template | llm
            summary = summary_chain.invoke({
                "query": query or "Provide a comprehensive summary of these legal documents",
                "documents": combined_content
            })
            
            # Extract content from response
            if hasattr(summary, 'content'):
                return summary.content
            else:
                return str(summary)
        except Exception as openai_error:
            logger.error(f"Error using OpenAI for summarization: {str(openai_error)}")
            
            # Try using DashScope as fallback
            try:
                logger.info("Falling back to DashScope for summarization")
                
                # Try to import DashScope
                try:
                    from langchain_dashscope import ChatDashScope
                    
                    # Initialize DashScope
                    dashscope_llm = ChatDashScope(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        model="qwen-max",
                        dashscope_api_base=os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                    )
                    
                    # Create prompt template
                    prompt_template = ChatPromptTemplate.from_template(summary_prompt)
                    
                    # Generate summary
                    summary_chain = prompt_template | dashscope_llm
                    summary = summary_chain.invoke({
                        "query": query or "Provide a comprehensive summary of these legal documents",
                        "documents": combined_content
                    })
                    
                    # Extract content from response
                    if hasattr(summary, 'content'):
                        return summary.content
                    else:
                        return str(summary)
                except ImportError:
                    logger.error("langchain_dashscope not available, trying direct API call")
                    
                    # If langchain_dashscope is not available, try direct API call
                    import requests
                    import json
                    
                    api_key = os.environ.get("OPENAI_API_KEY")
                    base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                    
                    data = {
                        "model": "qwen-max",
                        "messages": [
                            {"role": "system", "content": "You are a legal expert assistant."},
                            {"role": "user", "content": summary_prompt.format(
                                query=query or "Provide a comprehensive summary of these legal documents",
                                documents=combined_content[:15000]  # Limit content to avoid token limits
                            )}
                        ]
                    }
                    
                    response = requests.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("choices", [{}])[0].get("message", {}).get("content", "No summary generated.")
                    else:
                        logger.error(f"Error from DashScope API: {response.text}")
                        raise Exception(f"DashScope API error: {response.status_code}")
            except Exception as dashscope_error:
                logger.error(f"Error using DashScope for summarization: {str(dashscope_error)}")
                
                # Fallback to basic summarization if both methods fail
                return f"## Summary of Documents\n\n" + \
                       f"Found {len(documents)} documents related to your query.\n\n" + \
                       f"### Key Legal References\n" + \
                       (f"- {', '.join(legal_references)}" if legal_references else "- No specific legal references found.") + \
                       f"\n\n### Document Overview\n" + \
                       "\n".join([f"- **{doc.metadata.get('title', 'Untitled') if hasattr(doc, 'metadata') else doc.get('metadata', {}).get('title', 'Untitled')}**" 
                                 for doc in documents[:5]]) + \
                       "\n\n*Note: Detailed summarization failed due to API issues. Please try again later.*"
    except Exception as e:
        logger.error(f"Error summarizing documents: {str(e)}")
        return f"Error summarizing documents: {str(e)}"

def process_and_summarize(state: WorkflowState) -> WorkflowState:
    """
    Process and summarize the scraped documents.
    
    Args:
        state (WorkflowState): The current state of the workflow.
        
    Returns:
        WorkflowState: The updated state with summarized information.
    """
    try:
        # Get documents from scraping result
        scraping_result = state.get("scraping_result", {})
        if not scraping_result or not scraping_result.get("success", False):
            return {**state, "final_answer": "No documents were found to summarize."}
        
        documents = scraping_result.get("documents", [])
        if not documents:
            return {**state, "final_answer": "No documents were found to summarize."}
        
        # Get original query
        query = state.get("original_query", "")
        
        # Summarize documents
        logger.info(f"Summarizing {len(documents)} documents")
        summary = summarize_documents(documents, query)
        
        # Update state with summary
        return {**state, "final_answer": summary}
    except Exception as e:
        logger.error(f"Error in process_and_summarize: {str(e)}")
        return {**state, "final_answer": f"Error processing and summarizing documents: {str(e)}"}

def main():
    """Run the legal workflow."""
    # Get user query
    query = input("Processing query: ")
    logger.info(f"Processing query: {query}")
    
    # Initialize workflow state
    state = {
        "original_query": query,
        "legal_query": query,  # No conversion to legal language
        "keywords": [],
        "scraping_result": None,
        "final_answer": None
    }
    
    # Initialize scraper
    scraper = initialize_scraper()
    
    # Run workflow
    try:
        # Scrape web
        state = scrape_web(state)
        
        # Process and summarize documents
        state = process_and_summarize(state)
        
        # Save to vectorstore
        state = save_to_vectorstore(state)
        
        # Print final answer
        if state.get("final_answer"):
            print("="*50 + "Generated answer from scraped documents" + "="*50)
            print(state["final_answer"])
        else:
            print("No answer generated.")
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    main()