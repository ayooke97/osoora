"""
Retrieve Google CSE Query Results

This script retrieves and prints the results from Google CSE,
then opens each Hukumonline URL to scrape content.
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from googleapiclient.discovery import build
from colorama import Fore, Style, init
import time
import webbrowser
import io
import tempfile

# Try importing httpx for advanced HTTP requests
try:
    import httpx
    HAS_HTTPX = True
    print(f"{Fore.GREEN}httpx found. Advanced HTTP client enabled.{Style.RESET_ALL}")
except ImportError:
    HAS_HTTPX = False
    print(f"{Fore.YELLOW}httpx not found. Falling back to requests.{Style.RESET_ALL}")

# Try importing pypdf for PDF processing
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
    print(f"{Fore.GREEN}pypdf found. PDF scraping enabled.{Style.RESET_ALL}")
except ImportError:
    HAS_PYPDF = False
    print(f"{Fore.YELLOW}pypdf not found. PDF scraping disabled.{Style.RESET_ALL}")

# Initialize colorama
init()

def retrieve_cse_results(query, num_results=5):
    """
    Retrieve results from Google Custom Search Engine.
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return
        
    Returns:
        list: List of valid Hukumonline URLs
    """
    print(f"{Fore.CYAN}Retrieving results for query: {query}{Style.RESET_ALL}")
    
    # Get API key and CSE ID directly
    api_key = "AIzaSyD2MDuKtg2ZwMLCYYaOgDm4RWTMNnPIKHk"
    cse_id = "124f4fd8c7c694b40"
    
    valid_urls = []
    
    try:
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=num_results
        ).execute()
        
        # Print the search results
        if "items" in result:
            print(f"\n{Fore.GREEN}Found {len(result['items'])} results:{Style.RESET_ALL}")
            
            for i, item in enumerate(result["items"], 1):
                url = item.get('link', '')
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No snippet')
                
                print(f"\n{Fore.YELLOW}Result {i}:{Style.RESET_ALL}")
                print(f"Title: {title}")
                print(f"Link: {url}")
                print(f"Snippet: {snippet}")
                
                if 'hukumonline.com' in url:
                    valid_urls.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
        else:
            print(f"\n{Fore.RED}No results found{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
    
    return valid_urls

def scrape_article(article_url, article_title=None):
    """
    Scrape a Hukumonline article.
    
    Args:
        article_url (str): URL of the article to scrape
        article_title (str, optional): Title of the article
        
    Returns:
        tuple: (title, content)
    """
    print(f"\n{Fore.CYAN}Opening URL: {article_url}{Style.RESET_ALL}")
    
    try:
        # Create a session with headers
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Make the request
        response = session.get(article_url, headers=headers)
        
        if response.status_code != 200:
            print(f"{Fore.RED}Failed to retrieve article: HTTP {response.status_code}{Style.RESET_ALL}")
            return None, None
        
        print(f"{Fore.GREEN}Successfully retrieved article: HTTP {response.status_code}{Style.RESET_ALL}")
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = None
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.text.strip()
        else:
            # Try alternative title elements
            alt_title = soup.find('title')
            if alt_title:
                title = alt_title.text.strip()
        
        # Look for article content
        content = None
        
        # Try different selectors for the article content
        content_selectors = [
            'article', 
            '.article-content', 
            '.content', 
            '.post-content',
            '#article-content',
            '.entry-content',
            'main',
            '.main-content',
            '.article-detail',
            '.post',
            '#content'
        ]
        
        for selector in content_selectors:
            elems = soup.select(selector)
            if elems:
                # Remove script and style elements
                for script in elems[0].find_all(['script', 'style']):
                    script.decompose()
                
                # Get the text content
                content = elems[0].get_text(separator='\n').strip()
                break
        
        # If we couldn't find the content with selectors, try to find the main text content
        if not content:
            # Look for the largest text block
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Find the paragraph with the most text
                largest_p = max(paragraphs, key=lambda p: len(p.text.strip()))
                
                # Get the parent that likely contains the article content
                parent = largest_p.parent
                
                # Remove script and style elements
                for script in parent.find_all(['script', 'style']):
                    script.decompose()
                
                # Get the text content
                content = parent.get_text(separator='\n').strip()
        
        return title, content
    
    except Exception as e:
        print(f"{Fore.RED}Error scraping article: {str(e)}{Style.RESET_ALL}")
        return None, None

def extract_legal_references(text):
    """Extract legal references from text."""
    
    # Patterns for legal references
    patterns = [
        r'UU\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # UU No. 13 Tahun 2003
        r'Undang-Undang\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Undang-Undang No. 13 Tahun 2003
        r'PP\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # PP No. 78 Tahun 2015
        r'Peraturan\s+Pemerintah\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Peraturan Pemerintah No. 78 Tahun 2015
        r'Perpres\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Perpres No. 20 Tahun 2018
        r'Peraturan\s+Presiden\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Peraturan Presiden No. 20 Tahun 2018
        r'Permenaker\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Permenaker No. 5 Tahun 2018
        r'Permen\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Permen No. 5 Tahun 2018
        r'Peraturan\s+Menteri\s+\w+\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Peraturan Menteri Ketenagakerjaan No. 5 Tahun 2018
        r'Kepmen\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Kepmen No. 100 Tahun 2004
        r'Keputusan\s+Menteri\s+\w+\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Keputusan Menteri Ketenagakerjaan No. 100 Tahun 2004
        r'Perda\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Perda No. 6 Tahun 2020
        r'Peraturan\s+Daerah\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Peraturan Daerah No. 6 Tahun 2020
    ]
    
    references = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            reference = match.group(0)
            if reference not in references:
                references.append(reference)
    
    return references

def find_pdf_links(url):
    """
    Find all links on a webpage that might be PDFs or contain legal references.
    
    Args:
        url (str): URL of the webpage to search
        
    Returns:
        list: List of dictionaries containing URL and text of potential PDF links
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        # Headers for the request
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Send request to the URL
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links
        links = []
        
        # First, specifically look for links with class="css-140jb81" which are likely PDF links
        pdf_class_links = soup.find_all('a', class_='css-140jb81')
        if pdf_class_links:
            print(f"{Fore.GREEN}Found {len(pdf_class_links)} links with class='css-140jb81'{Style.RESET_ALL}")
            for a in pdf_class_links:
                link_url = a['href']
                link_text = a.get_text().strip()
                
                # Make the URL absolute
                absolute_url = urljoin(url, link_url)
                
                # Check if it's a PDF
                is_pdf = absolute_url.lower().endswith('.pdf')
                
                # Add to list
                links.append({
                    'url': absolute_url,
                    'text': link_text or absolute_url,
                    'is_pdf': is_pdf,
                    'has_legal_ref': True,  # Assume links with this class are relevant
                    'css_class': 'css-140jb81'
                })
        
        # Then look for all other links
        for a in soup.find_all('a', href=True):
            if a.get('class') and 'css-140jb81' in a.get('class'):
                continue  # Skip links we've already processed
                
            link_url = a['href']
            link_text = a.get_text().strip()
            
            # Make the URL absolute
            absolute_url = urljoin(url, link_url)
            
            # Check if it's a PDF or contains legal references
            is_pdf = absolute_url.lower().endswith('.pdf')
            has_legal_ref = False
            
            # Check for legal reference patterns in URL or text
            legal_patterns = [
                r'undang-?undang', r'uu[\s_-]no', r'peraturan', r'pp[\s_-]no', 
                r'perpres', r'kepres', r'keppres', r'permen', r'permenkeu',
                r'permendag', r'putusan', r'keputusan'
            ]
            
            for pattern in legal_patterns:
                if re.search(pattern, absolute_url, re.IGNORECASE) or re.search(pattern, link_text, re.IGNORECASE):
                    has_legal_ref = True
                    break
            
            # Add to list if it's a PDF or contains legal references
            if is_pdf or has_legal_ref:
                links.append({
                    'url': absolute_url,
                    'text': link_text or absolute_url,
                    'is_pdf': is_pdf,
                    'has_legal_ref': has_legal_ref,
                    'css_class': None
                })
        
        print(f"{Fore.GREEN}Found {len(links)} potential PDF/legal reference links{Style.RESET_ALL}")
        return links
        
    except Exception as e:
        print(f"{Fore.RED}Error finding PDF links: {str(e)}{Style.RESET_ALL}")
        return []

def download_pdf(pdf_url):
    """
    Download a PDF file from a URL.
    
    Args:
        pdf_url (str): URL of the PDF to download
        
    Returns:
        bytes: The PDF content as bytes or None if download failed
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        # Use httpx if available
        if HAS_HTTPX:
            try:
                print(f"{Fore.CYAN}Downloading PDF using httpx: {pdf_url}{Style.RESET_ALL}")
                
                # Create httpx client with custom settings
                with httpx.Client(
                    timeout=30.0,
                    follow_redirects=True,
                    headers={
                        'User-Agent': user_agent,
                        'Accept': 'application/pdf,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive',
                    }
                ) as client:
                    # Send GET request to download the PDF
                    response = client.get(pdf_url)
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        # Check if the content is a PDF
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf'):
                            print(f"{Fore.GREEN}Successfully downloaded PDF ({len(response.content)} bytes){Style.RESET_ALL}")
                            return response.content
                        else:
                            print(f"{Fore.YELLOW}Downloaded content is not a PDF (Content-Type: {content_type}){Style.RESET_ALL}")
                            # Try to return content anyway, might be a PDF with wrong content type
                            return response.content
                    else:
                        print(f"{Fore.RED}Failed to download PDF: HTTP {response.status_code}{Style.RESET_ALL}")
                        return None
                    
            except Exception as e:
                print(f"{Fore.RED}Error downloading PDF with httpx: {str(e)}{Style.RESET_ALL}")
                # Fall back to regular method
        
        # Fall back to regular method if httpx is not available or failed
        print(f"{Fore.CYAN}Downloading PDF using requests: {pdf_url}{Style.RESET_ALL}")
        headers = {
            'User-Agent': user_agent,
            'Accept': 'application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Try to get the content
        response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Check if the content is a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf'):
                pdf_content = response.content
                print(f"{Fore.GREEN}Successfully downloaded PDF ({len(pdf_content)} bytes){Style.RESET_ALL}")
                return pdf_content
            else:
                print(f"{Fore.YELLOW}Downloaded content is not a PDF (Content-Type: {content_type}){Style.RESET_ALL}")
                # Try to return content anyway, might be a PDF with wrong content type
                return response.content
        else:
            print(f"{Fore.RED}Failed to download PDF: HTTP {response.status_code}{Style.RESET_ALL}")
            return None
        
    except Exception as e:
        print(f"{Fore.RED}Error downloading PDF: {str(e)}{Style.RESET_ALL}")
        return None

def extract_text_from_pdf(pdf_content):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_content (bytes): The PDF content as bytes
        
    Returns:
        str: The extracted text from the PDF
    """
    if not HAS_PYPDF:
        print(f"{Fore.RED}pypdf not available. Cannot extract text from PDF.{Style.RESET_ALL}")
        return None
    
    try:
        # Create a PDF reader object
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from each page
        text = ""
        num_pages = len(pdf_reader.pages)
        print(f"{Fore.CYAN}Extracting text from PDF ({num_pages} pages){Style.RESET_ALL}")
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
            
            # Print progress for large PDFs
            if i % 10 == 0 and i > 0:
                print(f"{Fore.CYAN}Processed {i}/{num_pages} pages{Style.RESET_ALL}")
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.strip()
        
        print(f"{Fore.GREEN}Successfully extracted {len(text)} characters from PDF{Style.RESET_ALL}")
        return text
        
    except Exception as e:
        print(f"{Fore.RED}Error extracting text from PDF: {str(e)}{Style.RESET_ALL}")
        return None

def scrape_pdf(pdf_url, pdf_title=None):
    """
    Scrape a PDF file for its content.
    
    Args:
        pdf_url (str): URL of the PDF to scrape
        pdf_title (str, optional): Title of the PDF
        
    Returns:
        dict: Dictionary containing PDF information and extracted text
    """
    try:
        print(f"\n{Fore.BLUE}Scraping PDF: {pdf_title or pdf_url}{Style.RESET_ALL}")
        
        # Download the PDF
        pdf_content = download_pdf(pdf_url)
        
        if not pdf_content:
            print(f"{Fore.RED}Failed to download PDF{Style.RESET_ALL}")
            return {
                'url': pdf_url,
                'title': pdf_title or pdf_url,
                'success': False,
                'error': 'Failed to download PDF',
                'text': None,
                'legal_references': []
            }
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_content)
        
        if not pdf_text:
            print(f"{Fore.RED}Failed to extract text from PDF{Style.RESET_ALL}")
            return {
                'url': pdf_url,
                'title': pdf_title or pdf_url,
                'success': False,
                'error': 'Failed to extract text from PDF',
                'text': None,
                'legal_references': []
            }
        
        # Extract legal references from the PDF text
        legal_references = extract_legal_references(pdf_text)
        
        # Create a summary of the PDF content
        text_lines = pdf_text.split('\n')
        total_lines = len(text_lines)
        
        # Get a preview of the PDF content (first 10 lines and last 5 lines)
        preview_lines = []
        if total_lines <= 20:
            preview_lines = text_lines
        else:
            preview_lines = text_lines[:10] + ['...'] + text_lines[-5:]
        
        preview_text = '\n'.join(preview_lines)
        
        if legal_references:
            print(f"{Fore.GREEN}Found {len(legal_references)} legal references in PDF{Style.RESET_ALL}")
            for ref in legal_references:
                print(f"- {ref}")
        else:
            print(f"{Fore.YELLOW}No legal references found in PDF{Style.RESET_ALL}")
        
        # Return the PDF information
        return {
            'url': pdf_url,
            'title': pdf_title or pdf_url,
            'success': True,
            'text': pdf_text,
            'text_length': len(pdf_text),
            'preview': preview_text,
            'legal_references': legal_references
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error scraping PDF: {str(e)}{Style.RESET_ALL}")
        return {
            'url': pdf_url,
            'title': pdf_title or pdf_url,
            'success': False,
            'error': str(e),
            'text': None,
            'legal_references': []
        }

def check_pdf_for_legal_references(pdf_url, pdf_title=None):
    """
    Check if a PDF contains legal references without opening it in the browser.
    
    Args:
        pdf_url (str): URL of the PDF file
        pdf_title (str, optional): Title of the PDF
        
    Returns:
        dict: Information about the PDF including URL, title, and whether it's a PDF
    """
    try:
        print(f"{Fore.YELLOW}Checking link: {pdf_url}{Style.RESET_ALL}")
        
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        # Result dictionary
        result = {
            'url': pdf_url,
            'title': pdf_title or pdf_url,
            'is_pdf': False,
            'final_url': pdf_url,
            'content_type': None,
            'status_code': None,
            'success': False,
            'text': None,
            'legal_references': []
        }
        
        # Use httpx if available
        if HAS_HTTPX:
            try:
                print(f"{Fore.CYAN}Using httpx to access: {pdf_url}{Style.RESET_ALL}")
                
                # Create httpx client with custom settings
                with httpx.Client(
                    timeout=15.0,
                    follow_redirects=True,
                    headers={
                        'User-Agent': user_agent,
                        'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                ) as client:
                    # Send HEAD request first to check content type
                    head_response = client.head(pdf_url)
                    
                    # Update result with response info
                    result['status_code'] = head_response.status_code
                    result['content_type'] = head_response.headers.get('Content-Type', '')
                    result['final_url'] = str(head_response.url)
                    
                    # Check if it's a PDF
                    result['is_pdf'] = 'application/pdf' in result['content_type'] or result['final_url'].lower().endswith('.pdf')
                    
                    if result['is_pdf']:
                        print(f"{Fore.GREEN}Detected PDF document: {pdf_url}{Style.RESET_ALL}")
                        
                        # Scrape the PDF content if it's a PDF
                        pdf_info = scrape_pdf(result['final_url'], result['title'])
                        if pdf_info['success']:
                            result['text'] = pdf_info['text']
                            result['legal_references'] = pdf_info['legal_references']
                    
                    result['success'] = True
                    return result
                    
            except Exception as e:
                print(f"{Fore.RED}Error with httpx: {str(e)}{Style.RESET_ALL}")
                # Fall back to regular method
        
        # Fall back to regular method if httpx is not available or failed
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Try to get the content
        response = requests.head(pdf_url, headers=headers, timeout=10, allow_redirects=True)
        
        # Update result with response info
        result['status_code'] = response.status_code
        result['content_type'] = response.headers.get('Content-Type', '')
        result['final_url'] = response.url
        
        # Check if the content is a PDF
        result['is_pdf'] = 'application/pdf' in result['content_type'] or result['final_url'].lower().endswith('.pdf')
        
        if result['is_pdf']:
            print(f"{Fore.GREEN}Detected PDF document: {pdf_url}{Style.RESET_ALL}")
            
            # Scrape the PDF content if it's a PDF
            pdf_info = scrape_pdf(result['final_url'], result['title'])
            if pdf_info['success']:
                result['text'] = pdf_info['text']
                result['legal_references'] = pdf_info['legal_references']
        
        result['success'] = True
        return result
        
    except Exception as e:
        print(f"{Fore.RED}Error processing link: {str(e)}{Style.RESET_ALL}")
        return {
            'url': pdf_url,
            'title': pdf_title or pdf_url,
            'is_pdf': False,
            'final_url': pdf_url,
            'content_type': None,
            'status_code': None,
            'success': False,
            'error': str(e),
            'text': None,
            'legal_references': []
        }

def process_article_for_pdfs(article_url, article_title=None):
    """
    Process an article to find PDF links that might contain legal references.
    
    Args:
        article_url (str): URL of the article to process
        article_title (str, optional): Title of the article
        
    Returns:
        list: List of PDF links found in the article
    """
    try:
        print(f"\n{Fore.BLUE}Processing article for PDFs: {article_title or article_url}{Style.RESET_ALL}")
        
        # Find PDF links in the article
        pdf_links = find_pdf_links(article_url)
        
        if not pdf_links:
            print(f"{Fore.YELLOW}No PDF links found in the article{Style.RESET_ALL}")
            return []
        
        print(f"{Fore.GREEN}Found {len(pdf_links)} potential PDF links{Style.RESET_ALL}")
        
        # Process each PDF link
        processed_pdfs = []
        for link in pdf_links:
            pdf_info = check_pdf_for_legal_references(link['url'], link['text'])
            if pdf_info['success']:
                processed_pdfs.append(pdf_info)
                print(f"{Fore.GREEN}Successfully processed PDF: {link['text'] or link['url']}{Style.RESET_ALL}")
            
        return processed_pdfs
        
    except Exception as e:
        print(f"{Fore.RED}Error processing article for PDFs: {str(e)}{Style.RESET_ALL}")
        return []

def google_cse_search_and_scrape(query, num_results=10, site=None):
    """
    Search Google Custom Search Engine and scrape the results.
    
    Args:
        query (str): Search query
        num_results (int, optional): Number of results to return. Defaults to 10.
        site (str, optional): Site to search within. Defaults to None.
    
    Returns:
        dict: Dictionary containing search results and scraped content
    """
    # Add site: operator if specified
    if site:
        query = f"{query} site:{site}"
    
    print(f"\nRetrieving results for query: {query}\n")
    
    # Get search results
    results = retrieve_cse_results(query, num_results)
    
    if not results:
        print(f"{Fore.RED}No results found for query: {query}{Style.RESET_ALL}")
        return {"query": query, "results": [], "scraped_content": [], "legal_references": [], "pdfs": []}
    
    print(f"Found {len(results)} results:\n")
    
    # Process each result
    scraped_content = []
    all_legal_references = []
    all_pdfs = []
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        link = result.get("url", "")
        snippet = result.get("snippet", "No snippet")
        
        print(f"Result {i}:")
        print(f"Title: {title}")
        print(f"Link: {link}")
        
        # Scrape the article
        article_content = scrape_article(link, title)
        
        if article_content:
            scraped_content.append({"title": title, "link": link, "content": article_content})
            
            # Extract legal references
            legal_references = extract_legal_references(article_content)
            if legal_references:
                all_legal_references.extend(legal_references)
                print(f"{Fore.GREEN}Found {len(legal_references)} legal references{Style.RESET_ALL}")
                
                # Process the article for PDFs if legal references were found
                pdfs = process_article_for_pdfs(link, title)
                if pdfs:
                    all_pdfs.extend(pdfs)
                    
                    # Print summary of PDF content
                    for pdf in pdfs:
                        if pdf.get('text'):
                            print(f"\n{Fore.CYAN}PDF Content Summary for: {pdf.get('title', 'Unknown')}{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}URL: {pdf.get('url', 'Unknown')}{Style.RESET_ALL}")
                            
                            # Display preview if available, otherwise show a portion of the text
                            if pdf.get('preview'):
                                print(f"{Fore.CYAN}Content Preview:\n{pdf['preview']}{Style.RESET_ALL}")
                            else:
                                text_preview = pdf['text'][:300] + '...' if len(pdf['text']) > 300 else pdf['text']
                                print(f"{Fore.CYAN}Content Preview:\n{text_preview}{Style.RESET_ALL}")
                            
                            # Extract legal references from PDF
                            pdf_legal_refs = pdf.get('legal_references', [])
                            if pdf_legal_refs:
                                print(f"{Fore.GREEN}Found {len(pdf_legal_refs)} legal references in PDF:{Style.RESET_ALL}")
                                for ref in pdf_legal_refs:
                                    print(f"- {ref}")
                                all_legal_references.extend(pdf_legal_refs)
        
        # Wait a bit before the next request to avoid rate limiting
        if i < len(results):
            print(f"Waiting 2 seconds before next request...")
            time.sleep(2)
    
    # Print summary
    print("\nSummary:")
    print(f"Total articles processed: {len(results)}")
    print(f"Total PDFs found: {len(all_pdfs)}")
    print(f"Total legal references found: {len(all_legal_references)}")
    
    # Save results to a file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"search_results_{timestamp}.txt"
    
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Search Query: {query}\n")
            f.write(f"Total articles processed: {len(results)}\n")
            f.write(f"Total PDFs found: {len(all_pdfs)}\n")
            f.write(f"Total legal references found: {len(all_legal_references)}\n\n")
            
            f.write("Legal References Found:\n")
            for i, ref in enumerate(all_legal_references, 1):
                f.write(f"{i}. {ref}\n")
            
            f.write("\nPDFs Found:\n")
            for i, pdf in enumerate(all_pdfs, 1):
                f.write(f"\n--- PDF {i} ---\n")
                f.write(f"Title: {pdf.get('title', 'Unknown')}\n")
                f.write(f"URL: {pdf.get('url', 'Unknown')}\n")
                f.write(f"Is PDF: {pdf.get('is_pdf', False)}\n")
                
                # Include legal references found in the PDF
                pdf_legal_refs = pdf.get('legal_references', [])
                if pdf_legal_refs:
                    f.write(f"Legal References ({len(pdf_legal_refs)}):\n")
                    for ref in pdf_legal_refs:
                        f.write(f"- {ref}\n")
                
                # Include a preview of the PDF content
                if pdf.get('text'):
                    f.write("\nContent Preview:\n")
                    preview = pdf.get('preview', pdf['text'][:500] + '...' if len(pdf['text']) > 500 else pdf['text'])
                    f.write(f"{preview}\n")
        
        print(f"\n{Fore.GREEN}Results saved to: {results_file}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving results to file: {str(e)}{Style.RESET_ALL}")
    
    return {
        "query": query,
        "results": results,
        "scraped_content": scraped_content,
        "legal_references": all_legal_references,
        "pdfs": all_pdfs
    }

def test_pdf_scraping(url):
    """
    Test function to directly test PDF scraping functionality.
    
    Args:
        url (str): URL of the PDF or webpage to test
    """
    print(f"\n{Fore.BLUE}Testing PDF scraping for: {url}{Style.RESET_ALL}")
    
    # Check if the URL is a direct PDF or a webpage
    if url.lower().endswith('.pdf'):
        # Direct PDF URL
        process_direct_pdf(url)
    else:
        # Webpage that might contain PDF links
        process_webpage_for_pdfs(url)

def process_direct_pdf(pdf_url):
    """
    Process a direct PDF URL.
    
    Args:
        pdf_url (str): URL of the PDF to process
    """
    # Download the PDF
    start_time = time.time()
    pdf_content = download_pdf(pdf_url)
    download_time = time.time() - start_time
    
    if not pdf_content:
        print(f"{Fore.RED}Failed to download PDF. Exiting.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Successfully downloaded PDF ({len(pdf_content)} bytes) in {download_time:.2f} seconds{Style.RESET_ALL}")
    
    # Extract text from the PDF
    start_time = time.time()
    pdf_text = extract_text_from_pdf(pdf_content)
    extraction_time = time.time() - start_time
    
    if not pdf_text:
        print(f"{Fore.RED}Failed to extract text from PDF. Exiting.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Successfully extracted text from PDF ({len(pdf_text)} characters) in {extraction_time:.2f} seconds{Style.RESET_ALL}")
    
    # Extract legal references
    start_time = time.time()
    legal_references = extract_legal_references(pdf_text)
    reference_time = time.time() - start_time
    
    if legal_references:
        print(f"{Fore.GREEN}Found {len(legal_references)} legal references in {reference_time:.2f} seconds:{Style.RESET_ALL}")
        for i, ref in enumerate(legal_references, 1):
            print(f"{i}. {ref}")
    else:
        print(f"{Fore.YELLOW}No legal references found in the PDF (search took {reference_time:.2f} seconds){Style.RESET_ALL}")
    
    # Print a preview of the extracted text
    print(f"\n{Fore.CYAN}Text Preview (first 10 lines):{Style.RESET_ALL}")
    lines = pdf_text.split('\n')
    preview_lines = lines[:10]
    for line in preview_lines:
        print(line)
    
    # Save the extracted text
    save_path = f"pdf_extract_{int(time.time())}.txt"
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(pdf_text)
        print(f"{Fore.GREEN}Saved extracted text to: {save_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving text to file: {str(e)}{Style.RESET_ALL}")

def process_webpage_for_pdfs(url):
    """
    Process a webpage to find and process PDF links.
    
    Args:
        url (str): URL of the webpage to process
    """
    print(f"{Fore.CYAN}Searching for PDF links on webpage: {url}{Style.RESET_ALL}")
    
    # Find PDF links on the webpage
    start_time = time.time()
    pdf_links = find_pdf_links(url)
    search_time = time.time() - start_time
    
    if not pdf_links:
        print(f"{Fore.RED}No PDF links found on the webpage. Exiting.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}Found {len(pdf_links)} potential PDF links in {search_time:.2f} seconds{Style.RESET_ALL}")
    
    # Process each PDF link
    for i, link in enumerate(pdf_links, 1):
        print(f"\n{Fore.YELLOW}Processing PDF link {i}/{len(pdf_links)}: {link['text']}{Style.RESET_ALL}")
        print(f"URL: {link['url']}")
        print(f"Is PDF: {link['is_pdf']}")
        print(f"Has Legal Reference: {link['has_legal_ref']}")
        
        # Process the PDF link
        if link['is_pdf']:
            print(f"{Fore.CYAN}Processing direct PDF link...{Style.RESET_ALL}")
            process_direct_pdf(link['url'])
        else:
            print(f"{Fore.YELLOW}Link is not a direct PDF. Skipping.{Style.RESET_ALL}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PDF scraping functionality')
    parser.add_argument('url', nargs='?', help='URL of the PDF to scrape')
    parser.add_argument('--search', help='Search query for Google CSE')
    args = parser.parse_args()
    
    if args.url:
        test_pdf_scraping(args.url)
    elif args.search:
        results = google_cse_search_and_scrape(args.search)
        print(f"\n{Fore.GREEN}Completed search and scrape for: {args.search}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Please provide either a PDF URL or a search query.{Style.RESET_ALL}")
        print(f"Usage: python retrieve_cse_results.py [PDF_URL]")
        print(f"   or: python retrieve_cse_results.py --search \"your search query\"")
