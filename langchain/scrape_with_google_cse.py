"""
Scrape Hukumonline articles using Google CSE to find valid URLs

This script uses Google Custom Search Engine to find valid Hukumonline articles,
then scrapes them to extract content and look for legal references and PDF links.
"""

import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from colorama import Fore, Style, init
from dotenv import load_dotenv
import json
from googleapiclient.discovery import build

# Initialize colorama
init()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cse_scraper")

def google_cse_search(query, num_results=5):
    """
    Search for articles using Google Custom Search Engine.
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return
        
    Returns:
        list: List of search results
    """
    logger.info(f"Searching for: {query}")
    
    # Get API key and CSE ID from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        logger.error("Missing Google API key or CSE ID")
        return []
    
    try:
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=num_results
        ).execute()
        
        # Extract the search results
        search_results = []
        if "items" in result:
            for item in result["items"]:
                search_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            
            logger.info(f"Found {len(search_results)} search results")
        else:
            logger.warning("No search results found")
        
        return search_results
    
    except Exception as e:
        logger.error(f"Error in Google CSE search: {str(e)}")
        return []

def extract_legal_references(text):
    """Extract legal references from text."""
    logger.info("Extracting legal references from text")
    
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
                logger.info(f"Found legal reference: {reference}")
    
    return references

def scrape_article(article_url):
    """
    Scrape a Hukumonline article.
    
    Args:
        article_url (str): URL of the article to scrape
        
    Returns:
        tuple: (content, metadata, pdf_links)
    """
    logger.info(f"Scraping article: {article_url}")
    
    try:
        # Create a session with retries
        session = requests.Session()
        
        # Make the request
        logger.info("Making HTTP request")
        response = session.get(article_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve article: HTTP {response.status_code}")
            return None, {}, []
        
        logger.info(f"Successfully retrieved article: HTTP {response.status_code}")
        logger.info(f"Response content length: {len(response.content)} bytes")
        
        # Parse the HTML
        logger.info("Parsing HTML with BeautifulSoup")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract metadata
        metadata = {}
        
        # Extract title
        title_elem = soup.find('h1')
        if title_elem:
            metadata['title'] = title_elem.text.strip()
            logger.info(f"Found title: {metadata['title']}")
        else:
            logger.warning("Could not find h1 title element")
            
            # Try alternative title elements
            alt_title = soup.find('title')
            if alt_title:
                metadata['title'] = alt_title.text.strip()
                logger.info(f"Found alternative title: {metadata['title']}")
            else:
                logger.warning("Could not find any title element")
                metadata['title'] = "Untitled"
        
        # Print the HTML structure for debugging
        logger.info("HTML Structure Overview:")
        for i, tag in enumerate(soup.find_all(['div', 'article', 'section', 'main'])[:10]):
            class_attr = tag.get('class', [])
            id_attr = tag.get('id', '')
            class_str = ' '.join(class_attr) if class_attr else ''
            id_str = id_attr if id_attr else ''
            logger.info(f"Tag {i+1}: <{tag.name} class='{class_str}' id='{id_str}'>")
        
        # Look for article content
        logger.info("Looking for article content")
        
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
            '.article-detail'
        ]
        
        content_elem = None
        for selector in content_selectors:
            logger.info(f"Trying selector: {selector}")
            elem = soup.select_one(selector)
            if elem:
                content_elem = elem
                logger.info(f"Found content with selector: {selector}")
                break
        
        # If we couldn't find the content with selectors, try to find the main text content
        if not content_elem:
            logger.warning("Could not find content with selectors, trying to find main text content")
            
            # Look for the largest text block
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Find the paragraph with the most text
                largest_p = max(paragraphs, key=lambda p: len(p.text.strip()))
                
                # Get the parent that likely contains the article content
                content_elem = largest_p.parent
                logger.info(f"Found content using largest paragraph approach")
            else:
                logger.warning("Could not find any paragraphs")
        
        # Extract the text content and look for PDF links
        content = None
        pdf_links = []
        
        if content_elem:
            # Remove script and style elements
            for script in content_elem.find_all(['script', 'style']):
                script.decompose()
            
            # Get the text content
            content = content_elem.get_text(separator='\n').strip()
            logger.info(f"Extracted content: {len(content)} characters")
            
            # Look for PDF links
            logger.info("Looking for PDF links")
            for link in soup.find_all('a'):
                href = link.get('href')
                if not href:
                    continue
                
                # Make the URL absolute
                url = urljoin(article_url, href)
                
                # Check if it's a PDF
                if url.lower().endswith('.pdf'):
                    pdf_links.append({
                        'url': url,
                        'text': link.text.strip() or "PDF Document"
                    })
            
            if pdf_links:
                logger.info(f"Found {len(pdf_links)} PDF links")
            else:
                logger.warning("No PDF links found")
            
            # Look for "Dasar Hukum" section
            logger.info("Looking for 'Dasar Hukum' section")
            dasar_hukum_found = False
            
            # Check if "Dasar Hukum" is in the content
            if "Dasar Hukum" in content:
                dasar_hukum_found = True
                logger.info("Found 'Dasar Hukum' in content")
            
            # Look for headings that contain "Dasar Hukum"
            if not dasar_hukum_found:
                for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    if 'Dasar Hukum' in heading.text:
                        dasar_hukum_found = True
                        logger.info(f"Found 'Dasar Hukum' in heading: {heading.name}")
                        break
            
            # Look for strong or b tags containing "Dasar Hukum"
            if not dasar_hukum_found:
                for tag in soup.find_all(['strong', 'b']):
                    if 'Dasar Hukum' in tag.text:
                        dasar_hukum_found = True
                        logger.info(f"Found 'Dasar Hukum' in tag: {tag.name}")
                        break
            
            if dasar_hukum_found:
                metadata['has_dasar_hukum'] = True
            else:
                metadata['has_dasar_hukum'] = False
        else:
            logger.error("Could not find article content")
        
        return content, metadata, pdf_links
    
    except Exception as e:
        logger.error(f"Error scraping article: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}, []

def main():
    """Main function to search and scrape articles."""
    print(f"{Fore.CYAN}Searching and Scraping Hukumonline Articles{Style.RESET_ALL}")
    
    # Search queries to try
    search_queries = [
        "UU Cipta Kerja",
        "UU ITE",
        "Peraturan Pemerintah",
        "Peraturan Mahkamah Agung"
    ]
    
    for query in search_queries:
        print(f"\n{Fore.YELLOW}Searching for: {query}{Style.RESET_ALL}")
        
        # Search for articles using Google CSE
        search_results = google_cse_search(query, num_results=3)
        
        if not search_results:
            print(f"{Fore.RED}No search results found for: {query}{Style.RESET_ALL}")
            continue
        
        # Scrape each article
        for i, result in enumerate(search_results, 1):
            article_url = result['link']
            print(f"\n{Fore.GREEN}Article {i}: {result['title']}{Style.RESET_ALL}")
            print(f"URL: {article_url}")
            
            # Scrape the article
            content, metadata, pdf_links = scrape_article(article_url)
            
            if content:
                print(f"\n{Fore.GREEN}Successfully scraped article:{Style.RESET_ALL}")
                print(f"Title: {metadata.get('title', 'Untitled')}")
                
                # Show a preview of the content
                content_preview = content[:500] + "..." if len(content) > 500 else content
                print(f"\nContent Preview:\n{content_preview}")
                
                # Extract legal references
                references = extract_legal_references(content)
                if references:
                    print(f"\n{Fore.YELLOW}Legal References:{Style.RESET_ALL}")
                    for ref in references:
                        print(f"- {ref}")
                else:
                    print(f"\n{Fore.YELLOW}No legal references found{Style.RESET_ALL}")
                
                # Show PDF links
                if pdf_links:
                    print(f"\n{Fore.GREEN}PDF Links ({len(pdf_links)}):{Style.RESET_ALL}")
                    for pdf in pdf_links:
                        print(f"- {pdf['text']}: {pdf['url']}")
                else:
                    print(f"\n{Fore.YELLOW}No PDF links found{Style.RESET_ALL}")
                
                # Show if "Dasar Hukum" section was found
                if metadata.get('has_dasar_hukum', False):
                    print(f"\n{Fore.GREEN}Found 'Dasar Hukum' section in the article{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.YELLOW}No 'Dasar Hukum' section found in the article{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}Failed to scrape article content{Style.RESET_ALL}")
            
            print(f"\n{'-' * 80}")
        
        # Only process the first successful query
        if search_results:
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
