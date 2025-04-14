"""
Scrape a single Hukumonline article and print detailed logs

This script scrapes a specific Hukumonline article, extracts the content,
looks for legal references, and checks for PDF links.
"""

import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from colorama import Fore, Style, init

# Initialize colorama
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("article_scraper")

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
    """
    logger.info(f"Scraping article: {article_url}")
    
    try:
        # Create a session
        session = requests.Session()
        
        # Make the request
        logger.info("Making HTTP request")
        response = session.get(article_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve article: HTTP {response.status_code}")
            return
        
        logger.info(f"Successfully retrieved article: HTTP {response.status_code}")
        logger.info(f"Response content length: {len(response.content)} bytes")
        
        # Parse the HTML
        logger.info("Parsing HTML with BeautifulSoup")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = None
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.text.strip()
            logger.info(f"Found title: {title}")
        else:
            logger.warning("Could not find h1 title element")
            
            # Try alternative title elements
            alt_title = soup.find('title')
            if alt_title:
                title = alt_title.text.strip()
                logger.info(f"Found alternative title: {title}")
            else:
                logger.warning("Could not find any title element")
        
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
            '.main-content'
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
        
        # Extract the text content
        if content_elem:
            # Remove script and style elements
            for script in content_elem.find_all(['script', 'style']):
                script.decompose()
            
            # Get the text content
            content = content_elem.get_text(separator='\n').strip()
            logger.info(f"Extracted content: {len(content)} characters")
            
            # Print a preview of the content
            content_preview = content[:500] + "..." if len(content) > 500 else content
            print(f"\n{Fore.GREEN}Content Preview:{Style.RESET_ALL}\n{content_preview}")
            
            # Extract legal references
            references = extract_legal_references(content)
            if references:
                print(f"\n{Fore.YELLOW}Legal References:{Style.RESET_ALL}")
                for ref in references:
                    print(f"- {ref}")
            else:
                print(f"\n{Fore.YELLOW}No legal references found{Style.RESET_ALL}")
            
            # Look for "Dasar Hukum" section
            logger.info("Looking for 'Dasar Hukum' section")
            
            # Check if "Dasar Hukum" is in the content
            if "Dasar Hukum" in content:
                logger.info("Found 'Dasar Hukum' in content")
                print(f"\n{Fore.GREEN}Found 'Dasar Hukum' in content{Style.RESET_ALL}")
                
                # Try to extract the Dasar Hukum section
                dasar_hukum_index = content.find("Dasar Hukum")
                if dasar_hukum_index >= 0:
                    # Extract text after "Dasar Hukum"
                    dasar_hukum_text = content[dasar_hukum_index:]
                    # Limit to 500 characters for preview
                    dasar_hukum_preview = dasar_hukum_text[:500] + "..." if len(dasar_hukum_text) > 500 else dasar_hukum_text
                    print(f"\n{Fore.GREEN}Dasar Hukum Section:{Style.RESET_ALL}\n{dasar_hukum_preview}")
            
            # Look for headings that contain "Dasar Hukum"
            dasar_hukum_heading = None
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if 'Dasar Hukum' in heading.text:
                    dasar_hukum_heading = heading
                    logger.info(f"Found 'Dasar Hukum' in heading: {heading.name}")
                    print(f"\n{Fore.GREEN}Found 'Dasar Hukum' in heading: {heading.name}{Style.RESET_ALL}")
                    print(f"Heading text: {heading.text.strip()}")
                    break
            
            # If we found a Dasar Hukum heading, look for content after it
            if dasar_hukum_heading:
                # Get the next sibling elements
                next_elements = []
                current = dasar_hukum_heading.next_sibling
                while current and len(next_elements) < 5:
                    if hasattr(current, 'name') and current.name:
                        next_elements.append(current)
                    current = current.next_sibling
                
                # Print the elements after the Dasar Hukum heading
                if next_elements:
                    print(f"\n{Fore.GREEN}Elements after Dasar Hukum heading:{Style.RESET_ALL}")
                    for i, elem in enumerate(next_elements):
                        print(f"{i+1}. <{elem.name}>: {elem.text.strip()[:100]}...")
            
            # Look for PDF links
            logger.info("Looking for PDF links")
            pdf_links = []
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
                print(f"\n{Fore.GREEN}PDF Links:{Style.RESET_ALL}")
                for pdf in pdf_links:
                    print(f"- {pdf['text']}: {pdf['url']}")
            else:
                logger.warning("No PDF links found")
                print(f"\n{Fore.YELLOW}No PDF links found{Style.RESET_ALL}")
        else:
            logger.error("Could not find article content")
            print(f"\n{Fore.RED}Failed to extract article content{Style.RESET_ALL}")
    
    except Exception as e:
        logger.error(f"Error scraping article: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def main():
    """Main function to scrape a single article."""
    print(f"{Fore.CYAN}Scraping Single Hukumonline Article{Style.RESET_ALL}")
    
    # Article URL to scrape
    article_url = "https://www.hukumonline.com/berita/a/ini-isi-uu-cipta-kerja-yang-disahkan-dpr-lt5f7c0a4b1b331"
    
    print(f"\n{Fore.CYAN}Scraping article: {article_url}{Style.RESET_ALL}")
    scrape_article(article_url)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
