"""
Test HTML Content Scraping

This script tests the functionality to scrape HTML content from Hukumonline articles
and logs the process.
"""

import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
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
logger = logging.getLogger("html_scraper")

def scrape_article_content(article_url):
    """
    Scrape the HTML content from a Hukumonline article.
    
    Args:
        article_url (str): URL of the article to scrape
        
    Returns:
        tuple: (content, metadata)
    """
    logger.info(f"Scraping article content from: {article_url}")
    
    # Create a session with retries
    session = requests.Session()
    
    try:
        # Make the request
        logger.info(f"Making HTTP request to: {article_url}")
        response = session.get(article_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve article: HTTP {response.status_code}")
            return None, {}
            
        logger.info(f"Successfully retrieved article: HTTP {response.status_code}")
        
        # Parse the HTML
        logger.info("Parsing HTML content with BeautifulSoup")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract metadata
        metadata = {}
        
        # Try to get the title
        title_elem = soup.find('h1')
        if title_elem:
            metadata['title'] = title_elem.text.strip()
            logger.info(f"Found title: {metadata['title']}")
        else:
            logger.warning("Could not find title element (h1)")
            
            # Try alternative title elements
            alt_title = soup.find('title')
            if alt_title:
                metadata['title'] = alt_title.text.strip()
                logger.info(f"Found alternative title: {metadata['title']}")
        
        # Try to get the publication date
        date_elem = soup.find(class_=lambda c: c and ('date' in c.lower() or 'time' in c.lower() or 'publish' in c.lower()))
        if date_elem:
            metadata['date'] = date_elem.text.strip()
            logger.info(f"Found date: {metadata['date']}")
        else:
            logger.warning("Could not find date element")
        
        # Look for the article content
        logger.info("Looking for article content")
        
        # Try different selectors for the article content
        content_selectors = [
            'article', 
            '.article-content', 
            '.content', 
            '.post-content',
            '#article-content',
            '.entry-content'
        ]
        
        article_content = None
        for selector in content_selectors:
            logger.info(f"Trying selector: {selector}")
            content_elem = soup.select_one(selector)
            if content_elem:
                article_content = content_elem
                logger.info(f"Found content with selector: {selector}")
                break
        
        # If we couldn't find the content with selectors, try to find the main text content
        if not article_content:
            logger.warning("Could not find content with selectors, trying to find main text content")
            
            # Look for the largest text block
            paragraphs = soup.find_all('p')
            if paragraphs:
                # Find the paragraph with the most text
                largest_p = max(paragraphs, key=lambda p: len(p.text.strip()))
                
                # Get the parent that likely contains the article content
                article_content = largest_p.parent
                logger.info(f"Found content using largest paragraph approach")
        
        # Extract the text content
        if article_content:
            # Remove script and style elements
            for script in article_content.find_all(['script', 'style']):
                script.decompose()
            
            # Get the text content
            content = article_content.get_text(separator='\n').strip()
            logger.info(f"Extracted content: {len(content)} characters")
            
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
                logger.info(f"Found {len(pdf_links)} PDF links:")
                for pdf in pdf_links:
                    logger.info(f"- {pdf['text']}: {pdf['url']}")
            else:
                logger.warning("No PDF links found")
            
            return content, metadata
        else:
            logger.error("Could not find article content")
            return None, metadata
            
    except Exception as e:
        logger.error(f"Error scraping article content: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}

def main():
    """Main function to test HTML scraping."""
    print(f"{Fore.CYAN}Testing HTML Content Scraping{Style.RESET_ALL}")
    
    # Test article URLs
    article_urls = [
        "https://www.hukumonline.com/berita/a/aturan-ketenagakerjaan-di-uu-cipta-kerja-dinilai-memenuhi-teori-keadilan-lt605dd29a2680d",
        "https://www.hukumonline.com/berita/a/ini-isi-uu-cipta-kerja-yang-disahkan-dpr-lt5f7c0a4b1b331",
        "https://www.hukumonline.com/klinik/a/perbedaan-undang-undang-dan-peraturan-pemerintah-lt550c0a7474043"
    ]
    
    for i, article_url in enumerate(article_urls, 1):
        print(f"\n{Fore.YELLOW}Testing Article {i}: {article_url}{Style.RESET_ALL}")
        
        content, metadata = scrape_article_content(article_url)
        
        if content:
            print(f"\n{Fore.GREEN}Successfully scraped article:{Style.RESET_ALL}")
            print(f"Title: {metadata.get('title', 'Untitled')}")
            print(f"Date: {metadata.get('date', 'Unknown')}")
            
            # Show a preview of the content
            content_preview = content[:500] + "..." if len(content) > 500 else content
            print(f"\nContent Preview:\n{content_preview}")
        else:
            print(f"\n{Fore.RED}Failed to scrape article content.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
