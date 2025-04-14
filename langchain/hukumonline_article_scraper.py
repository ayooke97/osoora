"""
Hukumonline Article Scraper

This script scrapes specific Hukumonline articles to extract content,
legal references, and PDF links.
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from colorama import Fore, Style, init
import time

# Initialize colorama
init()

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("hukumonline_scraper")

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

def find_pdf_links(soup, base_url):
    """Find PDF links in the HTML."""
    logger.info("Looking for PDF links")
    
    pdf_links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if not href:
            continue
        
        # Make the URL absolute
        url = urljoin(base_url, href)
        
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
    
    return pdf_links

def find_dasar_hukum_section(soup, content):
    """Find the 'Dasar Hukum' section in the article."""
    logger.info("Looking for 'Dasar Hukum' section")
    
    dasar_hukum_found = False
    dasar_hukum_content = None
    
    # Check if "Dasar Hukum" is in the content
    if "Dasar Hukum" in content:
        dasar_hukum_found = True
        logger.info("Found 'Dasar Hukum' in content")
        
        # Try to extract the Dasar Hukum section
        dasar_hukum_index = content.find("Dasar Hukum")
        if dasar_hukum_index >= 0:
            # Extract text after "Dasar Hukum"
            dasar_hukum_content = content[dasar_hukum_index:dasar_hukum_index+500]
            logger.info(f"Dasar Hukum section preview: {dasar_hukum_content[:100]}...")
    
    # Look for headings that contain "Dasar Hukum"
    if not dasar_hukum_found:
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if 'Dasar Hukum' in heading.text:
                dasar_hukum_found = True
                logger.info(f"Found 'Dasar Hukum' in heading: {heading.name}")
                
                # Try to extract content after this heading
                dasar_hukum_content = ""
                current = heading.next_sibling
                
                # Collect text from the next few elements
                count = 0
                while current and count < 5:
                    if hasattr(current, 'text'):
                        dasar_hukum_content += current.text.strip() + "\n"
                        count += 1
                    current = current.next_sibling
                
                if dasar_hukum_content:
                    logger.info(f"Extracted content after 'Dasar Hukum' heading: {dasar_hukum_content[:100]}...")
                break
    
    # Look for strong or b tags containing "Dasar Hukum"
    if not dasar_hukum_found:
        for tag in soup.find_all(['strong', 'b']):
            if 'Dasar Hukum' in tag.text:
                dasar_hukum_found = True
                logger.info(f"Found 'Dasar Hukum' in tag: {tag.name}")
                
                # Try to extract content after this tag
                dasar_hukum_content = ""
                current = tag.next_sibling
                
                # Collect text from the next few elements
                count = 0
                while current and count < 5:
                    if hasattr(current, 'text'):
                        dasar_hukum_content += current.text.strip() + "\n"
                        count += 1
                    current = current.next_sibling
                
                if dasar_hukum_content:
                    logger.info(f"Extracted content after 'Dasar Hukum' tag: {dasar_hukum_content[:100]}...")
                break
    
    return dasar_hukum_found, dasar_hukum_content

def scrape_article(article_url):
    """
    Scrape a Hukumonline article.
    
    Args:
        article_url (str): URL of the article to scrape
        
    Returns:
        dict: Article data including content, legal references, and PDF links
    """
    logger.info(f"Scraping article: {article_url}")
    
    article_data = {
        'url': article_url,
        'title': None,
        'content': None,
        'legal_references': [],
        'pdf_links': [],
        'has_dasar_hukum': False,
        'dasar_hukum_content': None
    }
    
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
        logger.info("Making HTTP request")
        response = session.get(article_url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve article: HTTP {response.status_code}")
            return article_data
        
        logger.info(f"Successfully retrieved article: HTTP {response.status_code}")
        logger.info(f"Response content length: {len(response.content)} bytes")
        
        # Save the HTML content for debugging
        with open("last_article.html", "wb") as f:
            f.write(response.content)
            logger.info("Saved HTML content to last_article.html")
        
        # Parse the HTML
        logger.info("Parsing HTML with BeautifulSoup")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1')
        if title_elem:
            article_data['title'] = title_elem.text.strip()
            logger.info(f"Found title: {article_data['title']}")
        else:
            logger.warning("Could not find h1 title element")
            
            # Try alternative title elements
            alt_title = soup.find('title')
            if alt_title:
                article_data['title'] = alt_title.text.strip()
                logger.info(f"Found alternative title: {article_data['title']}")
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
            '.main-content',
            '.article-detail',
            '.post',
            '#content',
            '.klinik-detail',
            '.detail-content'
        ]
        
        content_elem = None
        for selector in content_selectors:
            logger.info(f"Trying selector: {selector}")
            elems = soup.select(selector)
            if elems:
                content_elem = elems[0]
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
            article_data['content'] = content_elem.get_text(separator='\n').strip()
            logger.info(f"Extracted content: {len(article_data['content'])} characters")
            
            # Extract legal references
            if article_data['content']:
                article_data['legal_references'] = extract_legal_references(article_data['content'])
            
            # Find PDF links
            article_data['pdf_links'] = find_pdf_links(soup, article_url)
            
            # Find "Dasar Hukum" section
            article_data['has_dasar_hukum'], article_data['dasar_hukum_content'] = find_dasar_hukum_section(
                soup, article_data['content']
            )
        else:
            logger.error("Could not find article content")
        
        return article_data
    
    except Exception as e:
        logger.error(f"Error scraping article: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return article_data

def main():
    """Main function to scrape articles."""
    print(f"{Fore.CYAN}Hukumonline Article Scraper{Style.RESET_ALL}")
    
    # List of specific Hukumonline URLs to scrape
    urls = [
        "https://www.hukumonline.com/berita/baca/lt5f7c0a4b1b331/ini-isi-uu-cipta-kerja-yang-disahkan-dpr",
        "https://www.hukumonline.com/klinik/a/cara-hitung-pesangon-lt515b7ec90fe0c/",
        "https://www.hukumonline.com/klinik/a/perbedaan-undang-undang-dan-peraturan-pemerintah-lt550c0a7474043/",
        "https://www.hukumonline.com/klinik/a/apa-yang-dimaksud-dengan-peraturan-perundang-undangan-lt4e8b93c11a0a8/",
        "https://www.hukumonline.com/berita/a/ini-isi-uu-cipta-kerja-yang-disahkan-dpr-lt5f7c0a4b1b331/"
    ]
    
    # Scrape each URL
    for i, url in enumerate(urls, 1):
        print(f"\n{Fore.YELLOW}Article {i}/{len(urls)}: {url}{Style.RESET_ALL}")
        
        # Scrape the article
        article_data = scrape_article(url)
        
        # Display the results
        if article_data['title'] and article_data['content']:
            print(f"\n{Fore.GREEN}Successfully scraped article:{Style.RESET_ALL}")
            print(f"Title: {article_data['title']}")
            
            # Show a preview of the content
            content_preview = article_data['content'][:500] + "..." if len(article_data['content']) > 500 else article_data['content']
            print(f"\nContent Preview:\n{content_preview}")
            
            # Show legal references
            if article_data['legal_references']:
                print(f"\n{Fore.YELLOW}Legal References ({len(article_data['legal_references'])}):{Style.RESET_ALL}")
                for ref in article_data['legal_references']:
                    print(f"- {ref}")
            else:
                print(f"\n{Fore.YELLOW}No legal references found{Style.RESET_ALL}")
            
            # Show PDF links
            if article_data['pdf_links']:
                print(f"\n{Fore.GREEN}PDF Links ({len(article_data['pdf_links'])}):{Style.RESET_ALL}")
                for pdf in article_data['pdf_links']:
                    print(f"- {pdf['text']}: {pdf['url']}")
            else:
                print(f"\n{Fore.YELLOW}No PDF links found{Style.RESET_ALL}")
            
            # Show if "Dasar Hukum" section was found
            if article_data['has_dasar_hukum']:
                print(f"\n{Fore.GREEN}Found 'Dasar Hukum' section in the article{Style.RESET_ALL}")
                if article_data['dasar_hukum_content']:
                    preview = article_data['dasar_hukum_content'][:300] + "..." if len(article_data['dasar_hukum_content']) > 300 else article_data['dasar_hukum_content']
                    print(f"Dasar Hukum Content:\n{preview}")
            else:
                print(f"\n{Fore.YELLOW}No 'Dasar Hukum' section found in the article{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}Failed to extract content from article{Style.RESET_ALL}")
        
        # Add delay between requests
        if i < len(urls):
            print(f"\nWaiting 2 seconds before next request...")
            time.sleep(2)
        
        print(f"\n{'-' * 80}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
