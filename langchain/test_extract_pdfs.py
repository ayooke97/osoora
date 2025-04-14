"""
Test PDF Extraction from "Dasar Hukum" Section

This script tests the functionality to extract PDFs from the "Dasar Hukum" section
of Hukumonline articles.
"""

import os
import sys
from colorama import Fore, Style, init
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Initialize colorama
init()

# Load environment variables
load_dotenv()

# Import the HukumonlineScraper
from hukumonline import HukumonlineScraper

def find_article_with_pdfs():
    """Find a Hukumonline article that likely has PDFs."""
    print(f"{Fore.CYAN}Searching for an article with PDFs...{Style.RESET_ALL}")
    
    # Try some search queries that are likely to return articles with legal references
    search_queries = [
        "UU Cipta Kerja",
        "UU ITE",
        "Peraturan Pemerintah",
        "Peraturan Mahkamah Agung"
    ]
    
    # Create a session with retries
    session = requests.Session()
    
    for query in search_queries:
        try:
            # Search on Hukumonline
            search_url = f"https://www.hukumonline.com/search/a/c/q/{query.replace(' ', '+')}"
            print(f"{Fore.CYAN}Searching: {search_url}{Style.RESET_ALL}")
            
            response = session.get(search_url)
            if response.status_code != 200:
                print(f"{Fore.RED}Failed to search: {response.status_code}{Style.RESET_ALL}")
                continue
                
            # Parse the search results
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if not href:
                    continue
                    
                # Look for article URLs
                if '/berita/' in href or '/klinik/' in href or '/pustaka/' in href:
                    article_links.append(href)
            
            # Check each article for PDFs
            for article_url in article_links[:5]:  # Check the first 5 articles
                # Make the URL absolute
                if not article_url.startswith('http'):
                    article_url = urljoin("https://www.hukumonline.com", article_url)
                    
                print(f"{Fore.YELLOW}Checking article: {article_url}{Style.RESET_ALL}")
                
                # Get the article
                article_response = session.get(article_url)
                if article_response.status_code != 200:
                    print(f"{Fore.RED}Failed to retrieve article: {article_response.status_code}{Style.RESET_ALL}")
                    continue
                    
                # Parse the article
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                # Look for PDF links
                pdf_links = []
                for link in article_soup.find_all('a'):
                    href = link.get('href')
                    if not href:
                        continue
                        
                    if href.lower().endswith('.pdf'):
                        pdf_links.append(href)
                
                # If we found PDFs, return this article URL
                if pdf_links:
                    print(f"{Fore.GREEN}Found article with PDFs: {article_url}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}PDF links: {pdf_links}{Style.RESET_ALL}")
                    return article_url
            
        except Exception as e:
            print(f"{Fore.RED}Error searching for articles: {str(e)}{Style.RESET_ALL}")
    
    # If we didn't find any articles with PDFs, return a default one
    return "https://www.hukumonline.com/berita/a/aturan-ketenagakerjaan-di-uu-cipta-kerja-dinilai-memenuhi-teori-keadilan-lt605dd29a2680d"

def test_extract_legal_basis_pdfs():
    """Test extracting PDFs from the 'Dasar Hukum' section of an article."""
    print(f"{Fore.CYAN}Initializing Hukumonline Scraper...{Style.RESET_ALL}")
    scraper = HukumonlineScraper()
    
    # Find an article with PDFs
    article_url = find_article_with_pdfs()
    
    print(f"\n{Fore.CYAN}Extracting PDFs from article: {article_url}{Style.RESET_ALL}")
    
    # Extract PDFs from the article
    pdf_documents = scraper.extract_legal_basis_pdfs(article_url)
    
    print(f"\n{Fore.GREEN}PDFs Found: {len(pdf_documents)}{Style.RESET_ALL}")
    
    # Display PDF information
    for i, pdf in enumerate(pdf_documents, 1):
        print(f"\n{Fore.YELLOW}PDF {i}:{Style.RESET_ALL}")
        print(f"Title: {pdf['title']}")
        print(f"URL: {pdf['url']}")
        print(f"Source: {pdf['metadata'].get('source', 'Unknown')}")
        
        # Show a preview of the content
        content_preview = pdf['content'][:200] + "..." if len(pdf['content']) > 200 else pdf['content']
        print(f"\nContent Preview:\n{content_preview}")
        
        # Extract legal references from the PDF content
        references = scraper.extract_legal_references(pdf['content'])
        if references:
            print(f"\nLegal References in PDF:")
            for ref in references:
                print(f"- {ref}")
    
    # If no PDFs were found, try to extract content from the article itself
    if not pdf_documents:
        print(f"\n{Fore.YELLOW}No PDFs found. Extracting article content...{Style.RESET_ALL}")
        
        try:
            # Extract article content
            article_content, article_metadata = scraper._extract_article_content(article_url)
            
            if article_content:
                print(f"\n{Fore.GREEN}Article Content:{Style.RESET_ALL}")
                print(f"Title: {article_metadata.get('title', 'Untitled')}")
                
                # Show a preview of the content
                content_preview = article_content[:500] + "..." if len(article_content) > 500 else article_content
                print(f"\nContent Preview:\n{content_preview}")
                
                # Extract legal references from the article content
                references = scraper.extract_legal_references(article_content)
                if references:
                    print(f"\n{Fore.GREEN}Legal References in Article:{Style.RESET_ALL}")
                    for ref in references:
                        print(f"- {ref}")
            else:
                print(f"\n{Fore.RED}Failed to extract article content.{Style.RESET_ALL}")
                
                # Try using the headless browser to extract content
                print(f"\n{Fore.YELLOW}Trying headless browser extraction...{Style.RESET_ALL}")
                try:
                    article_content, article_metadata = scraper._extract_article_content_headless(article_url)
                    
                    if article_content:
                        print(f"\n{Fore.GREEN}Article Content (Headless):{Style.RESET_ALL}")
                        print(f"Title: {article_metadata.get('title', 'Untitled')}")
                        
                        # Show a preview of the content
                        content_preview = article_content[:500] + "..." if len(article_content) > 500 else article_content
                        print(f"\nContent Preview:\n{content_preview}")
                        
                        # Extract legal references from the article content
                        references = scraper.extract_legal_references(article_content)
                        if references:
                            print(f"\n{Fore.GREEN}Legal References in Article:{Style.RESET_ALL}")
                            for ref in references:
                                print(f"- {ref}")
                    else:
                        print(f"\n{Fore.RED}Failed to extract article content with headless browser.{Style.RESET_ALL}")
                except Exception as e:
                    print(f"\n{Fore.RED}Error with headless browser extraction: {str(e)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"\n{Fore.RED}Error extracting article content: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        test_extract_legal_basis_pdfs()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
