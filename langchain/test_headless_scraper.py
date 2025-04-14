"""
Test Headless Browser Scraping

This script tests the headless browser scraping functionality in the HukumonlineScraper class.
"""

import os
import sys
import json
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize colorama
init()

# Load environment variables
load_dotenv()

# Import the HukumonlineScraper
from hukumonline import HukumonlineScraper

def test_headless_scraping():
    """Test the headless browser scraping functionality."""
    print(f"{Fore.CYAN}Initializing Hukumonline Scraper...{Style.RESET_ALL}")
    scraper = HukumonlineScraper()
    
    # Test query
    query = "UU Cipta Kerja"
    print(f"{Fore.CYAN}Testing search with query: '{query}'{Style.RESET_ALL}")
    
    # Search using Google CSE first, then fallback to headless browser if needed
    documents = scraper.search(query, max_results=3)
    
    print(f"\n{Fore.GREEN}Documents Found: {len(documents)}{Style.RESET_ALL}")
    
    # Display document information
    for i, doc in enumerate(documents, 1):
        print(f"\n{Fore.YELLOW}Document {i}:{Style.RESET_ALL}")
        print(f"Title: {doc.metadata.get('title', 'Untitled')}")
        print(f"URL: {doc.metadata.get('url', 'No URL')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Show a preview of the content
        content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
        print(f"\nContent Preview:\n{content_preview}")
        
        # Extract legal references
        references = scraper.extract_legal_references(doc.content)
        if references:
            print(f"\nLegal References:")
            for ref in references:
                print(f"- {ref}")
    
    # If documents were found, test summarization
    if documents:
        print(f"\n{Fore.CYAN}Testing document summarization...{Style.RESET_ALL}")
        summary = scraper.summarize_documents(query, documents)
        print(f"\n{Fore.GREEN}Summary:{Style.RESET_ALL}\n{summary}")
    else:
        print(f"\n{Fore.RED}No documents found to summarize.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        test_headless_scraping()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
