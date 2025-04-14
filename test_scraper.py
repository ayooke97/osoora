"""
Test script for the updated Peraturan Scraper with PDF extraction capabilities.
This script tests the scraping functionality for peraturan.go.id, including PDF extraction.
"""

import os
import sys
import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the scraper modules directly
from bpk_scraper import PeraturanScraper, BPKScraper, Document

def test_peraturan_scraper():
    """Test the PeraturanScraper class"""
    try:
        print(f"{Fore.GREEN}Testing PeraturanScraper...{Style.RESET_ALL}")
        scraper = PeraturanScraper()
        query = "undang-undang dasar"
        results = scraper.scrape_peraturan_go_id(query, max_pages=1)
        
        print(f"{Fore.GREEN}Found {len(results)} documents with PeraturanScraper{Style.RESET_ALL}")
        for i, doc in enumerate(results[:3]):
            print(f"{Fore.CYAN}Document {i+1}: {doc.metadata.get('title', 'No Title')} ({len(doc.page_content)} chars){Style.RESET_ALL}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Type: {doc.metadata.get('type', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:200]}...")
            print("-" * 50)
        
        return results
    except Exception as e:
        print(f"{Fore.RED}Error testing PeraturanScraper: {str(e)}{Style.RESET_ALL}")
        return []

if __name__ == "__main__":
    print(f"{Fore.YELLOW}Starting scraper tests...{Style.RESET_ALL}")
    
    # Test PeraturanScraper
    peraturan_results = test_peraturan_scraper()
    
    print(f"{Fore.YELLOW}Tests completed.{Style.RESET_ALL}")
