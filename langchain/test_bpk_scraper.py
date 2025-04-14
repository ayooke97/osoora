"""
Test script for the BPK Scraper with Legal LLM integration.
This script demonstrates how to use the BPK Scraper to retrieve legal documents
from peraturan.bpk.go.id and process queries using the legal LLM.
"""

import os
import sys
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables from .env file
load_dotenv()

# Import the BPKScraper class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bpk_scraper import BPKScraper

def test_bpk_scraper():
    """Test the BPK Scraper with a sample query."""
    print(f"\n{Fore.BLUE}Initializing BPK Scraper...{Style.RESET_ALL}")
    
    # Initialize the BPK Scraper
    scraper = BPKScraper()
    
    # Sample queries to test
    test_queries = [
        "Peraturan tentang keuangan negara",
        "Undang-undang tentang pemeriksaan pengelolaan keuangan negara",
        "Peraturan BPK tentang audit kinerja"
    ]
    
    # Process each query
    for query in test_queries:
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Testing query: {query}{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        # Set user preferences
        user_preferences = {
            'verbosity': 'detailed',
            'format': 'simple',
            'citations': True,
            'max_results': 5
        }
        
        try:
            # Process the query with the legal LLM
            result = scraper.process_query_with_llm(query, user_preferences)
            
            # Print the results
            print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"\n{Fore.GREEN}Original Query: {result['original_query']}{Style.RESET_ALL}")
            
            if 'keywords' in result and result['keywords']:
                print(f"\n{Fore.GREEN}Extracted Keywords: {', '.join(result['keywords'])}{Style.RESET_ALL}")
            
            print(f"\n{Fore.GREEN}Documents Retrieved: {len(result['documents'])}{Style.RESET_ALL}")
            
            # Print document details
            for i, doc in enumerate(result['documents']):
                print(f"\n{Fore.YELLOW}Document {i+1}:{Style.RESET_ALL}")
                print(f"  Title: {doc.metadata.get('title', 'Untitled')}")
                print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"  Type: {doc.metadata.get('type', 'Unknown')}")
                print(f"  Content Length: {len(doc.content)} characters")
            
            print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL}")
            print(f"\n{result['response']}")
            print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}Error processing query: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_bpk_scraper()
