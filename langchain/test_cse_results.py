"""
Test Google CSE Results

This script tests the Google Custom Search Engine functionality
and prints the query results.
"""

import os
import sys
import json
from dotenv import load_dotenv
from googleapiclient.discovery import build
from colorama import Fore, Style, init

# Initialize colorama
init()

# Load environment variables
load_dotenv()

def google_cse_search(query, num_results=5):
    """
    Search for articles using Google Custom Search Engine.
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return
        
    Returns:
        list: List of search results
    """
    print(f"Searching for: {query}")
    
    # Get API key and CSE ID directly from env_config.py
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    # If not found in environment, use hardcoded values from env_config.py
    if not api_key:
        api_key = "AIzaSyD2MDuKtg2ZwMLCYYaOgDm4RWTMNnPIKHk"
        print(f"Using hardcoded API key: {api_key[:5]}...{api_key[-5:]}")
    
    if not cse_id:
        cse_id = "124f4fd8c7c694b40"
        print(f"Using hardcoded CSE ID: {cse_id}")
    
    try:
        # Build the service
        print("Building Google API service...")
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute the search
        print(f"Executing search for: {query}")
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=num_results
        ).execute()
        
        # Print the raw result for debugging
        print(f"\n{Fore.CYAN}Raw API Response:{Style.RESET_ALL}")
        print(json.dumps(result, indent=2))
        
        # Extract the search results
        search_results = []
        if "items" in result:
            for item in result["items"]:
                search_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            
            print(f"\n{Fore.GREEN}Found {len(search_results)} search results{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}No search results found{Style.RESET_ALL}")
        
        return search_results
    
    except Exception as e:
        print(f"\n{Fore.RED}Error in Google CSE search: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main function to test Google CSE."""
    print(f"{Fore.CYAN}Testing Google CSE Results{Style.RESET_ALL}")
    
    # Test queries
    queries = [
        "UU Cipta Kerja",
        "UU ITE",
        "Peraturan Pemerintah"
    ]
    
    for query in queries:
        print(f"\n{Fore.YELLOW}Testing query: {query}{Style.RESET_ALL}")
        
        # Search using Google CSE
        search_results = google_cse_search(query, num_results=3)
        
        # Print the search results
        if search_results:
            print(f"\n{Fore.GREEN}Search Results:{Style.RESET_ALL}")
            for i, result in enumerate(search_results, 1):
                print(f"\n{Fore.YELLOW}Result {i}:{Style.RESET_ALL}")
                print(f"Title: {result['title']}")
                print(f"URL: {result['link']}")
                print(f"Snippet: {result['snippet']}")
        
        print(f"\n{'-' * 80}")
        
        # Only test the first query that returns results
        if search_results:
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
