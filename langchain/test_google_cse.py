"""
Test Google CSE Search and Scraping

This script tests the Google Custom Search Engine (CSE) integration
to search for information, open the most relevant result, and scrape the related information.
"""

import os
import sys
import json
from dotenv import load_dotenv
from langgraph import google_cse_search_and_scrape, convert_documents

# Load environment variables
load_dotenv()

def main():
    """Test Google CSE search and scraping."""
    # Check if API keys are configured
    try:
        from env_config import GOOGLE_API_KEY, GOOGLE_CSE_ID
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            print("Error: Google API Key or CSE ID not configured in env_config.py")
            print("Please add your Google API Key and CSE ID to env_config.py")
            return
    except ImportError:
        # Check environment variables
        if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
            print("Error: Google API Key or CSE ID not found in environment variables")
            print("Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables")
            return
    
    # Get search query from user
    query = input("Enter your search query: ")
    if not query:
        print("Error: No query provided")
        return
    
    # Set number of results
    try:
        num_results = int(input("Enter number of results to retrieve (default: 5): ") or "5")
    except ValueError:
        num_results = 5
    
    print(f"\nSearching for: {query}")
    print(f"Retrieving {num_results} results...\n")
    
    # Perform Google CSE search and scraping
    documents = google_cse_search_and_scrape(query, num_results=num_results)
    
    # Display results
    if not documents:
        print("No results found.")
        return
    
    print(f"Found {len(documents)} documents:\n")
    
    # Convert documents to ensure compatibility
    documents = convert_documents(documents)
    
    # Display document information
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:")
        # Check if doc is a dictionary or an object with metadata attribute
        if isinstance(doc, dict):
            metadata = doc.get('metadata', {})
            content = doc.get('page_content', '')
            print(f"Title: {metadata.get('title', 'Untitled')}")
            print(f"URL: {metadata.get('url', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
        else:
            # Assume it's an object with metadata attribute
            metadata = getattr(doc, 'metadata', {})
            content = getattr(doc, 'page_content', '')
            print(f"Title: {metadata.get('title', 'Untitled')}")
            print(f"URL: {metadata.get('url', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
        
        # Display content preview (first 200 characters)
        content_preview = content[:200] + "..." if len(content) > 200 else content
        print(f"Content Preview: {content_preview}")
        print("-" * 80)
    
    # Ask if user wants to save documents to a file
    save_option = input("Do you want to save the documents to a file? (y/n): ").lower()
    if save_option == 'y':
        filename = input("Enter filename (default: google_cse_results.json): ") or "google_cse_results.json"
        
        # Convert documents to JSON-serializable format
        docs_json = []
        for doc in documents:
            if isinstance(doc, dict):
                docs_json.append(doc)
            else:
                docs_json.append({
                    "page_content": getattr(doc, 'page_content', ''),
                    "metadata": getattr(doc, 'metadata', {})
                })
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(docs_json, f, ensure_ascii=False, indent=2)
        
        print(f"Documents saved to {filename}")

if __name__ == "__main__":
    main()
