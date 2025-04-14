"""
Test Document Summarization and Legal Reference Extraction

This script tests the document summarization and legal reference extraction functionality.
It searches for information using Google CSE, extracts legal references,
retrieves full legal documents, and provides a comprehensive summary.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_summarize")

# Load environment variables
load_dotenv()

def main():
    """Test document summarization and legal reference extraction."""
    try:
        # Import functions from langgraph
        from langgraph import (
            google_cse_search_and_scrape,
            extract_legal_references,
            retrieve_legal_documents,
            summarize_documents,
            convert_documents
        )
        
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
        
        print(f"\nSearching for: {query}")
        print("Retrieving and processing documents...\n")
        
        # Step 1: Perform Google CSE search and scraping
        documents = google_cse_search_and_scrape(query, num_results=5)
        
        if not documents:
            print("No results found.")
            return
        
        print(f"Found {len(documents)} documents from Google CSE")
        
        # Step 2: Convert documents to ensure compatibility
        documents = convert_documents(documents)
        
        # Step 3: Extract legal references from documents
        all_text = ""
        for doc in documents:
            if isinstance(doc, dict):
                all_text += doc.get('page_content', '') + " "
            else:
                all_text += getattr(doc, 'page_content', '') + " "
                
        legal_references = extract_legal_references(all_text)
        
        if legal_references:
            print(f"\nFound {len(legal_references)} legal references:")
            for ref in legal_references:
                print(f"- {ref}")
            
            # Step 4: Retrieve legal documents
            print("\nRetrieving legal documents...")
            legal_docs = retrieve_legal_documents(legal_references)
            
            if legal_docs:
                print(f"Retrieved {len(legal_docs)} legal documents")
                # Add legal documents to the collection
                documents.extend(legal_docs)
        else:
            print("\nNo legal references found in the documents")
        
        # Step 5: Summarize all documents
        print("\nSummarizing documents...")
        summary = summarize_documents(documents, query)
        
        # Display summary
        print("\n" + "="*80)
        print("DOCUMENT SUMMARY")
        print("="*80)
        print(summary)
        print("="*80)
    
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
        print("Make sure you have installed all required packages:")
        print("pip install langchain langchain_openai python-dotenv requests beautifulsoup4")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
