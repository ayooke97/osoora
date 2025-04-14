"""
Test script for the updated Peraturan Scraper with PDF extraction capabilities.
This script tests the scraping functionality for peraturan.go.id, including PDF extraction.
"""

import argparse
import sys
import os
import time
from colorama import init, Fore, Style

# Add the parent directory to the path to import the scraper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the scraper modules directly
from bpk_scraper import PeraturanScraper, Document
from langgraph import scrape_peraturan_go_id_direct, extract_pdf_content

# Initialize colorama
init()

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """Print colored text"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def print_header(text):
    """Print a header"""
    print("\n" + "=" * 35)
    print_colored(text, Fore.CYAN, Style.BRIGHT)
    print("=" * 35)

def print_success(text):
    """Print a success message"""
    print_colored(text, Fore.GREEN)

def print_error(text):
    """Print an error message"""
    print_colored(text, Fore.RED)

def print_warning(text):
    """Print a warning message"""
    print_colored(text, Fore.YELLOW)

def print_info(text):
    """Print an info message"""
    print_colored(text, Fore.BLUE)

def print_document_preview(doc, index=None):
    """Print a preview of a document"""
    if index is not None:
        print_colored(f"Document {index}: {doc.metadata.get('title', 'Untitled')} ({len(doc.page_content)} chars)", Fore.GREEN)
    else:
        print_colored(f"{doc.metadata.get('title', 'Untitled')} ({len(doc.page_content)} chars)", Fore.GREEN)
    
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Type: {doc.metadata.get('type', 'Unknown Type')}")
    
    # Print content preview
    content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
    print(f"Content preview: {content_preview}")
    print("-" * 50)

def test_manual_pdf_extraction():
    """Test manual PDF extraction"""
    print_header("Testing Manual PDF Extraction")
    
    try:
        print_info("Testing manual PDF extraction...")
        
        # Test PDF URLs
        pdf_urls = [
            "https://peraturan.go.id/files/UUD1945.pdf",
            # Add more test URLs as needed
        ]
        
        for pdf_url in pdf_urls:
            print_info(f"Processing PDF: {pdf_url}")
            
            # Use the extract_pdf_content function from langgraph
            pdf_content, pdf_metadata = extract_pdf_content(pdf_url)
            
            if pdf_content:
                print_success(f"Successfully extracted {len(pdf_content)} characters from PDF")
                
                # Print the first few lines of each page
                lines = pdf_content.split('\n\n')
                for i, line in enumerate(lines[:5]):
                    if line.strip():
                        print_info(f"Page {i+1} preview: {line[:100]}...")
            else:
                print_error(f"Failed to extract content from {pdf_url}")
    
    except Exception as e:
        print_error(f"Error in manual PDF extraction test: {str(e)}")

def test_peraturan_scraper(query="undang-undang dasar", max_pages=1):
    """Test the PeraturanScraper class"""
    print_header("Testing PeraturanScraper")
    
    try:
        print_info(f"Testing PeraturanScraper with query: '{query}' and max_pages: {max_pages}")
        
        # Initialize the scraper
        scraper = PeraturanScraper()
        
        # Start timer
        start_time = time.time()
        
        # Scrape documents
        documents = scraper.scrape_peraturan_go_id(query, max_pages)
        
        # End timer
        end_time = time.time()
        
        print_success(f"Scraped {len(documents)} documents in {end_time - start_time:.2f} seconds")
        
        # Print document previews
        for i, doc in enumerate(documents[:3], 1):
            print_document_preview(doc, i)
        
        if len(documents) > 3:
            print_info(f"... and {len(documents) - 3} more documents")
    
    except Exception as e:
        print_error(f"Error in PeraturanScraper test: {str(e)}")

def test_direct_scraping(query="undang-undang dasar", max_pages=1):
    """Test direct scraping function"""
    print_header("Testing Direct Scraping")
    
    try:
        print_info(f"Testing direct scraping with query: '{query}' and max_pages: {max_pages}")
        
        # Start timer
        start_time = time.time()
        
        # Scrape documents directly
        documents = scrape_peraturan_go_id_direct(query, max_pages)
        
        # End timer
        end_time = time.time()
        
        print_success(f"Found {len(documents)} documents with direct scraping in {end_time - start_time:.2f} seconds")
        
        # Count PDF documents
        pdf_docs = [doc for doc in documents if doc.metadata.get('type', '').lower().startswith('pdf')]
        print_success(f"Found {len(pdf_docs)} PDF documents")
        
        # Print document previews
        for i, doc in enumerate(documents[:3], 1):
            print_document_preview(doc, i)
        
        if len(documents) > 3:
            print_info(f"... and {len(documents) - 3} more documents")
    
    except Exception as e:
        print_error(f"Error in direct scraping test: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test the Peraturan Scraper')
    parser.add_argument('--query', type=str, default='undang-undang dasar', help='Query to search for')
    parser.add_argument('--max-pages', type=int, default=1, help='Maximum number of pages to scrape')
    parser.add_argument('--test', type=str, choices=['all', 'manual', 'scraper', 'direct'], default='all', 
                        help='Which test to run (all, manual, scraper, direct)')
    
    args = parser.parse_args()
    
    print_colored("Starting scraper tests...", Fore.CYAN, Style.BRIGHT)
    
    try:
        if args.test in ['all', 'manual']:
            test_manual_pdf_extraction()
        
        if args.test in ['all', 'scraper']:
            test_peraturan_scraper(args.query, args.max_pages)
        
        if args.test in ['all', 'direct']:
            test_direct_scraping(args.query, args.max_pages)
        
        print_success("\nTests completed.")
    
    except Exception as e:
        print_error(f"\nError in tests: {str(e)}")

if __name__ == "__main__":
    main()
