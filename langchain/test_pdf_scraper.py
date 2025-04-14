#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for PDF scraping functionality.
This script demonstrates how to use the PDF scraping functions to extract text and legal references from PDF files.
"""

import sys
import argparse
import time
import os
from colorama import Fore, Style, init
from retrieve_cse_results import scrape_pdf, extract_legal_references, download_pdf, extract_text_from_pdf

# Initialize colorama
init()

def main():
    """Main function to test PDF scraping functionality."""
    parser = argparse.ArgumentParser(description='Test PDF scraping functionality')
    parser.add_argument('url', help='URL of the PDF to scrape')
    parser.add_argument('--title', help='Title of the PDF (optional)')
    parser.add_argument('--save', action='store_true', help='Save the extracted text to a file')
    parser.add_argument('--output-dir', default='.', help='Directory to save output files (default: current directory)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"{Fore.CYAN}Testing PDF scraping for URL: {args.url}{Style.RESET_ALL}")
    
    start_time = time.time()
    
    # Download the PDF first
    print(f"{Fore.CYAN}Downloading PDF...{Style.RESET_ALL}")
    pdf_content = download_pdf(args.url)
    
    if not pdf_content:
        print(f"{Fore.RED}Failed to download PDF{Style.RESET_ALL}")
        sys.exit(1)
    
    download_time = time.time() - start_time
    print(f"{Fore.GREEN}PDF downloaded in {download_time:.2f} seconds ({len(pdf_content)} bytes){Style.RESET_ALL}")
    
    # Extract text from the PDF
    print(f"{Fore.CYAN}Extracting text from PDF...{Style.RESET_ALL}")
    extraction_start = time.time()
    pdf_text = extract_text_from_pdf(pdf_content)
    
    if not pdf_text:
        print(f"{Fore.RED}Failed to extract text from PDF{Style.RESET_ALL}")
        sys.exit(1)
    
    extraction_time = time.time() - extraction_start
    print(f"{Fore.GREEN}Text extracted in {extraction_time:.2f} seconds ({len(pdf_text)} characters){Style.RESET_ALL}")
    
    # Extract legal references
    print(f"{Fore.CYAN}Extracting legal references...{Style.RESET_ALL}")
    ref_start = time.time()
    legal_references = extract_legal_references(pdf_text)
    ref_time = time.time() - ref_start
    
    # Create a timestamp for filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save the PDF content to a file
    if args.save:
        title = args.title or "pdf_document"
        safe_title = ''.join(c if c.isalnum() else '_' for c in title)
        filename = os.path.join(args.output_dir, f"{safe_title}_{timestamp}.txt")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"PDF Title: {args.title or 'Unknown'}\n")
                f.write(f"PDF URL: {args.url}\n")
                f.write(f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Text Length: {len(pdf_text)} characters\n\n")
                f.write("=" * 80 + "\n")
                f.write("CONTENT\n")
                f.write("=" * 80 + "\n\n")
                f.write(pdf_text)
            
            print(f"{Fore.GREEN}Saved PDF content to: {filename}{Style.RESET_ALL}")
            
            # Save legal references to a separate file
            if legal_references:
                ref_filename = os.path.join(args.output_dir, f"{safe_title}_references_{timestamp}.txt")
                with open(ref_filename, 'w', encoding='utf-8') as f:
                    f.write(f"PDF Title: {args.title or 'Unknown'}\n")
                    f.write(f"PDF URL: {args.url}\n")
                    f.write(f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Legal References: {len(legal_references)}\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("LEGAL REFERENCES\n")
                    f.write("=" * 80 + "\n\n")
                    for i, ref in enumerate(legal_references, 1):
                        f.write(f"{i}. {ref}\n")
                
                print(f"{Fore.GREEN}Saved legal references to: {ref_filename}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}Error saving files: {str(e)}{Style.RESET_ALL}")
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{Fore.GREEN}PDF Processing Summary:{Style.RESET_ALL}")
    print(f"Title: {args.title or 'Unknown'}")
    print(f"URL: {args.url}")
    print(f"PDF Size: {len(pdf_content)} bytes")
    print(f"Extracted Text: {len(pdf_text)} characters")
    print(f"Legal References: {len(legal_references)}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    
    # Print legal references
    if legal_references:
        print(f"\n{Fore.GREEN}Found {len(legal_references)} legal references:{Style.RESET_ALL}")
        for i, ref in enumerate(legal_references, 1):
            print(f"{i}. {ref}")
    else:
        print(f"\n{Fore.YELLOW}No legal references found in the PDF{Style.RESET_ALL}")
    
    # Print content preview
    print(f"\n{Fore.CYAN}Content Preview:{Style.RESET_ALL}")
    # Show first 15 lines and last 5 lines
    lines = pdf_text.split('\n')
    if len(lines) <= 25:
        preview_lines = lines
    else:
        preview_lines = lines[:15] + ['...'] + lines[-5:]
    
    for line in preview_lines:
        print(line)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
