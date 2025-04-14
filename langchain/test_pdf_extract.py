#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for PDF extraction functionality.
This script tests the PDF downloading and text extraction functions.
"""

import sys
import argparse
from colorama import Fore, Style, init
from retrieve_cse_results import download_pdf, extract_text_from_pdf, extract_legal_references

# Initialize colorama
init()

def main():
    """Main function to test PDF extraction functionality."""
    parser = argparse.ArgumentParser(description='Test PDF extraction functionality')
    parser.add_argument('url', help='URL of the PDF to extract')
    parser.add_argument('--save', action='store_true', help='Save the extracted text to a file')
    parser.add_argument('--output', default='pdf_content.txt', help='Output file name')
    args = parser.parse_args()

    print(f"{Fore.CYAN}Testing PDF extraction for URL: {args.url}{Style.RESET_ALL}")
    
    # Download the PDF
    print(f"{Fore.CYAN}Downloading PDF...{Style.RESET_ALL}")
    pdf_content = download_pdf(args.url)
    
    if not pdf_content:
        print(f"{Fore.RED}Failed to download PDF. Exiting.{Style.RESET_ALL}")
        return 1
    
    print(f"{Fore.GREEN}Successfully downloaded PDF ({len(pdf_content)} bytes){Style.RESET_ALL}")
    
    # Extract text from the PDF
    print(f"{Fore.CYAN}Extracting text from PDF...{Style.RESET_ALL}")
    pdf_text = extract_text_from_pdf(pdf_content)
    
    if not pdf_text:
        print(f"{Fore.RED}Failed to extract text from PDF. Exiting.{Style.RESET_ALL}")
        return 1
    
    print(f"{Fore.GREEN}Successfully extracted text from PDF ({len(pdf_text)} characters){Style.RESET_ALL}")
    
    # Extract legal references
    print(f"{Fore.CYAN}Extracting legal references...{Style.RESET_ALL}")
    legal_references = extract_legal_references(pdf_text)
    
    if legal_references:
        print(f"{Fore.GREEN}Found {len(legal_references)} legal references:{Style.RESET_ALL}")
        for i, ref in enumerate(legal_references, 1):
            print(f"{i}. {ref}")
    else:
        print(f"{Fore.YELLOW}No legal references found in the PDF{Style.RESET_ALL}")
    
    # Print a preview of the extracted text
    print(f"\n{Fore.CYAN}Text Preview:{Style.RESET_ALL}")
    lines = pdf_text.split('\n')
    preview_lines = lines[:10]
    if len(lines) > 10:
        preview_lines.append("...")
    
    for line in preview_lines:
        print(line)
    
    # Save the extracted text to a file if requested
    if args.save:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(pdf_text)
            print(f"{Fore.GREEN}Saved extracted text to: {args.output}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving text to file: {str(e)}{Style.RESET_ALL}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
