#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Legal PDF Scraper
----------------
A specialized tool for extracting legal references and content from PDF documents.
This script is designed to work with Indonesian legal documents and can extract
various types of legal references such as UU, PP, Peraturan, etc.
"""

import os
import sys
import time
import argparse
import re
import io
from urllib.parse import urlparse, unquote
from colorama import Fore, Style, init

# Try importing required libraries
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    print("Warning: httpx not found. Using requests instead.")
    import requests

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("Error: pypdf not found. Please install it using 'pip install pypdf'")
    sys.exit(1)

# Initialize colorama
init()

# Patterns for legal references
LEGAL_PATTERNS = [
    r'UU\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # UU No. 13 Tahun 2003
    r'Undang-Undang\s+Nomor\s+\d+\s+Tahun\s+\d{4}',  # Undang-Undang Nomor 13 Tahun 2003
    r'PP\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # PP No. 14 Tahun 1993
    r'Peraturan\s+Pemerintah\s+Nomor\s+\d+\s+Tahun\s+\d{4}',  # Peraturan Pemerintah Nomor 14 Tahun 1993
    r'Perpres\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Perpres No. 20 Tahun 2018
    r'Peraturan\s+Presiden\s+Nomor\s+\d+\s+Tahun\s+\d{4}',  # Peraturan Presiden Nomor 20 Tahun 2018
    r'Permen\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Permen No. 13 Tahun 2003
    r'Peraturan\s+Menteri\s+\w+\s+Nomor\s+\d+\s+Tahun\s+\d{4}',  # Peraturan Menteri Ketenagakerjaan Nomor 13 Tahun 2003
    r'Kepmen\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Kepmen No. 232 Tahun 2003
    r'Keputusan\s+Menteri\s+\w+\s+Nomor\s+\d+\s+Tahun\s+\d{4}',  # Keputusan Menteri Ketenagakerjaan Nomor 232 Tahun 2003
    r'Perda\s+No\.?\s*\d+\s+Tahun\s+\d{4}',  # Perda No. 6 Tahun 2020
    r'Peraturan\s+Daerah\s+\w+\s+Nomor\s+\d+\s+Tahun\s+\d{4}',  # Peraturan Daerah Jakarta Nomor 6 Tahun 2020
]

def extract_legal_references(text):
    """
    Extract legal references from text.
    
    Args:
        text (str): Text to extract legal references from
        
    Returns:
        list: List of legal references found
    """
    if not text:
        return []
    
    all_references = []
    
    # Apply each pattern
    for pattern in LEGAL_PATTERNS:
        matches = re.findall(pattern, text)
        all_references.extend(matches)
    
    # Remove duplicates while preserving order
    unique_references = []
    for ref in all_references:
        if ref not in unique_references:
            unique_references.append(ref)
    
    return unique_references

def download_pdf(url):
    """
    Download a PDF file from a URL.
    
    Args:
        url (str): URL of the PDF to download
        
    Returns:
        bytes: The PDF content as bytes or None if download failed
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        print(f"{Fore.CYAN}Downloading PDF from: {url}{Style.RESET_ALL}")
        
        # Use httpx if available
        if HAS_HTTPX:
            try:
                # Create httpx client with custom settings
                with httpx.Client(
                    timeout=60.0,
                    follow_redirects=True,
                    headers={
                        'User-Agent': user_agent,
                        'Accept': 'application/pdf,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive',
                    }
                ) as client:
                    # Send GET request to download the PDF
                    response = client.get(url)
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        # Check if the content is a PDF
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                            print(f"{Fore.GREEN}Successfully downloaded PDF ({len(response.content)} bytes){Style.RESET_ALL}")
                            return response.content
                        else:
                            print(f"{Fore.YELLOW}Downloaded content is not a PDF (Content-Type: {content_type}){Style.RESET_ALL}")
                            # Try to return content anyway, might be a PDF with wrong content type
                            return response.content
                    else:
                        print(f"{Fore.RED}Failed to download PDF: HTTP {response.status_code}{Style.RESET_ALL}")
                        return None
                    
            except Exception as e:
                print(f"{Fore.RED}Error downloading PDF with httpx: {str(e)}{Style.RESET_ALL}")
                # Fall back to regular method
        
        # Fall back to regular method if httpx is not available or failed
        headers = {
            'User-Agent': user_agent,
            'Accept': 'application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Try to get the content
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Check if the content is a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                pdf_content = response.content
                print(f"{Fore.GREEN}Successfully downloaded PDF ({len(pdf_content)} bytes){Style.RESET_ALL}")
                return pdf_content
            else:
                print(f"{Fore.YELLOW}Downloaded content is not a PDF (Content-Type: {content_type}){Style.RESET_ALL}")
                # Try to return content anyway, might be a PDF with wrong content type
                return response.content
        else:
            print(f"{Fore.RED}Failed to download PDF: HTTP {response.status_code}{Style.RESET_ALL}")
            return None
        
    except Exception as e:
        print(f"{Fore.RED}Error downloading PDF: {str(e)}{Style.RESET_ALL}")
        return None

def extract_text_from_pdf(pdf_content):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_content (bytes): The PDF content as bytes
        
    Returns:
        dict: Dictionary containing extracted text and metadata
    """
    try:
        # Create a PDF reader object
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        
        # Get document info
        info = pdf_reader.metadata
        print(f"{Fore.CYAN}PDF Metadata: {info}{Style.RESET_ALL}")
        
        # Extract text from each page
        pages_text = []
        total_text = ""
        num_pages = len(pdf_reader.pages)
        print(f"{Fore.CYAN}Extracting text from PDF ({num_pages} pages){Style.RESET_ALL}")
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                print(f"{Fore.CYAN}Processing page {i+1}/{num_pages}...{Style.RESET_ALL}")
                page_text = page.extract_text()
                if page_text:
                    print(f"{Fore.GREEN}Extracted {len(page_text)} characters from page {i+1}{Style.RESET_ALL}")
                    pages_text.append(page_text)
                    total_text += f"--- Page {i+1} ---\n{page_text}\n\n"
                else:
                    print(f"{Fore.YELLOW}No text extracted from page {i+1}{Style.RESET_ALL}")
                    pages_text.append("")
                
                # Print progress for large PDFs
                if i % 10 == 0 and i > 0:
                    print(f"{Fore.CYAN}Processed {i}/{num_pages} pages{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error extracting text from page {i+1}: {str(e)}{Style.RESET_ALL}")
                pages_text.append("")
        
        # Clean up the text
        total_text = total_text.strip()
        
        print(f"{Fore.GREEN}Successfully extracted {len(total_text)} characters from PDF{Style.RESET_ALL}")
        
        # Return the extracted text and metadata
        return {
            'text': total_text,
            'pages_text': pages_text,
            'num_pages': num_pages,
            'title': info.title if info and hasattr(info, 'title') else None,
            'author': info.author if info and hasattr(info, 'author') else None,
            'creator': info.creator if info and hasattr(info, 'creator') else None,
            'producer': info.producer if info and hasattr(info, 'producer') else None,
            'subject': info.subject if info and hasattr(info, 'subject') else None,
        }
        
    except Exception as e:
        print(f"{Fore.RED}Error extracting text from PDF: {str(e)}{Style.RESET_ALL}")
        return None

def get_filename_from_url(url):
    """Extract a filename from a URL."""
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    return os.path.basename(path)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Legal PDF Scraper')
    parser.add_argument('url', help='URL of the PDF to scrape')
    parser.add_argument('--output-dir', default='pdf_output', help='Directory to save output files')
    parser.add_argument('--title', help='Title for the PDF (optional)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Start timer
    start_time = time.time()
    
    # Download the PDF
    pdf_content = download_pdf(args.url)
    if not pdf_content:
        print(f"{Fore.RED}Failed to download PDF. Exiting.{Style.RESET_ALL}")
        return 1
    
    download_time = time.time() - start_time
    
    # Extract text from the PDF
    extraction_start = time.time()
    pdf_data = extract_text_from_pdf(pdf_content)
    if not pdf_data:
        print(f"{Fore.RED}Failed to extract text from PDF. Exiting.{Style.RESET_ALL}")
        return 1
    
    extraction_time = time.time() - extraction_start
    
    # Extract legal references
    ref_start = time.time()
    legal_references = extract_legal_references(pdf_data['text'])
    ref_time = time.time() - ref_start
    
    # Get PDF title
    pdf_title = args.title or pdf_data.get('title') or get_filename_from_url(args.url)
    
    # Create a timestamp for filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_title = ''.join(c if c.isalnum() else '_' for c in pdf_title)
    
    # Save the PDF content to a file
    content_filename = os.path.join(args.output_dir, f"{safe_title}_{timestamp}.txt")
    try:
        with open(content_filename, 'w', encoding='utf-8') as f:
            # Write metadata
            f.write(f"PDF Title: {pdf_title}\n")
            f.write(f"PDF URL: {args.url}\n")
            f.write(f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Pages: {pdf_data['num_pages']}\n")
            f.write(f"Text Length: {len(pdf_data['text'])} characters\n")
            
            # Write document info if available
            if pdf_data.get('author'):
                f.write(f"Author: {pdf_data['author']}\n")
            if pdf_data.get('creator'):
                f.write(f"Creator: {pdf_data['creator']}\n")
            if pdf_data.get('producer'):
                f.write(f"Producer: {pdf_data['producer']}\n")
            if pdf_data.get('subject'):
                f.write(f"Subject: {pdf_data['subject']}\n")
            
            # Write legal references
            f.write(f"Legal References: {len(legal_references)}\n\n")
            
            # Write content
            f.write("=" * 80 + "\n")
            f.write("CONTENT\n")
            f.write("=" * 80 + "\n\n")
            f.write(pdf_data['text'])
        
        print(f"{Fore.GREEN}Saved PDF content to: {content_filename}{Style.RESET_ALL}")
        
        # Save legal references to a separate file if any were found
        if legal_references:
            ref_filename = os.path.join(args.output_dir, f"{safe_title}_references_{timestamp}.txt")
            with open(ref_filename, 'w', encoding='utf-8') as f:
                f.write(f"PDF Title: {pdf_title}\n")
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
    print(f"Title: {pdf_title}")
    print(f"URL: {args.url}")
    print(f"Pages: {pdf_data['num_pages']}")
    print(f"PDF Size: {len(pdf_content)} bytes")
    print(f"Extracted Text: {len(pdf_data['text'])} characters")
    print(f"Legal References: {len(legal_references)}")
    print(f"Processing Times:")
    print(f"  - Download: {download_time:.2f} seconds")
    print(f"  - Text Extraction: {extraction_time:.2f} seconds")
    print(f"  - Reference Extraction: {ref_time:.2f} seconds")
    print(f"  - Total: {total_time:.2f} seconds")
    
    # Print legal references
    if legal_references:
        print(f"\n{Fore.GREEN}Found {len(legal_references)} legal references:{Style.RESET_ALL}")
        for i, ref in enumerate(legal_references, 1):
            print(f"{i}. {ref}")
    else:
        print(f"\n{Fore.YELLOW}No legal references found in the PDF{Style.RESET_ALL}")
    
    # Print content preview
    print(f"\n{Fore.CYAN}Content Preview:{Style.RESET_ALL}")
    # Show first 10 lines
    lines = pdf_data['text'].split('\n')
    preview_lines = lines[:10]
    if len(lines) > 10:
        preview_lines.append("...")
    
    for line in preview_lines:
        print(line)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
