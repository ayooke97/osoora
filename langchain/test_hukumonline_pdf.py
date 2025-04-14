#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for hukumonline.com PDF extraction functionality.
This script specifically looks for links with class="css-140jb81" and downloads the PDFs.
"""

import sys
import time
import argparse
import requests
import httpx
import io
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from colorama import Fore, Style, init
from pypdf import PdfReader

# Initialize colorama
init()

def find_hukumonline_pdf_links(url):
    """
    Find PDF links on hukumonline.com pages, specifically looking for links with class="css-140jb81".
    
    Args:
        url (str): URL of the hukumonline.com page to search
        
    Returns:
        list: List of dictionaries containing URL and text of PDF links
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        # Headers for the request
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        print(f"{Fore.CYAN}Searching for PDF links on hukumonline.com: {url}{Style.RESET_ALL}")
        
        # Send request to the URL
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links with class="css-140jb81" which are likely PDF links
        pdf_links = []
        pdf_class_links = soup.find_all('a', class_='css-140jb81')
        
        if pdf_class_links:
            print(f"{Fore.GREEN}Found {len(pdf_class_links)} links with class='css-140jb81'{Style.RESET_ALL}")
            for a in pdf_class_links:
                link_url = a['href']
                link_text = a.get_text().strip()
                
                # Make the URL absolute
                absolute_url = urljoin(url, link_url)
                
                # Add to list
                pdf_links.append({
                    'url': absolute_url,
                    'text': link_text or absolute_url,
                    'source': 'hukumonline'
                })
        else:
            print(f"{Fore.YELLOW}No links with class='css-140jb81' found{Style.RESET_ALL}")
            
            # If no specific class links found, try to find any links that might be PDFs
            for a in soup.find_all('a', href=True):
                link_url = a['href']
                if link_url.lower().endswith('.pdf'):
                    link_text = a.get_text().strip()
                    absolute_url = urljoin(url, link_url)
                    pdf_links.append({
                        'url': absolute_url,
                        'text': link_text or absolute_url,
                        'source': 'hukumonline'
                    })
            
            # Print the HTML structure to help debug
            print(f"{Fore.YELLOW}HTML structure for debugging:{Style.RESET_ALL}")
            print(soup.prettify()[:1000])  # Print first 1000 chars of HTML
        
        print(f"{Fore.GREEN}Found {len(pdf_links)} potential PDF links{Style.RESET_ALL}")
        return pdf_links
        
    except Exception as e:
        print(f"{Fore.RED}Error finding PDF links on hukumonline.com: {str(e)}{Style.RESET_ALL}")
        return []

def download_pdf(pdf_url):
    """
    Download a PDF file from a URL.
    
    Args:
        pdf_url (str): URL of the PDF to download
        
    Returns:
        bytes: The PDF content as bytes or None if download failed
    """
    try:
        # Custom user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        print(f"{Fore.CYAN}Downloading PDF: {pdf_url}{Style.RESET_ALL}")
        
        # Create httpx client with custom settings
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={
                'User-Agent': user_agent,
                'Accept': 'application/pdf,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Referer': 'https://www.hukumonline.com/',
            }
        ) as client:
            # Send GET request to download the PDF
            response = client.get(pdf_url)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Check if the content is a PDF
                content_type = response.headers.get('Content-Type', '')
                if 'application/pdf' in content_type or pdf_url.lower().endswith('.pdf'):
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
        print(f"{Fore.RED}Error downloading PDF: {str(e)}{Style.RESET_ALL}")
        return None

def extract_text_from_pdf(pdf_content):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_content (bytes): The PDF content as bytes
        
    Returns:
        str: The extracted text from the PDF
    """
    try:
        # Create a PDF reader object
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        
        # Get document info
        info = pdf_reader.metadata
        print(f"{Fore.CYAN}PDF Metadata: {info}{Style.RESET_ALL}")
        
        # Extract text from each page
        text = ""
        num_pages = len(pdf_reader.pages)
        print(f"{Fore.CYAN}Extracting text from PDF ({num_pages} pages){Style.RESET_ALL}")
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {i+1} ---\n{page_text}\n\n"
                
                # Print progress for large PDFs
                if i % 10 == 0 and i > 0:
                    print(f"{Fore.CYAN}Processed {i}/{num_pages} pages{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error extracting text from page {i+1}: {str(e)}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}Successfully extracted {len(text)} characters from PDF{Style.RESET_ALL}")
        return text
        
    except Exception as e:
        print(f"{Fore.RED}Error extracting text from PDF: {str(e)}{Style.RESET_ALL}")
        return None

def main():
    """Main function to test PDF extraction functionality."""
    parser = argparse.ArgumentParser(description='Test hukumonline.com PDF extraction functionality')
    parser.add_argument('url', help='URL of the hukumonline.com page to extract PDFs from')
    parser.add_argument('--save', action='store_true', help='Save the extracted text to a file')
    parser.add_argument('--output', default='hukumonline_pdf.txt', help='Output file name')
    args = parser.parse_args()

    if 'hukumonline.com' not in args.url:
        print(f"{Fore.RED}Error: URL must be from hukumonline.com{Style.RESET_ALL}")
        return 1

    print(f"{Fore.CYAN}Testing PDF extraction for URL: {args.url}{Style.RESET_ALL}")
    
    # Find PDF links
    pdf_links = find_hukumonline_pdf_links(args.url)
    
    if not pdf_links:
        print(f"{Fore.RED}No PDF links found. Exiting.{Style.RESET_ALL}")
        return 1
    
    # Process each PDF link
    for i, link in enumerate(pdf_links, 1):
        print(f"\n{Fore.YELLOW}Processing PDF link {i}/{len(pdf_links)}: {link['text']}{Style.RESET_ALL}")
        print(f"URL: {link['url']}")
        
        # Download the PDF
        pdf_content = download_pdf(link['url'])
        
        if not pdf_content:
            print(f"{Fore.RED}Failed to download PDF. Skipping.{Style.RESET_ALL}")
            continue
        
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_content)
        
        if not pdf_text:
            print(f"{Fore.RED}Failed to extract text from PDF. Skipping.{Style.RESET_ALL}")
            continue
        
        # Print a preview of the extracted text
        print(f"\n{Fore.CYAN}Text Preview (first 10 lines):{Style.RESET_ALL}")
        lines = pdf_text.split('\n')
        preview_lines = lines[:10]
        for line in preview_lines:
            print(line)
        
        # Save the extracted text to a file if requested
        if args.save:
            output_file = f"{args.output.split('.')[0]}_{i}.{args.output.split('.')[1]}"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(pdf_text)
                print(f"{Fore.GREEN}Saved extracted text to: {output_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error saving text to file: {str(e)}{Style.RESET_ALL}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
