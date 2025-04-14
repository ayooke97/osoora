"""
PDF to Markdown Converter using LangChain and Sastrawi
This script demonstrates how to convert PDF documents to Markdown format
with Indonesian language optimization using Sastrawi.
"""

import os
from typing import List, Dict, Any
import argparse
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# Import our custom Sastrawi text splitter
from sastrawi import SastrawiTextSplitter, create_indonesian_text_splitter

def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    mode: str = "semantic",
    stemming: bool = False,
    remove_stopwords: bool = False
) -> str:
    """
    Convert a PDF file to Markdown format with Indonesian language optimization.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the Markdown file (if None, returns the content as string)
        chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between chunks
        mode: Chunking mode ("sentence", "paragraph", or "semantic")
        stemming: Whether to apply stemming
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Markdown content as string if output_path is None, otherwise None
    """
    print(f"Loading PDF: {pdf_path}")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages")
    
    # Create the Sastrawi text splitter
    text_splitter = create_indonesian_text_splitter(
        mode=mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        stemming=stemming,
        remove_stopwords=remove_stopwords
    )
    
    # Split the documents
    chunks = text_splitter.split_documents(pages)
    
    print(f"Split into {len(chunks)} chunks")
    
    # Convert to Markdown
    markdown_content = convert_chunks_to_markdown(chunks)
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Markdown saved to: {output_path}")
        return None
    
    return markdown_content

def convert_chunks_to_markdown(chunks: List[Document]) -> str:
    """
    Convert document chunks to Markdown format.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Markdown content as string
    """
    markdown_lines = []
    current_page = -1
    
    for chunk in chunks:
        # Get the page number from metadata
        page = chunk.metadata.get('page', 0)
        
        # Add page header if we've moved to a new page
        if page != current_page:
            current_page = page
            markdown_lines.append(f"\n## Halaman {page + 1}\n")
        
        # Add the chunk content
        content = chunk.page_content.strip()
        
        # Apply some basic Markdown formatting
        # Convert bullet points
        content = content.replace('•', '-')
        
        # Handle headings (simple heuristic - lines ending with ':' become headings)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().endswith(':') and len(line) < 100:
                lines[i] = f"### {line}"
        
        content = '\n'.join(lines)
        
        # Add the processed content
        markdown_lines.append(content)
        markdown_lines.append("\n---\n")  # Separator between chunks
    
    return '\n'.join(markdown_lines)

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to Markdown with Indonesian language optimization')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Path to save the Markdown file (default: same as PDF with .md extension)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Maximum size of text chunks')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Overlap between chunks')
    parser.add_argument('--mode', choices=['sentence', 'paragraph', 'semantic'], default='semantic',
                        help='Chunking mode')
    parser.add_argument('--stemming', action='store_true', help='Apply stemming to the text')
    parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords from the text')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        pdf_path = Path(args.pdf_path)
        args.output = str(pdf_path.with_suffix('.md'))
    
    convert_pdf_to_markdown(
        args.pdf_path,
        args.output,
        args.chunk_size,
        args.chunk_overlap,
        args.mode,
        args.stemming,
        args.remove_stopwords
    )

if __name__ == '__main__':
    main()
