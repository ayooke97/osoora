"""
Document Processor using LangChain and Sastrawi
This script demonstrates how to convert various document formats to Markdown
with Indonesian language optimization using Sastrawi.
"""

import os
from typing import List, Dict, Any, Optional, Union
import argparse
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    DirectoryLoader
)
from langchain.schema import Document

# Import our custom Sastrawi text splitter
from sastrawi import SastrawiTextSplitter, create_indonesian_text_splitter

# Map file extensions to appropriate loaders
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
}

def get_loader_for_file(file_path: str) -> Optional[object]:
    """
    Get the appropriate document loader for a given file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Document loader class or None if no suitable loader found
    """
    file_extension = Path(file_path).suffix.lower()
    return LOADER_MAPPING.get(file_extension)

def load_document(file_path: str) -> List[Document]:
    """
    Load a document using the appropriate LangChain loader.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects
    """
    loader_class = get_loader_for_file(file_path)
    
    if not loader_class:
        raise ValueError(f"No suitable loader found for file: {file_path}")
    
    loader = loader_class(file_path)
    return loader.load()

def load_documents_from_directory(directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
    """
    Load all documents from a directory using appropriate loaders.
    
    Args:
        directory_path: Path to the directory
        glob_pattern: Pattern to match files
        
    Returns:
        List of Document objects
    """
    # Define a loader selector function
    def loader_selector(file_path: str) -> Optional[object]:
        return get_loader_for_file(file_path)
    
    # Create a directory loader
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=loader_selector,
        show_progress=True,
        use_multithreading=True
    )
    
    return loader.load()

def process_document(
    input_path: str,
    output_path: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    mode: str = "semantic",
    stemming: bool = False,
    remove_stopwords: bool = False,
    is_directory: bool = False
) -> Union[str, None]:
    """
    Process a document or directory of documents with Indonesian language optimization.
    
    Args:
        input_path: Path to the document file or directory
        output_path: Path to save the Markdown file (if None, returns the content as string)
        chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between chunks
        mode: Chunking mode ("sentence", "paragraph", or "semantic")
        stemming: Whether to apply stemming
        remove_stopwords: Whether to remove stopwords
        is_directory: Whether the input path is a directory
        
    Returns:
        Markdown content as string if output_path is None, otherwise None
    """
    print(f"Loading {'directory' if is_directory else 'document'}: {input_path}")
    
    # Load the document(s)
    if is_directory:
        documents = load_documents_from_directory(input_path)
    else:
        documents = load_document(input_path)
    
    print(f"Loaded {len(documents)} document segments")
    
    # Create the Sastrawi text splitter
    text_splitter = create_indonesian_text_splitter(
        mode=mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        stemming=stemming,
        remove_stopwords=remove_stopwords
    )
    
    # Split the documents
    chunks = text_splitter.split_documents(documents)
    
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
    current_source = None
    
    for chunk in chunks:
        # Get the source from metadata
        source = chunk.metadata.get('source', None)
        page = chunk.metadata.get('page', None)
        
        # Add source header if we've moved to a new source
        if source != current_source:
            current_source = source
            if source:
                markdown_lines.append(f"\n# Source: {Path(source).name}\n")
        
        # Add page header if available
        if page is not None:
            markdown_lines.append(f"\n## Page {page + 1}\n")
        
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
    parser = argparse.ArgumentParser(description='Process documents with Indonesian language optimization')
    parser.add_argument('input_path', help='Path to the document file or directory')
    parser.add_argument('--output', '-o', help='Path to save the Markdown file (default: same name with .md extension)')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Maximum size of text chunks')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Overlap between chunks')
    parser.add_argument('--mode', choices=['sentence', 'paragraph', 'semantic'], default='semantic',
                        help='Chunking mode')
    parser.add_argument('--stemming', action='store_true', help='Apply stemming to the text')
    parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords from the text')
    parser.add_argument('--directory', '-d', action='store_true', help='Process all documents in the directory')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input_path)
        if args.directory:
            args.output = str(input_path / "processed_documents.md")
        else:
            args.output = str(input_path.with_suffix('.md'))
    
    process_document(
        args.input_path,
        args.output,
        args.chunk_size,
        args.chunk_overlap,
        args.mode,
        args.stemming,
        args.remove_stopwords,
        args.directory
    )

if __name__ == '__main__':
    main()
