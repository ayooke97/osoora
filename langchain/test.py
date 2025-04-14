import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def convert_pdf_to_markdown(pdf_path, output_md_path=None):
    """
    Convert a PDF file to Markdown format using LangChain.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_md_path (str, optional): Path where to save the Markdown file. 
                                       If None, will use the PDF filename with .md extension.
    
    Returns:
        str: Path to the generated Markdown file
    """
    # Validate that the PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    # Set default output path if not provided
    if output_md_path is None:
        output_md_path = os.path.splitext(pdf_path)[0] + ".md"
    
    print(f"Loading PDF from {pdf_path}...")
    
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"PDF loaded with {len(pages)} pages")
    
    # Process the PDF content into Markdown
    markdown_content = []
    
    # Add title based on filename
    filename = os.path.basename(pdf_path)
    title = os.path.splitext(filename)[0].replace("_", " ").title()
    markdown_content.append(f"# {title}\n\n")
    
    # Process each page
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}")
        
        # Get page content
        content = page.page_content
        
        # Basic content cleanup
        # Remove excessive newlines and spaces
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        # Detect and format headers
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Heuristic for headers: short lines with all caps or ending with colon
            if (len(line) < 50 and line.isupper()) or (len(line) < 30 and line.endswith(':')):
                # Make it a level 2 header
                formatted_lines.append(f"\n## {line}\n")
            else:
                formatted_lines.append(line)
        
        # Join the lines and add page separator
        page_content = '\n'.join(formatted_lines)
        markdown_content.append(page_content)
        markdown_content.append(f"\n\n---\n\n")
    
    # Write to markdown file
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(''.join(markdown_content))
    
    print(f"Markdown conversion complete. Output saved to {output_md_path}")
    return output_md_path

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PDF to Markdown using LangChain')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Path for the output Markdown file (optional)')
    
    args = parser.parse_args()
    
    try:
        output_path = convert_pdf_to_markdown(args.pdf_path, args.output)
        print(f"PDF successfully converted to Markdown: {output_path}")
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")