from typing import List, Dict, Any, Optional
from langchain.text_splitter import TextSplitter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

class SastrawiTextSplitter(TextSplitter):
    """
    Text splitter that uses Sastrawi for Indonesian text processing.
    This splitter can work in different modes:
    1. Sentence-based: Splits by Indonesian sentences
    2. Paragraph-based: Splits by paragraphs
    3. Semantic-based: Attempts to group related sentences based on content
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        mode: str = "sentence",
        stemming: bool = False,
        remove_stopwords: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the SastrawiTextSplitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            mode: The splitting mode ("sentence", "paragraph", or "semantic")
            stemming: Whether to apply stemming to the text
            remove_stopwords: Whether to remove stopwords from the text
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.mode = mode
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        
        # Initialize Sastrawi components if needed
        if self.stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        
        if self.remove_stopwords:
            stop_factory = StopWordRemoverFactory()
            self.stopword_remover = stop_factory.create_stop_word_remover()
    
    def preprocess_text(self, text: str) -> str:
        """Apply preprocessing steps to the text."""
        if self.stemming:
            text = self.stemmer.stem(text)
        
        if self.remove_stopwords:
            text = self.stopword_remover.remove(text)
        
        return text
    
    def split_text(self, text: str) -> List[str]:
        """
        Split incoming text and return chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        # Preprocess the text if needed
        processed_text = self.preprocess_text(text)
        
        # Split based on the selected mode
        if self.mode == "paragraph":
            # Split by paragraphs (double newlines)
            splits = re.split(r"\n\s*\n", processed_text)
            splits = [s for s in splits if s.strip()]  # Remove empty splits
        
        elif self.mode == "sentence":
            # Indonesian sentence splitting patterns
            # This handles common Indonesian sentence endings
            sentence_endings = r'(?<=[.!?;])\s+'
            splits = re.split(sentence_endings, processed_text)
            splits = [s for s in splits if s.strip()]  # Remove empty splits
        
        elif self.mode == "semantic":
            # For semantic mode, we'll first split into sentences
            sentence_endings = r'(?<=[.!?;])\s+'
            sentences = re.split(sentence_endings, processed_text)
            sentences = [s for s in sentences if s.strip()]  # Remove empty splits
            
            # Group sentences into semantic chunks
            # This is a simple approach - in a real implementation,
            # you might use embeddings to group related sentences
            current_chunk = []
            current_length = 0
            splits = []
            
            for sentence in sentences:
                sentence_len = len(sentence)
                
                # If adding this sentence would exceed chunk_size,
                # save the current chunk and start a new one
                if current_length + sentence_len > self.chunk_size and current_chunk:
                    splits.append(" ".join(current_chunk))
                    
                    # Start a new chunk with overlap
                    overlap_sentences = []
                    overlap_length = 0
                    
                    # Add sentences from the end of the previous chunk for overlap
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_len
            
            # Add the last chunk if it's not empty
            if current_chunk:
                splits.append(" ".join(current_chunk))
                
            return splits
        
        # For paragraph and sentence modes, merge splits using the parent class method
        return self._merge_splits(splits, separator=" ")

# Example usage function
def create_indonesian_text_splitter(
    mode: str = "sentence",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    stemming: bool = False,
    remove_stopwords: bool = False
) -> SastrawiTextSplitter:
    """
    Create a text splitter optimized for Indonesian text.
    
    Args:
        mode: Splitting mode ("sentence", "paragraph", or "semantic")
        chunk_size: Maximum size of chunks
        chunk_overlap: Overlap between chunks
        stemming: Whether to apply stemming
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Configured SastrawiTextSplitter
    """
    return SastrawiTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        mode=mode,
        stemming=stemming,
        remove_stopwords=remove_stopwords
    )