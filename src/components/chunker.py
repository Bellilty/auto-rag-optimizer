"""
Chunking Module - Adapted from rag-juridique
==============================================

Handles document chunking with configurable parameters.
Supports PDF and TXT extraction with flexible chunking strategies.
"""

import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Optional
from pathlib import Path


class DocumentChunker:
    """
    Handles document extraction and chunking with configurable parameters.
    """
    
    def __init__(self, document_directory: str = "data/raw_docs"):
        """
        Initialize the document chunker.
        
        Args:
            document_directory: Path to directory containing documents
        """
        self.document_directory = document_directory
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        doc = fitz.open(pdf_path)
        text = ""
        
        for page in doc:
            text += page.get_text("text")
            
        doc.close()
        return text
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            txt_path: Path to the TXT file
            
        Returns:
            Text content
        """
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by normalizing whitespace and newlines.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        # Normalize newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200,
        source: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Approximate chunk size in words
            overlap: Number of overlapping words between chunks
            source: Source filename (for metadata)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_dict = {
                "text": chunk_text,
                "chunk_id": len(chunks),
                "start_word": i,
                "end_word": min(i + chunk_size, len(words)),
                "source": source or "unknown"
            }
            chunks.append(chunk_dict)
        
        return chunks
    
    def process_document(
        self, 
        file_path: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[Dict[str, any]]:
        """
        Process a single document: extract, clean, and chunk.
        
        Args:
            file_path: Path to the document
            chunk_size: Chunk size in words
            overlap: Overlap in words
            
        Returns:
            List of chunks
        """
        filename = os.path.basename(file_path)
        
        # Extract based on file type
        if file_path.endswith('.pdf'):
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            raw_text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        # Chunk with source metadata
        chunks = self.chunk_text(clean_text, chunk_size, overlap, source=filename)
        
        return chunks
    
    def process_all_documents(
        self, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[Dict[str, any]]:
        """
        Process all documents in the directory.
        
        Args:
            chunk_size: Chunk size in words
            overlap: Overlap in words
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        if not os.path.exists(self.document_directory):
            print(f"Warning: Directory {self.document_directory} does not exist")
            return all_chunks
        
        # Get all supported files
        supported_extensions = ['.pdf', '.txt']
        files = [
            f for f in os.listdir(self.document_directory)
            if any(f.endswith(ext) for ext in supported_extensions)
        ]
        
        if not files:
            print(f"Warning: No documents found in {self.document_directory}")
            return all_chunks
        
        print(f"\n📚 Processing {len(files)} document(s)...")
        
        for filename in files:
            file_path = os.path.join(self.document_directory, filename)
            try:
                chunks = self.process_document(file_path, chunk_size, overlap)
                all_chunks.extend(chunks)
                print(f"  ✅ {filename}: {len(chunks)} chunks")
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
        
        print(f"\n✅ Total: {len(all_chunks)} chunks from {len(files)} document(s)\n")
        
        return all_chunks


if __name__ == "__main__":
    # Test the chunker
    chunker = DocumentChunker()
    chunks = chunker.process_all_documents()
    
    if chunks:
        print(f"Sample chunk from {chunks[0]['source']}:")
        print(chunks[0]['text'][:200] + "...")

