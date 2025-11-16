"""
Index Builder Module - Adapted from rag-juridique
==================================================

Handles embedding creation and FAISS index building.
Supports both vector (semantic) and BM25 (lexical) indexes.
"""

import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()


class IndexBuilder:
    """
    Builds and manages vector and BM25 indexes for RAG retrieval.
    """
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the index builder.
        
        Args:
            api_key: OpenAI API key (reads from env if not provided)
            embedding_model: OpenAI embedding model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = embedding_model
        
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def create_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            show_progress: Whether to show progress
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        total = len(texts)
        
        if show_progress:
            print(f"\n🔢 Creating {total} embeddings (batch size: {batch_size})...")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress = min(i + batch_size, total)
                print(f"  Progress: {progress}/{total} ({100*progress//total}%)")
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        if show_progress:
            print(f"✅ Embeddings created: shape = {embeddings_array.shape}\n")
        
        return embeddings_array
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        
        print(f"🏗️  Building FAISS index...")
        print(f"  Dimension: {dimension}")
        print(f"  Vectors: {embeddings.shape[0]}")
        
        # Use IndexFlatL2 for exact search (good for < 1M vectors)
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        print(f"✅ FAISS index built with {index.ntotal} vectors\n")
        
        return index
    
    def build_bm25_index(self, chunks: List[Dict[str, any]]) -> BM25Okapi:
        """
        Build a BM25 index for lexical search.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            BM25 index
        """
        print(f"🏗️  Building BM25 index from {len(chunks)} chunks...")
        
        # Tokenize texts for BM25
        tokenized_corpus = [chunk['text'].lower().split() for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_corpus)
        
        print(f"✅ BM25 index built\n")
        
        return bm25_index
    
    def save_indexes(
        self,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        chunks: List[Dict],
        output_dir: str = "data/index",
        config_name: str = "default"
    ):
        """
        Save FAISS index, BM25 index, and chunks to disk.
        
        Args:
            faiss_index: FAISS index to save
            bm25_index: BM25 index to save
            chunks: Chunks to save
            output_dir: Output directory
            config_name: Configuration name for the indexes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        faiss_path = os.path.join(output_dir, f"{config_name}_faiss.index")
        bm25_path = os.path.join(output_dir, f"{config_name}_bm25.pkl")
        chunks_path = os.path.join(output_dir, f"{config_name}_chunks.pkl")
        
        # Save FAISS index
        faiss.write_index(faiss_index, faiss_path)
        print(f"💾 FAISS index saved: {faiss_path}")
        
        # Save BM25 index
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        print(f"💾 BM25 index saved: {bm25_path}")
        
        # Save chunks
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"💾 Chunks saved: {chunks_path}")
        
        # Calculate sizes
        total_size = (
            os.path.getsize(faiss_path) + 
            os.path.getsize(bm25_path) + 
            os.path.getsize(chunks_path)
        ) / (1024 * 1024)
        print(f"  📦 Total size: {total_size:.2f} MB\n")
    
    def load_indexes(
        self,
        input_dir: str = "data/index",
        config_name: str = "default"
    ) -> tuple:
        """
        Load FAISS index, BM25 index, and chunks from disk.
        
        Args:
            input_dir: Input directory
            config_name: Configuration name
            
        Returns:
            Tuple of (faiss_index, bm25_index, chunks)
        """
        faiss_path = os.path.join(input_dir, f"{config_name}_faiss.index")
        bm25_path = os.path.join(input_dir, f"{config_name}_bm25.pkl")
        chunks_path = os.path.join(input_dir, f"{config_name}_chunks.pkl")
        
        print(f"📂 Loading indexes from {input_dir}...")
        
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_path)
        print(f"  ✅ FAISS index loaded: {faiss_index.ntotal} vectors")
        
        # Load BM25 index
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
        print(f"  ✅ BM25 index loaded")
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"  ✅ Chunks loaded: {len(chunks)} chunks\n")
        
        return faiss_index, bm25_index, chunks
    
    def indexes_exist(
        self,
        input_dir: str = "data/index",
        config_name: str = "default"
    ) -> bool:
        """
        Check if indexes exist.
        
        Args:
            input_dir: Input directory
            config_name: Configuration name
            
        Returns:
            True if all indexes exist
        """
        faiss_path = os.path.join(input_dir, f"{config_name}_faiss.index")
        bm25_path = os.path.join(input_dir, f"{config_name}_bm25.pkl")
        chunks_path = os.path.join(input_dir, f"{config_name}_chunks.pkl")
        
        return (
            os.path.exists(faiss_path) and 
            os.path.exists(bm25_path) and 
            os.path.exists(chunks_path)
        )


if __name__ == "__main__":
    # Test the index builder
    from chunker import DocumentChunker
    
    print("=== Testing Index Builder ===\n")
    
    # Process documents
    chunker = DocumentChunker()
    chunks = chunker.process_all_documents()
    
    if not chunks:
        print("No chunks to process. Add documents to data/raw_docs/")
        exit(1)
    
    # Build indexes
    builder = IndexBuilder()
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = builder.create_embeddings_batch(texts)
    
    faiss_index = builder.build_faiss_index(embeddings)
    bm25_index = builder.build_bm25_index(chunks)
    
    # Save
    builder.save_indexes(faiss_index, bm25_index, chunks)
    
    print("✅ Index building complete!")

