"""
Retriever Module - Adapted from rag-juridique
==============================================

Handles hybrid retrieval using both vector (semantic) and BM25 (lexical) search.
Supports configurable weighting between the two approaches.
"""

import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()


class HybridRetriever:
    """
    Hybrid retrieval system combining vector and BM25 search.
    """
    
    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        chunks: List[Dict],
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            faiss_index: FAISS index for semantic search
            bm25_index: BM25 index for lexical search
            chunks: List of document chunks
            api_key: OpenAI API key
            embedding_model: Embedding model name
        """
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=api_key)
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """
        Create embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        
        embedding = np.array([response.data[0].embedding]).astype('float32')
        return embedding
    
    def vector_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform vector-based semantic search.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (chunk_index, distance) tuples
        """
        query_embedding = self.create_query_embedding(query)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Convert to list of (index, distance) tuples
        results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]
        return results
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform BM25 lexical search.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (chunk_index, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices]
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        retrieval_pool: int = 20
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector and BM25 results.
        
        Args:
            query: Query text
            k: Number of final results to return
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            retrieval_pool: Number of candidates to retrieve from each method
            
        Returns:
            List of top k chunks with scores
        """
        # Normalize weights
        total_weight = vector_weight + bm25_weight
        vector_weight = vector_weight / total_weight
        bm25_weight = bm25_weight / total_weight
        
        # Get results from both methods
        vector_results = self.vector_search(query, k=retrieval_pool)
        bm25_results = self.bm25_search(query, k=retrieval_pool)
        
        # Normalize scores to [0, 1] range
        # For vector search (L2 distance), lower is better -> invert
        if vector_results:
            max_vector_dist = max(dist for _, dist in vector_results)
            vector_scores = {
                idx: 1 - (dist / (max_vector_dist + 1e-10)) 
                for idx, dist in vector_results
            }
        else:
            vector_scores = {}
        
        # For BM25, higher is better -> normalize to [0, 1]
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results) if bm25_results else 1
            bm25_scores = {
                idx: score / (max_bm25_score + 1e-10)
                for idx, score in bm25_results
            }
        else:
            bm25_scores = {}
        
        # Combine scores
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {}
        
        for idx in all_indices:
            vec_score = vector_scores.get(idx, 0)
            bm_score = bm25_scores.get(idx, 0)
            combined_scores[idx] = vector_weight * vec_score + bm25_weight * bm_score
        
        # Sort by combined score and get top k
        top_k_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Prepare results
        results = []
        for rank, (idx, score) in enumerate(top_k_indices, 1):
            chunk = self.chunks[idx].copy()
            chunk['retrieval_score'] = float(score)
            chunk['rank'] = rank
            chunk['vector_score'] = vector_scores.get(idx, 0)
            chunk['bm25_score'] = bm25_scores.get(idx, 0)
            results.append(chunk)
        
        return results
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        **kwargs
    ) -> List[Dict]:
        """
        Main retrieval method with flexible search strategy.
        
        Args:
            query: Query text
            k: Number of results
            method: Search method ("vector", "bm25", or "hybrid")
            **kwargs: Additional parameters for hybrid search
            
        Returns:
            List of retrieved chunks
        """
        if method == "vector":
            results = self.vector_search(query, k=k)
            return [
                {**self.chunks[idx], 'retrieval_score': 1 - dist, 'rank': rank}
                for rank, (idx, dist) in enumerate(results, 1)
            ]
        
        elif method == "bm25":
            results = self.bm25_search(query, k=k)
            return [
                {**self.chunks[idx], 'retrieval_score': score, 'rank': rank}
                for rank, (idx, score) in enumerate(results, 1)
            ]
        
        else:  # hybrid
            return self.hybrid_search(query, k=k, **kwargs)


class RAGGenerator:
    """
    RAG answer generation using retrieved context.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG generator.
        
        Args:
            api_key: OpenAI API key
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        model: str = "gpt-4o-mini",
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> Dict:
        """
        Generate answer using RAG approach.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            model: LLM model to use
            max_tokens: Max tokens in response
            temperature: Generation temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        # Build context from chunks
        context = ""
        for i, chunk in enumerate(context_chunks, 1):
            context += f"[Extract {i} - Source: {chunk.get('source', 'unknown')}]\n"
            context += chunk['text']
            context += "\n\n"
        
        # System and user prompts
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
Answer ONLY using information from the given context.
If the answer is not in the context, say "I cannot find this information in the provided documents."
Always cite sources when possible."""

        user_prompt = f"""Context:

{context}

Question: {query}

Answer clearly and cite sources."""

        # Generate answer
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        answer = completion.choices[0].message.content
        
        return {
            "query": query,
            "answer": answer,
            "sources": [chunk.get('source', 'unknown') for chunk in context_chunks],
            "num_chunks_used": len(context_chunks),
            "model": model,
            "tokens_used": {
                "prompt": completion.usage.prompt_tokens,
                "completion": completion.usage.completion_tokens,
                "total": completion.usage.total_tokens
            }
        }


if __name__ == "__main__":
    # Test retriever
    from index_builder import IndexBuilder
    
    print("=== Testing Retriever ===\n")
    
    builder = IndexBuilder()
    
    if not builder.indexes_exist():
        print("Indexes not found. Run index_builder.py first.")
        exit(1)
    
    faiss_index, bm25_index, chunks = builder.load_indexes()
    
    retriever = HybridRetriever(faiss_index, bm25_index, chunks)
    
    test_query = "What is data protection?"
    results = retriever.retrieve(test_query, k=3, method="hybrid")
    
    print(f"Query: {test_query}\n")
    for result in results:
        print(f"Rank {result['rank']}: {result['source']}")
        print(f"Score: {result['retrieval_score']:.3f}")
        print(f"Text preview: {result['text'][:150]}...")
        print()

