"""
Retriever Tools
===============

Utility functions for retrieval profiling and analysis.
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


def calculate_retrieval_metrics(
    retrieved_chunks: List[Dict],
    query: str
) -> Dict[str, Any]:
    """
    Calculate metrics for a single retrieval result.
    
    Args:
        retrieved_chunks: List of retrieved chunks with scores
        query: Original query
        
    Returns:
        Dictionary with metrics
    """
    if not retrieved_chunks:
        return {
            "num_retrieved": 0,
            "avg_score": 0,
            "score_std": 0,
            "unique_sources": 0
        }
    
    scores = [chunk.get('retrieval_score', 0) for chunk in retrieved_chunks]
    sources = [chunk.get('source', 'unknown') for chunk in retrieved_chunks]
    
    return {
        "num_retrieved": len(retrieved_chunks),
        "avg_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "unique_sources": len(set(sources)),
        "source_distribution": dict(Counter(sources))
    }


def calculate_score_distribution(
    all_results: List[List[Dict]]
) -> Dict[str, Any]:
    """
    Calculate score distribution across multiple queries.
    
    Args:
        all_results: List of retrieval results (each is a list of chunks)
        
    Returns:
        Dictionary with distribution statistics
    """
    all_scores = []
    for results in all_results:
        scores = [chunk.get('retrieval_score', 0) for chunk in results]
        all_scores.extend(scores)
    
    if not all_scores:
        return {
            "mean": 0,
            "std": 0,
            "percentiles": {}
        }
    
    return {
        "mean": float(np.mean(all_scores)),
        "std": float(np.std(all_scores)),
        "median": float(np.median(all_scores)),
        "min": float(np.min(all_scores)),
        "max": float(np.max(all_scores)),
        "percentiles": {
            "p25": float(np.percentile(all_scores, 25)),
            "p50": float(np.percentile(all_scores, 50)),
            "p75": float(np.percentile(all_scores, 75)),
            "p90": float(np.percentile(all_scores, 90)),
            "p95": float(np.percentile(all_scores, 95))
        }
    }


def calculate_diversity_metrics(
    retrieved_chunks: List[Dict]
) -> Dict[str, Any]:
    """
    Calculate diversity metrics for retrieved chunks.
    
    Args:
        retrieved_chunks: List of retrieved chunks
        
    Returns:
        Dictionary with diversity metrics
    """
    if not retrieved_chunks:
        return {
            "source_diversity": 0,
            "avg_text_length": 0
        }
    
    sources = [chunk.get('source', 'unknown') for chunk in retrieved_chunks]
    text_lengths = [len(chunk.get('text', '')) for chunk in retrieved_chunks]
    
    # Calculate source diversity (entropy-like metric)
    source_counts = Counter(sources)
    total = len(sources)
    diversity = len(source_counts) / total if total > 0 else 0
    
    return {
        "source_diversity": diversity,
        "unique_sources": len(source_counts),
        "total_chunks": total,
        "avg_text_length": float(np.mean(text_lengths)),
        "text_length_std": float(np.std(text_lengths))
    }


def analyze_vector_vs_bm25(
    retrieved_chunks: List[Dict]
) -> Dict[str, Any]:
    """
    Analyze the contribution of vector vs BM25 search.
    
    Args:
        retrieved_chunks: List of chunks with vector_score and bm25_score
        
    Returns:
        Analysis dictionary
    """
    if not retrieved_chunks:
        return {
            "vector_dominance": 0,
            "bm25_dominance": 0,
            "balanced_ratio": 0
        }
    
    vector_scores = [chunk.get('vector_score', 0) for chunk in retrieved_chunks]
    bm25_scores = [chunk.get('bm25_score', 0) for chunk in retrieved_chunks]
    
    # Count how many times each method "wins"
    vector_wins = sum(1 for v, b in zip(vector_scores, bm25_scores) if v > b)
    bm25_wins = sum(1 for v, b in zip(vector_scores, bm25_scores) if b > v)
    ties = len(retrieved_chunks) - vector_wins - bm25_wins
    
    total = len(retrieved_chunks)
    
    return {
        "vector_dominance": vector_wins / total if total > 0 else 0,
        "bm25_dominance": bm25_wins / total if total > 0 else 0,
        "ties": ties / total if total > 0 else 0,
        "avg_vector_score": float(np.mean(vector_scores)) if vector_scores else 0,
        "avg_bm25_score": float(np.mean(bm25_scores)) if bm25_scores else 0,
        "correlation": float(np.corrcoef(vector_scores, bm25_scores)[0, 1]) if len(vector_scores) > 1 else 0
    }


def detect_potential_issues(
    retrieval_results: List[Dict],
    score_threshold: float = 0.3
) -> List[str]:
    """
    Detect potential retrieval issues.
    
    Args:
        retrieval_results: List of retrieved chunks
        score_threshold: Threshold for low scores
        
    Returns:
        List of issue descriptions
    """
    issues = []
    
    if not retrieval_results:
        issues.append("No results retrieved")
        return issues
    
    scores = [chunk.get('retrieval_score', 0) for chunk in retrieval_results]
    
    # Check for low scores
    low_scores = [s for s in scores if s < score_threshold]
    if len(low_scores) > len(scores) * 0.5:
        issues.append(f"High proportion of low scores (>{len(low_scores)}/{len(scores)} below {score_threshold})")
    
    # Check for very low diversity
    sources = [chunk.get('source', 'unknown') for chunk in retrieval_results]
    unique_sources = len(set(sources))
    if unique_sources == 1 and len(retrieval_results) > 3:
        issues.append("Very low source diversity - all results from same document")
    
    # Check for score variance
    if len(scores) > 1:
        score_std = np.std(scores)
        if score_std < 0.05:
            issues.append("Very low score variance - retrieval may not be discriminative")
    
    return issues


if __name__ == "__main__":
    # Test retriever tools
    print("=== Testing Retriever Tools ===\n")
    
    # Sample data
    sample_chunks = [
        {'text': 'Sample text 1', 'retrieval_score': 0.8, 'vector_score': 0.9, 'bm25_score': 0.7, 'source': 'doc1.pdf'},
        {'text': 'Sample text 2', 'retrieval_score': 0.6, 'vector_score': 0.5, 'bm25_score': 0.7, 'source': 'doc2.pdf'},
        {'text': 'Sample text 3', 'retrieval_score': 0.4, 'vector_score': 0.3, 'bm25_score': 0.5, 'source': 'doc1.pdf'},
    ]
    
    metrics = calculate_retrieval_metrics(sample_chunks, "test query")
    print("Retrieval metrics:")
    print(metrics)
    print()
    
    diversity = calculate_diversity_metrics(sample_chunks)
    print("Diversity metrics:")
    print(diversity)
    print()
    
    analysis = analyze_vector_vs_bm25(sample_chunks)
    print("Vector vs BM25 analysis:")
    print(analysis)
    print()
    
    issues = detect_potential_issues(sample_chunks)
    print("Potential issues:")
    print(issues)

