"""
Chunking Tools
==============

Utility functions for analyzing and optimizing chunking strategies.
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


def analyze_chunk_statistics(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Analyze statistics about existing chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with chunk statistics
    """
    if not chunks:
        return {
            "num_chunks": 0,
            "avg_length": 0,
            "length_std": 0
        }
    
    lengths = [len(chunk['text']) for chunk in chunks]
    word_counts = [len(chunk['text'].split()) for chunk in chunks]
    sources = [chunk.get('source', 'unknown') for chunk in chunks]
    
    return {
        "num_chunks": len(chunks),
        "avg_length_chars": float(np.mean(lengths)),
        "length_std_chars": float(np.std(lengths)),
        "min_length_chars": int(np.min(lengths)),
        "max_length_chars": int(np.max(lengths)),
        "avg_length_words": float(np.mean(word_counts)),
        "length_std_words": float(np.std(word_counts)),
        "min_length_words": int(np.min(word_counts)),
        "max_length_words": int(np.max(word_counts)),
        "num_sources": len(set(sources)),
        "chunks_per_source": dict(Counter(sources))
    }


def estimate_optimal_chunk_size(
    current_chunk_size: int,
    current_metrics: Dict[str, Any],
    target: str = "balanced"
) -> Dict[str, Any]:
    """
    Estimate optimal chunk size based on current metrics.
    
    Args:
        current_chunk_size: Current chunk size in words
        current_metrics: Current retrieval metrics
        target: Optimization target ("balanced", "recall", "precision")
        
    Returns:
        Recommended chunking parameters
    """
    # Simple heuristic-based recommendations
    recommendations = {
        "current_chunk_size": current_chunk_size,
        "target": target
    }
    
    avg_score = current_metrics.get("avg_score", 0.5)
    score_std = current_metrics.get("score_std", 0.1)
    
    if target == "balanced":
        # Aim for moderate chunks with good overlap
        if avg_score < 0.4:
            # Low scores might benefit from smaller chunks
            recommended_size = max(300, int(current_chunk_size * 0.7))
            recommended_overlap = int(recommended_size * 0.25)
            reason = "Low retrieval scores - trying smaller chunks for more precision"
        elif avg_score > 0.7 and score_std < 0.1:
            # High scores but low variance - might increase size
            recommended_size = min(1500, int(current_chunk_size * 1.2))
            recommended_overlap = int(recommended_size * 0.2)
            reason = "Good scores with low variance - can try larger chunks"
        else:
            # Current size seems reasonable
            recommended_size = current_chunk_size
            recommended_overlap = int(current_chunk_size * 0.2)
            reason = "Current chunking appears reasonable"
    
    elif target == "recall":
        # Smaller chunks with more overlap for better recall
        recommended_size = max(200, int(current_chunk_size * 0.6))
        recommended_overlap = int(recommended_size * 0.3)
        reason = "Optimizing for recall - smaller chunks with higher overlap"
    
    else:  # precision
        # Larger chunks with less overlap for precision
        recommended_size = min(1500, int(current_chunk_size * 1.3))
        recommended_overlap = int(recommended_size * 0.15)
        reason = "Optimizing for precision - larger, more focused chunks"
    
    recommendations.update({
        "recommended_chunk_size": recommended_size,
        "recommended_overlap": recommended_overlap,
        "reason": reason
    })
    
    return recommendations


def validate_chunking_params(
    chunk_size: int,
    overlap: int
) -> Dict[str, Any]:
    """
    Validate chunking parameters.
    
    Args:
        chunk_size: Proposed chunk size in words
        overlap: Proposed overlap in words
        
    Returns:
        Validation result with warnings/errors
    """
    issues = []
    warnings = []
    
    # Check basic constraints
    if chunk_size < 50:
        issues.append("Chunk size too small (< 50 words) - may lose context")
    elif chunk_size > 2000:
        warnings.append("Chunk size very large (> 2000 words) - may hurt precision")
    
    if overlap < 0:
        issues.append("Overlap cannot be negative")
    elif overlap >= chunk_size:
        issues.append("Overlap must be smaller than chunk size")
    elif overlap > chunk_size * 0.5:
        warnings.append("Overlap > 50% of chunk size - may cause redundancy")
    elif overlap < chunk_size * 0.1:
        warnings.append("Overlap < 10% of chunk size - may lose context between chunks")
    
    return {
        "valid": len(issues) == 0,
        "errors": issues,
        "warnings": warnings,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "overlap_ratio": overlap / chunk_size if chunk_size > 0 else 0
    }


def suggest_chunking_strategy(
    document_stats: Dict[str, Any],
    retrieval_issues: List[str]
) -> Dict[str, Any]:
    """
    Suggest chunking strategy based on document stats and retrieval issues.
    
    Args:
        document_stats: Statistics about documents
        retrieval_issues: List of detected retrieval issues
        
    Returns:
        Suggested strategy
    """
    strategy = {
        "approach": "word_based",
        "reasoning": []
    }
    
    # Analyze issues
    has_low_scores = any("low scores" in issue.lower() for issue in retrieval_issues)
    has_low_diversity = any("diversity" in issue.lower() for issue in retrieval_issues)
    
    avg_doc_length = document_stats.get("avg_length_words", 1000)
    
    if has_low_scores:
        strategy["chunk_size"] = max(300, int(avg_doc_length * 0.3))
        strategy["overlap"] = int(strategy["chunk_size"] * 0.25)
        strategy["reasoning"].append("Using smaller chunks to improve retrieval precision")
    
    elif has_low_diversity:
        strategy["chunk_size"] = max(400, int(avg_doc_length * 0.4))
        strategy["overlap"] = int(strategy["chunk_size"] * 0.3)
        strategy["reasoning"].append("Moderate chunks with good overlap to improve diversity")
    
    else:
        # Default balanced approach
        strategy["chunk_size"] = 800
        strategy["overlap"] = 200
        strategy["reasoning"].append("Balanced approach for general use")
    
    return strategy


if __name__ == "__main__":
    # Test chunking tools
    print("=== Testing Chunking Tools ===\n")
    
    # Sample chunks
    sample_chunks = [
        {'text': 'word ' * 500, 'source': 'doc1.pdf'},
        {'text': 'word ' * 600, 'source': 'doc1.pdf'},
        {'text': 'word ' * 450, 'source': 'doc2.pdf'},
    ]
    
    stats = analyze_chunk_statistics(sample_chunks)
    print("Chunk statistics:")
    print(stats)
    print()
    
    current_metrics = {
        "avg_score": 0.35,
        "score_std": 0.08
    }
    
    optimal = estimate_optimal_chunk_size(1000, current_metrics, target="balanced")
    print("Optimal chunk size estimation:")
    print(optimal)
    print()
    
    validation = validate_chunking_params(800, 200)
    print("Chunking params validation:")
    print(validation)
    print()
    
    strategy = suggest_chunking_strategy(stats, ["Low scores detected"])
    print("Suggested strategy:")
    print(strategy)

