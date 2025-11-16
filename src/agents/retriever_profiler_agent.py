"""
Retriever Profiler Agent
=========================

Profiles retrieval behavior by running test queries and collecting metrics.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..components.retriever import HybridRetriever
from ..tools.retriever_tools import (
    calculate_retrieval_metrics,
    calculate_score_distribution,
    calculate_diversity_metrics,
    analyze_vector_vs_bm25,
    detect_potential_issues
)
from ..components.evaluator import load_test_queries


class RetrieverProfilerAgent:
    """
    Agent that profiles retrieval performance and characteristics.
    """
    
    def __init__(self, retriever: HybridRetriever):
        """
        Initialize the profiler agent.
        
        Args:
            retriever: Retriever to profile
        """
        self.retriever = retriever
    
    def profile_retrieval(
        self,
        queries: List[Dict[str, str]],
        k: int = 5,
        method: str = "hybrid",
        show_progress: bool = True,
        **retrieval_kwargs
    ) -> Dict[str, Any]:
        """
        Profile retrieval behavior on a set of queries.
        
        Args:
            queries: List of test queries
            k: Number of chunks to retrieve
            method: Retrieval method
            show_progress: Whether to show progress
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            Profiling report
        """
        all_results = []
        per_query_metrics = []
        
        total = len(queries)
        
        if show_progress:
            print(f"\n🔍 Profiling retrieval on {total} queries...")
        
        for i, query_dict in enumerate(queries, 1):
            query = query_dict.get('query', query_dict.get('question', ''))
            
            if show_progress:
                print(f"  [{i}/{total}] {query[:60]}...")
            
            try:
                # Retrieve
                results = self.retriever.retrieve(
                    query=query,
                    k=k,
                    method=method,
                    **retrieval_kwargs
                )
                
                all_results.append(results)
                
                # Calculate per-query metrics
                query_metrics = calculate_retrieval_metrics(results, query)
                query_metrics['query'] = query
                per_query_metrics.append(query_metrics)
                
                if show_progress:
                    print(f"    Avg score: {query_metrics['avg_score']:.3f}, "
                          f"Sources: {query_metrics['unique_sources']}")
            
            except Exception as e:
                if show_progress:
                    print(f"    Error: {e}")
                per_query_metrics.append({
                    'query': query,
                    'error': str(e)
                })
        
        # Aggregate metrics
        if show_progress:
            print(f"\n📊 Calculating aggregate metrics...")
        
        score_distribution = calculate_score_distribution(all_results)
        
        # Calculate diversity across all results
        all_chunks = [chunk for results in all_results for chunk in results]
        overall_diversity = calculate_diversity_metrics(all_chunks)
        
        # Analyze vector vs BM25 if hybrid method
        if method == "hybrid":
            vector_bm25_analysis = analyze_vector_vs_bm25(all_chunks)
        else:
            vector_bm25_analysis = {}
        
        # Detect issues
        issues = []
        for results in all_results:
            query_issues = detect_potential_issues(results)
            issues.extend(query_issues)
        
        # Count issue frequency
        unique_issues = list(set(issues))
        issue_counts = {issue: issues.count(issue) for issue in unique_issues}
        
        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": total,
            "retrieval_config": {
                "k": k,
                "method": method,
                **retrieval_kwargs
            },
            "score_distribution": score_distribution,
            "diversity_metrics": overall_diversity,
            "vector_bm25_analysis": vector_bm25_analysis,
            "detected_issues": issue_counts,
            "per_query_metrics": per_query_metrics,
            "summary": {
                "avg_retrieval_score": score_distribution.get('mean', 0),
                "score_std": score_distribution.get('std', 0),
                "avg_unique_sources": float(sum(
                    m.get('unique_sources', 0) for m in per_query_metrics
                ) / len(per_query_metrics)) if per_query_metrics else 0,
                "total_issues_detected": len(issues),
                "unique_issues": len(unique_issues)
            }
        }
        
        if show_progress:
            print(f"\n✅ Profiling complete!")
            print(f"   Average retrieval score: {report['summary']['avg_retrieval_score']:.3f}")
            print(f"   Issues detected: {report['summary']['total_issues_detected']}")
            print()
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """
        Save profiling report to file.
        
        Args:
            report: Profiling report
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Profiling report saved: {output_path}")
    
    def load_report(self, input_path: str) -> Dict[str, Any]:
        """
        Load profiling report from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Profiling report
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def run(
        self,
        queries_path: str,
        output_path: str,
        k: int = 5,
        method: str = "hybrid",
        **retrieval_kwargs
    ) -> Dict[str, Any]:
        """
        Run the profiler agent end-to-end.
        
        Args:
            queries_path: Path to test queries JSON
            output_path: Output path for report
            k: Number of chunks to retrieve
            method: Retrieval method
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            Profiling report
        """
        print("\n" + "="*60)
        print("RETRIEVER PROFILER AGENT")
        print("="*60)
        
        # Load queries
        queries = load_test_queries(queries_path)
        print(f"\n📋 Loaded {len(queries)} test queries from {queries_path}")
        
        # Profile retrieval
        report = self.profile_retrieval(
            queries=queries,
            k=k,
            method=method,
            show_progress=True,
            **retrieval_kwargs
        )
        
        # Save report
        self.save_report(report, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        print(f"Average Retrieval Score: {report['summary']['avg_retrieval_score']:.3f}")
        print(f"Score Std Dev: {report['summary']['score_std']:.3f}")
        print(f"Average Unique Sources: {report['summary']['avg_unique_sources']:.2f}")
        print(f"Issues Detected: {report['summary']['total_issues_detected']}")
        
        if report.get('detected_issues'):
            print("\nMost Common Issues:")
            for issue, count in sorted(
                report['detected_issues'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]:
                print(f"  - {issue} (occurred {count} times)")
        
        print("="*60 + "\n")
        
        return report


if __name__ == "__main__":
    print("=== Retriever Profiler Agent ===")
    print("This agent profiles retrieval behavior and collects metrics.")
    print("Use RetrieverProfilerAgent.run() to execute profiling.")

