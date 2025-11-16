"""
RAG Evaluator
=============

Evaluates RAG configurations using test queries and LLM-based judging.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..tools.llm_tools import LLMClient
from ..tools.evaluation_tools import (
    calculate_average_score,
    summarize_evaluations,
    create_comparison_report
)
from .retriever import HybridRetriever, RAGGenerator


class RAGEvaluator:
    """
    Evaluates RAG system performance using test queries.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: RAGGenerator,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize RAG evaluator.
        
        Args:
            retriever: Retriever instance
            generator: Generator instance
            llm_client: LLM client for judging (creates new if not provided)
        """
        self.retriever = retriever
        self.generator = generator
        self.llm_client = llm_client or LLMClient(temperature=0.3)
    
    def evaluate_single_query(
        self,
        query: str,
        reference_answer: Optional[str] = None,
        retrieval_k: int = 5,
        retrieval_method: str = "hybrid",
        **retrieval_kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate RAG on a single query.
        
        Args:
            query: Test query
            reference_answer: Optional reference answer for comparison
            retrieval_k: Number of chunks to retrieve
            retrieval_method: Retrieval method to use
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            Evaluation result dictionary
        """
        try:
            # Retrieve chunks
            retrieved_chunks = self.retriever.retrieve(
                query=query,
                k=retrieval_k,
                method=retrieval_method,
                **retrieval_kwargs
            )
            
            # Generate answer
            rag_result = self.generator.generate_answer(
                query=query,
                context_chunks=retrieved_chunks
            )
            
            # Build context summary for judge
            context_summary = "\n".join([
                f"Source: {chunk.get('source', 'unknown')}" 
                for chunk in retrieved_chunks[:3]
            ])
            
            # Judge answer quality
            judgment = self.llm_client.judge_answer_quality(
                query=query,
                answer=rag_result['answer'],
                context=context_summary,
                reference_answer=reference_answer
            )
            
            # Combine results
            evaluation = {
                "query": query,
                "answer": rag_result['answer'],
                "score": judgment.get('score', 0),
                "reasoning": judgment.get('reasoning', ''),
                "strengths": judgment.get('strengths', []),
                "weaknesses": judgment.get('weaknesses', []),
                "num_chunks_retrieved": len(retrieved_chunks),
                "sources": rag_result['sources'],
                "tokens_used": rag_result['tokens_used'],
                "retrieval_scores": [
                    chunk.get('retrieval_score', 0) 
                    for chunk in retrieved_chunks
                ],
                "status": "success"
            }
            
            return evaluation
            
        except Exception as e:
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "score": 0
            }
    
    def evaluate_query_set(
        self,
        queries: List[Dict[str, str]],
        retrieval_k: int = 5,
        retrieval_method: str = "hybrid",
        show_progress: bool = True,
        **retrieval_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate RAG on a set of queries.
        
        Args:
            queries: List of query dicts with 'query' and optional 'reference_answer'
            retrieval_k: Number of chunks to retrieve
            retrieval_method: Retrieval method
            show_progress: Whether to show progress
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            List of evaluation results
        """
        evaluations = []
        total = len(queries)
        
        if show_progress:
            print(f"\n🔍 Evaluating {total} queries...")
        
        for i, query_dict in enumerate(queries, 1):
            query = query_dict.get('query', query_dict.get('question', ''))
            reference = query_dict.get('reference_answer')
            
            if show_progress:
                print(f"  [{i}/{total}] {query[:60]}...")
            
            evaluation = self.evaluate_single_query(
                query=query,
                reference_answer=reference,
                retrieval_k=retrieval_k,
                retrieval_method=retrieval_method,
                **retrieval_kwargs
            )
            
            evaluations.append(evaluation)
            
            if show_progress and evaluation['status'] == 'success':
                print(f"    Score: {evaluation['score']}/10")
        
        if show_progress:
            avg_score = calculate_average_score(evaluations)
            print(f"\n✅ Evaluation complete. Average score: {avg_score:.2f}/10\n")
        
        return evaluations
    
    def save_evaluation_results(
        self,
        evaluations: List[Dict],
        output_path: str,
        config_name: str = "default",
        metadata: Optional[Dict] = None
    ):
        """
        Save evaluation results to file.
        
        Args:
            evaluations: List of evaluation results
            output_path: Output file path
            config_name: Configuration name
            metadata: Additional metadata to save
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        summary = summarize_evaluations(evaluations, config_name)
        
        results = {
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "evaluations": evaluations,
            "metadata": metadata or {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Evaluation results saved: {output_path}")
    
    def load_evaluation_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Evaluation results dictionary
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def compare_configurations(
        self,
        baseline_results_path: str,
        optimized_results_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two configuration evaluation results.
        
        Args:
            baseline_results_path: Path to baseline results
            optimized_results_path: Path to optimized results
            output_path: Optional path to save comparison report
            
        Returns:
            Comparison report
        """
        baseline_data = self.load_evaluation_results(baseline_results_path)
        optimized_data = self.load_evaluation_results(optimized_results_path)
        
        baseline_evals = baseline_data['evaluations']
        optimized_evals = optimized_data['evaluations']
        
        report = create_comparison_report(baseline_evals, optimized_evals)
        
        # Add metadata
        report['baseline_config'] = baseline_data.get('config_name', 'baseline')
        report['optimized_config'] = optimized_data.get('config_name', 'optimized')
        report['timestamp'] = datetime.now().isoformat()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"💾 Comparison report saved: {output_path}")
        
        return report


def load_test_queries(queries_path: str) -> List[Dict[str, str]]:
    """
    Load test queries from JSON file.
    
    Args:
        queries_path: Path to queries JSON file
        
    Returns:
        List of query dictionaries
    """
    with open(queries_path, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'queries' in data:
        return data['queries']
    else:
        raise ValueError("Invalid query file format")


if __name__ == "__main__":
    print("=== RAG Evaluator Module ===")
    print("This module provides evaluation functionality for RAG systems.")
    print("Import and use RAGEvaluator class to evaluate your RAG configurations.")

