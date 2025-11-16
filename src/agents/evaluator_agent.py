"""
Evaluator Agent
===============

Evaluates and compares baseline vs optimized RAG configurations.
"""

import json
import os
from typing import Dict, Any, Optional, List

from ..components.evaluator import RAGEvaluator, load_test_queries
from ..components.retriever import HybridRetriever, RAGGenerator
from ..tools.evaluation_tools import create_comparison_report


class EvaluatorAgent:
    """
    Agent that evaluates RAG configurations and compares results.
    """
    
    def __init__(
        self,
        baseline_retriever: HybridRetriever,
        baseline_generator: RAGGenerator,
        optimized_retriever: Optional[HybridRetriever] = None,
        optimized_generator: Optional[RAGGenerator] = None
    ):
        """
        Initialize the evaluator agent.
        
        Args:
            baseline_retriever: Baseline retriever
            baseline_generator: Baseline generator
            optimized_retriever: Optimized retriever (optional)
            optimized_generator: Optimized generator (optional)
        """
        self.baseline_evaluator = RAGEvaluator(baseline_retriever, baseline_generator)
        
        if optimized_retriever and optimized_generator:
            self.optimized_evaluator = RAGEvaluator(optimized_retriever, optimized_generator)
        else:
            self.optimized_evaluator = None
    
    def set_optimized_configuration(
        self,
        retriever: HybridRetriever,
        generator: RAGGenerator
    ):
        """
        Set the optimized configuration.
        
        Args:
            retriever: Optimized retriever
            generator: Optimized generator
        """
        self.optimized_evaluator = RAGEvaluator(retriever, generator)
    
    def evaluate_baseline(
        self,
        queries: List[Dict[str, str]],
        output_path: str,
        **eval_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate baseline configuration.
        
        Args:
            queries: Test queries
            output_path: Output path for results
            **eval_kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results
        """
        print("\n📊 Evaluating BASELINE configuration...")
        
        evaluations = self.baseline_evaluator.evaluate_query_set(
            queries=queries,
            show_progress=True,
            **eval_kwargs
        )
        
        self.baseline_evaluator.save_evaluation_results(
            evaluations=evaluations,
            output_path=output_path,
            config_name="baseline"
        )
        
        return evaluations
    
    def evaluate_optimized(
        self,
        queries: List[Dict[str, str]],
        output_path: str,
        **eval_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Evaluate optimized configuration.
        
        Args:
            queries: Test queries
            output_path: Output path for results
            **eval_kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results
        """
        if not self.optimized_evaluator:
            raise ValueError("Optimized configuration not set. Call set_optimized_configuration() first.")
        
        print("\n📊 Evaluating OPTIMIZED configuration...")
        
        evaluations = self.optimized_evaluator.evaluate_query_set(
            queries=queries,
            show_progress=True,
            **eval_kwargs
        )
        
        self.optimized_evaluator.save_evaluation_results(
            evaluations=evaluations,
            output_path=output_path,
            config_name="optimized"
        )
        
        return evaluations
    
    def compare_configurations(
        self,
        baseline_results_path: str,
        optimized_results_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Compare baseline and optimized configurations.
        
        Args:
            baseline_results_path: Path to baseline results
            optimized_results_path: Path to optimized results
            output_path: Output path for comparison report
            
        Returns:
            Comparison report
        """
        print("\n📈 Comparing baseline vs optimized configurations...")
        
        report = self.baseline_evaluator.compare_configurations(
            baseline_results_path=baseline_results_path,
            optimized_results_path=optimized_results_path,
            output_path=output_path
        )
        
        return report
    
    def run(
        self,
        queries_path: str,
        baseline_output_path: str,
        optimized_output_path: str,
        comparison_output_path: str,
        **eval_kwargs
    ) -> Dict[str, Any]:
        """
        Run the evaluator agent end-to-end.
        
        Args:
            queries_path: Path to test queries
            baseline_output_path: Output path for baseline results
            optimized_output_path: Output path for optimized results
            comparison_output_path: Output path for comparison report
            **eval_kwargs: Additional evaluation parameters
            
        Returns:
            Comparison report
        """
        print("\n" + "="*60)
        print("EVALUATOR AGENT")
        print("="*60)
        
        # Load queries
        queries = load_test_queries(queries_path)
        print(f"\n📋 Loaded {len(queries)} test queries from {queries_path}")
        
        # Evaluate baseline
        baseline_evals = self.evaluate_baseline(
            queries=queries,
            output_path=baseline_output_path,
            **eval_kwargs
        )
        
        # Evaluate optimized
        if self.optimized_evaluator:
            optimized_evals = self.evaluate_optimized(
                queries=queries,
                output_path=optimized_output_path,
                **eval_kwargs
            )
            
            # Compare
            report = self.compare_configurations(
                baseline_results_path=baseline_output_path,
                optimized_results_path=optimized_output_path,
                output_path=comparison_output_path
            )
        else:
            print("\n⚠️  Optimized configuration not set. Skipping comparison.")
            report = None
        
        # Print summary
        if report:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            
            baseline_avg = report['baseline_summary']['avg_score']
            optimized_avg = report['optimized_summary']['avg_score']
            improvement = report['improvement']['relative_improvement_pct']
            win_rate = report['win_rate']['win_rate'] * 100
            
            print(f"Baseline Average Score: {baseline_avg:.2f}/10")
            print(f"Optimized Average Score: {optimized_avg:.2f}/10")
            print(f"Improvement: {improvement:+.1f}%")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"\nRecommendation: {report['recommendation']}")
            print("="*60 + "\n")
        
        return report


if __name__ == "__main__":
    print("=== Evaluator Agent ===")
    print("This agent evaluates and compares RAG configurations.")
    print("Use EvaluatorAgent.run() to execute evaluation.")

