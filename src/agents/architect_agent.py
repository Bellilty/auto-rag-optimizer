"""
Architect Agent
===============

Synthesizes all reports and outputs final optimized RAG configuration.
"""

import json
import os
import yaml
from typing import Dict, Any, Optional
from datetime import datetime

from ..tools.llm_tools import LLMClient


class ArchitectAgent:
    """
    Agent that synthesizes profiling and evaluation data to create final configuration.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the architect agent.
        
        Args:
            llm_client: LLM client for reasoning (creates new if not provided)
        """
        self.llm_client = llm_client or LLMClient(temperature=0.5)
    
    def synthesize_configuration(
        self,
        profiling_report: Dict[str, Any],
        chunk_proposal: Dict[str, Any],
        evaluation_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize all reports into final optimized configuration.
        
        Args:
            profiling_report: Report from RetrieverProfilerAgent
            chunk_proposal: Proposal from ChunkArchitectAgent
            evaluation_report: Optional comparison report from EvaluatorAgent
            
        Returns:
            Final optimized configuration
        """
        print("\n🎯 Synthesizing final optimized RAG configuration...")
        
        # Extract key information
        retrieval_config = profiling_report.get('retrieval_config', {})
        profiling_summary = profiling_report.get('summary', {})
        vector_bm25_analysis = profiling_report.get('vector_bm25_analysis', {})
        
        proposed_chunking = chunk_proposal.get('proposed_config', {})
        chunk_reasoning = chunk_proposal.get('llm_recommendation', {}).get('reasoning', '')
        
        # Build context for LLM
        context = f"""RAG System Analysis Summary:

Current Retrieval Configuration:
- Method: {retrieval_config.get('method', 'hybrid')}
- Top-k: {retrieval_config.get('k', 5)}
- Vector weight: {retrieval_config.get('vector_weight', 0.7)}
- BM25 weight: {retrieval_config.get('bm25_weight', 0.3)}

Retrieval Performance:
- Average retrieval score: {profiling_summary.get('avg_retrieval_score', 0):.3f}
- Score std dev: {profiling_summary.get('score_std', 0):.3f}

Vector vs BM25 Analysis:
- Vector dominance: {vector_bm25_analysis.get('vector_dominance', 0):.2%}
- BM25 dominance: {vector_bm25_analysis.get('bm25_dominance', 0):.2%}
- Correlation: {vector_bm25_analysis.get('correlation', 0):.3f}

Proposed Chunking:
- Chunk size: {proposed_chunking.get('chunk_size', 1000)} words
- Overlap: {proposed_chunking.get('overlap', 200)} words
- Reasoning: {chunk_reasoning}
"""
        
        if evaluation_report:
            improvement = evaluation_report.get('improvement', {})
            context += f"""
Evaluation Results:
- Baseline avg score: {improvement.get('baseline_avg', 0):.2f}/10
- Optimized avg score: {improvement.get('optimized_avg', 0):.2f}/10
- Improvement: {improvement.get('relative_improvement_pct', 0):+.1f}%
- Recommendation: {evaluation_report.get('recommendation', 'N/A')}
"""
        
        # Prompt LLM for final configuration
        system_message = """You are an expert RAG architect making final configuration decisions.

Based on profiling, chunking proposals, and evaluation results, determine the optimal configuration.

Consider:
1. Chunking parameters (size, overlap)
2. Hybrid retrieval weights (vector vs BM25)
3. Top-k retrieval parameter
4. Any other optimizations

Respond in JSON format:
{
    "chunking": {
        "chunk_size": <number>,
        "overlap": <number>,
        "strategy": "<description>"
    },
    "retrieval": {
        "method": "hybrid",
        "top_k": <number>,
        "vector_weight": <0-1>,
        "bm25_weight": <0-1>,
        "retrieval_pool": <number>
    },
    "reasoning": "<detailed explanation>",
    "expected_benefits": ["<benefit1>", "<benefit2>", ...],
    "confidence": "<high/medium/low>"
}"""

        user_message = f"""{context}

Based on this analysis, what is the optimal final RAG configuration?
Provide your recommendation in JSON format."""

        # Get LLM recommendation
        recommendation = self.llm_client.prompt_json(
            user_message=user_message,
            system_message=system_message,
            temperature=0.4  # Lower temperature for consistent decisions
        )
        
        # Build final configuration
        final_config = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "configuration": {
                "chunking": recommendation.get('chunking', proposed_chunking),
                "retrieval": recommendation.get('retrieval', retrieval_config),
                "generation": {
                    "model": "gpt-4o-mini",
                    "max_tokens": 500,
                    "temperature": 0.3
                }
            },
            "reasoning": recommendation.get('reasoning', ''),
            "expected_benefits": recommendation.get('expected_benefits', []),
            "confidence": recommendation.get('confidence', 'medium'),
            "source_reports": {
                "profiling_summary": profiling_summary,
                "chunk_proposal_summary": {
                    "proposed_chunk_size": proposed_chunking.get('chunk_size'),
                    "proposed_overlap": proposed_chunking.get('overlap')
                }
            }
        }
        
        if evaluation_report:
            final_config['source_reports']['evaluation_summary'] = {
                "baseline_avg": evaluation_report.get('improvement', {}).get('baseline_avg'),
                "optimized_avg": evaluation_report.get('improvement', {}).get('optimized_avg'),
                "improvement_pct": evaluation_report.get('improvement', {}).get('relative_improvement_pct')
            }
        
        print(f"✅ Final configuration synthesized")
        print(f"   Chunk size: {final_config['configuration']['chunking'].get('chunk_size')} words")
        print(f"   Top-k: {final_config['configuration']['retrieval'].get('top_k')}")
        print(f"   Confidence: {final_config['confidence']}")
        
        return final_config
    
    def save_configuration(
        self,
        config: Dict[str, Any],
        yaml_path: str,
        json_path: Optional[str] = None
    ):
        """
        Save configuration to YAML (and optionally JSON).
        
        Args:
            config: Configuration dictionary
            yaml_path: Output path for YAML file
            json_path: Optional output path for JSON file
        """
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        # Save as YAML
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"💾 Configuration saved (YAML): {yaml_path}")
        
        # Optionally save as JSON for easier programmatic access
        if json_path:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"💾 Configuration saved (JSON): {json_path}")
    
    def load_configuration(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from file (YAML or JSON).
        
        Args:
            path: Input file path
            
        Returns:
            Configuration dictionary
        """
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                return json.load(f)
    
    def run(
        self,
        profiling_report_path: str,
        chunk_proposal_path: str,
        evaluation_report_path: Optional[str],
        output_yaml_path: str,
        output_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the architect agent end-to-end.
        
        Args:
            profiling_report_path: Path to profiling report
            chunk_proposal_path: Path to chunk proposal
            evaluation_report_path: Optional path to evaluation report
            output_yaml_path: Output path for YAML configuration
            output_json_path: Optional output path for JSON configuration
            
        Returns:
            Final configuration
        """
        print("\n" + "="*60)
        print("ARCHITECT AGENT")
        print("="*60)
        
        # Load reports
        print(f"\n📋 Loading reports...")
        
        with open(profiling_report_path, 'r') as f:
            profiling_report = json.load(f)
        print(f"  ✓ Profiling report: {profiling_report_path}")
        
        with open(chunk_proposal_path, 'r') as f:
            chunk_proposal = json.load(f)
        print(f"  ✓ Chunk proposal: {chunk_proposal_path}")
        
        evaluation_report = None
        if evaluation_report_path and os.path.exists(evaluation_report_path):
            with open(evaluation_report_path, 'r') as f:
                evaluation_report = json.load(f)
            print(f"  ✓ Evaluation report: {evaluation_report_path}")
        
        # Synthesize configuration
        final_config = self.synthesize_configuration(
            profiling_report=profiling_report,
            chunk_proposal=chunk_proposal,
            evaluation_report=evaluation_report
        )
        
        # Save configuration
        self.save_configuration(
            config=final_config,
            yaml_path=output_yaml_path,
            json_path=output_json_path
        )
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL OPTIMIZED CONFIGURATION")
        print("="*60)
        
        chunking = final_config['configuration']['chunking']
        retrieval = final_config['configuration']['retrieval']
        
        print(f"\nChunking:")
        print(f"  - Size: {chunking.get('chunk_size')} words")
        print(f"  - Overlap: {chunking.get('overlap')} words")
        
        print(f"\nRetrieval:")
        print(f"  - Method: {retrieval.get('method')}")
        print(f"  - Top-k: {retrieval.get('top_k')}")
        print(f"  - Vector weight: {retrieval.get('vector_weight', 0):.2f}")
        print(f"  - BM25 weight: {retrieval.get('bm25_weight', 0):.2f}")
        
        if final_config.get('expected_benefits'):
            print(f"\nExpected Benefits:")
            for benefit in final_config['expected_benefits'][:3]:
                print(f"  - {benefit}")
        
        print(f"\nConfidence: {final_config['confidence'].upper()}")
        print("="*60 + "\n")
        
        return final_config


if __name__ == "__main__":
    print("=== Architect Agent ===")
    print("This agent synthesizes all reports into final optimized configuration.")
    print("Use ArchitectAgent.run() to execute.")

