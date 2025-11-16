"""
Chunk Architect Agent
======================

Proposes optimal chunking parameters based on retrieval profiling.
Uses LLM reasoning to suggest improvements.
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

from ..tools.llm_tools import LLMClient
from ..tools.chunking_tools import (
    validate_chunking_params,
    suggest_chunking_strategy
)


class ChunkArchitectAgent:
    """
    Agent that proposes optimized chunking parameters.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the chunk architect agent.
        
        Args:
            llm_client: LLM client for reasoning (creates new if not provided)
        """
        self.llm_client = llm_client or LLMClient(temperature=0.7)
    
    def analyze_profiling_report(
        self,
        profiling_report: Dict[str, Any],
        current_chunk_size: int,
        current_overlap: int
    ) -> Dict[str, Any]:
        """
        Analyze profiling report and propose new chunking parameters.
        
        Args:
            profiling_report: Report from RetrieverProfilerAgent
            current_chunk_size: Current chunk size in words
            current_overlap: Current overlap in words
            
        Returns:
            Proposed chunking configuration
        """
        print("\n🏗️  Analyzing retrieval profile to propose chunking improvements...")
        
        # Extract key metrics
        summary = profiling_report.get('summary', {})
        issues = profiling_report.get('detected_issues', {})
        score_dist = profiling_report.get('score_distribution', {})
        
        # Build context for LLM
        context = f"""Current RAG Configuration:
- Chunk size: {current_chunk_size} words
- Overlap: {current_overlap} words
- Overlap ratio: {current_overlap/current_chunk_size*100:.1f}%

Retrieval Performance Metrics:
- Average retrieval score: {summary.get('avg_retrieval_score', 0):.3f}
- Score standard deviation: {summary.get('score_std', 0):.3f}
- Average unique sources per query: {summary.get('avg_unique_sources', 0):.2f}
- Score distribution: mean={score_dist.get('mean', 0):.3f}, median={score_dist.get('median', 0):.3f}
- Percentiles: p25={score_dist.get('percentiles', {}).get('p25', 0):.3f}, p75={score_dist.get('percentiles', {}).get('p75', 0):.3f}

Detected Issues:
"""
        
        if issues:
            for issue, count in issues.items():
                context += f"- {issue} (occurred {count} times)\n"
        else:
            context += "- No significant issues detected\n"
        
        # Prompt LLM for chunking recommendations
        system_message = """You are an expert in RAG optimization, specifically in document chunking strategies.

Your task is to analyze retrieval performance metrics and propose optimal chunking parameters.

Consider:
1. If retrieval scores are low, smaller chunks might improve precision
2. If diversity is low, different chunk sizes or higher overlap might help
3. If scores are already high, maintain or slightly optimize current approach
4. Overlap should typically be 15-30% of chunk size
5. Chunk size typically ranges from 200-1500 words

Respond in JSON format:
{
    "recommended_chunk_size": <number>,
    "recommended_overlap": <number>,
    "reasoning": "<detailed explanation>",
    "expected_improvements": ["<improvement1>", "<improvement2>", ...],
    "risks": ["<risk1>", ...],
    "confidence": "<high/medium/low>"
}"""

        user_message = f"""{context}

Based on these metrics, what chunking parameters would you recommend?
Provide your analysis and recommendations in JSON format."""

        # Get LLM recommendation
        recommendation = self.llm_client.prompt_json(
            user_message=user_message,
            system_message=system_message,
            temperature=0.5  # Lower temperature for more consistent recommendations
        )
        
        # Validate proposed parameters
        proposed_size = recommendation.get('recommended_chunk_size', current_chunk_size)
        proposed_overlap = recommendation.get('recommended_overlap', current_overlap)
        
        validation = validate_chunking_params(proposed_size, proposed_overlap)
        
        # Build final proposal
        proposal = {
            "timestamp": datetime.now().isoformat(),
            "current_config": {
                "chunk_size": current_chunk_size,
                "overlap": current_overlap
            },
            "proposed_config": {
                "chunk_size": proposed_size,
                "overlap": proposed_overlap,
                "overlap_ratio": proposed_overlap / proposed_size if proposed_size > 0 else 0
            },
            "llm_recommendation": recommendation,
            "validation": validation,
            "profiling_summary": summary
        }
        
        print(f"✅ Proposed chunk size: {proposed_size} words (overlap: {proposed_overlap})")
        print(f"   Reasoning: {recommendation.get('reasoning', 'N/A')[:100]}...")
        print(f"   Confidence: {recommendation.get('confidence', 'N/A')}")
        
        if not validation['valid']:
            print(f"⚠️  Warning: Validation errors detected:")
            for error in validation['errors']:
                print(f"   - {error}")
        
        return proposal
    
    def save_proposal(self, proposal: Dict[str, Any], output_path: str):
        """
        Save chunking proposal to file.
        
        Args:
            proposal: Chunking proposal
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(proposal, f, indent=2)
        
        print(f"💾 Chunking proposal saved: {output_path}")
    
    def load_proposal(self, input_path: str) -> Dict[str, Any]:
        """
        Load chunking proposal from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Chunking proposal
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def run(
        self,
        profiling_report_path: str,
        output_path: str,
        current_chunk_size: int = 1000,
        current_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Run the chunk architect agent end-to-end.
        
        Args:
            profiling_report_path: Path to profiling report JSON
            output_path: Output path for proposal
            current_chunk_size: Current chunk size
            current_overlap: Current overlap
            
        Returns:
            Chunking proposal
        """
        print("\n" + "="*60)
        print("CHUNK ARCHITECT AGENT")
        print("="*60)
        
        # Load profiling report
        with open(profiling_report_path, 'r') as f:
            profiling_report = json.load(f)
        
        print(f"\n📋 Loaded profiling report from {profiling_report_path}")
        
        # Analyze and propose
        proposal = self.analyze_profiling_report(
            profiling_report=profiling_report,
            current_chunk_size=current_chunk_size,
            current_overlap=current_overlap
        )
        
        # Save proposal
        self.save_proposal(proposal, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("CHUNKING PROPOSAL SUMMARY")
        print("="*60)
        print(f"Current: {current_chunk_size} words, {current_overlap} overlap")
        print(f"Proposed: {proposal['proposed_config']['chunk_size']} words, "
              f"{proposal['proposed_config']['overlap']} overlap")
        
        change_pct = (proposal['proposed_config']['chunk_size'] - current_chunk_size) / current_chunk_size * 100
        print(f"Change: {change_pct:+.1f}% in chunk size")
        
        if proposal['llm_recommendation'].get('expected_improvements'):
            print("\nExpected Improvements:")
            for improvement in proposal['llm_recommendation']['expected_improvements'][:3]:
                print(f"  - {improvement}")
        
        print("="*60 + "\n")
        
        return proposal


if __name__ == "__main__":
    print("=== Chunk Architect Agent ===")
    print("This agent proposes optimal chunking parameters based on profiling.")
    print("Use ChunkArchitectAgent.run() to execute.")

