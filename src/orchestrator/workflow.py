"""
Workflow Orchestrator
=====================

Orchestrates the multi-agent RAG optimization workflow.
"""

import os
import yaml
from typing import Dict, Any, Optional
from datetime import datetime

from ..components.chunker import DocumentChunker
from ..components.index_builder import IndexBuilder
from ..components.retriever import HybridRetriever, RAGGenerator
from ..agents.retriever_profiler_agent import RetrieverProfilerAgent
from ..agents.chunk_architect_agent import ChunkArchitectAgent
from ..agents.evaluator_agent import EvaluatorAgent
from ..agents.architect_agent import ArchitectAgent


class RAGOptimizationWorkflow:
    """
    Orchestrates the full RAG optimization pipeline.
    """
    
    def __init__(
        self,
        base_config_path: str = "src/configs/base_config.yaml",
        test_queries_path: str = "src/configs/test_queries.json",
        documents_dir: str = "data/raw_docs",
        index_dir: str = "data/index",
        output_dir: str = "outputs"
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            base_config_path: Path to base configuration
            test_queries_path: Path to test queries
            documents_dir: Directory containing documents
            index_dir: Directory for indexes
            output_dir: Output directory for reports
        """
        self.base_config_path = base_config_path
        self.test_queries_path = test_queries_path
        self.documents_dir = documents_dir
        self.index_dir = index_dir
        self.output_dir = output_dir
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        print(f"✅ Workflow initialized")
        print(f"   Base config: {base_config_path}")
        print(f"   Test queries: {test_queries_path}")
        print(f"   Documents: {documents_dir}")
    
    def run_full_optimization(
        self,
        skip_baseline_indexing: bool = False,
        skip_evaluation: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full optimization workflow.
        
        Args:
            skip_baseline_indexing: Skip baseline indexing if already exists
            skip_evaluation: Skip evaluation step (faster, for testing)
            
        Returns:
            Dictionary with all results and paths
        """
        print("\n" + "="*80)
        print(" "*20 + "AUTO-RAG OPTIMIZER")
        print(" "*15 + "Multi-Agent RAG Optimization Pipeline")
        print("="*80)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            "started_at": datetime.now().isoformat(),
            "config": self.base_config
        }
        
        # ===================================================================
        # STEP 1: Build Baseline Index
        # ===================================================================
        print("\n" + "─"*80)
        print("STEP 1/6: Build Baseline Index")
        print("─"*80)
        
        baseline_config = self.base_config['chunking']
        chunk_size = baseline_config['chunk_size']
        overlap = baseline_config['overlap']
        
        print(f"\nBaseline chunking: {chunk_size} words, {overlap} overlap")
        
        builder = IndexBuilder()
        
        baseline_index_exists = builder.indexes_exist(
            input_dir=self.index_dir,
            config_name="baseline"
        )
        
        if skip_baseline_indexing and baseline_index_exists:
            print("📂 Loading existing baseline index...")
            faiss_index, bm25_index, chunks = builder.load_indexes(
                input_dir=self.index_dir,
                config_name="baseline"
            )
        else:
            print("🔨 Building baseline index...")
            chunker = DocumentChunker(self.documents_dir)
            chunks = chunker.process_all_documents(
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            if not chunks:
                raise ValueError(f"No documents found in {self.documents_dir}")
            
            texts = [chunk['text'] for chunk in chunks]
            embeddings = builder.create_embeddings_batch(texts)
            
            faiss_index = builder.build_faiss_index(embeddings)
            bm25_index = builder.build_bm25_index(chunks)
            
            builder.save_indexes(
                faiss_index, bm25_index, chunks,
                output_dir=self.index_dir,
                config_name="baseline"
            )
        
        baseline_retriever = HybridRetriever(
            faiss_index, bm25_index, chunks
        )
        baseline_generator = RAGGenerator()
        
        results['baseline_chunks'] = len(chunks)
        
        # ===================================================================
        # STEP 2: Profile Retrieval
        # ===================================================================
        print("\n" + "─"*80)
        print("STEP 2/6: Profile Baseline Retrieval")
        print("─"*80)
        
        profiler = RetrieverProfilerAgent(baseline_retriever)
        
        retrieval_config = self.base_config.get('retrieval', {})
        profiling_report = profiler.run(
            queries_path=self.test_queries_path,
            output_path=os.path.join(self.output_dir, "reports", "retrieval_report.json"),
            k=retrieval_config.get('top_k', 5),
            method=retrieval_config.get('method', 'hybrid'),
            vector_weight=retrieval_config.get('vector_weight', 0.7),
            bm25_weight=retrieval_config.get('bm25_weight', 0.3)
        )
        
        results['profiling_report_path'] = os.path.join(self.output_dir, "reports", "retrieval_report.json")
        
        # ===================================================================
        # STEP 3: Propose Optimized Chunking
        # ===================================================================
        print("\n" + "─"*80)
        print("STEP 3/6: Propose Optimized Chunking")
        print("─"*80)
        
        chunk_architect = ChunkArchitectAgent()
        
        chunk_proposal = chunk_architect.run(
            profiling_report_path=results['profiling_report_path'],
            output_path=os.path.join(self.output_dir, "reports", "chunk_proposal.json"),
            current_chunk_size=chunk_size,
            current_overlap=overlap
        )
        
        results['chunk_proposal_path'] = os.path.join(self.output_dir, "reports", "chunk_proposal.json")
        
        # ===================================================================
        # STEP 4: Build Optimized Index
        # ===================================================================
        print("\n" + "─"*80)
        print("STEP 4/6: Build Optimized Index")
        print("─"*80)
        
        proposed_config = chunk_proposal['proposed_config']
        new_chunk_size = proposed_config['chunk_size']
        new_overlap = proposed_config['overlap']
        
        print(f"\n🔨 Building optimized index with new chunking:")
        print(f"   Chunk size: {new_chunk_size} words")
        print(f"   Overlap: {new_overlap} words")
        
        chunker = DocumentChunker(self.documents_dir)
        optimized_chunks = chunker.process_all_documents(
            chunk_size=new_chunk_size,
            overlap=new_overlap
        )
        
        optimized_texts = [chunk['text'] for chunk in optimized_chunks]
        optimized_embeddings = builder.create_embeddings_batch(optimized_texts)
        
        optimized_faiss_index = builder.build_faiss_index(optimized_embeddings)
        optimized_bm25_index = builder.build_bm25_index(optimized_chunks)
        
        builder.save_indexes(
            optimized_faiss_index, optimized_bm25_index, optimized_chunks,
            output_dir=self.index_dir,
            config_name="optimized"
        )
        
        optimized_retriever = HybridRetriever(
            optimized_faiss_index, optimized_bm25_index, optimized_chunks
        )
        optimized_generator = RAGGenerator()
        
        results['optimized_chunks'] = len(optimized_chunks)
        
        # ===================================================================
        # STEP 5: Evaluate Both Configurations (Optional)
        # ===================================================================
        evaluation_report_path = None
        
        if not skip_evaluation:
            print("\n" + "─"*80)
            print("STEP 5/6: Evaluate Baseline vs Optimized")
            print("─"*80)
            
            evaluator = EvaluatorAgent(
                baseline_retriever=baseline_retriever,
                baseline_generator=baseline_generator,
                optimized_retriever=optimized_retriever,
                optimized_generator=optimized_generator
            )
            
            evaluation_report = evaluator.run(
                queries_path=self.test_queries_path,
                baseline_output_path=os.path.join(self.output_dir, "metrics", "baseline_evaluation.json"),
                optimized_output_path=os.path.join(self.output_dir, "metrics", "optimized_evaluation.json"),
                comparison_output_path=os.path.join(self.output_dir, "reports", "evaluation_report.json"),
                retrieval_k=retrieval_config.get('top_k', 5),
                retrieval_method=retrieval_config.get('method', 'hybrid')
            )
            
            evaluation_report_path = os.path.join(self.output_dir, "reports", "evaluation_report.json")
            results['evaluation_report_path'] = evaluation_report_path
        else:
            print("\n⏭️  Skipping evaluation step")
        
        # ===================================================================
        # STEP 6: Generate Final Configuration
        # ===================================================================
        print("\n" + "─"*80)
        print("STEP 6/6: Generate Final Optimized Configuration")
        print("─"*80)
        
        architect = ArchitectAgent()
        
        final_config = architect.run(
            profiling_report_path=results['profiling_report_path'],
            chunk_proposal_path=results['chunk_proposal_path'],
            evaluation_report_path=evaluation_report_path,
            output_yaml_path=os.path.join(self.output_dir, "optimized_config.yaml"),
            output_json_path=os.path.join(self.output_dir, "optimized_config.json")
        )
        
        results['final_config_path'] = os.path.join(self.output_dir, "optimized_config.yaml")
        results['final_config'] = final_config
        results['completed_at'] = datetime.now().isoformat()
        
        # ===================================================================
        # SUMMARY
        # ===================================================================
        print("\n" + "="*80)
        print(" "*25 + "OPTIMIZATION COMPLETE!")
        print("="*80)
        
        print(f"\n📊 Summary:")
        print(f"   Baseline chunks: {results['baseline_chunks']}")
        print(f"   Optimized chunks: {results['optimized_chunks']}")
        print(f"   Change: {results['optimized_chunks'] - results['baseline_chunks']:+d} chunks")
        
        print(f"\n📁 Outputs:")
        print(f"   Reports: {os.path.join(self.output_dir, 'reports/')}")
        print(f"   Metrics: {os.path.join(self.output_dir, 'metrics/')}")
        print(f"   Final config: {results['final_config_path']}")
        
        print(f"\n⏱️  Total time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        return results


if __name__ == "__main__":
    print("=== RAG Optimization Workflow ===")
    print("This module orchestrates the full multi-agent optimization pipeline.")
    print("Use RAGOptimizationWorkflow.run_full_optimization() to execute.")

