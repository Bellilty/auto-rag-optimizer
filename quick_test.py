"""
Quick Test Script - Fast RAG Optimization Test
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.orchestrator.workflow import RAGOptimizationWorkflow

def main():
    print("\n" + "="*80)
    print("QUICK TEST - Auto-RAG Optimizer")
    print("="*80)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set!")
        print("The .env file will be loaded automatically by the modules.")
        
    # Initialize workflow
    workflow = RAGOptimizationWorkflow(
        base_config_path="src/configs/base_config.yaml",
        test_queries_path="src/configs/test_queries.json",
        documents_dir="data/raw_docs",
        index_dir="data/index",
        output_dir="outputs"
    )
    
    print("\n🚀 Starting quick test (evaluation skipped for speed)...\n")
    
    try:
        # Run with skip_evaluation=True for faster testing
        results = workflow.run_full_optimization(
            skip_baseline_indexing=False,
            skip_evaluation=True  # Skip evaluation for quick test
        )
        
        print("\n✅ Test completed successfully!")
        print(f"\n📊 Quick Results:")
        print(f"   - Baseline chunks: {results.get('baseline_chunks', 'N/A')}")
        print(f"   - Optimized chunks: {results.get('optimized_chunks', 'N/A')}")
        print(f"   - Final config: {results.get('final_config_path', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

