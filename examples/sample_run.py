"""
Sample Run Script
=================

Example of how to run the complete Auto-RAG Optimizer pipeline.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestrator.workflow import RAGOptimizationWorkflow


def main():
    """
    Run the full RAG optimization workflow.
    """
    print("\n" + "="*80)
    print("AUTO-RAG OPTIMIZER - Sample Run")
    print("="*80)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Initialize workflow
    workflow = RAGOptimizationWorkflow(
        base_config_path="src/configs/base_config.yaml",
        test_queries_path="src/configs/test_queries.json",
        documents_dir="data/raw_docs",
        index_dir="data/index",
        output_dir="outputs"
    )
    
    # Check if documents exist
    docs_exist = os.path.exists("data/raw_docs") and \
                 len([f for f in os.listdir("data/raw_docs") 
                      if f.endswith(('.pdf', '.txt'))]) > 0
    
    if not docs_exist:
        print("\n⚠️  WARNING: No documents found in data/raw_docs/")
        print("\nPlease add PDF or TXT documents to data/raw_docs/ before running.")
        print("\nYou can:")
        print("  1. Copy documents from the rag-juridique project:")
        print("     cp ../rag-juridique/data/pdfs/* data/raw_docs/")
        print("  2. Or add your own PDF/TXT documents to data/raw_docs/")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    # Run optimization
    try:
        # Set skip_evaluation=True for faster testing
        # Set skip_evaluation=False for full evaluation
        results = workflow.run_full_optimization(
            skip_baseline_indexing=False,  # Set to True to reuse existing baseline index
            skip_evaluation=False          # Set to True to skip evaluation (faster)
        )
        
        print("\n✅ Optimization completed successfully!")
        print(f"\n📊 Results summary:")
        print(f"   - Baseline chunks: {results['baseline_chunks']}")
        print(f"   - Optimized chunks: {results['optimized_chunks']}")
        print(f"   - Final config: {results['final_config_path']}")
        
        print(f"\n📁 All reports saved to: {workflow.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Review the optimized configuration: {results['final_config_path']}")
        print(f"  2. Check the reports in: outputs/reports/")
        print(f"  3. Review evaluation metrics in: outputs/metrics/")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

