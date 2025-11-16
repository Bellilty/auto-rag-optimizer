"""
Auto-RAG Optimizer - Interactive Demo
======================================

A Gradio interface for demonstrating the RAG optimization pipeline.
"""

import gradio as gr
import os
import sys
import json
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.orchestrator.workflow import RAGOptimizationWorkflow
from src.components.index_builder import IndexBuilder
from src.components.retriever import HybridRetriever, RAGGenerator

class RAGOptimizerDemo:
    """Demo application for Auto-RAG Optimizer."""
    
    def __init__(self):
        self.workflow = None
        self.baseline_retriever = None
        self.optimized_retriever = None
        self.baseline_generator = None
        self.optimized_generator = None
        
    def check_setup(self):
        """Check if system is properly configured."""
        issues = []
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("⚠️ OPENAI_API_KEY not set in environment")
        
        # Check documents
        docs_dir = "data/raw_docs"
        if os.path.exists(docs_dir):
            docs = [f for f in os.listdir(docs_dir) if f.endswith(('.pdf', '.txt'))]
            if not docs:
                issues.append("⚠️ No documents found in data/raw_docs/")
        else:
            issues.append("❌ data/raw_docs/ directory not found")
        
        if issues:
            return "❌ Setup Issues:\n" + "\n".join(issues)
        
        return "✅ System ready!"
    
    def run_optimization(self, skip_evaluation, progress=gr.Progress()):
        """Run the full optimization pipeline."""
        try:
            progress(0, desc="Initializing...")
            
            # Initialize workflow
            self.workflow = RAGOptimizationWorkflow(
                base_config_path="src/configs/base_config.yaml",
                test_queries_path="src/configs/test_queries.json",
                documents_dir="data/raw_docs",
                index_dir="data/index",
                output_dir="outputs"
            )
            
            progress(0.1, desc="Running optimization...")
            
            # Run optimization
            results = self.workflow.run_full_optimization(
                skip_baseline_indexing=False,
                skip_evaluation=skip_evaluation
            )
            
            progress(1.0, desc="Complete!")
            
            # Format results
            summary = f"""
## ✅ Optimization Complete!

### Results Summary
- **Baseline chunks**: {results.get('baseline_chunks', 'N/A')}
- **Optimized chunks**: {results.get('optimized_chunks', 'N/A')}
- **Change**: {results.get('optimized_chunks', 0) - results.get('baseline_chunks', 0):+d} chunks

### Output Files
- 📊 Reports: `outputs/reports/`
- 📈 Metrics: `outputs/metrics/`
- ⚙️ Final config: `{results.get('final_config_path', 'N/A')}`

### Next Steps
1. Review the optimized configuration
2. Check profiling and evaluation reports
3. Use the optimized parameters in your RAG system
"""
            
            # Load indexes for Q&A
            self._load_indexes()
            
            return summary
            
        except Exception as e:
            return f"❌ Error during optimization:\n{str(e)}"
    
    def _load_indexes(self):
        """Load baseline and optimized indexes for Q&A."""
        try:
            builder = IndexBuilder()
            
            # Load baseline
            if builder.indexes_exist(input_dir="data/index", config_name="baseline"):
                baseline_faiss, baseline_bm25, baseline_chunks = builder.load_indexes(
                    input_dir="data/index", config_name="baseline"
                )
                self.baseline_retriever = HybridRetriever(
                    baseline_faiss, baseline_bm25, baseline_chunks
                )
                self.baseline_generator = RAGGenerator()
            
            # Load optimized
            if builder.indexes_exist(input_dir="data/index", config_name="optimized"):
                opt_faiss, opt_bm25, opt_chunks = builder.load_indexes(
                    input_dir="data/index", config_name="optimized"
                )
                self.optimized_retriever = HybridRetriever(
                    opt_faiss, opt_bm25, opt_chunks
                )
                self.optimized_generator = RAGGenerator()
                
        except Exception as e:
            print(f"Warning: Could not load indexes: {e}")
    
    def query_rag(self, query, config_type, top_k):
        """Query the RAG system."""
        if not query:
            return "Please enter a query."
        
        try:
            # Select retriever and generator
            if config_type == "Baseline":
                if not self.baseline_retriever:
                    return "❌ Baseline index not loaded. Run optimization first."
                retriever = self.baseline_retriever
                generator = self.baseline_generator
            else:
                if not self.optimized_retriever:
                    return "❌ Optimized index not loaded. Run optimization first."
                retriever = self.optimized_retriever
                generator = self.optimized_generator
            
            # Retrieve chunks
            chunks = retriever.retrieve(
                query=query,
                k=top_k,
                method="hybrid"
            )
            
            # Generate answer
            result = generator.generate_answer(
                query=query,
                context_chunks=chunks
            )
            
            # Format response
            response = f"""
### Answer ({config_type} Configuration)

{result['answer']}

---

### Retrieved Sources
"""
            for i, chunk in enumerate(chunks[:3], 1):
                response += f"\n**{i}. {chunk.get('source', 'unknown')}** (score: {chunk.get('retrieval_score', 0):.3f})\n"
                response += f"```\n{chunk['text'][:200]}...\n```\n"
            
            response += f"\n---\n*Tokens used: {result['tokens_used']['total']}*"
            
            return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_reports(self):
        """Load and display optimization reports."""
        reports = {}
        
        # Profiling report
        prof_path = "outputs/reports/retrieval_report.json"
        if os.path.exists(prof_path):
            with open(prof_path) as f:
                reports['profiling'] = json.load(f)
        
        # Chunk proposal
        chunk_path = "outputs/reports/chunk_proposal.json"
        if os.path.exists(chunk_path):
            with open(chunk_path) as f:
                reports['chunking'] = json.load(f)
        
        # Evaluation
        eval_path = "outputs/reports/evaluation_report.json"
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                reports['evaluation'] = json.load(f)
        
        # Final config
        config_path = "outputs/optimized_config.yaml"
        if os.path.exists(config_path):
            with open(config_path) as f:
                reports['final_config'] = yaml.safe_load(f)
        
        if not reports:
            return "No reports found. Run optimization first."
        
        # Format reports
        output = "# Optimization Reports\n\n"
        
        if 'profiling' in reports:
            summary = reports['profiling'].get('summary', {})
            output += f"""
## 📊 Retrieval Profiling

- **Average retrieval score**: {summary.get('avg_retrieval_score', 0):.3f}
- **Score std dev**: {summary.get('score_std', 0):.3f}
- **Average unique sources**: {summary.get('avg_unique_sources', 0):.2f}
- **Issues detected**: {summary.get('total_issues_detected', 0)}

"""
        
        if 'chunking' in reports:
            proposed = reports['chunking'].get('proposed_config', {})
            current = reports['chunking'].get('current_config', {})
            output += f"""
## 🔨 Chunking Proposal

**Current**: {current.get('chunk_size')} words, {current.get('overlap')} overlap
**Proposed**: {proposed.get('chunk_size')} words, {proposed.get('overlap')} overlap

**Reasoning**: {reports['chunking'].get('llm_recommendation', {}).get('reasoning', 'N/A')[:200]}...

"""
        
        if 'evaluation' in reports:
            improvement = reports['evaluation'].get('improvement', {})
            output += f"""
## 📈 Evaluation Results

- **Baseline avg score**: {improvement.get('baseline_avg', 0):.2f}/10
- **Optimized avg score**: {improvement.get('optimized_avg', 0):.2f}/10
- **Improvement**: {improvement.get('relative_improvement_pct', 0):+.1f}%

**Recommendation**: {reports['evaluation'].get('recommendation', 'N/A')}

"""
        
        if 'final_config' in reports:
            config = reports['final_config'].get('configuration', {})
            chunking = config.get('chunking', {})
            retrieval = config.get('retrieval', {})
            output += f"""
## ⚙️ Final Configuration

### Chunking
- Size: {chunking.get('chunk_size')} words
- Overlap: {chunking.get('overlap')} words

### Retrieval
- Method: {retrieval.get('method')}
- Top-k: {retrieval.get('top_k')}
- Vector weight: {retrieval.get('vector_weight', 0):.2f}
- BM25 weight: {retrieval.get('bm25_weight', 0):.2f}

"""
        
        return output


def create_demo():
    """Create the Gradio demo interface."""
    demo_app = RAGOptimizerDemo()
    
    with gr.Blocks(title="Auto-RAG Optimizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🤖 Auto-RAG Optimizer
## Multi-Agent RAG Evaluation and Architecture Refinement

An automated pipeline for profiling, evaluating, and optimizing Retrieval-Augmented Generation (RAG) systems.

### How It Works
1. **Profile** retrieval behavior on test queries
2. **Analyze** performance with LLM reasoning
3. **Propose** optimized chunking parameters
4. **Evaluate** baseline vs optimized configurations
5. **Generate** production-ready configuration
""")
        
        with gr.Tabs():
            # Tab 1: Setup & Optimization
            with gr.Tab("🚀 Optimization"):
                gr.Markdown("### Run RAG Optimization Pipeline")
                
                setup_status = gr.Textbox(
                    label="System Status",
                    value=demo_app.check_setup(),
                    interactive=False
                )
                
                gr.Markdown("### Configuration")
                skip_eval = gr.Checkbox(
                    label="Skip evaluation (faster, ~5-7 min instead of ~10-15 min)",
                    value=True
                )
                
                run_btn = gr.Button("🚀 Run Optimization", variant="primary", size="lg")
                
                optimization_output = gr.Markdown(label="Results")
                
                run_btn.click(
                    fn=demo_app.run_optimization,
                    inputs=[skip_eval],
                    outputs=[optimization_output]
                )
            
            # Tab 2: Q&A Testing
            with gr.Tab("💬 Test Q&A"):
                gr.Markdown("### Test Baseline vs Optimized RAG")
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Enter your question",
                        placeholder="What are the main principles of data protection?",
                        lines=2
                    )
                
                with gr.Row():
                    config_choice = gr.Radio(
                        choices=["Baseline", "Optimized"],
                        label="Configuration",
                        value="Baseline"
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top-K chunks"
                    )
                
                query_btn = gr.Button("🔍 Query RAG", variant="primary")
                answer_output = gr.Markdown(label="Answer")
                
                query_btn.click(
                    fn=demo_app.query_rag,
                    inputs=[query_input, config_choice, top_k_slider],
                    outputs=[answer_output]
                )
                
                # Example queries
                gr.Markdown("### Example Queries")
                gr.Examples(
                    examples=[
                        ["What are the main principles of data protection?"],
                        ["What rights do data subjects have?"],
                        ["What is the role of a data controller?"],
                        ["What are the penalties for non-compliance?"],
                    ],
                    inputs=[query_input]
                )
            
            # Tab 3: Reports
            with gr.Tab("📊 Reports"):
                gr.Markdown("### View Optimization Reports")
                
                load_reports_btn = gr.Button("📂 Load Reports", variant="secondary")
                reports_output = gr.Markdown(label="Reports")
                
                load_reports_btn.click(
                    fn=demo_app.load_reports,
                    outputs=[reports_output]
                )
            
            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
### About Auto-RAG Optimizer

This system uses **4 specialized AI agents** to optimize RAG pipelines:

1. **RetrieverProfilerAgent**: Profiles retrieval behavior and collects metrics
2. **ChunkArchitectAgent**: Uses LLM reasoning to propose optimal chunking
3. **EvaluatorAgent**: Evaluates configurations with LLM-based judging
4. **ArchitectAgent**: Synthesizes reports into final configuration

### Technologies

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: text-embedding-3-small
- **Vector Search**: FAISS
- **Lexical Search**: BM25
- **Framework**: Python 3.11+

### Cost Estimate

Typical optimization run (10 queries, 3 documents):
- Embeddings: ~$0.002
- Agent reasoning: ~$0.005
- Evaluation: ~$0.01
- **Total: ~$0.02**

### Links

- **GitHub**: [Bellilty/auto-rag-optimizer](https://github.com/Bellilty/auto-rag-optimizer)
- **Documentation**: See README.md
- **Quick Start**: See QUICKSTART.md

### Architecture

```
Baseline Index → Profiling → Chunking Proposal → 
Optimized Index → Evaluation → Final Configuration
```

Each step is driven by specialized agents that use LLM reasoning
to make informed optimization decisions.
""")
        
        gr.Markdown("""
---
**Auto-RAG Optimizer** | Multi-Agent RAG Optimization Pipeline | MIT License
""")
    
    return demo


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting Auto-RAG Optimizer Demo...")
    print("="*80 + "\n")
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


