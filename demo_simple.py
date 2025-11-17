"""
Simple Demo - Auto-RAG Optimizer
"""

import gradio as gr
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from src.components.index_builder import IndexBuilder
from src.components.retriever import HybridRetriever, RAGGenerator
from src.orchestrator.workflow import RAGOptimizationWorkflow

# Global variables
baseline_retriever = None
optimized_retriever = None
generator = None

def load_indexes():
    """Load indexes at startup."""
    global baseline_retriever, optimized_retriever, generator
    
    builder = IndexBuilder()
    
    # Load baseline
    if builder.indexes_exist(input_dir="data/index", config_name="baseline"):
        faiss_idx, bm25_idx, chunks = builder.load_indexes(input_dir="data/index", config_name="baseline")
        baseline_retriever = HybridRetriever(faiss_idx, bm25_idx, chunks)
        print("✅ Baseline loaded")
    
    # Load optimized
    if builder.indexes_exist(input_dir="data/index", config_name="optimized"):
        faiss_idx, bm25_idx, chunks = builder.load_indexes(input_dir="data/index", config_name="optimized")
        optimized_retriever = HybridRetriever(faiss_idx, bm25_idx, chunks)
        print("✅ Optimized loaded")
    
    generator = RAGGenerator()

def run_optimization(skip_eval):
    """Run optimization."""
    try:
        workflow = RAGOptimizationWorkflow(
            base_config_path="src/configs/base_config.yaml",
            test_queries_path="src/configs/test_queries.json",
            documents_dir="data/raw_docs",
            index_dir="data/index",
            output_dir="outputs"
        )
        
        yield "🚀 Starting optimization...\n\n"
        
        results = workflow.run_full_optimization(
            skip_baseline_indexing=False,
            skip_evaluation=skip_eval
        )
        
        # Reload indexes
        load_indexes()
        
        summary = f"""
✅ Optimization Complete!

📊 Results:
- Baseline chunks: {results.get('baseline_chunks', 'N/A')}
- Optimized chunks: {results.get('optimized_chunks', 'N/A')}
- Change: {results.get('optimized_chunks', 0) - results.get('baseline_chunks', 0):+d} chunks

📁 Outputs:
- Final config: {results.get('final_config_path', 'N/A')}
- Reports: outputs/reports/
- Metrics: outputs/metrics/

✅ Now you can test Q&A in the other tab!
"""
        yield summary
        
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def query_rag(query, config_type, top_k):
    """Query the RAG."""
    if not query:
        return "Please enter a question."
    
    if config_type == "Baseline":
        if not baseline_retriever:
            return "❌ Baseline not loaded. Run optimization first or restart demo."
        retriever = baseline_retriever
    else:
        if not optimized_retriever:
            return "❌ Optimized not loaded. Run optimization first."
        retriever = optimized_retriever
    
    try:
        # Retrieve
        chunks = retriever.retrieve(query=query, k=top_k, method="hybrid")
        
        # Generate
        result = generator.generate_answer(query=query, context_chunks=chunks)
        
        # Format
        response = f"""### Answer ({config_type})

{result['answer']}

---

### Retrieved Sources
"""
        for i, chunk in enumerate(chunks[:3], 1):
            response += f"\n**{i}. {chunk.get('source', 'unknown')}** (score: {chunk.get('retrieval_score', 0):.3f})\n"
            response += f"```\n{chunk['text'][:200]}...\n```\n"
        
        response += f"\n*Tokens: {result['tokens_used']['total']}*"
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load indexes at startup
print("Loading indexes...")
load_indexes()

# Create Gradio interface
with gr.Blocks(title="Auto-RAG Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🤖 Auto-RAG Optimizer
## Multi-Agent RAG Optimization Pipeline

Automatically optimize your RAG system with 4 AI agents.
""")
    
    with gr.Tabs():
        # Tab 1: Optimization
        with gr.Tab("🚀 Run Optimization"):
            gr.Markdown("### Run the full optimization pipeline")
            
            skip_eval = gr.Checkbox(label="Skip evaluation (faster: 5-7 min)", value=True)
            run_btn = gr.Button("🚀 Start Optimization", variant="primary", size="lg")
            output = gr.Textbox(label="Progress", lines=15)
            
            run_btn.click(fn=run_optimization, inputs=[skip_eval], outputs=[output])
        
        # Tab 2: Q&A
        with gr.Tab("💬 Test Q&A"):
            gr.Markdown("### Compare Baseline vs Optimized")
            
            query = gr.Textbox(label="Question", placeholder="Quels sont les principes de la République française ?", lines=2)
            
            with gr.Row():
                config = gr.Radio(["Baseline", "Optimized"], label="Configuration", value="Baseline")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K")
            
            query_btn = gr.Button("🔍 Query", variant="primary")
            answer = gr.Markdown(label="Answer")
            
            query_btn.click(fn=query_rag, inputs=[query, config, top_k], outputs=[answer])
            
            gr.Examples(
                examples=[
                    ["Quels sont les principes de la République française ?"],
                    ["Quel est le rôle du Président de la République ?"],
                    ["Quelle est la durée légale du travail ?"],
                ],
                inputs=[query]
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
### Multi-Agent Architecture

4 AI agents working together:

1. **Profiler Agent** - Analyzes retrieval behavior
2. **Architect Agent** - Optimizes chunking with LLM reasoning
3. **Evaluator Agent** - Compares configurations
4. **Synthesizer Agent** - Generates final config

### Tech Stack

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: text-embedding-3-small
- **Vector Search**: FAISS
- **Lexical Search**: BM25
- **Cost**: ~$0.02 per optimization run

### GitHub

🔗 [Bellilty/auto-rag-optimizer](https://github.com/Bellilty/auto-rag-optimizer)

Open source • Python • MIT License
""")

if __name__ == "__main__":
    print("\n🚀 Starting Auto-RAG Optimizer Demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)




