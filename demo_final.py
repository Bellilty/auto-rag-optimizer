"""
Final Demo - Auto-RAG Optimizer
Shows real workflow execution with live output!
"""

import gradio as gr
import os
import sys
import io
import contextlib
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

class OutputCapture:
    """Capture stdout in real-time."""
    def __init__(self):
        self.output = ""
    
    def write(self, text):
        self.output += text
        return len(text)
    
    def flush(self):
        pass

def run_optimization_with_capture(skip_eval):
    """Run optimization and capture output in real-time."""
    
    header = """
╔══════════════════════════════════════════════════════════════╗
║  🤖 AUTO-RAG OPTIMIZER - MULTI-AGENT SYSTEM ACTIVATED       ║
╚══════════════════════════════════════════════════════════════╝

🎯 Mission: Optimize RAG pipeline automatically
👥 Agents: 4 AI agents ready to collaborate
⚡ Mode: """ + ("Fast (skip eval)" if skip_eval else "Full pipeline") + """

"""
    yield header
    
    try:
        workflow = RAGOptimizationWorkflow(
            base_config_path="src/configs/base_config.yaml",
            test_queries_path="src/configs/test_queries.json",
            documents_dir="data/raw_docs",
            index_dir="data/index",
            output_dir="outputs"
        )
        
        # Capture output
        captured = OutputCapture()
        
        # Redirect stdout
        original_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            # Run the actual workflow
            results = workflow.run_full_optimization(
                skip_baseline_indexing=False,
                skip_evaluation=skip_eval
            )
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
        
        # Add emojis and formatting to make it prettier
        output = captured.output
        
        # Add some visual enhancements
        output = output.replace("STEP 1/6", "🔨 STEP 1/6")
        output = output.replace("STEP 2/6", "🤖 STEP 2/6 - AGENT #1: Profiler")
        output = output.replace("STEP 3/6", "🧠 STEP 3/6 - AGENT #2: Chunk Architect")
        output = output.replace("STEP 4/6", "⚙️  STEP 4/6")
        output = output.replace("STEP 5/6", "⚖️  STEP 5/6 - AGENT #3: Evaluator")
        output = output.replace("STEP 6/6", "🎯 STEP 6/6 - AGENT #4: Final Architect")
        
        full_output = header + "\n" + output + f"""

╔══════════════════════════════════════════════════════════════╗
║  ✅ OPTIMIZATION COMPLETE!                                   ║
╚══════════════════════════════════════════════════════════════╝

📊 Results:
   • Baseline: {results.get('baseline_chunks', 'N/A')} chunks
   • Optimized: {results.get('optimized_chunks', 'N/A')} chunks
   • Change: {results.get('optimized_chunks', 0) - results.get('baseline_chunks', 0):+d} chunks

📁 Outputs:
   • Final config: {results.get('final_config_path', 'N/A')}
   • Reports: outputs/reports/
   • Metrics: outputs/metrics/

🚀 Next Step: Go to "💬 Test Q&A" tab to compare results!

"""
        yield full_output
        
        # Reload indexes
        load_indexes()
        
    except Exception as e:
        import traceback
        error_output = header + f"\n\n❌ ERROR: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        yield error_output

def query_rag(query, config_type, top_k):
    """Query the RAG."""
    if not query:
        return "❓ Please enter a question."
    
    if config_type == "📊 Baseline":
        if not baseline_retriever:
            return "❌ Baseline not loaded. Run optimization first!"
        retriever = baseline_retriever
        emoji = "📊"
        label = "Baseline"
    else:
        if not optimized_retriever:
            return "❌ Optimized not loaded. Run optimization first!"
        retriever = optimized_retriever
        emoji = "✨"
        label = "Optimized"
    
    try:
        # Retrieve
        chunks = retriever.retrieve(query=query, k=top_k, method="hybrid")
        
        # Generate
        result = generator.generate_answer(query=query, context_chunks=chunks)
        
        # Format
        response = f"""## {emoji} {label} Configuration

### Answer:
{result['answer']}

---

### 📚 Sources Retrieved (Top 3):

"""
        for i, chunk in enumerate(chunks[:3], 1):
            score = chunk.get('retrieval_score', 0)
            score_emoji = "🔥" if score > 0.7 else "✅" if score > 0.5 else "⚠️"
            response += f"""
**{i}. {chunk.get('source', 'unknown')}** {score_emoji} Score: {score:.3f}

```
{chunk['text'][:250]}...
```
"""
        
        response += f"\n\n💬 *Tokens used: {result['tokens_used']['total']} | Cost: ~${result['tokens_used']['total'] * 0.000002:.4f}*"
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load indexes at startup
print("🔄 Loading existing indexes...")
load_indexes()

# Create Gradio interface
with gr.Blocks(title="Auto-RAG Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🤖 Auto-RAG Optimizer
## Multi-Agent RAG Optimization Pipeline

Automatically optimize your RAG system with 4 collaborating AI agents!
""")
    
    with gr.Tabs():
        # Tab 1: Optimization
        with gr.Tab("🚀 Run Optimization"):
            gr.Markdown("""
### Watch 4 AI Agents Work Together!

The agents will:
1. **🔍 Profiler Agent** → Profile baseline retrieval behavior
2. **🧠 Chunk Architect Agent** → Propose optimized chunking (with GPT-4)
3. **⚖️ Evaluator Agent** → Compare baseline vs optimized (optional)
4. **🎯 Final Architect Agent** → Generate production config

⏱️ Takes ~5-7 minutes with "Skip evaluation" enabled
""")
            
            skip_eval = gr.Checkbox(
                label="⚡ Skip evaluation (recommended for demo - saves time)", 
                value=True
            )
            run_btn = gr.Button("🚀 START OPTIMIZATION", variant="primary", size="lg")
            output = gr.Textbox(
                label="Live Agent Activity", 
                lines=30, 
                max_lines=35, 
                show_copy_button=True
            )
            
            run_btn.click(
                fn=run_optimization_with_capture, 
                inputs=[skip_eval], 
                outputs=[output]
            )
        
        # Tab 2: Q&A
        with gr.Tab("💬 Test Q&A"):
            gr.Markdown("### Compare Baseline vs Optimized RAG")
            
            query = gr.Textbox(
                label="Your Question (in French)", 
                placeholder="Quels sont les principes de la République française ?",
                lines=2
            )
            
            with gr.Row():
                config = gr.Radio(
                    ["📊 Baseline", "✨ Optimized"], 
                    label="Configuration", 
                    value="📊 Baseline",
                    info="Switch between before/after optimization"
                )
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K chunks to retrieve")
            
            query_btn = gr.Button("🔍 Query RAG", variant="primary", size="lg")
            answer = gr.Markdown(label="Answer & Sources")
            
            query_btn.click(fn=query_rag, inputs=[query, config, top_k], outputs=[answer])
            
            gr.Examples(
                examples=[
                    ["Quels sont les principes de la République française ?"],
                    ["Quel est le rôle du Président de la République ?"],
                    ["Comment sont élus les députés à l'Assemblée nationale ?"],
                    ["Quelle est la durée légale du travail en France ?"],
                    ["Quels sont les droits du salarié en cas de licenciement ?"],
                ],
                inputs=[query],
                label="📝 Example Questions (French Legal Documents)"
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## 🎯 What Is This?

An **automated multi-agent system** that optimizes Retrieval-Augmented Generation (RAG) pipelines.

No manual tuning. Just AI agents working together to improve your RAG.

---

## 👥 The 4 AI Agents

### 1. 🔍 Profiler Agent
- Runs test queries on your baseline RAG
- Measures: retrieval scores, source diversity, score distributions
- Identifies patterns and issues
- **Output**: `retrieval_report.json`

### 2. 🧠 Chunk Architect Agent
- Reads the profiling report
- Uses **GPT-4o-mini** to reason about optimal chunking
- Proposes new `chunk_size` and `chunk_overlap`
- **Output**: `chunk_proposal.json`

### 3. ⚖️ Evaluator Agent (Optional)
- Compares baseline vs optimized RAG
- Uses **LLM-as-a-judge** to score answer quality
- Generates win/loss statistics
- **Output**: `evaluation_report.json`

### 4. 🎯 Final Architect Agent
- Synthesizes all reports from other agents
- Generates complete production-ready configuration
- Includes chunking, retrieval, and hybrid search parameters
- **Output**: `optimized_config.yaml`

---

## 🔧 Tech Stack

- **LLM**: OpenAI GPT-4o-mini (for agent reasoning & evaluation)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Search**: FAISS (IndexFlatL2)
- **Lexical Search**: BM25 (rank-bm25)
- **Framework**: Python, Gradio
- **Documents**: French legal texts (Constitution + Labor Code)

---

## 💰 Cost

~**$0.02-0.05** per optimization run

Breakdown:
- Embeddings: ~$0.01 (one-time per index)
- Agent reasoning: ~$0.01-0.03
- Evaluation (if enabled): ~$0.01-0.02

---

## 📂 Demo Data

This demo uses **French legal documents**:
- 📜 Constitution de la République Française
- ⚖️ Code du Travail (extracts)

You can replace with your own documents in `data/raw_docs/`

---

## 🔗 Open Source

**GitHub**: [Bellilty/auto-rag-optimizer](https://github.com/Bellilty/auto-rag-optimizer)

- Full source code
- MIT License
- Python 3.11+
- Contributions welcome!

---

## 🚀 Try It Yourself

1. Run optimization (5-7 min)
2. Compare baseline vs optimized in Q&A tab
3. Check generated reports in `outputs/`

The agents will show you exactly what they're doing!
""")

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "🤖 AUTO-RAG OPTIMIZER - LIVE DEMO")
    print("="*70)
    print("\n✨ Starting Gradio interface at http://localhost:7860")
    print("📚 Demo uses French legal documents")
    print("👥 4 AI agents ready to optimize your RAG\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


