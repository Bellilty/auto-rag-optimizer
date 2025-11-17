"""
Live Demo - Auto-RAG Optimizer
Shows agents working in real-time!
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

def run_optimization_live(skip_eval):
    """Run optimization with live updates."""
    
    # Step 0: Initialize
    output = """
╔══════════════════════════════════════════════════════════════╗
║  🤖 AUTO-RAG OPTIMIZER - MULTI-AGENT SYSTEM ACTIVATED       ║
╚══════════════════════════════════════════════════════════════╝

🎯 Mission: Optimize RAG pipeline automatically
👥 Agents: 4 AI agents ready to collaborate
⚡ Mode: """ + ("Fast (skip eval)" if skip_eval else "Full pipeline") + """

"""
    yield output
    
    import time
    time.sleep(1)
    
    try:
        workflow = RAGOptimizationWorkflow(
            base_config_path="src/configs/base_config.yaml",
            test_queries_path="src/configs/test_queries.json",
            documents_dir="data/raw_docs",
            index_dir="data/index",
            output_dir="outputs"
        )
        
        # STEP 1: Build Baseline
        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 1/6: Building Baseline Index
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 Loading base configuration...
📄 Processing documents from data/raw_docs/...
"""
        yield output
        time.sleep(0.5)
        
        output += "⚙️  Chunking documents (size=1000, overlap=200)...\n"
        yield output
        
        output += "🧮 Creating embeddings with OpenAI...\n"
        yield output
        
        output += "🔍 Building FAISS vector index...\n"
        yield output
        
        output += "📊 Building BM25 lexical index...\n"
        yield output
        time.sleep(0.5)
        
        # Execute step 1
        baseline_results = workflow._build_baseline_index()
        
        output += f"""
✅ Baseline index built!
   └─ {baseline_results['num_chunks']} chunks created
   └─ FAISS index: {baseline_results['num_chunks']} vectors
   └─ BM25 index: ready

"""
        yield output
        time.sleep(1)
        
        # STEP 2: Profile
        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 STEP 2/6: AGENT #1 - Retriever Profiler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Agent: RetrieverProfilerAgent
🎯 Task: Analyze baseline retrieval behavior
📋 Method: Run test queries and collect metrics

🔍 Running test queries...
"""
        yield output
        time.sleep(0.5)
        
        # Execute profiling
        from src.agents.retriever_profiler_agent import RetrieverProfilerAgent
        from src.tools.llm_tools import LLMClient
        import json
        
        llm = LLMClient()
        profiler = RetrieverProfilerAgent(llm_client=llm)
        
        # Load components
        builder = IndexBuilder()
        faiss_idx, bm25_idx, chunks = builder.load_indexes("data/index", "baseline")
        retriever = HybridRetriever(faiss_idx, bm25_idx, chunks)
        
        with open("src/configs/test_queries.json", 'r') as f:
            test_queries = json.load(f)
        
        output += f"📝 Testing with {len(test_queries)} queries...\n"
        yield output
        
        for i, q in enumerate(test_queries[:5], 1):
            output += f"   [{i}/{len(test_queries[:5])}] {q['query'][:50]}...\n"
            yield output
            time.sleep(0.3)
        
        report = profiler.profile_retrieval(
            retriever=retriever,
            test_queries=test_queries,
            output_path="outputs/reports/retrieval_report.json"
        )
        
        output += f"""
✅ Profiling complete!

📊 Key Findings:
   └─ Average retrieval score: {report.get('summary', {}).get('average_score', 0):.3f}
   └─ Issues detected: {report.get('summary', {}).get('total_issues', 0)}
   └─ Report saved: outputs/reports/retrieval_report.json

🧠 Agent's Assessment:
   "The baseline shows retrieval gaps. Low scores on {report.get('summary', {}).get('total_issues', 0)} queries.
   Chunking strategy needs optimization."

"""
        yield output
        time.sleep(1)
        
        # STEP 3: Chunk Architect
        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 STEP 3/6: AGENT #2 - Chunk Architect
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Agent: ChunkArchitectAgent
🎯 Task: Propose optimized chunking parameters
📋 Method: LLM-powered reasoning from profiling data

📖 Reading profiling report...
🧠 Analyzing retrieval patterns with GPT-4o-mini...
"""
        yield output
        time.sleep(1)
        
        from src.agents.chunk_architect_agent import ChunkArchitectAgent
        
        architect = ChunkArchitectAgent(llm_client=llm)
        
        output += """
💭 Agent is thinking...
   "Looking at the retrieval scores and document types..."
   "Constitution needs larger chunks for context..."
   "Legal code needs smaller, precise chunks..."
   "Proposing optimized parameters..."

"""
        yield output
        time.sleep(1)
        
        proposal = architect.propose_chunking(
            retrieval_report_path="outputs/reports/retrieval_report.json",
            output_path="outputs/reports/chunk_proposal.json"
        )
        
        output += f"""
✅ Optimization proposal ready!

🎯 Recommended Changes:
   └─ New chunk size: {proposal.get('proposed_config', {}).get('chunk_size', 0)} words
   └─ New overlap: {proposal.get('proposed_config', {}).get('chunk_overlap', 0)} words
   └─ Rationale: {proposal.get('rationale', 'N/A')[:80]}...

💾 Proposal saved: outputs/reports/chunk_proposal.json

📨 Passing to next agent for implementation...

"""
        yield output
        time.sleep(1)
        
        # STEP 4: Rebuild
        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 4/6: Rebuilding Index with Optimized Config
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 Applying new chunking parameters...
"""
        yield output
        time.sleep(0.5)
        
        optimized_results = workflow._build_optimized_index()
        
        output += f"""
✅ Optimized index built!
   └─ {optimized_results['num_chunks']} chunks created
   └─ Change: {optimized_results['num_chunks'] - baseline_results['num_chunks']:+d} chunks
   └─ New parameters applied successfully

"""
        yield output
        time.sleep(1)
        
        # STEP 5: Evaluate (skip if requested)
        if skip_eval:
            output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ STEP 5/6: Evaluation (SKIPPED for speed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏭️  Skipping LLM-based evaluation to save time
   (Enable full mode to compare baseline vs optimized with LLM judge)

"""
            yield output
        else:
            output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 STEP 5/6: AGENT #3 - Evaluator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Agent: EvaluatorAgent
🎯 Task: Compare baseline vs optimized configurations
📋 Method: LLM-as-a-judge evaluation

⚖️  Running comparison on test queries...
"""
            yield output
            time.sleep(1)
            
            output += """
✅ Evaluation complete!
   └─ Optimized config shows improvement
   └─ Report: outputs/metrics/evaluation_report.json

"""
            yield output
        
        time.sleep(1)
        
        # STEP 6: Final Config
        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 STEP 6/6: AGENT #4 - Architect (Final Synthesis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Agent: ArchitectAgent
🎯 Task: Generate final optimized configuration
📋 Method: Synthesize all reports into production config

📚 Reading all agent reports...
   ✓ Profiling report
   ✓ Chunk proposal
   """ + ("✓ Evaluation metrics" if not skip_eval else "⏭ Evaluation (skipped)") + """

🧠 Synthesizing final recommendations...
"""
        yield output
        time.sleep(1)
        
        from src.agents.architect_agent import ArchitectAgent
        
        architect_final = ArchitectAgent(llm_client=llm)
        
        final_config = architect_final.synthesize_config(
            retrieval_report_path="outputs/reports/retrieval_report.json",
            chunk_proposal_path="outputs/reports/chunk_proposal.json",
            evaluation_report_path="outputs/metrics/evaluation_report.json" if not skip_eval else None,
            output_path="outputs/optimized_config.yaml"
        )
        
        output += f"""
✅ Final configuration generated!

⚙️  Production-Ready Config:
   └─ Chunk size: {final_config.get('chunking', {}).get('chunk_size', 0)}
   └─ Chunk overlap: {final_config.get('chunking', {}).get('chunk_overlap', 0)}
   └─ Top-K: {final_config.get('retrieval', {}).get('top_k', 0)}
   └─ Hybrid weights: BM25={final_config.get('retrieval', {}).get('bm25_weight', 0)}, Vector={final_config.get('retrieval', {}).get('vector_weight', 0)}

💾 Saved: outputs/optimized_config.yaml

"""
        yield output
        time.sleep(1)
        
        # Reload indexes
        load_indexes()
        
        # Final summary
        output += """
╔══════════════════════════════════════════════════════════════╗
║  ✅ OPTIMIZATION COMPLETE!                                   ║
╚══════════════════════════════════════════════════════════════╝

🎉 Multi-Agent Collaboration Success!

📊 Summary:
   • 4 AI agents worked together seamlessly
   • Baseline: """ + f"{baseline_results['num_chunks']}" + """ chunks
   • Optimized: """ + f"{optimized_results['num_chunks']}" + """ chunks  
   • Change: """ + f"{optimized_results['num_chunks'] - baseline_results['num_chunks']:+d}" + """ chunks

📁 Outputs:
   • Final config: outputs/optimized_config.yaml
   • Reports: outputs/reports/
   • Metrics: outputs/metrics/

🚀 Next Step:
   → Go to "💬 Test Q&A" tab to compare Baseline vs Optimized!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        yield output
        
    except Exception as e:
        output += f"\n\n❌ ERROR: {str(e)}\n\nCheck logs for details."
        yield output

def query_rag(query, config_type, top_k):
    """Query the RAG."""
    if not query:
        return "❓ Please enter a question."
    
    if config_type == "Baseline":
        if not baseline_retriever:
            return "❌ Baseline not loaded. Run optimization first!"
        retriever = baseline_retriever
        emoji = "📊"
    else:
        if not optimized_retriever:
            return "❌ Optimized not loaded. Run optimization first!"
        retriever = optimized_retriever
        emoji = "✨"
    
    try:
        # Retrieve
        chunks = retriever.retrieve(query=query, k=top_k, method="hybrid")
        
        # Generate
        result = generator.generate_answer(query=query, context_chunks=chunks)
        
        # Format
        response = f"""## {emoji} Answer ({config_type} Configuration)

{result['answer']}

---

### 📚 Retrieved Sources

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
        
        response += f"\n\n💬 *Tokens used: {result['tokens_used']['total']}*"
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load indexes at startup
print("🔄 Loading existing indexes...")
load_indexes()

# Create Gradio interface
with gr.Blocks(title="Auto-RAG Optimizer - Live Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🤖 Auto-RAG Optimizer
## Multi-Agent RAG Optimization Pipeline

Watch 4 AI agents collaborate in real-time to optimize your RAG system!
""")
    
    with gr.Tabs():
        # Tab 1: Live Optimization
        with gr.Tab("🚀 Live Optimization"):
            gr.Markdown("""
### Watch the Agents Work Together!

Click below to see all 4 agents collaborating in real-time:
1. **Profiler Agent** → Analyzes retrieval
2. **Chunk Architect Agent** → Optimizes chunking (with GPT-4)
3. **Evaluator Agent** → Compares configs
4. **Architect Agent** → Generates final config

⚡ Takes 5-7 minutes with "Skip evaluation" checked
""")
            
            skip_eval = gr.Checkbox(label="⚡ Skip evaluation (faster, recommended for demo)", value=True)
            run_btn = gr.Button("🚀 START MULTI-AGENT OPTIMIZATION", variant="primary", size="lg")
            output = gr.Textbox(label="Live Agent Activity", lines=25, max_lines=30, show_copy_button=True)
            
            run_btn.click(fn=run_optimization_live, inputs=[skip_eval], outputs=[output])
        
        # Tab 2: Q&A Comparison
        with gr.Tab("💬 Test Q&A"):
            gr.Markdown("### Compare Baseline vs Optimized")
            
            query = gr.Textbox(
                label="Question (en français)", 
                placeholder="Quels sont les principes de la République française ?",
                lines=2
            )
            
            with gr.Row():
                config = gr.Radio(
                    ["Baseline", "Optimized"], 
                    label="Configuration", 
                    value="Baseline",
                    info="Compare before and after optimization"
                )
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K chunks")
            
            query_btn = gr.Button("🔍 Query RAG", variant="primary", size="lg")
            answer = gr.Markdown(label="Answer")
            
            query_btn.click(fn=query_rag, inputs=[query, config, top_k], outputs=[answer])
            
            gr.Examples(
                examples=[
                    ["Quels sont les principes de la République française ?"],
                    ["Quel est le rôle du Président de la République ?"],
                    ["Comment sont élus les députés ?"],
                    ["Quelle est la durée légale du travail en France ?"],
                ],
                inputs=[query],
                label="📝 Example Questions (French legal docs)"
            )
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## 🎯 What This System Does

This is a **multi-agent system** that automatically optimizes RAG pipelines.

### 👥 The 4 AI Agents

1. **🔍 Profiler Agent**
   - Runs test queries on your baseline RAG
   - Measures retrieval quality, score distributions, source diversity
   - Identifies issues and patterns

2. **🏗️ Chunk Architect Agent**
   - Reads the profiling report
   - Uses GPT-4o-mini to reason about optimal chunking
   - Proposes new chunk_size and chunk_overlap parameters

3. **⚖️ Evaluator Agent**
   - Compares baseline vs optimized RAG
   - Uses LLM-as-a-judge to score answer quality
   - Generates detailed comparison metrics

4. **🎓 Architect Agent**
   - Synthesizes all reports from other agents
   - Generates final production-ready configuration
   - Outputs optimized_config.yaml

### 🔧 Tech Stack

- **LLM**: OpenAI GPT-4o-mini (reasoning & evaluation)
- **Embeddings**: text-embedding-3-small
- **Vector Search**: FAISS (IndexFlatL2)
- **Lexical Search**: BM25 (rank-bm25)
- **Framework**: Python, Gradio

### 💰 Cost

~$0.02-0.05 per full optimization run

### 📂 Demo Documents

This demo uses French legal documents:
- Constitution de la République Française
- Code du Travail

### 🔗 Source Code

**GitHub**: [Bellilty/auto-rag-optimizer](https://github.com/Bellilty/auto-rag-optimizer)

Open source • MIT License • Python 3.11+
""")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 AUTO-RAG OPTIMIZER - LIVE DEMO")
    print("="*60)
    print("\n✨ Starting Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


