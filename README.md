# 🤖 Auto-RAG Optimizer

> **Multi-Agent System for Automated RAG Pipeline Optimization**  
> No manual tuning. Just AI agents collaborating to improve your Retrieval-Augmented Generation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![FAISS](https://img.shields.io/badge/vector-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Problem → Solution

**Problem**: RAG pipelines need manual tuning (chunk size, overlap, top-k, hybrid weights...)  
**Solution**: Let AI agents analyze, experiment, and optimize automatically.

```
Traditional Approach:        Auto-RAG Optimizer:
─────────────────           ───────────────────
Manual tuning               4 AI Agents collaborate
Trial & error               Data-driven decisions
Hours of work               5-10 minutes automated
Guesswork                   LLM reasoning + metrics
```

---

## 🏗️ Architecture: 4 Specialized AI Agents

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG OPTIMIZATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

    Documents                    Test Queries
        │                             │
        ▼                             ▼
   ┌──────────────────────────────────────────┐
   │  STEP 1: Build Baseline Index            │
   │  • Chunk documents (default params)      │
   │  • Create embeddings + FAISS index       │
   │  • Build BM25 lexical index              │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  🤖 AGENT #1: Retriever Profiler         │
   │  • Run test queries                      │
   │  • Measure: recall, diversity, scores    │
   │  • Detect issues (low scores, gaps)      │
   │  Output: retrieval_report.json           │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  🧠 AGENT #2: Chunk Architect            │
   │  • Analyze profiling report              │
   │  • Use GPT-4o-mini to reason             │
   │  • Propose optimal chunk_size + overlap  │
   │  Output: chunk_proposal.json             │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  STEP 4: Rebuild Index (Optimized)       │
   │  • Re-chunk with new parameters          │
   │  • Rebuild FAISS + BM25 indexes          │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  ⚖️ AGENT #3: Evaluator (Optional)       │
   │  • Compare baseline vs optimized         │
   │  • LLM-as-Judge: score answers           │
   │  • Win/Loss statistics                   │
   │  Output: evaluation_report.json          │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  🎯 AGENT #4: Final Architect            │
   │  • Synthesize all reports                │
   │  • Generate production config            │
   │  Output: optimized_config.yaml           │
   └──────────────────────────────────────────┘
```

---

## 📊 Real Results (French Legal Documents)

**Test Case**: 9 queries on French Constitution & Labor Code

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Chunk Size** | 4000 words | 600 words | **-85%** (more granular) |
| **Chunk Overlap** | 0 words | 150 words | **Context preserved** |
| **Total Chunks** | 24 | 220 | **+196** (+817%) |
| **Answer Quality** | 8.89/10 | 9.00/10 | **+1.2%** |
| **Avg Retrieval Score** | 0.448 | 0.496 | **+10.9%** |
| **Top-1 Retrieval** | 0.494 | 0.540 | **+9.2%** |
| **Precision@3** | 63.0% | 85.2% | **+22.2pp** |
| **Excellent Answers (8-10)** | 100% | 100% | Maintained |

### 💡 Key Insights

1. **Smaller chunks = Better precision**: Going from 4000 → 600 words improved retrieval by **10.9%**
2. **Overlap matters**: Adding 150 words overlap preserved context across chunks
3. **Granularity helps**: 24 → 220 chunks means better coverage and diversity
4. **Precision improved dramatically**: Precision@3 went from 63% → 85% (**+22.2pp**)

---

## 🗣️ Real Example: Before vs After Optimization

### Example Question (French)
**"Quels sont les principes fondamentaux de la République française ?"**  
*(What are the fundamental principles of the French Republic?)*

---

### ❌ **BASELINE** (4000 words, 0 overlap)

**Score**: 9/10  
**Retrieval Scores**: [0.583, 0.510, 0.438]  
**Retrieved**: 3 chunks (very large chunks)

**Answer**:
> Les principes fondamentaux de la République française sont énoncés dans le préambule et les articles de la Constitution. Ils incluent :
> 
> 1. **Indivisibilité** : La France est une République indivisible, laïque, démocratique et sociale (Article 1).
> 2. **Égalité** : Elle assure l'égalité devant la loi de tous les citoyens sans distinction d'origine, de race ou de religion (Article 1).
> 3. **Respect des croyances** : La République respecte toutes les croyances (Article 1).
> 4. **Souveraineté nationale** : La souveraineté nationale appartient au peuple qui l'exerce par ses représentants et par la voie du référendum (Article 3).
> 5. **Liberté, Égalité, Fraternité** : La devise de la République est « Liberté, Égalité, Fraternité » (Article 2).

**Issues**:
- ⚠️ Large chunks (4000 words) capture too much irrelevant content
- ⚠️ Low retrieval precision (0.438 for 3rd chunk)
- ⚠️ No overlap = context fragmentation

---

### ✅ **OPTIMIZED** (600 words, 150 overlap)

**Score**: 9/10  
**Retrieval Scores**: [0.661, 0.519, 0.513]  
**Retrieved**: 5 chunks (focused, precise chunks)

**Answer**:
> Les principes fondamentaux de la République française, tels qu'énoncés dans la Constitution, incluent :
>
> 1. **Indivisibilité** : La France est une République indivisible, laïque, démocratique et sociale (Article 1).
> 2. **Égalité** : Elle assure l'égalité devant la loi de tous les citoyens sans distinction d'origine, de race ou de religion (Article 1).
> 3. **Souveraineté nationale** : La souveraineté nationale appartient au peuple, qui l'exerce par ses représentants et par la voie du référendum (Article 3).
> 4. **Liberté** : La devise de la République est « Liberté, Égalité, Fraternité » (Article 2).
> 5. **Respect des croyances** : La République respecte toutes les croyances.
> 6. **Démocratie** : Le principe est un gouvernement du peuple, par le peuple et pour le peuple (Article 2).
>
> Ces principes sont également renforcés par l'attachement aux Droits de l'homme et aux droits et devoirs définis dans la Charte de l'environnement de 2004 (Préambule de la Constitution).

**Improvements**:
- ✅ **Higher retrieval scores** (0.661 vs 0.583 for top-1)
- ✅ **Better precision** across all retrieved chunks
- ✅ **More comprehensive answer** with additional context (point 6 + Préambule)
- ✅ **Better source citations** thanks to overlap preserving context

---

## 📈 Why Did It Improve?

### Problem with Large Chunks (4000 words)
```
[────────────────────────────────────────────────────────────────]
│  Introduction │ Relevant Info │ Irrelevant Content │ More Text │
└────────────────────────────────────────────────────────────────┘
                      ↑
                Only this part is relevant
                but entire chunk is scored
```

### Solution with Optimized Chunks (600 words + overlap)
```
[─────────────]  [─────────────]  [─────────────]
│  Relevant 1 │  │  Relevant 2 │  │  Relevant 3 │
└──overlap──┘    └──overlap──┘    └──overlap──┘
      ↑              ↑               ↑
   Each chunk focused on one topic
   Overlap preserves context
   Better retrieval precision
```

**Key Improvements**:
1. **Smaller chunks** = each chunk focuses on ONE topic → higher semantic similarity
2. **Overlap** = context flows between chunks → no information loss
3. **More chunks** = better coverage of document → higher recall
4. **Higher precision** = less irrelevant content → better answer quality

---

## 🛠️ Tech Stack

```
┌─────────────────┬──────────────────────────────────────────┐
│ Component       │ Technology                               │
├─────────────────┼──────────────────────────────────────────┤
│ Agent LLM       │ OpenAI GPT-4o-mini (reasoning)          │
│ Embeddings      │ OpenAI text-embedding-3-small           │
│ Vector Search   │ FAISS (IndexFlatL2)                     │
│ Lexical Search  │ BM25 (rank-bm25)                        │
│ Orchestration   │ Python 3.11+ (custom multi-agent)       │
│ Evaluation      │ LLM-as-Judge (GPT-4o-mini)              │
│ Storage         │ JSON reports + YAML configs             │
│ Cost            │ ~$0.02-0.05 per optimization run        │
└─────────────────┴──────────────────────────────────────────┘
```

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Clone & Install
```bash
git clone https://github.com/Bellilty/auto-rag-optimizer.git
cd auto-rag-optimizer
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Set OpenAI API Key
```bash
export OPENAI_API_KEY="sk-your-key-here"
# Or create .env file with OPENAI_API_KEY=sk-...
```

### 3️⃣ Run Optimization
```bash
# Add your documents to data/raw_docs/
# Add test queries to src/configs/test_queries.json

python examples/sample_run.py
```

**That's it!** The agents will:
- Profile your baseline RAG
- Propose optimized chunking
- Rebuild indexes
- Evaluate improvements
- Generate production config

**Output**: `outputs/optimized_config.yaml` + detailed reports

---

## 📂 Project Structure

```
auto-rag-optimizer/
├── src/
│   ├── agents/                 # 4 specialized AI agents
│   │   ├── retriever_profiler_agent.py
│   │   ├── chunk_architect_agent.py
│   │   ├── evaluator_agent.py
│   │   └── architect_agent.py
│   ├── orchestrator/
│   │   └── workflow.py         # Multi-agent pipeline
│   ├── components/             # RAG building blocks
│   │   ├── chunker.py
│   │   ├── index_builder.py
│   │   ├── retriever.py
│   │   └── evaluator.py
│   ├── tools/                  # Utilities
│   │   ├── llm_tools.py
│   │   ├── retriever_tools.py
│   │   └── evaluation_tools.py
│   └── configs/
│       ├── base_config.yaml    # Starting point
│       └── test_queries.json   # Evaluation data
├── data/
│   ├── raw_docs/               # Your documents (PDF, TXT)
│   └── index/                  # Generated indexes
├── outputs/
│   ├── optimized_config.yaml   # 🎯 Final result
│   └── reports/                # Agent reports (JSON)
└── examples/
    └── sample_run.py           # Full demo script
```

---

## 💡 Key Features

✅ **Fully Automated** – No manual parameter tuning  
✅ **Multi-Agent** – 4 specialized LLM agents collaborate  
✅ **Data-Driven** – Decisions based on metrics + LLM reasoning  
✅ **Hybrid Search** – Combines vector (FAISS) + lexical (BM25)  
✅ **LLM-as-Judge** – Evaluates answer quality objectively  
✅ **Production-Ready** – Outputs clean YAML configuration  
✅ **Cost-Efficient** – ~$0.02-0.05 per optimization run  
✅ **Extensible** – Easy to add custom agents or metrics  

---

## 🧪 Example Use Cases

| Domain | Documents | Optimization Focus |
|--------|-----------|-------------------|
| **Legal** | Laws, court decisions | Precise chunking for citations |
| **Medical** | Research papers, protocols | Context preservation across chunks |
| **Customer Support** | FAQs, tickets | Fast retrieval, diverse sources |
| **Technical Docs** | API docs, guides | Code snippet integrity |
| **Finance** | Reports, regulations | Numerical data accuracy |

---

## 🔬 Evaluation Metrics Explained

### 1. **Answer Quality (LLM-as-Judge)**
GPT-4o-mini scores each answer (1-10) based on:
- Relevance
- Completeness  
- Accuracy
- Conciseness

### 2. **Retrieval Score (Cosine Similarity)**
- Semantic similarity between query and retrieved chunks
- Range: 0.0 (completely different) → 1.0 (identical)
- **Higher is better**

### 3. **Top-1 Retrieval Score**
- Similarity score of the BEST retrieved chunk
- Critical for answer quality
- **Target: > 0.5**

### 4. **Precision@K**
- Percentage of top-K chunks that are relevant (score > 0.4)
- Measures retrieval accuracy
- **Target: > 70%**

---

## 📈 Agent Reasoning Example

**Chunk Architect Agent Analysis** (from actual run):

```yaml
Input (Profiling Report):
  - Current chunk_size: 4000 words
  - Current overlap: 0 words
  - Avg retrieval score: 0.413
  - Issues: Low source diversity, many low scores

Agent Reasoning (GPT-4o-mini):
  "The current average retrieval score of 0.413 indicates room 
   for improvement. The large chunk size (4000 words) captures 
   too much irrelevant content, diluting semantic similarity.
   
   Reducing chunk size to 600 words will:
   • Increase precision by focusing each chunk on one topic
   • Improve retrieval scores by reducing noise
   • Enable better source diversity
   
   Adding 150 words overlap (30%) will:
   • Preserve context across chunk boundaries
   • Prevent information fragmentation
   • Maintain answer completeness"

Proposed Output:
  - chunk_size: 600 words (-85%)
  - overlap: 150 words (+150 words)
  - confidence: HIGH
  
Expected Impact:
  ✓ +10-15% retrieval score improvement
  ✓ +20-30pp precision improvement
  ✓ Better answer consistency
```

**Result**: Retrieval improved by **+10.9%**, Precision@3 by **+22.2pp** ✅

---

## 🌟 Why This Matters

**Traditional RAG Development**:
- ⏰ Hours of manual experimentation
- 🎲 Trial and error, guesswork
- 📉 Suboptimal configurations
- 💸 Wasted API costs on poor retrievals

**With Auto-RAG Optimizer**:
- ⚡ 5-10 minutes automated
- 🤖 AI reasoning + data analysis
- 📈 Measurable, reproducible results
- 💰 Optimized for quality AND cost

---

## 🤝 Contributing

Contributions welcome! Ideas:
- Add more agents (e.g., RerankerAgent, PromptAgent)
- Support more vector DBs (Pinecone, Weaviate, Qdrant)
- Custom evaluation metrics
- Multi-language support
- Web UI (Gradio/Streamlit)

---

## 📝 License

MIT License - Free for personal and commercial use.

---

## 🔗 Links

- **GitHub**: [Bellilty/auto-rag-optimizer](https://github.com/Bellilty/auto-rag-optimizer)
- **Issues**: [Report bugs or request features](https://github.com/Bellilty/auto-rag-optimizer/issues)

---

<div align="center">

**Built with ❤️ for the RAG community**

*If you find this useful, star the repo ⭐ and share on LinkedIn!*

[![Star on GitHub](https://img.shields.io/github/stars/Bellilty/auto-rag-optimizer?style=social)](https://github.com/Bellilty/auto-rag-optimizer)

</div>
