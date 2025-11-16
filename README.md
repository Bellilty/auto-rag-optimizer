# 🤖 AutoRAG Optimizer

**Multi-Agent RAG Evaluation and Architecture Refinement**

An automated pipeline for profiling, evaluating, and optimizing Retrieval-Augmented Generation (RAG) systems using cooperating AI agents.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Outputs](#-outputs)
- [Demo (Coming Soon)](#-demo-coming-soon)
- [Development](#-development)
- [License](#-license)

---

## 🎯 Overview

AutoRAG Optimizer is a sophisticated multi-agent system that automatically analyzes and optimizes RAG pipelines. Instead of manually tweaking parameters, this system uses specialized AI agents to:

1. **Profile** your retrieval behavior
2. **Analyze** performance bottlenecks
3. **Propose** optimized configurations
4. **Evaluate** improvements quantitatively
5. **Output** a production-ready optimized configuration

### Why AutoRAG Optimizer?

Building an effective RAG system requires careful tuning of multiple parameters:

- Chunk size and overlap
- Retrieval methods (vector vs BM25 vs hybrid)
- Hybrid search weights
- Top-k parameters

This project automates the optimization process using LLM-powered agents that analyze your specific data and use cases.

---

## ✨ Features

### Multi-Agent Architecture

- **RetrieverProfilerAgent**: Profiles retrieval behavior, collects metrics (scores, diversity, BM25 vs vector analysis)
- **ChunkArchitectAgent**: Uses LLM reasoning to propose optimal chunking parameters based on profiling
- **EvaluatorAgent**: Runs comparative evaluations with LLM-based judging
- **ArchitectAgent**: Synthesizes all data into a final optimized configuration

### Hybrid Retrieval

- Vector search (semantic) using FAISS
- BM25 search (lexical) using rank-bm25
- Configurable hybrid weighting

### Comprehensive Evaluation

- LLM-based answer quality judging
- Baseline vs optimized comparisons
- Win rate calculations
- Per-query detailed metrics

### Clean Architecture

- Modular components (chunking, retrieval, evaluation)
- Reusable tools and utilities
- Clear separation of concerns
- Type hints and docstrings throughout

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Optimization Workflow                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Baseline Index   │
                    │ (chunking +      │
                    │  vector + BM25)  │
                    └──────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ RetrieverProfilerAgent       │
                │ - Run test queries           │
                │ - Collect metrics            │
                │ - Detect issues              │
                │ → retrieval_report.json      │
                └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ ChunkArchitectAgent          │
                │ - Analyze profiling          │
                │ - LLM proposes chunking      │
                │ → chunk_proposal.json        │
                └──────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Rebuild Index    │
                    │ (new chunking)   │
                    └──────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ EvaluatorAgent               │
                │ - Evaluate baseline          │
                │ - Evaluate optimized         │
                │ - Compare results            │
                │ → evaluation_report.json     │
                └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ ArchitectAgent               │
                │ - Synthesize reports         │
                │ - Generate final config      │
                │ → optimized_config.yaml      │
                └──────────────────────────────┘
```

### Agent Responsibilities

| Agent                      | Input               | Output                   | Purpose                                       |
| -------------------------- | ------------------- | ------------------------ | --------------------------------------------- |
| **RetrieverProfilerAgent** | Test queries        | `retrieval_report.json`  | Profiles retrieval behavior and metrics       |
| **ChunkArchitectAgent**    | Profiling report    | `chunk_proposal.json`    | Proposes optimized chunking parameters        |
| **EvaluatorAgent**         | Both configurations | `evaluation_report.json` | Compares baseline vs optimized quantitatively |
| **ArchitectAgent**         | All reports         | `optimized_config.yaml`  | Synthesizes final production configuration    |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Bellilty/auto-rag-optimizer.git
cd auto-rag-optimizer
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key-here
```

Or export directly:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

5. **Add documents**

Place your PDF or TXT documents in `data/raw_docs/`:

```bash
# Example: copy from existing rag-juridique project
cp ../rag-juridique/data/pdfs/* data/raw_docs/

# Or add your own documents
cp /path/to/your/documents/*.pdf data/raw_docs/
```

---

## 🚀 Quick Start

### Run Complete Optimization

```bash
python examples/sample_run.py
```

This will:

1. Build baseline index from your documents
2. Profile retrieval performance
3. Propose optimized chunking
4. Rebuild index with new chunking
5. Evaluate both configurations
6. Generate final optimized configuration

### Expected Output

```
================================================================================
                          AUTO-RAG OPTIMIZER
                Multi-Agent RAG Optimization Pipeline
================================================================================

STEP 1/6: Build Baseline Index
────────────────────────────────────────────────────────────────────────────────
🔨 Building baseline index...
...

STEP 2/6: Profile Baseline Retrieval
────────────────────────────────────────────────────────────────────────────────
🔍 Profiling retrieval on 10 queries...
...

STEP 3/6: Propose Optimized Chunking
────────────────────────────────────────────────────────────────────────────────
🏗️  Analyzing retrieval profile...
✅ Proposed chunk size: 800 words (overlap: 180)
...

STEP 4/6: Build Optimized Index
────────────────────────────────────────────────────────────────────────────────
🔨 Building optimized index...
...

STEP 5/6: Evaluate Baseline vs Optimized
────────────────────────────────────────────────────────────────────────────────
📊 Evaluating BASELINE configuration...
📊 Evaluating OPTIMIZED configuration...
...

STEP 6/6: Generate Final Optimized Configuration
────────────────────────────────────────────────────────────────────────────────
🎯 Synthesizing final configuration...
...

================================================================================
                        OPTIMIZATION COMPLETE!
================================================================================
```

### Using Individual Agents

You can also run agents individually for more control:

```python
from src.components.index_builder import IndexBuilder
from src.components.retriever import HybridRetriever
from src.agents.retriever_profiler_agent import RetrieverProfilerAgent

# Build index
builder = IndexBuilder()
faiss_index, bm25_index, chunks = builder.load_indexes()

# Create retriever
retriever = HybridRetriever(faiss_index, bm25_index, chunks)

# Run profiler
profiler = RetrieverProfilerAgent(retriever)
report = profiler.run(
    queries_path="src/configs/test_queries.json",
    output_path="outputs/reports/retrieval_report.json"
)
```

---

## 📂 Project Structure

```
auto-rag-optimizer/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
│
├── src/
│   ├── __init__.py
│   │
│   ├── orchestrator/           # Workflow coordination
│   │   ├── __init__.py
│   │   └── workflow.py         # Main orchestration logic
│   │
│   ├── agents/                 # LLM-driven decision agents
│   │   ├── __init__.py
│   │   ├── retriever_profiler_agent.py
│   │   ├── chunk_architect_agent.py
│   │   ├── evaluator_agent.py
│   │   └── architect_agent.py
│   │
│   ├── components/             # Core RAG components
│   │   ├── __init__.py
│   │   ├── chunker.py          # Document chunking
│   │   ├── index_builder.py    # Embeddings + FAISS + BM25
│   │   ├── retriever.py        # Hybrid retrieval
│   │   └── evaluator.py        # RAG evaluation
│   │
│   ├── tools/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── llm_tools.py        # LLM interactions
│   │   ├── retriever_tools.py  # Retrieval analysis
│   │   ├── chunking_tools.py   # Chunking analysis
│   │   └── evaluation_tools.py # Evaluation metrics
│   │
│   └── configs/                # Configuration files
│       ├── __init__.py
│       ├── base_config.yaml    # Baseline configuration
│       └── test_queries.json   # Test queries
│
├── data/                       # Data directories
│   ├── raw_docs/              # Place your PDF/TXT documents here
│   ├── processed_docs/        # Processed documents (future use)
│   └── index/                 # FAISS and BM25 indexes
│
├── outputs/                   # Generated outputs
│   ├── reports/              # Agent reports (JSON)
│   ├── metrics/              # Evaluation metrics (JSON)
│   └── optimized_config.yaml # Final optimized configuration
│
├── examples/
│   └── sample_run.py         # Example usage script
│
└── notebooks/
    └── exploration.ipynb     # Jupyter notebook for exploration
```

---

## 🔧 How It Works

### 1. Baseline Index Creation

The system first chunks your documents using baseline parameters (from `base_config.yaml`) and builds:

- **FAISS index**: For semantic (vector) search
- **BM25 index**: For lexical (keyword) search

### 2. Retrieval Profiling

The **RetrieverProfilerAgent** runs test queries and collects:

- Retrieval scores
- Score distributions
- Source diversity
- Vector vs BM25 contribution
- Potential issues (low scores, low diversity, etc.)

### 3. Chunking Optimization

The **ChunkArchitectAgent**:

- Analyzes the profiling report
- Uses an LLM to reason about optimal chunking
- Proposes new `chunk_size` and `overlap` parameters
- Validates the proposal

### 4. Index Rebuilding

The system re-chunks documents with the proposed parameters and rebuilds both indexes.

### 5. Evaluation

The **EvaluatorAgent**:

- Runs the same test queries on both configurations
- Uses an LLM judge to score answer quality (0-10)
- Compares baseline vs optimized
- Calculates win rate and improvement metrics

### 6. Final Configuration

The **ArchitectAgent**:

- Synthesizes all reports
- Uses LLM reasoning to finalize configuration
- Considers chunking, retrieval weights, top-k, etc.
- Outputs `optimized_config.yaml`

---

## ⚙️ Configuration

### Base Configuration (`src/configs/base_config.yaml`)

```yaml
chunking:
  chunk_size: 1000 # Words per chunk
  overlap: 200 # Overlapping words
  strategy: "word_based"

retrieval:
  method: "hybrid" # vector | bm25 | hybrid
  top_k: 5
  vector_weight: 0.7 # Hybrid weight for semantic search
  bm25_weight: 0.3 # Hybrid weight for lexical search

generation:
  model: "gpt-4o-mini"
  max_tokens: 500
  temperature: 0.3
```

### Test Queries (`src/configs/test_queries.json`)

```json
{
  "queries": [
    {
      "query": "What are the main principles?",
      "category": "general",
      "difficulty": "medium"
    },
    ...
  ]
}
```

Add your own domain-specific queries for better optimization results.

---

## 📊 Outputs

### Reports (`outputs/reports/`)

- **`retrieval_report.json`**: Profiling metrics
- **`chunk_proposal.json`**: Proposed chunking parameters
- **`evaluation_report.json`**: Baseline vs optimized comparison

### Metrics (`outputs/metrics/`)

- **`baseline_evaluation.json`**: Detailed baseline evaluation
- **`optimized_evaluation.json`**: Detailed optimized evaluation

### Final Configuration

- **`outputs/optimized_config.yaml`**: Production-ready configuration
- **`outputs/optimized_config.json`**: Same in JSON format

### Example Output Structure

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "configuration": {
    "chunking": {
      "chunk_size": 800,
      "overlap": 180
    },
    "retrieval": {
      "method": "hybrid",
      "top_k": 5,
      "vector_weight": 0.65,
      "bm25_weight": 0.35
    }
  },
  "reasoning": "Reduced chunk size improves precision...",
  "expected_benefits": ["Better retrieval precision", "Improved diversity"],
  "confidence": "high"
}
```

---

## 🎬 Demo (Coming Soon)

Future additions:

- **CLI interface** for interactive configuration
- **Web dashboard** for visualization
- **Jupyter notebooks** with step-by-step walkthroughs
- **API endpoints** for programmatic access

---

## 🛠️ Development

### Running Tests

```bash
# Test individual components
python src/components/chunker.py
python src/tools/llm_tools.py

# Test agents
python src/agents/retriever_profiler_agent.py
```

### Adding Custom Agents

Create a new agent in `src/agents/`:

```python
class MyCustomAgent:
    def __init__(self, ...):
        pass

    def run(self, ...):
        # Your agent logic
        pass
```

Register it in the workflow orchestrator.

### Customizing Evaluation

Modify `test_queries.json` with domain-specific queries for your use case.

---

## 💰 Costs

This project uses OpenAI APIs. Approximate costs for a typical run (10 queries, 3 documents):

| Operation                         | Tokens | Cost       |
| --------------------------------- | ------ | ---------- |
| Embeddings (baseline + optimized) | ~100K  | $0.002     |
| Profiling queries                 | ~10K   | $0.002     |
| LLM reasoning (agents)            | ~20K   | $0.005     |
| Evaluation (baseline + optimized) | ~50K   | $0.01      |
| **Total**                         | ~180K  | **~$0.02** |

For larger document sets, costs scale with:

- Number of chunks → embedding costs
- Number of test queries → evaluation costs

---

## 📚 References

This project adapts and extends concepts from:

- [rag-juridique](https://github.com/Bellilty/rag-juridique) - Base RAG implementation
- LangChain - RAG patterns and best practices
- FAISS - Vector similarity search
- Rank-BM25 - Lexical search

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional chunking strategies (sentence-based, semantic, etc.)
- More sophisticated evaluation metrics
- Support for other LLM providers (Anthropic, local models)
- Web UI for visualization
- Additional optimization targets (latency, cost, etc.)

---

## 📝 License

This project is open source and available for educational and commercial use.

---

## 🎉 Acknowledgments

Built as a practical exploration of multi-agent systems for RAG optimization.

**Questions?** Open an issue or reach out!

---

**Happy Optimizing! 🚀**
