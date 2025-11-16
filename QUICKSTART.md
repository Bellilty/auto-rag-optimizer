# 🚀 QuickStart Guide

Get Auto-RAG Optimizer running in 5 minutes!

## Prerequisites

- **Python 3.10 or 3.11** (stable release - NOT alpha/beta versions)
- **OpenAI API key** - [Get one here](https://platform.openai.com/api-keys)
- **Documents** - PDF or TXT files to optimize

---

## Step 1: Setup Environment

### Option A: Using the setup script (Recommended)

```bash
cd auto-rag-optimizer
chmod +x setup_with_sample_data.sh
./setup_with_sample_data.sh
```

### Option B: Manual setup

```bash
# Create virtual environment with stable Python
python3.11 -m venv venv  # Or python3.10
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Step 2: Add Documents

Place your PDF or TXT documents in `data/raw_docs/`:

```bash
# Example: Copy from existing project
cp ../rag-juridique/data/pdfs/* data/raw_docs/

# Or add your own
cp /path/to/your/docs/*.pdf data/raw_docs/
```

---

## Step 3: Verify Setup

```bash
python verify_setup.py
```

You should see:
- ✅ All packages imported successfully
- ✅ Environment configured
- ✅ All directories present
- ✅ Documents ready

---

## Step 4: Run Optimization!

### Full optimization (recommended)

```bash
python examples/sample_run.py
```

This will:
1. Build baseline index (2-5 min)
2. Profile retrieval (~1 min)
3. Propose optimized chunking (~30 sec)
4. Rebuild optimized index (2-5 min)
5. Evaluate both configurations (~2 min)
6. Generate final configuration (~30 sec)

**Total time**: ~10-15 minutes for a small dataset

### Quick test (skip evaluation)

Edit `examples/sample_run.py` and change:

```python
results = workflow.run_full_optimization(
    skip_evaluation=True  # Faster!
)
```

Then run:
```bash
python examples/sample_run.py
```

**Total time**: ~5-7 minutes

---

## Step 5: View Results

After completion, check:

### Final Configuration
```bash
cat outputs/optimized_config.yaml
```

### Reports
```bash
# Retrieval profiling
cat outputs/reports/retrieval_report.json

# Chunking proposal
cat outputs/reports/chunk_proposal.json

# Evaluation comparison
cat outputs/reports/evaluation_report.json
```

### Metrics
```bash
ls outputs/metrics/
# - baseline_evaluation.json
# - optimized_evaluation.json
```

---

## Expected Output

```
================================================================================
                          AUTO-RAG OPTIMIZER
                Multi-Agent RAG Optimization Pipeline
================================================================================

STEP 1/6: Build Baseline Index
────────────────────────────────────────────────────────────────────────────────
📚 Processing 3 document(s)...
🔢 Creating 150 embeddings...
✅ FAISS index built with 150 vectors

STEP 2/6: Profile Baseline Retrieval
────────────────────────────────────────────────────────────────────────────────
🔍 Profiling retrieval on 10 queries...
✅ Profiling complete!
   Average retrieval score: 0.652

STEP 3/6: Propose Optimized Chunking
────────────────────────────────────────────────────────────────────────────────
🏗️  Analyzing retrieval profile...
✅ Proposed chunk size: 800 words (overlap: 180)

STEP 4/6: Build Optimized Index
────────────────────────────────────────────────────────────────────────────────
🔨 Building optimized index...
✅ FAISS index built with 185 vectors

STEP 5/6: Evaluate Baseline vs Optimized
────────────────────────────────────────────────────────────────────────────────
📊 Evaluating BASELINE configuration...
📊 Evaluating OPTIMIZED configuration...
✅ Evaluation complete. Average score: 7.2/10

STEP 6/6: Generate Final Optimized Configuration
────────────────────────────────────────────────────────────────────────────────
🎯 Synthesizing final configuration...
✅ Configuration saved: outputs/optimized_config.yaml

================================================================================
                        OPTIMIZATION COMPLETE!
================================================================================
📊 Summary:
   Baseline chunks: 150
   Optimized chunks: 185
   
📁 Outputs:
   Reports: outputs/reports/
   Final config: outputs/optimized_config.yaml
================================================================================
```

---

## Costs

Typical run (10 queries, 3 documents):
- **Embeddings**: ~$0.002
- **LLM reasoning**: ~$0.005  
- **Evaluation**: ~$0.01

**Total**: ~**$0.02 per optimization run**

---

## What's Next?

1. **Review the optimized configuration**: `outputs/optimized_config.yaml`
2. **Implement in your RAG system**: Use the recommended parameters
3. **Run with your own queries**: Edit `src/configs/test_queries.json`
4. **Tune further**: Adjust base config and re-run

---

## Troubleshooting

### Python version issues?
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#1-python-version-issues)

### API key not working?
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#2-openai-api-key-not-set)

### Other issues?
Check the full [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide

---

## Need Help?

- 📖 Read the full [README.md](README.md)
- 🐛 Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 💬 Open an issue on GitHub

---

**Happy Optimizing! 🚀**


