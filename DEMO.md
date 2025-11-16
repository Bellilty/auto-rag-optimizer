# 🎬 Demo Guide - Auto-RAG Optimizer

## Quick Start

### 1. Launch the Demo

```bash
# Activate environment
source venv/bin/activate

# Launch Gradio interface
python demo_app.py
```

The interface will open at: **http://localhost:7860**

---

## Demo Features

### Tab 1: 🚀 Optimization

**Run the full optimization pipeline**

1. Check system status
2. Choose whether to skip evaluation (faster)
3. Click "Run Optimization"
4. Wait 5-15 minutes (depending on settings)
5. View results summary

**Outputs**:
- Baseline vs optimized chunk counts
- Final configuration path
- Reports location

---

### Tab 2: 💬 Test Q&A

**Compare baseline vs optimized RAG**

1. Enter your question
2. Choose configuration (Baseline or Optimized)
3. Adjust Top-K chunks (1-10)
4. Click "Query RAG"
5. See answer with sources

**Example Questions**:
- "What are the main principles of data protection?"
- "What rights do data subjects have?"
- "What is the role of a data controller?"

**Try both configurations** to see the difference!

---

### Tab 3: 📊 Reports

**View detailed optimization reports**

1. Click "Load Reports"
2. Review:
   - Retrieval profiling metrics
   - Chunking proposal reasoning
   - Evaluation results (if run)
   - Final configuration details

**Insights**:
- Average retrieval scores
- Proposed vs current chunking
- Performance improvements
- LLM reasoning

---

### Tab 4: ℹ️ About

**Learn about the system**

- Architecture overview
- Technologies used
- Cost estimates
- Links to documentation

---

## Demo Workflow

### Recommended Flow

1. **Start with Optimization**
   - Run with "Skip evaluation" checked (5-7 min)
   - Review the results summary

2. **Test Q&A**
   - Try example questions
   - Compare Baseline vs Optimized
   - Notice differences in answers/sources

3. **Check Reports**
   - Load and review all reports
   - Understand optimization decisions
   - See performance metrics

4. **For Full Demo** (optional)
   - Re-run without skipping evaluation (10-15 min)
   - Get complete comparison metrics
   - See win rates and improvements

---

## Recording a Demo

### Preparation

1. **Clean start**: Delete `data/index/*` and `outputs/*` for fresh demo
2. **Test documents**: Ensure 2-3 PDFs are in `data/raw_docs/`
3. **API key**: Verify `OPENAI_API_KEY` is set
4. **Screen recording**: Use QuickTime, OBS, or similar

### Demo Script (5 minutes)

**Intro (30 sec)**
- "This is Auto-RAG Optimizer, a multi-agent system for optimizing RAG pipelines"
- Show the interface

**Optimization (2 min)**
- "Let's optimize a RAG system with 3 legal documents"
- Click "Run Optimization" (with skip evaluation)
- While running: "4 AI agents are analyzing retrieval, proposing improvements"
- Show results when complete

**Q&A Comparison (2 min)**
- "Now let's test both configurations"
- Try a question with Baseline
- Try same question with Optimized
- "Notice the difference in answers and sources"

**Reports (30 sec)**
- Click "Load Reports"
- "Here's the LLM reasoning behind the optimization"
- Show key metrics

**Wrap-up (30 sec)**
- "Full evaluation takes 10-15 minutes"
- "Cost per run: ~$0.02"
- "GitHub: Bellilty/auto-rag-optimizer"

---

## Tips for Live Demo

### Do's
✅ Use "Skip evaluation" for faster demos
✅ Prepare interesting questions beforehand
✅ Show both configurations side-by-side
✅ Highlight LLM reasoning in reports
✅ Mention the cost-effectiveness (~$0.02/run)

### Don'ts
❌ Don't wait for full optimization (show pre-run results)
❌ Don't skip the comparison (key demo feature)
❌ Don't forget to mention the multi-agent architecture
❌ Don't rush through the reports tab

---

## Sharing the Demo

### Generate Shareable Link

```bash
python demo_app.py
```

Then set `share=True` in `demo_app.py`:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Creates public URL
    show_error=True
)
```

Gradio will generate a public URL (valid for 72 hours).

---

## Troubleshooting

### Demo won't start
- Check: `source venv/bin/activate`
- Check: `pip list | grep gradio`
- Reinstall: `pip install gradio`

### API errors
- Verify API key in `.env`
- Check: `echo $OPENAI_API_KEY`
- Export: `export OPENAI_API_KEY='your-key'`

### No documents
- Add PDFs to `data/raw_docs/`
- Check: `ls data/raw_docs/`

### Indexes not loading
- Run optimization first
- Check: `ls data/index/`

---

## Advanced: Custom Demo Setup

### Use Your Own Documents

1. Clear existing data:
```bash
rm -rf data/index/* outputs/*
```

2. Add your documents:
```bash
cp /path/to/your/docs/*.pdf data/raw_docs/
```

3. Update test queries in `src/configs/test_queries.json`

4. Run optimization and demo

---

## Demo Metrics

Typical performance (3 documents, 10 queries):

| Metric | Time | Cost |
|--------|------|------|
| With evaluation skip | 5-7 min | $0.01 |
| Full evaluation | 10-15 min | $0.02 |

Scale:
- +5 documents: +2-3 min
- +10 queries: +1-2 min (evaluation only)

---

## Support

- **Issues**: https://github.com/Bellilty/auto-rag-optimizer/issues
- **Docs**: See README.md
- **Quick Start**: See QUICKSTART.md

---

**Ready to record your demo? Launch with `python demo_app.py`! 🎬**

