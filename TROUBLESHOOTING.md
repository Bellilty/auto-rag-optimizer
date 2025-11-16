# Troubleshooting Guide

## Common Issues and Solutions

### 1. Python Version Issues

#### Problem: `TypeError: rmtree() got an unexpected keyword argument 'onexc'`

**Cause**: You're using Python 3.12.0a3 (alpha release), which has compatibility issues with pip.

**Solution**:

Use a stable Python version (3.10 or 3.11):

```bash
# Check your Python versions
python3 --version
python3.10 --version  # Try specific version
python3.11 --version

# Recreate venv with stable Python
rm -rf venv
python3.11 -m venv venv  # Or python3.10
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative**: Install via conda with a stable Python:

```bash
conda create -n auto-rag-optimizer python=3.11
conda activate auto-rag-optimizer
pip install -r requirements.txt
```

---

### 2. OpenAI API Key Not Set

#### Problem: `OpenAI API key not found`

**Solution**:

1. Create a `.env` file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

3. Or export temporarily:
```bash
export OPENAI_API_KEY='sk-your-actual-api-key-here'
```

---

### 3. No Documents Found

#### Problem: `No documents found in data/raw_docs/`

**Solution**:

Add PDF or TXT documents to `data/raw_docs/`:

```bash
# Option 1: Copy from rag-juridique
cp ../rag-juridique/data/pdfs/* data/raw_docs/

# Option 2: Add your own documents
cp /path/to/your/documents/*.pdf data/raw_docs/
```

---

### 4. FAISS Installation Issues

#### Problem: `faiss-cpu` installation fails

**Solutions**:

**macOS**:
```bash
pip install --no-cache-dir faiss-cpu
```

**Windows**:
```bash
# Use conda for easier FAISS installation
conda install -c conda-forge faiss-cpu
```

**Linux**:
```bash
pip install faiss-cpu
```

---

### 5. Memory Issues with Large Documents

#### Problem: Out of memory during indexing

**Solution**:

Reduce batch size in `src/components/index_builder.py`:

```python
# Change batch_size from 100 to 20
embeddings = builder.create_embeddings_batch(texts, batch_size=20)
```

---

### 6. Import Errors

#### Problem: `ModuleNotFoundError: No module named 'src'`

**Solution**:

Make sure you're running scripts from the project root:

```bash
cd /path/to/auto-rag-optimizer
python examples/sample_run.py
```

Not from subdirectories.

---

### 7. Rate Limiting from OpenAI

#### Problem: `Rate limit exceeded`

**Solution**:

1. **Wait and retry**: OpenAI has rate limits

2. **Reduce batch size**: Slow down embedding creation

3. **Use tier limits**: Check your OpenAI tier at https://platform.openai.com/account/limits

4. **Skip evaluation**: Run with `skip_evaluation=True` for faster testing

```python
results = workflow.run_full_optimization(
    skip_evaluation=True  # Skips evaluation step
)
```

---

### 8. Slow Execution

#### Problem: Optimization takes too long

**Solutions**:

1. **Reduce test queries**: Edit `src/configs/test_queries.json` to use fewer queries

2. **Skip evaluation**: Set `skip_evaluation=True`

3. **Reuse baseline index**: Set `skip_baseline_indexing=True`

```python
results = workflow.run_full_optimization(
    skip_baseline_indexing=True,  # Reuse existing baseline
    skip_evaluation=True           # Skip evaluation for speed
)
```

---

### 9. Permission Denied Errors

#### Problem: Cannot create directories or files

**Solution**:

Check directory permissions:

```bash
# Make sure you own the project directory
ls -la auto-rag-optimizer/

# Fix permissions if needed
chmod -R u+w auto-rag-optimizer/
```

---

### 10. Git Issues

#### Problem: Git not recognizing the repository

**Solution**:

Initialize git if needed:

```bash
cd auto-rag-optimizer
git init
git add .
git commit -m "Initial commit: Auto-RAG Optimizer"
```

---

## Getting Help

If you encounter other issues:

1. **Check the logs**: Look for detailed error messages
2. **Verify setup**: Run `python verify_setup.py`
3. **Check Python version**: Must be 3.10+ (stable release)
4. **Check API key**: Make sure OpenAI API key is valid
5. **Check documents**: Ensure PDFs/TXTs are in `data/raw_docs/`

## Reporting Issues

When reporting issues, please include:

- Python version (`python --version`)
- Error message (full traceback)
- Steps to reproduce
- Output of `python verify_setup.py`


