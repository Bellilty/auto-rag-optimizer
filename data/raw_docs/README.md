# Raw Documents

Place your PDF or TXT documents in this directory.

## Supported Formats

- **PDF** (`.pdf`) - Extracted using PyMuPDF
- **TXT** (`.txt`) - Plain text files

## Getting Started

### Option 1: Copy from rag-juridique

If you have the rag-juridique project:

```bash
cp ../../rag-juridique/data/pdfs/* .
```

### Option 2: Add Your Own Documents

Simply copy or download your documents here:

```bash
cp /path/to/your/documents/*.pdf .
```

### Option 3: Download Sample Documents

Example legal documents (public domain):

- **GDPR**: https://gdpr-info.eu/
- **US Constitution**: https://www.archives.gov/founding-docs
- **French Legal Texts**: https://www.legifrance.gouv.fr/

## Document Requirements

- **Size**: No strict limits, but very large documents (>100 pages) will increase processing time
- **Language**: The system works with any language, but evaluation quality depends on LLM understanding
- **Content**: Works best with structured documents (legal, technical, academic)

## Next Steps

After adding documents, run:

```bash
python examples/sample_run.py
```

The system will automatically:

1. Extract text from your documents
2. Chunk them according to configuration
3. Create embeddings and indexes
4. Optimize the RAG pipeline
