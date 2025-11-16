#!/bin/bash

# Setup Script for Auto-RAG Optimizer
# ====================================
# This script helps you quickly set up the project with sample data

set -e

echo "================================"
echo "Auto-RAG Optimizer Setup"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the auto-rag-optimizer root directory"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Check for .env
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your OpenAI API key:"
    echo "   OPENAI_API_KEY=your-api-key-here"
    echo ""
fi

# Check for documents
echo "📄 Checking for documents..."

if [ -d "../rag-juridique/data/pdfs" ]; then
    read -p "Found rag-juridique project. Copy documents? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📋 Copying documents from rag-juridique..."
        cp ../rag-juridique/data/pdfs/*.pdf data/raw_docs/ 2>/dev/null || true
        cp ../rag-juridique/data/pdfs/*.txt data/raw_docs/ 2>/dev/null || true
        
        doc_count=$(ls data/raw_docs/*.{pdf,txt} 2>/dev/null | wc -l | tr -d ' ')
        echo "✅ Copied $doc_count document(s)"
    fi
else
    echo "ℹ️  No rag-juridique project found"
    echo "   Please add your own documents to data/raw_docs/"
fi

# Verify setup
echo ""
echo "🔍 Running setup verification..."
python verify_setup.py

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Make sure your OpenAI API key is set in .env"
echo "  2. Add documents to data/raw_docs/ (if not done)"
echo "  3. Run: python examples/sample_run.py"
echo ""

