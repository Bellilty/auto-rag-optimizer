"""
Setup Verification Script
==========================

Quick script to verify that all dependencies and modules are correctly set up.
"""

import sys
import os


def check_imports():
    """Check that all required packages can be imported."""
    print("🔍 Checking imports...\n")
    
    packages = [
        ("openai", "OpenAI"),
        ("faiss", "FAISS (vector search)"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("rank_bm25", "Rank-BM25"),
        ("dotenv", "python-dotenv"),
        ("pydantic", "Pydantic"),
        ("fitz", "PyMuPDF"),
        ("tqdm", "tqdm"),
    ]
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - {e}")
            failed.append(name)
    
    return failed


def check_environment():
    """Check environment variables."""
    print("\n🔧 Checking environment...\n")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        # Mask the key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"  ✅ OPENAI_API_KEY is set: {masked_key}")
    else:
        print("  ⚠️  OPENAI_API_KEY is not set")
        print("     Set it with: export OPENAI_API_KEY='your-key'")
        print("     Or create a .env file")
    
    return api_key is not None


def check_directories():
    """Check that required directories exist."""
    print("\n📁 Checking directories...\n")
    
    dirs = [
        "src/agents",
        "src/components",
        "src/tools",
        "src/orchestrator",
        "src/configs",
        "data/raw_docs",
        "data/index",
        "outputs/reports",
        "outputs/metrics",
        "examples",
    ]
    
    missing = []
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - missing")
            missing.append(dir_path)
    
    return missing


def check_documents():
    """Check if documents are present."""
    print("\n📄 Checking documents...\n")
    
    docs_dir = "data/raw_docs"
    
    if os.path.exists(docs_dir):
        docs = [f for f in os.listdir(docs_dir) if f.endswith(('.pdf', '.txt'))]
        
        if docs:
            print(f"  ✅ Found {len(docs)} document(s):")
            for doc in docs[:5]:  # Show first 5
                print(f"     - {doc}")
            if len(docs) > 5:
                print(f"     ... and {len(docs) - 5} more")
            return True
        else:
            print(f"  ⚠️  No documents found in {docs_dir}")
            print(f"     Add PDF or TXT files to start")
            return False
    else:
        print(f"  ❌ Directory {docs_dir} not found")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("AUTO-RAG OPTIMIZER - Setup Verification")
    print("="*60 + "\n")
    
    # Check imports
    failed_imports = check_imports()
    
    # Check environment
    env_ok = check_environment()
    
    # Check directories
    missing_dirs = check_directories()
    
    # Check documents
    docs_present = check_documents()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("✅ All packages imported successfully")
    
    if not env_ok:
        print("⚠️  OpenAI API key not set")
        print("   The system will not work without it")
    else:
        print("✅ Environment configured")
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {len(missing_dirs)}")
    else:
        print("✅ All directories present")
    
    if not docs_present:
        print("⚠️  No documents found")
        print("   Add documents to data/raw_docs/ to get started")
    else:
        print("✅ Documents ready")
    
    # Overall status
    print()
    if not failed_imports and env_ok and not missing_dirs:
        print("🎉 Setup complete! Ready to optimize RAG systems.")
        print("\nNext step: python examples/sample_run.py")
        return 0
    else:
        print("⚠️  Setup incomplete. Please address the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

