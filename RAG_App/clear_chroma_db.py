#!/usr/bin/env python3
"""
Standalone script to clear all data from ChromaDB.

This script:
1. Deletes the existing ChromaDB collection
2. Recreates an empty collection
3. Optionally deletes the entire database directory

Usage:
    python clear_chroma_db.py           # Clear collection only
    python clear_chroma_db.py --full    # Delete entire database directory
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Path to ChromaDB directory
CHROMA_DB_PATH = "./db/chroma_db"


def clear_collection():
    """Clear the ChromaDB collection programmatically."""
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_chroma import Chroma
        
        print("Initializing ChromaDB connection...")
        
        # Initialize embeddings (needed for ChromaDB)
        embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
        
        # Connect to existing ChromaDB instance
        vectordb = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        
        print("Deleting collection 'documents'...")
        
        # Delete the collection
        try:
            vectordb.delete_collection()
            print("✓ Collection deleted successfully")
        except Exception as e:
            error_msg = str(e).lower()
            if "does not exist" in error_msg or "not found" in error_msg:
                print("⚠ Collection does not exist (may already be empty)")
            else:
                raise
        
        # Recreate empty collection
        print("Recreating empty collection...")
        vectordb = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
        )
        print("✓ Empty collection recreated successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: Missing required package: {e}")
        print("Please install required packages: pip install langchain-chroma langchain-ollama")
        return False
    except Exception as e:
        print(f"❌ Error clearing collection: {e}")
        return False


def delete_directory():
    """Delete the entire ChromaDB directory."""
    db_path = Path(CHROMA_DB_PATH)
    
    if not db_path.exists():
        print(f"⚠ Database directory does not exist: {CHROMA_DB_PATH}")
        return True
    
    try:
        print(f"Deleting database directory: {CHROMA_DB_PATH}")
        shutil.rmtree(db_path)
        print("✓ Database directory deleted successfully")
        
        # Recreate empty directory
        db_path.mkdir(parents=True, exist_ok=True)
        print("✓ Empty database directory recreated")
        
        return True
    except Exception as e:
        print(f"❌ Error deleting directory: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clear ChromaDB data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_chroma_db.py           # Clear collection only (recommended)
  python clear_chroma_db.py --full    # Delete entire database directory
        """
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Delete entire database directory (more thorough, but slower)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ChromaDB Data Clearing Script")
    print("=" * 60)
    print()
    
    # Confirmation prompt
    if not args.force:
        action = "delete the entire database directory" if args.full else "clear the collection"
        response = input(f"Are you sure you want to {action}? (yes/no): ").strip().lower()
        
        if response not in ["yes", "y"]:
            print("Operation cancelled.")
            sys.exit(0)
        print()
    
    # Clear or delete
    if args.full:
        success = delete_directory()
    else:
        success = clear_collection()
    
    print()
    print("=" * 60)
    if success:
        print("✓ ChromaDB cleared successfully!")
    else:
        print("❌ Failed to clear ChromaDB")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

