"""
Hybrid RAG Agent - Extract Graph Package
=========================================

This package implements a Hybrid Retrieval-Augmented Generation (RAG) system that
combines vector similarity search with graph-based knowledge retrieval.

What This Package Does
----------------------
- Ingests text files and stores them in PostgreSQL
- Creates vector embeddings using OpenAI and stores them in pgvector
- Extracts structured entities from text using LLMs (GPT-4o or Gemini)
- Stores entity relationships in an Apache AGE knowledge graph
- Enables hybrid queries combining semantic search and graph traversal

Modules
-------
- `db`: Database connectivity and operations (PostgreSQL, pgvector, AGE)
- `readers`: File ingestion utilities for loading text documents
- `ingest`: Vectorization pipeline using LlamaIndex
- `extractors`: LLM-based entity extraction using langextract

Example Usage
-------------
    # CLI usage
    $ extract-graph --init-db                    # Initialize database schema
    $ extract-graph --folder ./transcripts       # Ingest text files
    $ extract-graph --vectorize                  # Create vector embeddings

    # Programmatic usage
    from extract_graph.db.PostgresConnection import PostgresConnection
    from extract_graph.ingest.vectorizer import Vectorizer

    with PostgresConnection() as db:
        results = db.similarity_search("embeddings", query_vector, limit=5)

See Also
--------
- README.md for installation and configuration instructions
- db/migrations/ for database schema definitions
"""

__version__ = "1.0.0"
__author__ = "Gustavo Herrera"
__email__ = "gustavo@knowmeglobal.com"
