# Hybrid RAG Agent - Extract Graph

An example demonstrating how to build a **Hybrid Retrieval-Augmented Generation (RAG)** system that combines **vector similarity search** with **graph-based knowledge retrieval**.

## Overview

This project shows how to:
1. **Receive text files** (e.g., interview transcripts, conversations)
2. **Vectorize them** using OpenAI embeddings and pgvector
3. **Extract structured information** from unstructured text using LLMs
4. **Build a knowledge graph** with Apache AGE
5. **Query using Hybrid RAG** combining semantic search and graph traversal

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                 │
│  main.py - REST API (Flask) and CLI entry points                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────────┐
│                          SERVICE LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────────┐ │
│  │   readers/   │  │   ingest/    │  │         extractors/                │ │
│  │ filereader   │  │  vectorizer  │  │  langextractor (base)              │ │
│  │              │  │              │  │  extracttechnologies (specialized) │ │
│  └──────────────┘  └──────────────┘  └────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────────┐
│                       DATA ACCESS LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                 db/PostgresConnection + VectorStoreFactory              ││
│  │  • Connection pooling (psycopg2)                                        ││
│  │  • Vector operations (pgvector + LlamaIndex)                            ││
│  │  • Graph operations (Apache AGE + Cypher)                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                                   │
│  PostgreSQL 16 with extensions: pgvector, Apache AGE                        │
└─────────────────────────────────────────────────────────────────────────────┘
```


## Project Structure

```
hybrid-agent/
├── extract_graph/
│   ├── __init__.py              # Package documentation and version
│   ├── main.py                  # Flask REST API and CLI entry point
│   ├── db/
│   │   ├── PostgresConnection.py # Database abstraction (pooling, SQL, Cypher)
│   │   ├── vector_store.py       # VectorStoreFactory (SRP-compliant)
│   │   ├── init_database.py      # Schema migration runner
│   │   └── migrations/
│   │       └── 001_initial_schema.sql
│   ├── readers/
│   │   └── filereader.py        # Text file ingestion
│   ├── ingest/
│   │   └── vectorizer.py        # Document vectorization pipeline
│   └── extractors/
│       ├── langextractor.py      # Base LLM entity extractor
│       └── extracttechnologies.py # Work history extractor
├── pyproject.toml               # Python package configuration
├── docker-compose.yaml          # PostgreSQL + AGE + pgvector
├── Dockerfile                   # Custom PostgreSQL image
└── .env.template                # Environment variables template
```

## Installation

### 1. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install Apache AGE Python client (special handling required)
pip install apache-age-python --no-deps
```

> **Note**: `apache-age-python` is installed with `--no-deps` because it tries to compile psycopg2 from source. We use `psycopg2-binary` instead.

### 2. Install Development Dependencies (Optional)

```bash
pip install -e ".[dev]"
```

### 3. Configure Environment Variables

```bash
cp .env.template .env
# Edit .env with your values
```

Required environment variables:
```env
OPENAI_API_KEY=sk-...           # For embeddings and GPT-4o extraction
LANGEXTRACT_API_KEY=AI...       # For Gemini extraction (optional)

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=graph_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

LOG_LEVEL=INFO                  # DEBUG for verbose output
```

### 4. Start the Database

```bash
# Build the custom PostgreSQL image with pgvector and AGE
docker-compose build --no-cache

# Start the database
docker-compose up -d
```

### 5. Initialize the Schema

```bash
extract-graph --init-db
```

## Usage

### CLI Commands

```bash
# Initialize database schema
extract-graph --init-db

# Ingest text files from a folder (recursive)
extract-graph --folder ./data/transcripts

# Vectorize all unprocessed documents
extract-graph --vectorize
```

# Extract graph data
```
extract-graph --extract
```

# Show results
```
extract-graph --show
```

### REST API

Start the server:
```bash
python -m extract_graph.main
```

Endpoints:
- `GET /init-db` - Initialize database schema
- `POST /readfiles/<folder>` - Ingest text files from folder
- `GET /vectorize` - Vectorize unprocessed documents

### Programmatic Usage

```python
from extract_graph.db.PostgresConnection import PostgresConnection
from extract_graph.db.vector_store import VectorStoreFactory
from extract_graph.ingest.vectorizer import Vectorizer
from extract_graph.readers.filereader import read_txt_files_to_database
from extract_graph.extractors.extracttechnologies import ExtractorFocusOnWork

# Ingest documents
read_txt_files_to_database("./transcripts")

# Vectorize
Vectorizer().ingest()

# Query with SQL
with PostgresConnection() as db:
    docs = db.execute_query(
        "SELECT * FROM documents WHERE processed = true",
        dict_cursor=True
    )

# Query the graph
with PostgresConnection() as db:
    results = db.execute_cypher(
        "knowledge_graph",
        "MATCH (p:Person)-[:WORKED_AT]->(c:Company) RETURN p.name, c.name"
    )

# Similarity search
with PostgresConnection() as db:
    similar = db.similarity_search("embeddings", query_vector, limit=5)

# Extract entities from text
extractor = ExtractorFocusOnWork("OPENAI")
extractor.setInputText(transcript)
result = extractor.extract()
```

## Data Flow

```
┌─────────────────────────┐
│  Text Files (.txt)      │
│  in Folder Structure    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  File Reader (filereader.py)            │
│  - Scan recursively for .txt files      │
│  - Store in 'documents' table           │
└────────────┬────────────────────────────┘
             │
             ├─────────────────────────────────┐
             │                                 │
             ▼                                 ▼
    ┌──────────────────────┐      ┌─────────────────────────┐
    │  VECTORIZATION PATH  │      │  EXTRACTION PATH        │
    │  (vectorizer.py)     │      │  (langextractor.py)     │
    └──────────┬───────────┘      └────────────┬────────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │  LLM Processing                                          │
    │  - OpenAI text-embedding-3-small (vectors)               │
    │  - GPT-4o or Gemini 2.5 Pro (extraction)                 │
    └──────────┬───────────────────────────────────────────────┘
               │
    ┌──────────┴──────────────────────────────┐
    │                                         │
    ▼                                         ▼
┌──────────────────────────┐    ┌──────────────────────┐
│ PostgreSQL pgvector      │    │ Apache AGE Graph     │
│ - embeddings table       │    │ - knowledge_graph    │
│ - 1536-dim vectors       │    │ - Nodes & Edges      │
│ - HNSW index             │    │ - Cypher queries     │
└──────────────────────────┘    └──────────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
                    ┌────────────────────┐
                    │  Hybrid RAG Agent  │
                    │  - Vector Search   │
                    │  - Graph Queries   │
                    │  - Combined QA     │
                    └────────────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Store** | pgvector + LlamaIndex | Semantic similarity search |
| **Graph Database** | Apache AGE | Knowledge graph storage |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim vectors |
| **LLM Extraction** | GPT-4o / Gemini 2.5 Pro | Entity extraction |
| **Database** | PostgreSQL 16 | Unified storage |
| **Web Framework** | Flask + Flask-RESTful | REST API |
| **Connection Pool** | psycopg2 ThreadedConnectionPool | Efficient DB access |

## Configuration

### HNSW Index Parameters

The vector store uses HNSW (Hierarchical Navigable Small World) indexing:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hnsw_m` | 16 | Bi-directional links per node |
| `hnsw_ef_construction` | 64 | Build-time quality |
| `hnsw_ef_search` | 40 | Query-time quality |

Customize via `VectorStoreFactory.create()`:
```python
vector_store = VectorStoreFactory.create(
    hnsw_m=32,
    hnsw_ef_construction=128,
    hnsw_ef_search=100
)
```

### Connection Pool

Default pool size: 1-10 connections. Customize:
```python
db = PostgresConnection(min_connections=5, max_connections=20)
```

## Extending the System

### Creating Custom Extractors

```python
from extract_graph.extractors.langextractor import LangExtractor
import langextract as lx

class MyCustomExtractor(LangExtractor):
    def __init__(self, model: str):
        super().__init__(model)
        self.prompt = "Extract product names and prices..."
        self.examples = [
            lx.data.ExampleData(
                text="The iPhone costs $999.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="product",
                        extraction_text="iPhone"
                    ),
                    lx.data.Extraction(
                        extraction_class="price",
                        extraction_text="$999"
                    ),
                ]
            )
        ]
```

### Adding Database Migrations

Create a new file in `extract_graph/db/migrations/`:
```sql
-- 002_add_indexes.sql
CREATE INDEX IF NOT EXISTS idx_documents_processed
ON documents(processed);
```

Run migrations:
```bash
extract-graph --init-db
```

## License

LGPL - See LICENSE file for details.

## Author

Gustavo Herrera - gustavo@knowmeglobal.com
