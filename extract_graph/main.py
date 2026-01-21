"""
Application Entry Point - REST API and CLI
==========================================

This module is the main entry point for the Hybrid RAG application,
providing both a REST API (Flask) and a Command-Line Interface (CLI).

What It Does
------------
- Configures logging with rotating file handlers
- Provides REST endpoints for database init, file ingestion, and vectorization
- Provides CLI commands for the same operations
- Orchestrates calls to service layer components

Logging Configuration
---------------------
Uses Python's logging.config with:
- Rotating file handler (10MB files, 5 backups)
- Console (stdout) handler
- Configurable log level via LOG_LEVEL environment variable

REST API Endpoints
------------------
GET  /init-db           Initialize database schema and extensions
GET  /vectorize         Process unvectorized documents in the database
POST /readfiles/<folder> Read .txt files from specified folder into database

CLI Commands
------------
--init-db     Initialize the database schema
--vectorize   Vectorize all unprocessed documents
--folder PATH Read text files from the specified folder

Example Usage
-------------
    # Start REST API server
    $ python -m extract_graph.main

    # Or use CLI commands
    $ extract-graph --init-db
    $ extract-graph --folder ./data/transcripts
    $ extract-graph --vectorize

Environment Variables
---------------------
LOG_LEVEL : str
    Logging level (DEBUG, INFO, WARN, ERROR, CRITICAL). Default: INFO

See Also
--------
- extract_graph.db.init_database : Database initialization logic
- extract_graph.ingest.vectorizer : Document vectorization pipeline
- extract_graph.readers.filereader : File ingestion utilities
"""

import logging
import logging.config
import os
import sys
from dotenv import load_dotenv
from datetime import date
from flask import Flask
from flask_restful import abort, Api, Resource

# Load environment variables from .env file before any other initialization.
# This ensures all configuration is available to subsequent imports.
load_dotenv()

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Create logs directory if it doesn't exist.
# Using exist_ok=True makes this operation idempotent (safe to call multiple times).
os.makedirs("./logs", exist_ok=True)

# Global logging configuration using dictConfig for flexibility.
# This approach allows:
# - Multiple handlers (console + file) with different levels
# - Rotating file logs to prevent disk space issues
# - Consistent formatting across all loggers
log_config = {
    "version": 1,
    "disable_existing_loggers": False,  # Don't disable loggers from imported modules
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        # Console handler for immediate feedback during development/debugging
        "default": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        },
        # Rotating file handler for production logging and audit trails
        # Max 10MB per file, keeps 5 backup files (50MB total max)
        "file": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"./logs/app_{date.today().isoformat()}.log",
            "maxBytes": 10485760,  # 10MB per file
            "backupCount": 5,  # Keep 5 backups
            "formatter": "standard",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
}

# Apply logging configuration before importing modules that may log during import.
# This ensures consistent logging behavior from the start.
logging.config.dictConfig(log_config)

# ---------------------------------------------------------------------------
# Service Layer Imports (after logging is configured)
# ---------------------------------------------------------------------------
import argparse
from extract_graph.ingest.vectorizer import Vectorizer
from extract_graph.db.init_database import init_database
from extract_graph.readers.filereader import read_txt_files_to_database

# ---------------------------------------------------------------------------
# Flask Application Setup
# ---------------------------------------------------------------------------
# Create Flask application instance.
# Flask-RESTful provides a cleaner Resource-based API abstraction.
app = Flask(__name__)
api = Api(app)


# ---------------------------------------------------------------------------
# REST API Resources
# ---------------------------------------------------------------------------
# Each Resource class follows the Single Responsibility Principle (SRP),
# handling one specific operation. Error handling is consistent across all
# resources: log the error details internally, return user-friendly message externally.


class InitDB(Resource):
    """
    REST Resource for database initialization.

    This endpoint triggers the creation of database schema and extensions
    required for the hybrid RAG system (pgvector, Apache AGE, tables).

    HTTP Methods
    ------------
    GET /init-db
        Initialize or reinitialize the database schema.
        Idempotent: safe to call multiple times.

    Returns
    -------
    tuple
        ("OK", 200) on success
        Aborts with 500 on database initialization failure

    Example
    -------
        curl http://localhost:5000/init-db

    Notes
    -----
    This operation is idempotent due to IF NOT EXISTS clauses in schema SQL.
    Safe to call on an already-initialized database.
    """

    def get(self):
        try:
            init_database()
            return "OK", 200
        except Exception as e:
            # Log full error details for debugging, return generic message to client
            logging.getLogger(__name__).error(f"Error initializing database: {e}")
            abort(500, message="Error initializing database")


class ReadFiles(Resource):
    """
    REST Resource for file ingestion.

    This endpoint reads text files from a specified folder and stores their
    contents in the documents table for subsequent vectorization.

    HTTP Methods
    ------------
    POST /readfiles/<folder>
        Read all .txt files from the specified folder (recursively).
        Files are stored with processed=False, awaiting vectorization.

    Parameters
    ----------
    folder : str
        Path to the folder containing .txt files to ingest.
        Can be absolute or relative path.

    Returns
    -------
    tuple
        ("OK", 200) on success
        Aborts with 400 if folder is invalid or inaccessible

    Example
    -------
        curl -X POST http://localhost:5000/readfiles/data/transcripts

    Notes
    -----
    Only .txt files are processed. Binary files and other formats are ignored.
    Folder is scanned recursively, so nested .txt files are also ingested.
    """

    def post(self, folder):
        try:
            read_txt_files_to_database(folder)
            return "OK", 200
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error reading folder {folder}. Error: {e}"
            )
            abort(400, message=f"Invalid folder provided: {folder}")


class Vectorize(Resource):
    """
    REST Resource for document vectorization.

    This endpoint triggers vectorization of all unprocessed documents in the
    database. Documents are converted to vector embeddings using OpenAI's
    text-embedding-3-small model and stored in pgvector for similarity search.

    HTTP Methods
    ------------
    GET /vectorize
        Process all documents where processed=False.
        Updates processed flag and processed_at timestamp on completion.

    Returns
    -------
    tuple
        ("OK", 200) on success
        Aborts with 500 on vectorization failure

    Example
    -------
        curl http://localhost:5000/vectorize

    Notes
    -----
    Requires OPENAI_API_KEY environment variable to be set.
    This operation can be slow for large document sets due to API calls.
    Consider running in background for production workloads.

    See Also
    --------
    extract_graph.ingest.vectorizer.Vectorizer : The vectorization implementation
    """

    def get(self):
        try:
            Vectorizer().ingest()
            return "OK", 200
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during vectorization: {e}")
            abort(500, message="Error during vectorization")


# ---------------------------------------------------------------------------
# Route Registration
# ---------------------------------------------------------------------------
# Register REST resources with their URL endpoints.
# Flask-RESTful handles HTTP method routing automatically based on
# method names in the Resource classes (get, post, put, delete, etc.).
api.add_resource(InitDB, "/init-db")
api.add_resource(Vectorize, "/vectorize")
api.add_resource(ReadFiles, "/readfiles/<folder>")


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------


def main():
    """
    CLI entry point for the Hybrid RAG application.

    Provides command-line access to the core operations:
    database initialization, file ingestion, vectorization, entity extraction,
    and database content display.

    This function is registered as 'extract-graph' console script
    in pyproject.toml, allowing direct invocation from the shell.

    Arguments
    ---------
    --init-db : flag
        Initialize database schema and extensions, then exit.
    --folder PATH : str
        Read .txt files from PATH recursively, store in database, then exit.
    --vectorize : flag
        Process all unvectorized documents in database, then exit.
    --extract : flag
        Extract entities from processed documents and build knowledge graph.
    --show : flag
        Display sample data from documents, vectors, and graph.

    Returns
    -------
    None
        Exits after completing the requested operation.

    Examples
    --------
    Initialize a fresh database:
        $ extract-graph --init-db

    Ingest interview transcripts:
        $ extract-graph --folder ./data/transcripts

    Create vector embeddings for ingested documents:
        $ extract-graph --vectorize

    Extract entities and build knowledge graph:
        $ extract-graph --extract

    Display database contents:
        $ extract-graph --show

    Complete workflow:
        $ extract-graph --init-db
        $ extract-graph --folder ./data/transcripts
        $ extract-graph --vectorize
        $ extract-graph --extract
        $ extract-graph --show

    Notes
    -----
    Operations are mutually exclusive - only one operation per invocation.
    For multiple operations, run the command multiple times or use the REST API.

    See Also
    --------
    InitDB : REST endpoint equivalent for --init-db
    ReadFiles : REST endpoint equivalent for --folder
    Vectorize : REST endpoint equivalent for --vectorize
    """
    parser = argparse.ArgumentParser(
        description="Analyze texts to extract vectors and structured data, "
        "build a graph and use it for Hybrid RAG"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="the folder containing the .txt files to be read. "
        "It can read subfolders. Only .txt files will be picked up.",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        required=False,
        help="Initialize the database schema and return",
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="Find all unprocessed documents in the db and vectorize them",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract entities from processed documents and build knowledge graph",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show sample data from documents, vectors, and graph",
    )
    args = parser.parse_args()

    # Execute the requested operation.
    # Operations are mutually exclusive - first matching argument wins.
    if args.init_db:
        init_database()
        return

    if args.vectorize:
        Vectorizer().ingest()
        return

    if args.extract:
        from extract_graph.ingest.graph_extractor import GraphExtractor
        GraphExtractor().extract_and_store()
        return

    if args.show:
        from extract_graph.display.db_display import show_database_contents
        show_database_contents()
        return

    if args.folder:
        read_txt_files_to_database(args.folder)
        return


# ---------------------------------------------------------------------------
# Module Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dual-mode execution: CLI with arguments, or Flask server without.
    # This allows the same module to serve both development (server) and
    # production (CLI) use cases without separate entry points.
    if len(sys.argv) > 1:
        main()
    else:
        # Development server - do not use in production.
        # For production, use a proper WSGI server like gunicorn:
        #   gunicorn extract_graph.main:app
        app.run(debug=True)
