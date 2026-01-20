"""
Document Vectorization Service
==============================

This module converts text documents into vector embeddings and stores them
in pgvector for semantic similarity search.

What It Does
------------
- Fetches unprocessed documents from the database
- Generates embeddings using OpenAI's text-embedding-3-small model
- Stores vectors in pgvector with HNSW indexing
- Marks documents as processed to prevent re-vectorization

Data Flow
---------
1. Query unprocessed documents from PostgreSQL
2. Convert each document to LlamaIndex Document with metadata
3. Generate embeddings via OpenAI API (text-embedding-3-small)
4. Store embeddings in pgvector with HNSW indexing
5. Mark source documents as processed with timestamp

Environment Variables
---------------------
OPENAI_API_KEY : str
    Required for generating embeddings via OpenAI API.

Example Usage
-------------
    # Programmatic usage
    from extract_graph.ingest.vectorizer import Vectorizer

    vectorizer = Vectorizer()
    vectorizer.ingest()

    # CLI usage
    $ extract-graph --vectorize

See Also
--------
- extract_graph.db.PostgresConnection : Database operations
- extract_graph.db.vector_store.VectorStoreFactory : Vector store creation
- extract_graph.readers.filereader : Document ingestion
- llama_index.core.VectorStoreIndex : Vector index creation
"""

import logging
from extract_graph.db.PostgresConnection import PostgresConnection
from extract_graph.db.vector_store import VectorStoreFactory
from llama_index.core import Document, VectorStoreIndex, StorageContext
from datetime import date


class Vectorizer:
    """
    Service for vectorizing documents and storing embeddings.

    The Vectorizer processes unprocessed documents from the database,
    generates vector embeddings using LlamaIndex and OpenAI, and stores
    them in pgvector for efficient similarity search.

    This class follows the Service pattern, encapsulating the business
    logic for document vectorization while delegating data access to
    the PostgresConnection repository.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for this class.

    Examples
    --------
    Basic usage:

        >>> vectorizer = Vectorizer()
        >>> vectorizer.ingest()  # Process all unprocessed documents

    Integration with file ingestion:

        >>> from extract_graph.readers.filereader import read_txt_files_to_database
        >>> read_txt_files_to_database("./transcripts")  # Ingest files
        >>> Vectorizer().ingest()  # Then vectorize them

    Notes
    -----
    Processing Flow:
        1. Fetches documents where processed=False
        2. Creates LlamaIndex Document objects with metadata
        3. Generates embeddings via VectorStoreIndex.from_documents()
        4. Updates documents table to mark as processed

    The vectorization uses OpenAI's text-embedding-3-small model (1536 dims)
    as configured in VectorStoreFactory. HNSW indexing is used for
    efficient approximate nearest neighbor search.

    See Also
    --------
    PostgresConnection : Handles database operations
    VectorStoreFactory : Creates configured PGVectorStore instances
    VectorStoreIndex : LlamaIndex class for vector index management
    """

    def __init__(self):
        """
        Initialize the Vectorizer service.

        Sets up logging for the vectorization process.
        Database connections and vector stores are created on-demand during ingest().
        """
        self.logger = logging.getLogger(__name__)

    def ingest(self):
        """
        Process and vectorize all unprocessed documents.

        This method performs the complete vectorization pipeline:
        1. Creates vector store (independent of database connection lifecycle)
        2. Queries documents where processed=False
        3. Converts them to LlamaIndex Document format with metadata
        4. Generates embeddings and creates vector index
        5. Marks documents as processed in the database

        The method is idempotent - running it multiple times will only
        process documents that haven't been processed yet.

        Raises
        ------
        ValueError
            If OPENAI_API_KEY is not set.
        openai.error.AuthenticationError
            If OPENAI_API_KEY is invalid.
        psycopg2.OperationalError
            If database connection fails.

        Examples
        --------
        Process pending documents:

            >>> vectorizer = Vectorizer()
            >>> vectorizer.ingest()
            INFO - found and created 5
            INFO - Index creation completed.
            INFO - Vectorized and stored 5 documents.

        Notes
        -----
        - Vector store is created first (manages its own connection)
        - Documents are fetched with a managed connection
        - Document status updates use a separate managed connection
        - Each step is logged for monitoring and debugging

        The metadata preserved in vectors includes:
        - filename: Original source file name
        - orig_id: Database ID for traceability

        This metadata enables tracking vector results back to source
        documents, which is essential for RAG applications.
        """
        # -----------------------------------------------------------------
        # Step 1: Create vector store (independent of db connection lifecycle)
        # -----------------------------------------------------------------
        # VectorStoreFactory creates a PGVectorStore that manages its own
        # database connection. This is created BEFORE the PostgresConnection
        # context manager to avoid the bug where db.vector_store was accessed
        # after the connection was closed.
        vector_store = VectorStoreFactory.create()

        # -----------------------------------------------------------------
        # Step 2: Fetch unprocessed documents from database
        # -----------------------------------------------------------------
        # Use context manager for automatic connection cleanup.
        # Only documents with processed=False are selected.
        docs = []
        with PostgresConnection() as db:
            rows = db.execute_query(
                "SELECT id, filename, content FROM documents WHERE processed = false"
            )
            # Convert database rows to LlamaIndex Document objects.
            # Metadata is preserved for traceability from vector results
            # back to source documents.
            for row in rows:
                doc_id, filename, content = row
                doc = Document(
                    text=content, metadata={"filename": filename, "orig_id": doc_id}
                )
                docs.append(doc)

        self.logger.info(f"Found {len(docs)} unprocessed documents")

        if not docs:
            self.logger.info("No documents to process")
            return

        # -----------------------------------------------------------------
        # Step 3: Generate embeddings and create vector index
        # -----------------------------------------------------------------
        # VectorStoreIndex.from_documents() handles:
        # - Chunking documents (if needed)
        # - Calling OpenAI API for embeddings
        # - Storing vectors in pgvector
        # - Creating HNSW index for similarity search
        #
        # Note: vector_store is safe to use here because it manages its own
        # connection, independent of the PostgresConnection above.
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, show_progress=True
        )

        self.logger.info("Index creation completed.")

        # -----------------------------------------------------------------
        # Step 4: Mark documents as processed
        # -----------------------------------------------------------------
        # Update each document's processed flag and timestamp.
        # This prevents re-processing on subsequent ingest() calls.
        with PostgresConnection() as db:
            for doc in docs:
                db.execute_query(
                    "UPDATE documents SET processed = true, processed_at = %s WHERE id = %s",
                    (date.today(), doc.metadata["orig_id"]),
                    fetch=False,
                )

        self.logger.info(f"Vectorized and stored {len(docs)} documents.")
