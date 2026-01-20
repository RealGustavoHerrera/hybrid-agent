"""
Vector Store Factory
====================

This module creates configured PGVectorStore instances for LlamaIndex vector
operations. It handles embedding model configuration and HNSW index parameters.

What It Does
------------
- Configures OpenAI embedding model (text-embedding-3-small by default)
- Creates PGVectorStore with connection to PostgreSQL
- Sets up HNSW index parameters for similarity search

Configuration
-------------
Default HNSW parameters are optimized for balanced recall/performance:
- hnsw_m=16: Number of bi-directional links per node
- hnsw_ef_construction=64: Build-time quality parameter
- hnsw_ef_search=40: Query-time quality parameter
- vector_cosine_ops: Cosine similarity distance metric

Environment Variables
---------------------
POSTGRES_HOST : str
    Database host
POSTGRES_PORT : int
    Database port
POSTGRES_DB : str
    Database name
POSTGRES_USER : str
    Database user
POSTGRES_PASSWORD : str
    Database password
OPENAI_API_KEY : str
    OpenAI API key for embedding model
LOG_LEVEL : str
    If "DEBUG", enables SQL logging

Example Usage
-------------
    from extract_graph.db.vector_store import VectorStoreFactory

    # Create with defaults
    vector_store = VectorStoreFactory.create()

    # Create with custom embedding model
    vector_store = VectorStoreFactory.create(
        embed_model="text-embedding-3-large",
        embed_dim=3072
    )

    # Use with LlamaIndex
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

See Also
--------
- extract_graph.db.PostgresConnection : Database connection management
- extract_graph.ingest.vectorizer : Uses this factory for vectorization
"""

import os
import logging
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings


class VectorStoreFactory:
    """
    Factory for creating configured PGVectorStore instances.

    This factory encapsulates all vector store configuration including:
    - OpenAI embedding model setup
    - PostgreSQL connection parameters
    - HNSW index configuration

    The factory creates PGVectorStore instances that manage their own
    database connections, independent of PostgresConnection instances.

    Class Attributes
    ----------------
    DEFAULT_EMBED_MODEL : str
        Default OpenAI embedding model (text-embedding-3-small)
    DEFAULT_EMBED_DIM : int
        Default embedding dimension (1536)
    DEFAULT_TABLE_NAME : str
        Default table name for embeddings

    Examples
    --------
    Create vector store with defaults:

        >>> vector_store = VectorStoreFactory.create()

    Create with custom configuration:

        >>> vector_store = VectorStoreFactory.create(
        ...     embed_model="text-embedding-3-large",
        ...     embed_dim=3072,
        ...     table_name="custom_embeddings"
        ... )

    Notes
    -----
    The factory sets the global `Settings.embed_model` which affects
    all LlamaIndex operations in the current process. This is a
    LlamaIndex design constraint.
    """

    DEFAULT_EMBED_MODEL = "text-embedding-3-small"
    DEFAULT_EMBED_DIM = 1536
    DEFAULT_TABLE_NAME = "embeddings"

    @staticmethod
    def create(
        embed_model: str = None,
        embed_dim: int = None,
        table_name: str = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 40,
    ) -> PGVectorStore:
        """
        Create a configured PGVectorStore instance.

        Parameters
        ----------
        embed_model : str, optional
            OpenAI embedding model name. Default: text-embedding-3-small
        embed_dim : int, optional
            Embedding dimension. Must match the model's output dimension.
            Default: 1536 (for text-embedding-3-small)
        table_name : str, optional
            PostgreSQL table name for storing embeddings.
            Default: "embeddings"
        hnsw_m : int, default=16
            HNSW parameter: number of bi-directional links per node.
            Higher values improve recall but increase memory usage.
        hnsw_ef_construction : int, default=64
            HNSW parameter: size of dynamic candidate list during index build.
            Higher values improve index quality but slow down construction.
        hnsw_ef_search : int, default=40
            HNSW parameter: size of dynamic candidate list during search.
            Higher values improve recall but slow down queries.

        Returns
        -------
        PGVectorStore
            Configured vector store ready for use with LlamaIndex.

        Raises
        ------
        ValueError
            If OPENAI_API_KEY environment variable is not set.

        Examples
        --------
        Basic usage:

            >>> store = VectorStoreFactory.create()
            >>> storage_ctx = StorageContext.from_defaults(vector_store=store)

        High-quality configuration (slower, better recall):

            >>> store = VectorStoreFactory.create(
            ...     hnsw_m=32,
            ...     hnsw_ef_construction=128,
            ...     hnsw_ef_search=100
            ... )

        Notes
        -----
        HNSW Parameter Guidelines:
            - For development/testing: defaults are fine
            - For production with <1M vectors: m=16, ef_construction=64
            - For production with >1M vectors: m=32, ef_construction=128
            - For high recall requirements: increase ef_search (50-200)
        """
        logger = logging.getLogger(__name__)

        # Apply defaults
        embed_model = embed_model or VectorStoreFactory.DEFAULT_EMBED_MODEL
        embed_dim = embed_dim or VectorStoreFactory.DEFAULT_EMBED_DIM
        table_name = table_name or VectorStoreFactory.DEFAULT_TABLE_NAME

        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Configure the global embedding model for LlamaIndex.
        # This is a LlamaIndex design pattern - the Settings object
        # provides default values for all LlamaIndex operations.
        Settings.embed_model = OpenAIEmbedding(model=embed_model, api_key=api_key)
        logger.debug(f"Configured embedding model: {embed_model}")

        # Create PGVectorStore with HNSW index configuration.
        # PGVectorStore manages its own connection pool internally,
        # separate from our PostgresConnection class.
        vector_store = PGVectorStore.from_params(
            database=os.getenv("POSTGRES_DB"),
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            table_name=table_name,
            embed_dim=embed_dim,
            hnsw_kwargs={
                "hnsw_m": hnsw_m,
                "hnsw_ef_construction": hnsw_ef_construction,
                "hnsw_ef_search": hnsw_ef_search,
                "hnsw_dist_method": "vector_cosine_ops",
            },
            perform_setup=True,  # Auto-create table and index if not exists
            debug=(os.getenv("LOG_LEVEL") == "DEBUG"),
        )

        logger.debug(
            f"Created PGVectorStore: table={table_name}, dim={embed_dim}, "
            f"hnsw_m={hnsw_m}, ef_construction={hnsw_ef_construction}"
        )

        return vector_store
