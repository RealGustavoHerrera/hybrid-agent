"""
PostgreSQL Connection Manager with Graph Support
================================================

This module provides database connectivity for PostgreSQL with pgvector
(vector similarity search) and Apache AGE (graph database) extensions.

What It Does
------------
- Manages a thread-safe connection pool using psycopg2
- Executes SQL queries with parameterized inputs
- Executes Cypher queries against Apache AGE graphs
- Performs vector similarity searches using pgvector operators
- Provides context manager support for automatic cleanup

Operations Available
--------------------
- SQL: execute_query() for standard SQL operations
- Graph: execute_cypher(), create_graph(), drop_graph() for Apache AGE
- Vector: insert_vector(), similarity_search() for pgvector

Note: For LlamaIndex vector store operations, use VectorStoreFactory instead.

Graph Database Configuration
----------------------------
- **Extension**: Apache AGE (A Graph Extension for PostgreSQL)
- **Query Language**: Cypher (property graph query language)
- **Graph Name**: Configurable, default 'knowledge_graph'

Environment Variables Required
------------------------------
POSTGRES_HOST : str
    Database server hostname
POSTGRES_PORT : int
    Database server port (typically 5432)
POSTGRES_DB : str
    Database name
POSTGRES_USER : str
    Database user
POSTGRES_PASSWORD : str
    Database password

Example Usage
-------------
    # Using context manager (recommended)
    with PostgresConnection() as db:
        # Execute SQL query
        results = db.execute_query("SELECT * FROM documents WHERE processed = false")

        # Execute Cypher query on graph
        nodes = db.execute_cypher("knowledge_graph",
                                   "MATCH (n:Person) RETURN n LIMIT 10")

        # Perform similarity search
        similar = db.similarity_search("embeddings", query_vector, limit=5)

    # Manual connection management
    db = PostgresConnection()
    try:
        db.execute_query("INSERT INTO documents ...")
    finally:
        db.close()

See Also
--------
- extract_graph.db.vector_store.VectorStoreFactory : Vector store creation
- psycopg2.pool.ThreadedConnectionPool : Connection pooling
- Apache AGE documentation : https://age.apache.org/

Notes
-----
Thread Safety:
    This class uses ThreadedConnectionPool, making it safe for multi-threaded
    applications. Each thread gets its own connection from the pool.

Connection Lifecycle:
    Connections are automatically returned to the pool after each operation
    when using context managers. Manual management requires explicit close().
"""

import os
import logging
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pgvector.psycopg2 import register_vector


class PostgresConnection:
    """
    Connection manager for PostgreSQL with Apache AGE and pgvector support.

    This class provides a unified interface to PostgreSQL with extensions for:
    - Standard SQL operations with connection pooling
    - Vector similarity search via pgvector (low-level operations)
    - Graph database operations via Apache AGE (Cypher queries)

    The class implements the context manager protocol for safe resource management
    and uses connection pooling for efficient multi-threaded access.

    Note: For LlamaIndex vector store operations, use VectorStoreFactory instead.
    This class focuses on connection management and direct SQL/Cypher operations.

    Attributes
    ----------
    pool : psycopg2.pool.ThreadedConnectionPool
        Thread-safe connection pool for database access.
    logger : logging.Logger
        Logger instance for this class.

    Parameters
    ----------
    min_connections : int, default=1
        Minimum number of connections to maintain in the pool.
    max_connections : int, default=10
        Maximum number of connections allowed in the pool.

    Raises
    ------
    psycopg2.OperationalError
        If unable to connect to the database.
    ValueError
        If required environment variables are not set.

    Examples
    --------
    Basic usage with context manager:

        >>> with PostgresConnection() as db:
        ...     results = db.execute_query("SELECT COUNT(*) FROM documents")
        ...     print(f"Document count: {results[0][0]}")

    Custom pool size for high-concurrency scenarios:

        >>> db = PostgresConnection(min_connections=5, max_connections=20)

    See Also
    --------
    VectorStoreFactory : For creating LlamaIndex PGVectorStore instances
    """

    def __init__(self, min_connections: int = 1, max_connections: int = 10):

        self.logger = logging.getLogger(__name__)

        # -----------------------------------------------------------------------
        # Connection Pool Setup
        # -----------------------------------------------------------------------
        # ThreadedConnectionPool is chosen for thread-safety in multi-threaded
        # environments (Flask with threading, concurrent API calls, etc.).
        # The pool manages connection lifecycle, reusing connections to avoid
        # the overhead of establishing new connections for each operation.
        self.pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        self.logger.debug(
            f"connection pool created with min: {min_connections} "
            f"and max: {max_connections} connections"
        )

        # -----------------------------------------------------------------------
        # Extension Initialization
        # -----------------------------------------------------------------------
        # Load Apache AGE and register pgvector types on initial connection.
        # This must happen before any graph or vector operations.
        self._initialize_extensions()

    def _initialize_extensions(self):
        """
        Load AGE extension and register pgvector types.

        This private method is called during initialization to prepare
        the database connection for graph and vector operations.

        The method:
        1. Loads the Apache AGE shared library into PostgreSQL
        2. Registers pgvector custom types with psycopg2

        Notes
        -----
        AGE must be loaded in each session with LOAD 'age' before
        executing Cypher queries. This is a PostgreSQL requirement
        for dynamically loaded extensions.

        pgvector type registration allows psycopg2 to properly handle
        VECTOR column types in query results.
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Load Apache AGE shared library for Cypher query support.
                # This must be done in each PostgreSQL session.
                cur.execute("LOAD 'age';")
                conn.commit()

            # Register pgvector types with psycopg2 so that VECTOR columns
            # are properly serialized/deserialized as Python lists.
            register_vector(conn)
        finally:
            self.logger.debug("Extensions initialized")
            self.pool.putconn(conn)

    @contextmanager
    def get_connection(self):
        """
        Context manager for acquiring a database connection from the pool.

        Provides a connection that is automatically committed on success
        or rolled back on exception. The connection is returned to the pool
        when the context exits.

        Yields
        ------
        psycopg2.extensions.connection
            A database connection from the pool.

        Raises
        ------
        Exception
            Re-raises any exception after rolling back the transaction.

        Examples
        --------
        Basic usage with manual cursor management:

            >>> with db.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("SELECT * FROM documents")
            ...         results = cur.fetchall()

        Transaction with multiple operations:

            >>> with db.get_connection() as conn:
            ...     with conn.cursor() as cur:
            ...         cur.execute("INSERT INTO documents ...")
            ...         cur.execute("UPDATE metadata SET ...")
            ...     # Both operations committed together

        Notes
        -----
        Connection is committed automatically on successful context exit.
        Any exception triggers a rollback before the exception is re-raised.
        The connection is always returned to the pool, even on error.
        """
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Context manager for acquiring a database cursor with automatic cleanup.

        Provides a higher-level abstraction than get_connection() when only
        cursor operations are needed. Handles connection acquisition, cursor
        creation, and cleanup automatically.

        Parameters
        ----------
        cursor_factory : type, optional
            psycopg2 cursor factory class. Use RealDictCursor for
            dictionary-style results. Default is standard tuple cursor.

        Yields
        ------
        psycopg2.extensions.cursor
            A database cursor configured with the specified factory.

        Examples
        --------
        Standard cursor (returns tuples):

            >>> with db.get_cursor() as cur:
            ...     cur.execute("SELECT id, filename FROM documents")
            ...     for row in cur.fetchall():
            ...         print(row[0], row[1])  # Access by index

        Dictionary cursor (returns dicts):

            >>> with db.get_cursor(cursor_factory=RealDictCursor) as cur:
            ...     cur.execute("SELECT id, filename FROM documents")
            ...     for row in cur.fetchall():
            ...         print(row['id'], row['filename'])  # Access by name

        Notes
        -----
        Cursor is automatically closed on context exit.
        Transaction commit/rollback is handled by the underlying get_connection().
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def execute_query(
        self,
        query: str,
        params: tuple = None,
        fetch: bool = True,
        dict_cursor: bool = False,
    ) -> Optional[List[tuple]]:
        """
        Execute a SQL query with optional result fetching.

        This is the primary method for executing SQL queries against the database.
        Supports parameterized queries (protecting against SQL injection) and
        flexible result format options.

        Parameters
        ----------
        query : str
            SQL query string. Use %s placeholders for parameters.
        params : tuple, optional
            Query parameters to substitute into placeholders.
            Always use this for user-provided values to prevent SQL injection.
        fetch : bool, default=True
            If True, fetch and return query results.
            If False, return None (useful for INSERT/UPDATE/DELETE).
        dict_cursor : bool, default=False
            If True, return results as list of dictionaries.
            If False, return results as list of tuples.

        Returns
        -------
        Optional[List[tuple]] or Optional[List[dict]]
            Query results if fetch=True, None otherwise.
            Result type depends on dict_cursor parameter.

        Examples
        --------
        Select with parameters (tuple results):

            >>> results = db.execute_query(
            ...     "SELECT * FROM documents WHERE processed = %s",
            ...     (False,)
            ... )
            >>> for row in results:
            ...     print(row[0], row[1])  # id, filename

        Select with dictionary results:

            >>> results = db.execute_query(
            ...     "SELECT id, filename FROM documents",
            ...     dict_cursor=True
            ... )
            >>> for row in results:
            ...     print(row['filename'])

        Insert without fetching:

            >>> db.execute_query(
            ...     "INSERT INTO documents (filename, content) VALUES (%s, %s)",
            ...     ("doc.txt", "content here"),
            ...     fetch=False
            ... )

        Notes
        -----
        Always use parameterized queries (%s placeholders with params tuple)
        for any user-provided or dynamic values. Never use string formatting
        or concatenation to build queries with dynamic values.
        """
        cursor_factory = RealDictCursor if dict_cursor else None

        with self.get_cursor(cursor_factory=cursor_factory) as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return None

    def execute_cypher(
        self, graph_name: str, cypher_query: str, dict_cursor: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Apache AGE graph database.

        Cypher is a declarative graph query language that allows for
        expressive and efficient querying and updating of graph data.
        This method wraps AGE's cypher() function for PostgreSQL.

        Parameters
        ----------
        graph_name : str
            Name of the graph to query (e.g., 'knowledge_graph').
            The graph must exist (see create_graph method).
        cypher_query : str
            Cypher query string to execute. Supports all AGE-compatible
            Cypher operations: MATCH, CREATE, MERGE, DELETE, SET, etc.
        dict_cursor : bool, default=True
            If True, return results as list of dictionaries.
            If False, return results as list of tuples.

        Returns
        -------
        List[Dict[str, Any]] or List[tuple]
            Query results. Each row contains an 'agtype' result column.

        Examples
        --------
        Create a node:

            >>> db.execute_cypher(
            ...     "knowledge_graph",
            ...     "CREATE (p:Person {name: 'Alice', role: 'Engineer'}) RETURN p"
            ... )

        Query nodes:

            >>> results = db.execute_cypher(
            ...     "knowledge_graph",
            ...     "MATCH (p:Person) WHERE p.role = 'Engineer' RETURN p.name"
            ... )

        Create relationship:

            >>> db.execute_cypher(
            ...     "knowledge_graph",
            ...     '''
            ...     MATCH (p:Person {name: 'Alice'}), (t:Technology {name: 'Python'})
            ...     CREATE (p)-[:USES]->(t)
            ...     '''
            ... )

        Pattern matching with relationships:

            >>> results = db.execute_cypher(
            ...     "knowledge_graph",
            ...     '''
            ...     MATCH (p:Person)-[:WORKED_AT]->(c:Company)
            ...     RETURN p.name, c.name
            ...     '''
            ... )

        Notes
        -----
        The method temporarily changes the PostgreSQL search_path to include
        ag_catalog (AGE's schema) and resets it after execution. This is
        required for AGE functions to be accessible.

        AGE Cypher queries use $$ delimiters for the query string, allowing
        single and double quotes within the Cypher query itself.

        See Also
        --------
        create_graph : Create a new graph before querying
        drop_graph : Remove a graph and all its data
        """
        with self.get_cursor(
            cursor_factory=RealDictCursor if dict_cursor else None
        ) as cur:
            # Set search path to include AGE catalog for Cypher functions.
            # ag_catalog contains AGE's Cypher execution functions.
            cur.execute("SET search_path = ag_catalog, '$user', public;")

            # Execute Cypher via AGE's cypher() function.
            # The $$ delimiter allows safe embedding of the query string.
            query = f"""
                SELECT * FROM cypher('{graph_name}', $$
                    {cypher_query}
                $$) as (result agtype);
            """
            cur.execute(query)
            results = cur.fetchall()

            # Reset search path to default to avoid affecting subsequent operations.
            cur.execute("SET search_path = '$user', public;")

            return results

    def create_graph(self, graph_name: str) -> None:
        """
        Create a new graph in the Apache AGE graph database.

        Creates an empty graph that can then be populated with nodes
        and edges using Cypher queries via execute_cypher().

        Parameters
        ----------
        graph_name : str
            Name for the new graph. Must be unique within the database.
            Conventionally uses snake_case (e.g., 'knowledge_graph').

        Raises
        ------
        psycopg2.errors.DuplicateSchema
            If a graph with this name already exists.

        Examples
        --------
        Create a new graph:

            >>> db.create_graph("knowledge_graph")

        Then populate it:

            >>> db.execute_cypher(
            ...     "knowledge_graph",
            ...     "CREATE (n:Person {name: 'Alice'})"
            ... )

        Notes
        -----
        Each graph is stored in its own PostgreSQL schema within ag_catalog.
        Creating a graph is a prerequisite for any Cypher operations on it.
        """
        with self.get_cursor() as cur:
            cur.execute("SET search_path = ag_catalog, '$user', public;")
            cur.execute(f"SELECT create_graph('{graph_name}');")
            cur.execute("SET search_path = '$user', public;")

    def drop_graph(self, graph_name: str, cascade: bool = True) -> None:
        """
        Drop a graph and optionally all dependent objects.

        Permanently deletes the graph and all its nodes, edges, and labels.
        This operation cannot be undone.

        Parameters
        ----------
        graph_name : str
            Name of the graph to drop.
        cascade : bool, default=True
            If True, drop all dependent objects (labels, constraints).
            If False, fail if any dependent objects exist.

        Raises
        ------
        psycopg2.errors.InvalidSchemaName
            If the graph does not exist.
        psycopg2.errors.DependentObjectsStillExist
            If cascade=False and dependent objects exist.

        Examples
        --------
        Drop graph and all contents:

            >>> db.drop_graph("old_knowledge_graph")

        Cautious drop (fails if not empty):

            >>> db.drop_graph("test_graph", cascade=False)

        Warnings
        --------
        This operation is irreversible. All nodes, edges, and their
        properties will be permanently deleted.
        """
        cascade_str = "true" if cascade else "false"
        with self.get_cursor() as cur:
            cur.execute("SET search_path = ag_catalog, '$user', public;")
            cur.execute(f"SELECT drop_graph('{graph_name}', {cascade_str});")
            cur.execute("SET search_path = '$user', public;")

    def insert_vector(
        self, table: str, vector: List[float], metadata: Dict[str, Any] = None
    ) -> None:
        """
        Insert a vector embedding with optional metadata into a pgvector table.

        This method provides low-level vector insertion for cases where
        LlamaIndex's VectorStoreIndex is not being used. For most use cases,
        prefer using the Vectorizer service which handles this automatically.

        Parameters
        ----------
        table : str
            Name of the table containing the embedding column.
            Table must have: embedding VECTOR(dimension) column.
        vector : List[float]
            The embedding vector to insert. Must match the VECTOR dimension
            defined in the table schema.
        metadata : Dict[str, Any], optional
            Additional column values to insert alongside the vector.
            Keys must match existing column names in the table.

        Examples
        --------
        Insert vector only:

            >>> embedding = [0.1, 0.2, 0.3, ...]  # 1536 dimensions
            >>> db.insert_vector("embeddings", embedding)

        Insert vector with metadata:

            >>> db.insert_vector(
            ...     "embeddings",
            ...     embedding,
            ...     {"text": "Original document text", "source": "interview.txt"}
            ... )

        Notes
        -----
        For batch insertions, consider using execute_query with
        executemany or COPY for better performance.

        See Also
        --------
        similarity_search : Query vectors by similarity
        Vectorizer : Higher-level vectorization service
        """
        if metadata:
            columns = ["embedding"] + list(metadata.keys())
            values = [vector] + list(metadata.values())
            placeholders = ", ".join(["%s"] * len(values))
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({placeholders})
            """
            self.execute_query(query, tuple(values), fetch=False)
        else:
            query = f"INSERT INTO {table} (embedding) VALUES (%s)"
            self.execute_query(query, (vector,), fetch=False)

    def similarity_search(
        self,
        table: str,
        query_vector: List[float],
        limit: int = 10,
        distance_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using pgvector distance operators.

        Finds the most similar vectors to a query vector using one of
        several distance metrics. This is the core operation for semantic
        search and retrieval in RAG applications.

        Parameters
        ----------
        table : str
            Name of the table containing embeddings.
        query_vector : List[float]
            The query embedding vector. Must have same dimension as stored vectors.
        limit : int, default=10
            Maximum number of results to return.
        distance_metric : str, default='cosine'
            Distance function to use:
            - 'cosine': Cosine distance (1 - cosine similarity). Best for normalized vectors.
            - 'l2': Euclidean (L2) distance. Good for absolute distances.
            - 'inner_product': Negative inner product. Fastest but requires normalized vectors.

        Returns
        -------
        List[Dict[str, Any]]
            List of matching rows with all columns plus 'distance' score.
            Ordered by distance ascending (closest first).

        Examples
        --------
        Find similar documents:

            >>> query_embedding = embed_model.embed("What is RAG?")
            >>> results = db.similarity_search(
            ...     "embeddings",
            ...     query_embedding,
            ...     limit=5,
            ...     distance_metric="cosine"
            ... )
            >>> for r in results:
            ...     print(f"Distance: {r['distance']:.4f}, Text: {r['text'][:50]}...")

        Use L2 distance:

            >>> results = db.similarity_search(
            ...     "embeddings",
            ...     query_embedding,
            ...     distance_metric="l2"
            ... )

        Notes
        -----
        Distance Metrics Comparison:
        - Cosine: Values 0-2, where 0 is identical. Best for semantic similarity.
        - L2: Values 0+, where 0 is identical. Better for spatial clustering.
        - Inner Product: Negative values, more negative = more similar.

        For optimal performance, ensure the table has an HNSW or IVFFlat index
        on the embedding column with matching distance function.
        """
        # Map metric names to pgvector operators
        operators = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}
        op = operators.get(distance_metric, "<=>")

        query = f"""
            SELECT *, embedding {op} %s as distance
            FROM {table}
            ORDER BY distance
            LIMIT %s
        """
        return self.execute_query(
            query, (query_vector, limit), fetch=True, dict_cursor=True
        )

    def close(self):
        """
        Close all connections in the pool and release resources.

        This method should be called when the PostgresConnection instance
        is no longer needed, unless using the context manager protocol
        (which calls this automatically).

        Notes
        -----
        After calling close(), the instance cannot be used for further
        database operations. Attempting to do so will raise an error.

        Examples
        --------
        Manual resource management:

            >>> db = PostgresConnection()
            >>> try:
            ...     db.execute_query("SELECT 1")
            ... finally:
            ...     db.close()
        """
        if self.pool:
            self.pool.closeall()

    def __enter__(self):
        """
        Enter the context manager, returning self for use in 'with' statements.

        Returns
        -------
        PostgresConnection
            This instance, ready for database operations.

        Examples
        --------
            >>> with PostgresConnection() as db:
            ...     results = db.execute_query("SELECT 1")
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, closing all connections.

        Parameters
        ----------
        exc_type : type
            Exception type if an exception was raised, None otherwise.
        exc_val : Exception
            Exception instance if raised, None otherwise.
        exc_tb : traceback
            Traceback if exception was raised, None otherwise.

        Returns
        -------
        None
            Does not suppress exceptions (returns None/False).
        """
        self.close()


# ---------------------------------------------------------------------------
# Module Usage Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Demonstration of PostgresConnection capabilities.

    This example shows:
    1. Context manager usage for safe resource management
    2. Graph creation with Apache AGE
    3. Cypher query execution for node creation
    4. Standard SQL query execution
    5. Vector operations (commented, requires existing data)

    Prerequisites:
    - PostgreSQL with pgvector and AGE extensions
    - Environment variables configured (see .env.template)
    - Database initialized (run: extract-graph --init-db)
    """
    # Using context manager ensures proper cleanup
    with PostgresConnection() as db:
        # ---- Graph Operations ----
        # Create a new graph for demonstration
        db.create_graph("my_graph")

        # Create a node using Cypher query language
        results = db.execute_cypher(
            "my_graph", "CREATE (n:Person {name: 'Alice', age: 30}) RETURN n"
        )
        print("Cypher results:", results)

        # ---- SQL Operations ----
        # Query PostgreSQL system tables
        tables = db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
            dict_cursor=True,
        )
        print("Tables:", tables)

        # ---- Vector Operations ----
        # These require existing embeddings table and vector data
        # db.insert_vector('embeddings', [0.1] * 1536, {'text': 'example'})
        # similar = db.similarity_search('embeddings', [0.1] * 1536, limit=5)
