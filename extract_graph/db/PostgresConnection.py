import os, logging
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pgvector.psycopg2 import register_vector
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings


class PostgresConnection:
    """
    Connection manager for PostgreSQL with AGE and pgvector support.
    Uses connection pooling for better performance.
    """

    def __init__(self, min_connections: int = 1, max_connections: int = 10):

        self.logger = logging.getLogger(__name__)

        # Create connection pool loading from environment variables
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
            f"connection pool created with min: {min_connections} and max: {max_connections} connections"
        )

        # Initialize AGE and pgvector on first connection
        self._initialize_extensions()

        # Create a vector_store
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
        )

        self.vector_store = PGVectorStore.from_params(
            database=os.getenv("POSTGRES_DB"),
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            table_name="embeddings",
            embed_dim=1536,  # Dimension for text-embedding-3-small;
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
            perform_setup=True,  # Ensures table/index creation
            debug=(
                os.getenv("LOG_LEVEL") == "DEBUG"
            ),  # Logs SQL for troubleshooting in DEBUG mode
        )
        self.logger.debug("vector store created")

    def _initialize_extensions(self):
        """Load AGE extension and register pgvector types."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Load AGE
                cur.execute("LOAD 'age';")
                conn.commit()

            # Register pgvector types
            register_vector(conn)
        finally:
            self.logger.debug("Extensions initialized")
            self.pool.putconn(conn)

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.
        Usage:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ...")
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
        Context manager for getting a cursor with auto-commit/rollback.
        Usage:
            with db.get_cursor() as cur:
                cur.execute("SELECT ...")
                results = cur.fetchall()
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
        Execute a query and optionally fetch results.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            dict_cursor: Return results as dictionaries

        Returns:
            List of results or None
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
        Execute a Cypher query using AGE.

        Args:
            graph_name: Name of the graph
            cypher_query: Cypher query string
            dict_cursor: Return results as dictionaries

        Returns:
            List of query results
        """
        with self.get_cursor(
            cursor_factory=RealDictCursor if dict_cursor else None
        ) as cur:
            # Set search path for AGE
            cur.execute("SET search_path = ag_catalog, '$user', public;")
            query = f"""
                SELECT * FROM cypher('{graph_name}', $$
                    {cypher_query}
                $$) as (result agtype);
            """
            cur.execute(query)
            results = cur.fetchall()

            # Reset to default
            cur.execute("SET search_path = '$user', public;")

            return results

    def create_graph(self, graph_name: str) -> None:
        """Create a new graph in AGE."""
        with self.get_cursor() as cur:
            cur.execute("SET search_path = ag_catalog, '$user', public;")
            cur.execute(f"SELECT create_graph('{graph_name}');")
            # Reset to default
            cur.execute("SET search_path = '$user', public;")

    def drop_graph(self, graph_name: str, cascade: bool = True) -> None:
        """Drop a graph from AGE."""
        cascade_str = "true" if cascade else "false"
        with self.get_cursor() as cur:
            cur.execute("SET search_path = ag_catalog, '$user', public;")
            cur.execute(f"SELECT drop_graph('{graph_name}', {cascade_str});")
            # Reset to default
            cur.execute("SET search_path = '$user', public;")

    def insert_vector(
        self, table: str, vector: List[float], metadata: Dict[str, Any] = None
    ) -> None:
        """
        Insert a vector with optional metadata.
        Table should have columns: id SERIAL, embedding VECTOR(dimension), ...
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
        Perform similarity search using pgvector.

        Args:
            table: Table name
            query_vector: Query vector
            limit: Number of results
            distance_metric: 'cosine', 'l2', or 'inner_product'

        Returns:
            List of similar vectors with distances
        """
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
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Usage example
if __name__ == "__main__":
    # Using environment variables
    with PostgresConnection() as db:
        # Create a graph
        db.create_graph("my_graph")

        # Execute Cypher query
        results = db.execute_cypher(
            "my_graph", "CREATE (n:Person {name: 'Alice', age: 30}) RETURN n"
        )
        print("Cypher results:", results)

        # Regular SQL query
        tables = db.execute_query(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
            dict_cursor=True,
        )
        print("Tables:", tables)

        # Vector operations (assuming you have a vectors table)
        # db.insert_vector('embeddings', [0.1, 0.2, 0.3], {'text': 'example'})
        # similar = db.similarity_search('embeddings', [0.1, 0.2, 0.3], limit=5)
