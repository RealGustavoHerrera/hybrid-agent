"""
Database Initialization Module
==============================

This module initializes the database schema by executing SQL migration files.

What It Does
------------
- Reads .sql files from the db/migrations/ directory
- Executes them in alphabetical order against PostgreSQL
- Sets up tables, extensions, and graphs needed by the application

Migration File Naming
---------------------
Files in db/migrations/ are sorted alphabetically and executed in order.
Convention: NNN_description.sql (e.g., 001_initial_schema.sql)

Example Usage
-------------
    # CLI usage
    $ extract-graph --init-db

    # Programmatic usage
    from extract_graph.db.init_database import init_database
    init_database()

See Also
--------
- extract_graph.db.migrations/ : SQL migration files
- extract_graph.db.PostgresConnection : Database connection management
"""

import sys
import logging
from path import Path
import glob
from extract_graph.db.PostgresConnection import PostgresConnection


def init_database() -> None:
    """
    Initialize the database schema by executing SQL migration files.

    Reads all .sql files from the migrations directory (sorted alphabetically)
    and executes them against the database. This sets up:
    - The documents table for storing ingested text
    - The Apache AGE knowledge_graph for entity relationships
    - Any additional schema defined in migration files

    The function is idempotent when migrations use IF NOT EXISTS clauses.

    Raises
    ------
    SystemExit
        If any migration fails to execute. Exit code 1.
    psycopg2.OperationalError
        If unable to connect to the database.

    Examples
    --------
    Initialize a fresh database:

        >>> init_database()
        INFO - Initializing database...
        INFO - schemas found in .../migrations/**/*.sql: 1
        INFO - file 001_initial_schema.sql executed
        INFO - Database initialized successfully!

    Notes
    -----
    Migration files should be designed to be idempotent (safe to run
    multiple times). Use constructs like:
    - CREATE TABLE IF NOT EXISTS
    - CREATE EXTENSION IF NOT EXISTS
    - DO $$ BEGIN ... EXCEPTION WHEN ... END $$

    The function creates its own PostgresConnection and closes it
    when done, ensuring proper resource cleanup.

    See Also
    --------
    PostgresConnection : Database connection management
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing database...")

    db = PostgresConnection()

    # Locate all SQL migration files in the migrations directory.
    # Files are sorted to ensure consistent execution order.
    migrations = Path(__file__).parent / "migrations" / "**/*.sql"
    schemas = sorted(glob.glob(migrations, recursive=True))
    logger.info(f"Migration files found in {migrations}: {len(schemas)}")

    try:
        for schema_file_str in schemas:
            schema_file = Path(schema_file_str)
            if schema_file.exists():
                with open(schema_file, "r") as f:
                    schema_sql = f.read()
                    db.execute_query(schema_sql, fetch=False)
                    logger.info(f"Executed migration: {schema_file.name}")
            else:
                logger.warning(f"Migration file not found: {schema_file}")

        logger.info("Database initialized successfully!")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        sys.exit(1)

    finally:
        db.close()
