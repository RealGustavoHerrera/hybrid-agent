import sys, logging
from path import Path
import glob
from extract_graph.db.PostgresConnection import PostgresConnection


def init_database():
    logger = logging.getLogger(__name__)
    """Initialize database schema."""
    logger.info("Initializing database...")

    db = PostgresConnection()

    # try to read .sql files in migrations
    migrations = Path(__file__).parent / "migrations" / "**/*.sql"
    schemas = sorted(glob.glob(migrations, recursive=True))
    logger.info(f"schemas found in {migrations}: {len(schemas)}")

    try:
        for schema_file_str in schemas:
            schema_file = Path(schema_file_str)
            if schema_file.exists():
                with open(schema_file, "r") as f:
                    schema_sql = f.read()
                    db.execute_query(schema_sql, fetch=False)
                    logger.info(f"file {schema_file} executed")
            else:
                logger.info(f"file {schema_file} not found")

        logger.info("✓ Database initialized successfully!")

    except Exception as e:
        logger.info(f"✗ Error initializing database: {e}")
        sys.exit(1)

    finally:
        db.close()
        return
