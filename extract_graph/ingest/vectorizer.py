import logging
from extract_graph.db.PostgresConnection import PostgresConnection
from llama_index.core import Document, VectorStoreIndex, StorageContext
from datetime import date


class Vectorizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        pass

    def ingest(self):
        docs = []
        with PostgresConnection() as db:
            rows = db.execute_query(
                "SELECT id, filename, content FROM documents WHERE processed = false"
            )
            for row in rows:
                id, filename, content = row
                doc = Document(
                    text=content, metadata={"filename": filename, "orig_id": id}
                )
                docs.append(doc)

        self.logger.info(f"found and created {len(docs)}")

        storage_context = StorageContext.from_defaults(vector_store=db.vector_store)
        VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, show_progress=True
        )

        self.logger.info("✅ Index creation completed.")

        with PostgresConnection() as db:
            for doc in docs:
                db.execute_query(
                    "UPDATE documents SET processed = true, processed_at = %s WHERE id = %s",
                    (date.today(), doc.metadata["orig_id"]),
                    False,
                )

        self.logger.info(f"Vectorized and stored {len(docs)} documents.")
