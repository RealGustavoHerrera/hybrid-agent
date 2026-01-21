"""
Database Display Module
=======================

This module provides functions to display sample contents from the database,
useful for verifying the pipeline results.

What It Does
------------
- Shows document records from the documents table
- Shows embedding count from pgvector
- Shows nodes and relationships from the knowledge graph

Example Usage
-------------
    from extract_graph.display.db_display import show_database_contents

    # Display all database contents
    show_database_contents()

    # CLI usage
    $ extract-graph --show

See Also
--------
- extract_graph.db.PostgresConnection : Database operations
"""

import logging
from extract_graph.db.PostgresConnection import PostgresConnection


def show_database_contents():
    """
    Display sample contents from documents, vectors, and knowledge graph.

    Prints a formatted summary of:
    - Documents table (first 5 records with id, filename, processed status)
    - Vector embeddings count from the data_embeddings table
    - Person nodes from the knowledge graph (up to 10)
    - Company nodes from the knowledge graph (up to 10)
    - Technology nodes from the knowledge graph (up to 15)
    - Sample relationships showing connections between entities

    This function is read-only and safe to run at any time.

    Examples
    --------
    Display database contents:

        >>> show_database_contents()
        ============================================================
        DOCUMENTS TABLE
        ============================================================
          [1] interview_transcript.txt (processed: True)
        ...
    """
    logger = logging.getLogger(__name__)

    with PostgresConnection() as db:
        # Documents table
        print("\n" + "=" * 60)
        print("DOCUMENTS TABLE")
        print("=" * 60)
        docs = db.execute_query(
            "SELECT id, filename, processed FROM documents LIMIT 5",
            dict_cursor=True
        )
        if docs:
            for doc in docs:
                status = "processed" if doc["processed"] else "pending"
                print(f"  [{doc['id']}] {doc['filename']} ({status})")
        else:
            print("  (no documents)")

        # Vector embeddings count
        print("\n" + "=" * 60)
        print("VECTOR EMBEDDINGS")
        print("=" * 60)
        try:
            count = db.execute_query("SELECT COUNT(*) FROM data_embeddings")
            print(f"  Total embeddings: {count[0][0]}")
        except Exception:
            print("  (embeddings table not found or empty)")

        # Knowledge graph - People
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH - People")
        print("=" * 60)
        try:
            people = db.execute_cypher(
                "knowledge_graph",
                "MATCH (p:Person) RETURN p.name LIMIT 10"
            )
            if people:
                for p in people:
                    print(f"  - {p['result']}")
            else:
                print("  (no people nodes)")
        except Exception as e:
            print(f"  (error querying graph: {e})")

        # Knowledge graph - Companies
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH - Companies")
        print("=" * 60)
        try:
            companies = db.execute_cypher(
                "knowledge_graph",
                "MATCH (c:Company) RETURN c.name LIMIT 10"
            )
            if companies:
                for c in companies:
                    print(f"  - {c['result']}")
            else:
                print("  (no company nodes)")
        except Exception as e:
            print(f"  (error querying graph: {e})")

        # Knowledge graph - Technologies
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH - Technologies")
        print("=" * 60)
        try:
            techs = db.execute_cypher(
                "knowledge_graph",
                "MATCH (t:Technology) RETURN t.name LIMIT 15"
            )
            if techs:
                for t in techs:
                    print(f"  - {t['result']}")
            else:
                print("  (no technology nodes)")
        except Exception as e:
            print(f"  (error querying graph: {e})")

        # Knowledge graph - Job Titles
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH - Job Titles")
        print("=" * 60)
        try:
            titles = db.execute_cypher(
                "knowledge_graph",
                "MATCH (j:JobTitle) RETURN j.name LIMIT 10"
            )
            if titles:
                for t in titles:
                    print(f"  - {t['result']}")
            else:
                print("  (no job title nodes)")
        except Exception as e:
            print(f"  (error querying graph: {e})")

        # Knowledge graph - Sample Relationships
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH - Sample Relationships")
        print("=" * 60)
        try:
            rels = db.execute_cypher(
                "knowledge_graph",
                "MATCH (p:Person)-[r]->(n) "
                "RETURN p.name + ' --[' + type(r) + ']--> ' + n.name "
                "LIMIT 15"
            )
            if rels:
                for r in rels:
                    print(f"  {r['result']}")
            else:
                print("  (no relationships)")
        except Exception as e:
            print(f"  (error querying graph: {e})")

        print("\n" + "=" * 60)

    logger.info("Database contents displayed successfully")
