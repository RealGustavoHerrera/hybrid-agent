"""
Graph Extractor Service
=======================

This module extracts entities from documents using LLMs and stores them
in the Apache AGE knowledge graph.

What It Does
------------
- Fetches processed documents from the database
- Runs entity extraction using ExtractorFocusOnWork
- Creates nodes (Person, Company, Technology, JobTitle) in the graph
- Creates relationships (WORKED_AT, USED_TECH, HAS_TITLE)

Graph Schema
------------
Nodes:
    - Person: {name, source_doc}
    - Company: {name}
    - Technology: {name}
    - JobTitle: {name}

Relationships:
    - (Person)-[:WORKED_AT]->(Company)
    - (Person)-[:USED_TECH]->(Technology)
    - (Person)-[:HAS_TITLE]->(JobTitle)

Example Usage
-------------
    from extract_graph.ingest.graph_extractor import GraphExtractor

    # Extract and store entities
    extractor = GraphExtractor()
    extractor.extract_and_store()

    # CLI usage
    $ extract-graph --extract

See Also
--------
- extract_graph.extractors.extracttechnologies : Entity extraction logic
- extract_graph.db.PostgresConnection : Graph database operations
"""

import logging
import random
import re
import time
from extract_graph.db.PostgresConnection import PostgresConnection
from extract_graph.extractors.extracttechnologies import ExtractorFocusOnWork


class GraphExtractor:
    """
    Service for extracting entities from documents and storing them in
    the knowledge graph.

    This service connects the extraction layer (LLM-based entity extraction)
    with the data access layer (Apache AGE graph storage).

    Parameters
    ----------
    model : str, default="OPENAI"
        LLM provider to use for extraction. Must be 'GEMINI' or 'OPENAI'.

    Attributes
    ----------
    extractor : ExtractorFocusOnWork
        The entity extractor instance.
    logger : logging.Logger
        Logger instance for this class.

    Examples
    --------
    Basic usage:

        >>> extractor = GraphExtractor()
        >>> extractor.extract_and_store()
        INFO - Extracting from: interview_transcript.txt
        INFO - Stored 45 entities for Emilio Arenas

    With Gemini model:

        >>> extractor = GraphExtractor(model="GEMINI")
        >>> extractor.extract_and_store()
    """

    # Rate limit handling configuration
    MAX_RETRIES = 3
    BASE_DELAY = 5  # seconds
    INTER_DOCUMENT_DELAY = 3  # seconds between documents

    def __init__(self, model: str = "OPENAI"):
        """
        Initialize the GraphExtractor service.

        Parameters
        ----------
        model : str, default="OPENAI"
            LLM provider to use. Must be 'GEMINI' or 'OPENAI'.

        Raises
        ------
        ValueError
            If required API key is not set.
        """
        self.extractor = ExtractorFocusOnWork(model)
        self.logger = logging.getLogger(__name__)

    def _extract_with_retry(self, content: str) -> object:
        """
        Execute extraction with exponential backoff retry for rate limits.

        Retries on rate limit (429) errors with increasing delays.
        Uses exponential backoff with jitter to avoid thundering herd.

        Parameters
        ----------
        content : str
            The document content to extract entities from.

        Returns
        -------
        object
            The extraction result from langextract.

        Raises
        ------
        Exception
            If extraction fails after all retries, or for non-rate-limit errors.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                self.extractor.setInputText(content)
                return self.extractor.extract()
            except Exception as e:
                error_str = str(e)
                # Retryable errors:
                # - Rate limits (429)
                # - LLM data quality issues (malformed JSON from GPT)
                is_rate_limit = "429" in error_str or "rate limit" in error_str.lower()
                is_llm_data_error = "extraction text must be" in error_str.lower()

                if is_rate_limit or is_llm_data_error:
                    if attempt < self.MAX_RETRIES - 1:
                        # Exponential backoff with jitter
                        delay = (self.BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
                        error_type = "Rate limit" if is_rate_limit else "LLM data error"
                        self.logger.warning(
                            f"{error_type} hit, waiting {delay:.1f}s before retry "
                            f"(attempt {attempt + 1}/{self.MAX_RETRIES})"
                        )
                        time.sleep(delay)
                    else:
                        raise  # Re-raise on final attempt
                else:
                    raise  # Non-rate-limit errors should not retry

        # Should not reach here, but just in case
        raise RuntimeError("Max retries exceeded")

    def _extract_person_name(self, filename: str) -> str:
        """
        Extract the interviewee's name from the filename.

        The filename format is expected to be:
        "Gustavo Herrera and Emilio Arenas Bosch - 2026_01_16..."

        Parameters
        ----------
        filename : str
            The document filename.

        Returns
        -------
        str
            The extracted person name, or "Unknown" if extraction fails.
        """
        # Pattern: "Interviewer and Interviewee Name - date..."
        # We want the second person's name (the interviewee)
        match = re.match(r".*? and ([^-]+)", filename)
        if match:
            return match.group(1).strip()
        return "Unknown"

    def _escape_cypher_string(self, text: str) -> str:
        """
        Escape special characters for Cypher queries.

        Parameters
        ----------
        text : str
            The text to escape.

        Returns
        -------
        str
            The escaped text safe for Cypher queries.
        """
        # Apache AGE Cypher uses backslash escaping for single quotes
        return text.replace("\\", "\\\\").replace("'", "\\'")

    def extract_and_store(self):
        """
        Extract entities from all processed documents and store in the graph.

        Processes each document that has been vectorized (processed=True),
        extracts entities using the LLM, and creates nodes and relationships
        in the knowledge graph.

        The method is idempotent due to MERGE operations - running multiple
        times will not create duplicate nodes or relationships.

        Raises
        ------
        psycopg2.OperationalError
            If database connection fails.
        Exception
            If LLM extraction fails.

        Notes
        -----
        Processing Flow:
            1. Query documents where processed=True
            2. For each document, run entity extraction
            3. Create Person node from filename
            4. Create entity nodes (Company, Technology, JobTitle)
            5. Create relationships to Person node
        """
        with PostgresConnection() as db:
            # Get documents that have been vectorized
            rows = db.execute_query(
                "SELECT id, filename, content FROM documents WHERE processed = true"
            )

            if not rows:
                self.logger.info("No processed documents found for extraction")
                return

            self.logger.info(f"Found {len(rows)} documents for extraction")

            for idx, (doc_id, filename, content) in enumerate(rows):
                self.logger.info(f"Extracting from: {filename}")

                # Run extraction with retry logic for rate limits
                try:
                    result = self._extract_with_retry(content)
                except Exception as e:
                    self.logger.error(
                        f"Extraction failed for {filename}: {e}. Skipping.",
                        exc_info=True
                    )
                    continue

                # Get person name from filename
                person_name = self._extract_person_name(filename)
                person_name_escaped = self._escape_cypher_string(person_name)
                filename_escaped = self._escape_cypher_string(filename)

                # Create Person node
                db.execute_cypher(
                    "knowledge_graph",
                    f"MERGE (p:Person {{name: '{person_name_escaped}', "
                    f"source_doc: '{filename_escaped}'}})"
                )

                # Track counts for logging
                entity_counts = {"company": 0, "technology": 0, "job_title": 0}

                # Create nodes and relationships for each extraction
                for entity in result.extractions:
                    entity_class = entity.extraction_class
                    # Skip entities with None or empty text
                    if not entity.extraction_text:
                        continue
                    entity_text = self._escape_cypher_string(
                        str(entity.extraction_text)
                    )

                    if entity_class == "company":
                        db.execute_cypher(
                            "knowledge_graph",
                            f"MERGE (c:Company {{name: '{entity_text}'}})"
                        )
                        db.execute_cypher(
                            "knowledge_graph",
                            f"MATCH (p:Person {{name: '{person_name_escaped}'}}), "
                            f"(c:Company {{name: '{entity_text}'}}) "
                            f"MERGE (p)-[:WORKED_AT]->(c)"
                        )
                        entity_counts["company"] += 1

                    elif entity_class == "technology":
                        db.execute_cypher(
                            "knowledge_graph",
                            f"MERGE (t:Technology {{name: '{entity_text}'}})"
                        )
                        db.execute_cypher(
                            "knowledge_graph",
                            f"MATCH (p:Person {{name: '{person_name_escaped}'}}), "
                            f"(t:Technology {{name: '{entity_text}'}}) "
                            f"MERGE (p)-[:USED_TECH]->(t)"
                        )
                        entity_counts["technology"] += 1

                    elif entity_class == "job_title":
                        db.execute_cypher(
                            "knowledge_graph",
                            f"MERGE (j:JobTitle {{name: '{entity_text}'}})"
                        )
                        db.execute_cypher(
                            "knowledge_graph",
                            f"MATCH (p:Person {{name: '{person_name_escaped}'}}), "
                            f"(j:JobTitle {{name: '{entity_text}'}}) "
                            f"MERGE (p)-[:HAS_TITLE]->(j)"
                        )
                        entity_counts["job_title"] += 1

                self.logger.info(
                    f"Stored entities for {person_name}: "
                    f"{entity_counts['company']} companies, "
                    f"{entity_counts['technology']} technologies, "
                    f"{entity_counts['job_title']} job titles"
                )

                # Add delay between documents to avoid rate limits
                if idx < len(rows) - 1:  # Don't delay after last document
                    self.logger.info(
                        f"Waiting {self.INTER_DOCUMENT_DELAY}s before next document..."
                    )
                    time.sleep(self.INTER_DOCUMENT_DELAY)

            self.logger.info("Graph extraction completed successfully")
