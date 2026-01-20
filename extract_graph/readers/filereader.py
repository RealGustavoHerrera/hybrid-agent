"""
File Reader - Text Document Ingestion
=====================================

This module reads text files from the file system and stores them in the
database for subsequent vectorization.

What It Does
------------
- Recursively scans a folder for .txt files
- Reads each file's content as UTF-8 text
- Stores filename and content in the 'documents' table
- Marks documents as unprocessed (processed=False) for vectorization

Supported File Types
--------------------
Currently supports .txt (plain text) files only. Files are:
- Scanned recursively in subdirectories
- Read as UTF-8 encoded text
- Stored with filename and full content

Example Usage
-------------
    # Programmatic usage
    from extract_graph.readers.filereader import read_txt_files_to_database

    read_txt_files_to_database("./data/transcripts")

    # CLI usage
    $ extract-graph --folder ./data/transcripts

See Also
--------
- extract_graph.ingest.vectorizer : Next step - vectorize ingested documents
- extract_graph.db.PostgresConnection : Database operations
"""

import os
import logging
import pandas as pd
from datetime import date
from glob import glob
from extract_graph.db.PostgresConnection import PostgresConnection


def _readText(filepath: str) -> str:
    """
    Read the contents of a text file.

    This is a private helper function that handles file reading with
    proper validation and error handling.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the text file.

    Returns
    -------
    str
        The complete contents of the file as a string.

    Raises
    ------
    ValueError
        If the file does not exist or is not readable.

    Notes
    -----
    Files are read using UTF-8 encoding (Python default).
    The entire file content is loaded into memory, so this is
    not suitable for very large files (>100MB).
    """
    if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
        with open(filepath, "r") as file:
            content = file.read()
            return content
    else:
        raise ValueError("Invalid filepath provided")


def read_txt_files_to_database(folder: str) -> None:
    """
    Read all .txt files from a folder and store them in the database.

    Recursively scans the specified folder for .txt files, reads their
    contents, and inserts them into the 'documents' table with
    processed=False, making them available for vectorization.

    Parameters
    ----------
    folder : str
        Path to the folder containing .txt files. Can be absolute or
        relative. The folder is scanned recursively for .txt files.

    Raises
    ------
    ValueError
        If the folder does not exist or is not accessible.
    psycopg2.OperationalError
        If database connection fails.

    Examples
    --------
    Ingest interview transcripts:

        >>> read_txt_files_to_database("./data/interviews")
        INFO - the provided ./data/interviews is a folder
        INFO - Saving 15 texts to the database
        INFO - 15 documents saved to the database

    Ingest from nested folder structure:

        >>> read_txt_files_to_database("/opt/documents/2024")
        # Processes /opt/documents/2024/**/*.txt recursively

    Notes
    -----
    File Processing:
        - Only .txt files are processed (case-sensitive)
        - Subdirectories are scanned recursively
        - Empty files are stored with empty content
        - Filename (without path) is preserved as metadata

    Database Schema:
        Documents are inserted with:
        - filename: Base name of the file (e.g., "interview.txt")
        - content: Full text content of the file
        - processed: False (ready for vectorization)
        - created_at: Current date

    Duplicate Handling:
        Currently no deduplication is performed. Re-running on the
        same folder will create duplicate entries. Consider checking
        for existing filenames before production use.

    Memory Considerations:
        Files are read entirely into memory. For very large files
        (>100MB), consider implementing streaming or chunked reading.

    See Also
    --------
    Vectorizer.ingest : Process stored documents into vectors
    """
    logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Validate input folder
    # -------------------------------------------------------------------------
    # Verify the folder exists and is readable before attempting to process.
    if os.path.isdir(folder) and os.access(folder, os.R_OK):
        logger.info(f"the provided {folder} is a folder")
    else:
        raise ValueError(f"Invalid folder provided: {folder}")

    # -------------------------------------------------------------------------
    # Scan for .txt files and read contents
    # -------------------------------------------------------------------------
    # glob with recursive=True and **/ pattern finds all .txt files
    # in any subdirectory depth.
    data = []
    for file in glob(f"{folder}/**/*.txt", recursive=True):
        data.append({"filename": os.path.basename(file), "content": _readText(file)})

    logger.info(f"Saving {len(data)} texts to the database")

    # -------------------------------------------------------------------------
    # Store documents in database
    # -------------------------------------------------------------------------
    # Use pandas DataFrame for convenient iteration.
    # Each document is inserted with processed=False for later vectorization.
    texts = pd.DataFrame(data)
    today = date.today()

    with PostgresConnection() as db:
        for _, row in texts.iterrows():
            db.execute_query(
                "INSERT INTO documents (filename, content, processed, created_at) "
                "VALUES (%s, %s, %s, %s)",
                (row["filename"], row["content"], False, today),
                fetch=False,
            )

    logger.info(f"{texts.shape[0]} documents saved to the database")
