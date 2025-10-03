import os, logging
import pandas as pd
from datetime import date
from glob import glob
from extract_graph.db.PostgresConnection import PostgresConnection


def _readText(filepath):
    if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
        with open(filepath, "r") as file:
            content = file.read()
            return content
    else:
        raise ValueError("Invalid filepath provided")


def read_txt_files_to_database(folder):
    logger = logging.getLogger(__name__)
    # check if it's a folder
    if os.path.isdir(folder) and os.access(folder, os.R_OK):
        logger.info(f"the provided {folder} is a folder")
    else:
        raise ValueError(f"Invalid folder provided: {folder}")

    # and try to read .txt files inside
    data = []
    for file in glob(f"{folder}/**/*.txt", recursive=True):
        data.append({"filename": os.path.basename(file), "content": _readText(file)})

    logger.info(f"Saving {len(data)} texts to the database")

    texts = pd.DataFrame(data)
    today = date.today()
    with PostgresConnection() as db:
        for _, row in texts.iterrows():
            db.execute_query(
                "INSERT INTO documents (filename, content, processed, created_at) VALUES (%s, %s, %s, %s)",
                (row["filename"], row["content"], False, today),
                False,
            )
    logger.info(f"{texts.shape[0]} documents saved to the database")
