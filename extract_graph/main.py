import logging
import logging.config
import os
import sys
from dotenv import load_dotenv
from datetime import date
from flask import Flask
from flask_restful import abort, Api, Resource

load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)

# Global logging config with rotation
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "default": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        },
        "file": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"./logs/app_{date.today().isoformat()}.log",
            "maxBytes": 10485760,  # 10MB per file
            "backupCount": 5,  # Keep 5 backups
            "formatter": "standard",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
}

# Apply config (before other imports that might log)
logging.config.dictConfig(log_config)
import argparse
from extract_graph.ingest.vectorizer import Vectorizer
from extract_graph.db.init_database import init_database
from extract_graph.readers.filereader import read_txt_files_to_database

app = Flask(__name__)
api = Api(app)


class InitDB(Resource):
    def get(self):
        try:
            init_database()
            return "OK", 200
        except Exception as e:
            logging.getLogger(__name__).error(f"Error initializing database: {e}")
            abort(500, message="Error initializing database")


class ReadFiles(Resource):
    def post(self, folder):
        try:
            read_txt_files_to_database(folder)
            return "OK", 200
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error reading folder {folder}. Error: {e}"
            )
            abort(400, message=f"Invalid folder provided: {folder}")


class Vectorize(Resource):
    def get(self):
        try:
            Vectorizer().ingest()
            return "OK", 200
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during vectorization: {e}")
            abort(500, message="Error during vectorization")


api.add_resource(InitDB, "/init-db")
api.add_resource(Vectorize, "/vectorize")
api.add_resource(ReadFiles, "/readfiles/<folder>")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze texts to extract vectors and structured data, build a graph and use it for Hybrid RAG"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="the folder containing the .txt files to be read. It can read subfolders. Only .txt files will be picked up.",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        required=False,
        help="Initialize the database schema and return",
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="Find all unprocessed documents in the db and vectorize them",
    )
    args = parser.parse_args()

    if args.init_db:
        init_database()
        return

    if args.vectorize:
        Vectorizer().ingest()
        return

    if args.folder:
        read_txt_files_to_database(args.folder)
        return


if __name__ == "__main__":
    # Optionally run CLI mode if args provided, else run server
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        app.run(debug=True)
