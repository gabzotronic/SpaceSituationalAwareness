"""Configuration: credentials, DB path, constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Space-Track credentials
SPACETRACK_IDENTITY = os.environ["SPACETRACK_IDENTITY"]
SPACETRACK_PASSWORD = os.environ["SPACETRACK_PASSWORD"]

# Database
DB_PATH = Path(__file__).parent / "satcat.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Ingestion settings
BATCH_SIZE = 5000  # rows per executemany batch
