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

# Maneuver detection thresholds (absolute deltas between consecutive GP epochs)
MANEUVER_THRESHOLDS = {
    "SEMIMAJOR_AXIS": 1.0,    # km
    "ECCENTRICITY":   0.001,
    "INCLINATION":    0.05,   # degrees
    "RA_OF_ASC_NODE": 0.5,    # degrees
    "PERIOD":         0.05,   # minutes
}
