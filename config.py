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
# Retained for legacy detect_maneuvers() in ingest.py until Phase 6 removes it.
MANEUVER_THRESHOLDS = {
    "SEMIMAJOR_AXIS": 1.0,    # km
    "ECCENTRICITY":   0.001,
    "INCLINATION":    0.05,   # degrees
    "RA_OF_ASC_NODE": 0.5,    # degrees
    "PERIOD":         0.05,   # minutes
}

# OD-based maneuver detection parameters
OD_SIGMA_MULTIPLIER   = 5.0    # threshold = median + N × MAD of vel residuals
OD_KP_THRESHOLD       = 5.0    # Kp above which space weather is 'disturbed'
OD_MAX_GAP_DAYS       = 10.0   # skip epoch pairs with larger gaps
OD_BSTAR_NOISE_MAX    = 1e-3   # skip TLEs with |B*| above this (poor fit)
OD_BSTAR_DELTA_THRESH = 5e-5   # B* step-change secondary signal threshold
MIN_HISTORY_DAYS      = 30     # warn if gp_history coverage is below this
