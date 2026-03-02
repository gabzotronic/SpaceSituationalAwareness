# CLAUDE.md

## Project overview

Local satellite catalog database for Space Situational Awareness (SSA) conjunction prediction analysis. Ingests GP (orbital elements/TLE) and SATCAT (object metadata) from Space-Track.org into SQLite. Part of the broader OSTIN/SSA analysis project at Adroitly.

## File structure

- `config.py` — Loads `.env` credentials, defines `DB_PATH` (`satcat.db`), `SCHEMA_PATH`, `BATCH_SIZE`
- `schema.sql` — SQLite DDL: tables `gp`, `gp_history`, `satcat`, `sync_meta`; indexes; `catalog_view`
- `ingest.py` — Space-Track API client + DB ingestion. CLI: `--full`, `--update`, `--status`
- `query.py` — Helper functions: `get_object()`, `get_tle()`, `get_objects_in_altitude_band()`, `get_objects_by_type()`, `get_catalog_stats()`
- `satcat.db` — SQLite database (~30MB). WAL mode. ~33K GP records, ~68K SATCAT records.
- `.env` — Space-Track credentials (gitignored, never commit)

## Key conventions

- Python 3.8+ compatibility required (no `X | Y` union types — use `from __future__ import annotations`)
- All Space-Track field names are UPPER_CASE to match the API JSON keys exactly
- `NORAD_CAT_ID` is the universal join key between tables
- `gp` table has a UNIQUE constraint on `NORAD_CAT_ID` (one row per object); `gp_history` does not
- Timestamps: `EPOCH` is Space-Track's element set epoch (ISO 8601); `ingested_at` is local insertion time
- Altitude values (`APOAPSIS`, `PERIAPSIS`, `APOGEE`, `PERIGEE`) are in km above Earth's surface
- The `spacetrack` library (v1.3.1) uses `None` for null-value filtering (not `op.isnull()` — that doesn't exist in this version)

## Common tasks

### Refresh the database
```bash
python ingest.py --update    # Incremental (archives old GP to gp_history, fetches new epochs)
python ingest.py --full      # Full reload (replaces all GP and SATCAT)
python ingest.py --status    # Check record counts and last sync time
```

### Query patterns
- Single object lookup: `SELECT * FROM catalog_view WHERE NORAD_CAT_ID = ?`
- Altitude-band conjunction screening: `SELECT * FROM gp WHERE PERIAPSIS <= ? AND APOAPSIS >= ?`
- TLE for SGP4 propagation: `SELECT TLE_LINE0, TLE_LINE1, TLE_LINE2 FROM gp WHERE NORAD_CAT_ID = ?`

## Dependencies

`spacetrack` (API client with rate limiting), `python-dotenv`. Install with `pip install -r requirements.txt`.

## Things to watch out for

- `.env` contains real Space-Track credentials — never commit or log them
- Space-Track rate limits: 30 req/min, 300 req/hr (handled by the `spacetrack` library)
- `gp_history` grows with each `--update` cycle — may need periodic pruning for long-running deployments
- The `spacetrack` library returns JSON as a string; `ingest.py` handles both `str` and `list` return types
- SQLite WAL mode is set via `schema.sql` PRAGMA — concurrent readers are fine, but only one writer at a time
