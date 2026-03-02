# Satellite Catalog Local Database — Technical Specification

## Context

Build a local satellite catalog database ingested from Space-Track.org's API, storing both GP (TLE/OMM orbital elements) and SATCAT (object metadata) for 30,000+ tracked objects. This database will serve as the foundation for conjunction prediction analysis. Python + SQLite stack, credentials already available.

## Project Structure

```
SATCAT/
├── config.py              # Credentials loading, constants, DB path
├── schema.sql             # SQLite DDL (tables, indexes, views)
├── ingest.py              # Space-Track API client + DB ingestion logic
├── query.py               # Helper queries (for later conjunction work)
├── requirements.txt       # Dependencies
├── .env                   # Space-Track credentials (gitignored)
└── .gitignore
```

## Step 1: Project Setup & Dependencies

**Files:** `requirements.txt`, `.env`, `.gitignore`, `config.py`

**Environment:** Use the `orbit` conda environment.

```bash
conda activate orbit
pip install -r requirements.txt
```

- **Dependencies:** `spacetrack` (API client with built-in rate limiting), `python-dotenv` (credential management)
- **`.env`** stores `SPACETRACK_IDENTITY` and `SPACETRACK_PASSWORD`
- **`config.py`** loads env vars, defines DB path (`satcat.db`), constants (refresh intervals, batch sizes)

## Step 2: SQLite Schema

**File:** `schema.sql`, applied by `ingest.py`

### Table: `gp` (latest GP element sets)
Stores the most recent OMM/TLE for each object. Key columns:
- `GP_ID` (PK) — Space-Track unique element set ID
- `NORAD_CAT_ID` — catalog number (indexed, unique constraint for "latest" table)
- `OBJECT_NAME`, `OBJECT_ID` (international designator)
- Orbital elements: `EPOCH`, `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`
- Derived: `SEMIMAJOR_AXIS`, `PERIOD`, `APOAPSIS`, `PERIAPSIS`
- TLE strings: `TLE_LINE0`, `TLE_LINE1`, `TLE_LINE2`
- Metadata: `OBJECT_TYPE`, `RCS_SIZE`, `COUNTRY_CODE`, `DECAY_DATE`
- `ingested_at` — local timestamp of when the record was inserted

### Table: `gp_history` (historical element sets)
Same schema as `gp` but without the unique constraint on `NORAD_CAT_ID`. Old GP records move here when a newer epoch arrives. Enables tracking orbital element evolution over time.

### Table: `satcat` (satellite catalog metadata)
- `NORAD_CAT_ID` (PK)
- `SATNAME`, `OBJECT_NAME`, `OBJECT_ID`, `INTLDES`
- `OBJECT_TYPE` (PAYLOAD / ROCKET BODY / DEBRIS / UNKNOWN)
- `COUNTRY` (Space-Track country codes: US, PRC, CIS, etc.)
- `LAUNCH` (date), `SITE`, `DECAY` (date, NULL if in orbit)
- `PERIOD`, `INCLINATION`, `APOGEE`, `PERIGEE`
- `RCS_SIZE`, `CURRENT` (Y/N)
- `ingested_at`

### Indexes (for conjunction prediction queries)
- `gp`: index on `(NORAD_CAT_ID)`, `(EPOCH)`, `(APOAPSIS, PERIAPSIS)` for altitude-band filtering
- `satcat`: index on `(OBJECT_TYPE)`, `(COUNTRY)`, `(CURRENT)`

### View: `catalog_view`
Joins `gp` + `satcat` on `NORAD_CAT_ID` for convenient querying with both orbital and metadata fields.

## Step 3: Ingestion Script

**File:** `ingest.py`

### Authentication
- Uses `spacetrack.SpaceTrackClient` with credentials from `.env`
- Cookie-based session, automatic rate-limit handling (30 req/min, 300 req/hr)

### Initial Load (full catalog)
1. Fetch all current GP data in one request: `gp(decay_date=op.isnull(), orderby="norad_cat_id", format="json")` — returns ~30K records as JSON
2. Fetch full SATCAT: `satcat(orderby="norad_cat_id", format="json")`
3. Parse JSON, insert into SQLite using `executemany` with `INSERT OR REPLACE`
4. Log record counts and elapsed time

### Incremental Update (daily/hourly refresh)
1. Fetch GP data with `epoch > last_sync_timestamp` to get only updated element sets
2. For each updated object: move the existing GP record to `gp_history`, insert the new one
3. Refresh SATCAT once daily (lightweight — metadata changes rarely)
4. Store `last_sync_timestamp` in a `sync_meta` table

### CLI Interface
```
python ingest.py --full        # Full initial load
python ingest.py --update      # Incremental update since last sync
python ingest.py --status      # Show DB stats (record counts, last sync time)
```

## Step 4: Query Helpers

**File:** `query.py`

Basic query functions to verify the database and prepare for conjunction work:
- `get_object(norad_id)` — return GP + SATCAT for a single object
- `get_objects_in_altitude_band(min_km, max_km)` — objects whose perigee/apogee overlaps the band (key for conjunction screening)
- `get_objects_by_type(object_type)` — filter by PAYLOAD/DEBRIS/etc.
- `get_catalog_stats()` — total objects, breakdown by type/country, oldest/newest epoch
- `get_tle(norad_id)` — return TLE lines ready for SGP4 propagation

## Step 5: Verification

1. Run `python ingest.py --full` — confirm ~30K GP records and ~60K SATCAT records ingested
2. Run `python ingest.py --status` — verify record counts and sync timestamp
3. Run queries: look up ISS (NORAD 25544), filter LEO debris, check altitude-band query
4. Run `python ingest.py --update` — confirm incremental update works (only fetches new epochs)
5. Verify `gp_history` table captures old element sets after an update cycle

## Key Design Decisions

- **`spacetrack` library over raw requests** — handles rate limiting, retries, URL encoding automatically
- **Single-request bulk fetch** — Space-Track returns the full catalog in one response; avoids pagination overhead
- **GP + GP_HISTORY split** — keeps the "latest" table lean for conjunction queries while preserving history
- **SQLite WAL mode** — enabled for better concurrent read performance with <10 users
- **`ingested_at` timestamps** — track data freshness independently of Space-Track's EPOCH
