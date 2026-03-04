# Satellite Catalog Database

Local SQLite database of the US Space Command satellite catalog, ingested from [Space-Track.org](https://www.space-track.org). Stores orbital elements (GP/TLE) and object metadata (SATCAT) for all tracked objects on orbit. Built as the foundation for conjunction prediction analysis.

## What's in the database

| Table | Records | Description |
|-------|---------|-------------|
| `gp` | ~33K | Latest GP element set per object (Keplerian elements, TLE lines, drag term, derived orbital params). One row per tracked object currently on orbit. |
| `gp_history` | grows over time | Previous GP records archived when a newer epoch arrives during incremental updates. |
| `satcat` | ~68K | Full satellite catalog — every object ever tracked, including decayed. Contains launch info, country, object type, RCS size. |
| `maneuvers` | grows over time | Detected orbital maneuvers — epoch pairs with element deltas that exceeded configurable thresholds. |
| `sync_meta` | 2 rows | Timestamps of last GP and SATCAT sync. |
| `catalog_view` | (view) | Convenience join of `gp` + `satcat` on `NORAD_CAT_ID`. |

## Setup

```bash
pip install -r requirements.txt
```

Edit `.env` with your Space-Track credentials:

```
SPACETRACK_IDENTITY=you@example.com
SPACETRACK_PASSWORD=yourpassword
```

## Ingestion

```bash
python ingest.py --full      # Initial load (~18s, fetches full GP + SATCAT)
python ingest.py --update    # Incremental update (new epochs since last sync + maneuver detection)
python ingest.py --status    # Show record counts and sync timestamps
python ingest.py --maneuvers # Show recent maneuver detections
```

## Querying

### Python helpers (`query.py`)

```python
from query import (
    get_object, get_tle, get_objects_in_altitude_band, get_catalog_stats,
    get_maneuvers, get_maneuvering_objects, get_maneuver_history,
)

# Look up a single object by NORAD ID
iss = get_object(25544)
print(iss["OBJECT_NAME"], iss["EPOCH"], iss["APOAPSIS"])

# Get TLE lines for SGP4 propagation
line0, line1, line2 = get_tle(25544)

# Find all objects passing through an altitude band (conjunction screening)
leo_objects = get_objects_in_altitude_band(390, 430)  # km, returns ~1500 objects

# Filter by type
debris = get_objects_by_type("DEBRIS")

# Database summary
stats = get_catalog_stats()

# Maneuver detection — objects with orbital element jumps between GP epochs
recent = get_maneuvers(since="2026-03-01")           # all recent detections
iss_maneuvers = get_maneuver_history(25544)           # full timeline for one object
active_ids = get_maneuvering_objects(since="2026-03-01")  # NORAD IDs that maneuvered
```

### Direct SQL (`satcat.db`)

The database is a standard SQLite file. Open it with any SQLite client, Python's `sqlite3`, or tools like DB Browser for SQLite.

**Look up an object:**

```sql
SELECT * FROM catalog_view WHERE NORAD_CAT_ID = 25544;
```

**Objects in an altitude band** (the key query for conjunction screening — finds anything whose perigee-to-apogee range overlaps the band):

```sql
SELECT NORAD_CAT_ID, OBJECT_NAME, PERIAPSIS, APOAPSIS, OBJECT_TYPE
FROM gp
WHERE PERIAPSIS <= 430 AND APOAPSIS >= 390
ORDER BY NORAD_CAT_ID;
```

**Count by object type:**

```sql
SELECT OBJECT_TYPE, COUNT(*) AS cnt
FROM gp
GROUP BY OBJECT_TYPE
ORDER BY cnt DESC;
-- PAYLOAD: ~17K, DEBRIS: ~12K, ROCKET BODY: ~2K, UNKNOWN: ~1.5K
```

**All Starlink satellites:**

```sql
SELECT g.NORAD_CAT_ID, g.OBJECT_NAME, g.APOAPSIS, g.PERIAPSIS, g.INCLINATION
FROM gp g
WHERE g.OBJECT_NAME LIKE 'STARLINK%'
ORDER BY g.NORAD_CAT_ID;
```

**Objects by country of origin:**

```sql
SELECT s.COUNTRY, COUNT(*) AS cnt
FROM satcat s
WHERE s.CURRENT = 'Y'
GROUP BY s.COUNTRY
ORDER BY cnt DESC
LIMIT 10;
```

**Recent maneuver detections** (objects whose orbital elements jumped between consecutive GP epochs):

```sql
SELECT m.NORAD_CAT_ID, g.OBJECT_NAME, m.EPOCH_AFTER,
       m.DELTA_SMA, m.DELTA_PERIOD, m.DELTA_INCLINATION
FROM maneuvers m
LEFT JOIN gp g ON m.NORAD_CAT_ID = g.NORAD_CAT_ID
ORDER BY m.EPOCH_AFTER DESC
LIMIT 20;
```

**Maneuver history for a specific object:**

```sql
SELECT EPOCH_BEFORE, EPOCH_AFTER, DELTA_SMA, DELTA_PERIOD, DELTA_INCLINATION
FROM maneuvers
WHERE NORAD_CAT_ID = 25544
ORDER BY EPOCH_AFTER;
```

**Track orbital element history** (after running `--update` at least once):

```sql
SELECT EPOCH, SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION
FROM gp_history
WHERE NORAD_CAT_ID = 25544
ORDER BY EPOCH;
```

**Get TLE for propagation:**

```sql
SELECT TLE_LINE0, TLE_LINE1, TLE_LINE2
FROM gp
WHERE NORAD_CAT_ID = 25544;
```

## Schema reference

### `gp` table — key columns

| Column | Type | Description |
|--------|------|-------------|
| `NORAD_CAT_ID` | INTEGER | NORAD catalog number (unique in `gp`, indexed) |
| `OBJECT_NAME` | TEXT | Common name (e.g., "ISS (ZARYA)") |
| `EPOCH` | TEXT | Epoch of the element set (ISO 8601) |
| `MEAN_MOTION` | REAL | Revolutions per day |
| `ECCENTRICITY` | REAL | Orbital eccentricity |
| `INCLINATION` | REAL | Orbital inclination (degrees) |
| `RA_OF_ASC_NODE` | REAL | Right ascension of ascending node (degrees) |
| `ARG_OF_PERICENTER` | REAL | Argument of perigee (degrees) |
| `MEAN_ANOMALY` | REAL | Mean anomaly (degrees) |
| `BSTAR` | REAL | Drag term (1/Earth radii) |
| `SEMIMAJOR_AXIS` | REAL | Semi-major axis (km) |
| `PERIOD` | REAL | Orbital period (minutes) |
| `APOAPSIS` | REAL | Apogee altitude (km) |
| `PERIAPSIS` | REAL | Perigee altitude (km) |
| `OBJECT_TYPE` | TEXT | PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, TBA |
| `TLE_LINE0`, `TLE_LINE1`, `TLE_LINE2` | TEXT | Standard TLE format for SGP4 |
| `ingested_at` | TEXT | When this record was written to the local DB |

### `satcat` table — key columns

| Column | Type | Description |
|--------|------|-------------|
| `NORAD_CAT_ID` | INTEGER | Primary key, matches `gp` |
| `SATNAME` | TEXT | Official satellite name |
| `INTLDES` | TEXT | International designator (e.g., "1998-067A") |
| `OBJECT_TYPE` | TEXT | PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN |
| `COUNTRY` | TEXT | Registering country/org code (US, CIS, PRC, etc.) |
| `LAUNCH` | TEXT | Launch date |
| `DECAY` | TEXT | Decay date (NULL if still on orbit) |
| `APOGEE` | INTEGER | Apogee height (km) |
| `PERIGEE` | INTEGER | Perigee height (km) |
| `RCS_SIZE` | TEXT | Radar cross-section category (SMALL, MEDIUM, LARGE) |
| `CURRENT` | TEXT | "Y" if currently tracked, "N" if historical |

### `maneuvers` table — key columns

| Column | Type | Description |
|--------|------|-------------|
| `NORAD_CAT_ID` | INTEGER | Object that maneuvered |
| `EPOCH_BEFORE` | TEXT | GP epoch of the prior element set |
| `EPOCH_AFTER` | TEXT | GP epoch of the new element set |
| `DELTA_SMA` | REAL | Absolute change in semi-major axis (km) |
| `DELTA_ECCENTRICITY` | REAL | Absolute change in eccentricity |
| `DELTA_INCLINATION` | REAL | Absolute change in inclination (degrees) |
| `DELTA_RAAN` | REAL | Absolute change in RAAN (degrees) |
| `DELTA_PERIOD` | REAL | Absolute change in period (minutes) |
| `DELTA_APOAPSIS` | REAL | Absolute change in apogee altitude (km) |
| `DELTA_PERIAPSIS` | REAL | Absolute change in perigee altitude (km) |
| `detected_at` | TEXT | When the detection was recorded |

Thresholds for triggering a detection are configured in `config.py` (`MANEUVER_THRESHOLDS`). A row is inserted when **any** tracked element exceeds its threshold. Detection runs automatically after each `--update` cycle.

### Indexes

Optimized for conjunction prediction workflows:
- `gp(NORAD_CAT_ID)` — fast single-object lookup
- `gp(EPOCH)` — filtering by data freshness
- `gp(APOAPSIS, PERIAPSIS)` — altitude-band screening
- `maneuvers(NORAD_CAT_ID)` — maneuver lookup by object
- `maneuvers(EPOCH_AFTER)` — recent maneuver queries
- `satcat(OBJECT_TYPE)`, `satcat(COUNTRY)`, `satcat(CURRENT)` — catalog filtering

## Data source

All data comes from the [18th Space Defense Squadron](https://www.space-track.org) via their public API. GP data uses the OMM (Orbiting Mean-elements Message) format, which is the modern replacement for raw TLE distribution. Requires a free Space-Track.org account.
