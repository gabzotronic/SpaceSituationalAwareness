# Maneuver Detection from TLE History

## Context

For conjunction prediction, knowing whether a satellite has recently maneuvered (or maneuvers frequently) is critical — a maneuvering object invalidates propagated trajectories. The `gp_history` table already accumulates a TLE time-series on each `--update` cycle. We can compare consecutive GP epochs for the same object and flag abrupt changes in orbital elements as probable maneuvers.

## Approach: Threshold-Based Anomaly Detection on `gp_history`

After each `--update` ingestion, scan newly updated objects' TLE histories and compare the fresh GP epoch against the most recent prior epoch. If any orbital element delta exceeds a configurable threshold, insert a row into a new `maneuvers` table.

## Files to Modify

| File | Change |
|---|---|
| `schema.sql` | Add `maneuvers` table + index |
| `config.py` | Add maneuver detection thresholds |
| `ingest.py` | Call `detect_maneuvers()` after GP update in `ingest_update()` |
| `query.py` | Add query helpers for maneuver data |

## 1. Schema — new `maneuvers` table in `schema.sql`

```sql
CREATE TABLE IF NOT EXISTS maneuvers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    NORAD_CAT_ID        INTEGER NOT NULL,
    EPOCH_BEFORE        TEXT NOT NULL,
    EPOCH_AFTER         TEXT NOT NULL,
    DELTA_SMA           REAL,    -- km
    DELTA_ECCENTRICITY  REAL,
    DELTA_INCLINATION   REAL,    -- degrees
    DELTA_RAAN          REAL,    -- degrees
    DELTA_PERIOD        REAL,    -- minutes
    DELTA_APOAPSIS      REAL,    -- km
    DELTA_PERIAPSIS     REAL,    -- km
    detected_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(NORAD_CAT_ID, EPOCH_BEFORE, EPOCH_AFTER)
);

CREATE INDEX IF NOT EXISTS idx_maneuvers_norad ON maneuvers (NORAD_CAT_ID);
CREATE INDEX IF NOT EXISTS idx_maneuvers_epoch ON maneuvers (EPOCH_AFTER);
```

## 2. Thresholds in `config.py`

```python
MANEUVER_THRESHOLDS = {
    "SEMIMAJOR_AXIS": 1.0,    # km
    "ECCENTRICITY":   0.001,
    "INCLINATION":    0.05,   # degrees
    "RA_OF_ASC_NODE": 0.5,    # degrees
    "PERIOD":         0.05,   # minutes
}
```

Starting values — conservative enough to catch real maneuvers while avoiding noise from drag/perturbations. Tune after reviewing initial detections.

## 3. Detection logic in `ingest.py`

`detect_maneuvers(con, norad_ids)` called at end of `ingest_update()`:

1. Query the two most recent GP epochs (current `gp` row + most recent `gp_history` row)
2. Compute absolute deltas for each tracked element
3. If any delta exceeds its threshold, INSERT into `maneuvers`
4. Log summary: `"Detected N maneuvers across M objects"`

Runs only on the set of objects just updated — no full-catalog scan.

## 4. Query helpers in `query.py`

- `get_maneuvers(norad_id=None, since=None)` — list detections, optionally filtered
- `get_maneuvering_objects(since=None)` — distinct NORAD_CAT_IDs that maneuvered recently
- `get_maneuver_history(norad_id)` — full maneuver timeline for one object

## 5. CLI integration in `ingest.py`

- Detection runs automatically after `--update`
- Add `--maneuvers` flag to display recent detections

## Verification

1. `python ingest.py --update` — maneuver detection log after GP ingestion
2. `python ingest.py --maneuvers` — display detected maneuvers
3. `python ingest.py --status` — maneuver count in status output
4. Validate against ISS (NORAD 25544) which maneuvers regularly
5. `SELECT * FROM maneuvers ORDER BY detected_at DESC LIMIT 20`
