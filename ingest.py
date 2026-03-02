"""Space-Track API ingestion into local SQLite database."""

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

import spacetrack.operators as op
from spacetrack import SpaceTrackClient

from config import (
    BATCH_SIZE,
    DB_PATH,
    SCHEMA_PATH,
    SPACETRACK_IDENTITY,
    SPACETRACK_PASSWORD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# GP columns we store (order must match INSERT placeholders)
GP_COLS = [
    "GP_ID", "NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_ID", "EPOCH",
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH",
    "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
    "OBJECT_TYPE", "RCS_SIZE", "COUNTRY_CODE", "LAUNCH_DATE", "SITE",
    "DECAY_DATE", "FILE", "GP_ID",  # GP_ID_ORIG gets the same value
    "TLE_LINE0", "TLE_LINE1", "TLE_LINE2",
]

SATCAT_COLS = [
    "NORAD_CAT_ID", "SATNAME", "OBJECT_NAME", "OBJECT_ID", "INTLDES",
    "OBJECT_TYPE", "COUNTRY", "LAUNCH", "SITE", "DECAY",
    "PERIOD", "INCLINATION", "APOGEE", "PERIGEE",
    "RCS_SIZE", "CURRENT", "COMMENT", "FILE",
    "LAUNCH_YEAR", "LAUNCH_NUM", "LAUNCH_PIECE", "RCSVALUE",
]


def _coerce(val, col):
    """Coerce a JSON string value to the appropriate Python type."""
    if val is None or val == "":
        return None
    if col in (
        "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT",
        "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS",
        "PERIAPSIS", "RCSVALUE",
    ):
        return float(val)
    if col in (
        "GP_ID", "NORAD_CAT_ID", "EPHEMERIS_TYPE", "ELEMENT_SET_NO",
        "REV_AT_EPOCH", "FILE", "APOGEE", "PERIGEE",
        "LAUNCH_YEAR", "LAUNCH_NUM",
    ):
        return int(val)
    return val


def init_db():
    """Create/open DB and apply schema."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    with open(SCHEMA_PATH) as f:
        con.executescript(f.read())
    return con


def _build_gp_row(rec):
    """Extract a tuple of values from a GP JSON record."""
    vals = []
    for col in GP_COLS:
        raw = rec.get(col)
        vals.append(_coerce(raw, col))
    return tuple(vals)


def _build_satcat_row(rec):
    """Extract a tuple of values from a SATCAT JSON record."""
    vals = []
    for col in SATCAT_COLS:
        raw = rec.get(col)
        vals.append(_coerce(raw, col))
    return tuple(vals)


def _gp_placeholders():
    n = len(GP_COLS)
    return ", ".join(["?"] * n)


def _gp_col_names():
    cols = list(GP_COLS)
    cols[cols.index("GP_ID", 1)] = "GP_ID_ORIG"  # second GP_ID -> GP_ID_ORIG
    return ", ".join(cols)


def _satcat_placeholders():
    return ", ".join(["?"] * len(SATCAT_COLS))


def _satcat_col_names():
    return ", ".join(SATCAT_COLS)


# ── Full ingestion ──────────────────────────────────────────────

def ingest_full(con, st):
    """Full load of GP and SATCAT catalogs."""
    t0 = time.time()

    # — GP data —
    log.info("Fetching full GP catalog from Space-Track …")
    gp_json = st.gp(
        decay_date=None,
        orderby="norad_cat_id",
        format="json",
    )
    gp_records = json.loads(gp_json) if isinstance(gp_json, str) else gp_json
    log.info("Received %d GP records", len(gp_records))

    rows = [_build_gp_row(r) for r in gp_records]
    sql = f"INSERT OR REPLACE INTO gp ({_gp_col_names()}) VALUES ({_gp_placeholders()})"
    for i in range(0, len(rows), BATCH_SIZE):
        con.executemany(sql, rows[i : i + BATCH_SIZE])
    con.commit()
    log.info("Inserted %d GP records", len(rows))

    # — SATCAT data —
    log.info("Fetching full SATCAT from Space-Track …")
    sat_json = st.satcat(
        orderby="norad_cat_id",
        format="json",
    )
    sat_records = json.loads(sat_json) if isinstance(sat_json, str) else sat_json
    log.info("Received %d SATCAT records", len(sat_records))

    rows = [_build_satcat_row(r) for r in sat_records]
    sql = f"INSERT OR REPLACE INTO satcat ({_satcat_col_names()}) VALUES ({_satcat_placeholders()})"
    for i in range(0, len(rows), BATCH_SIZE):
        con.executemany(sql, rows[i : i + BATCH_SIZE])
    con.commit()
    log.info("Inserted %d SATCAT records", len(rows))

    # Update sync timestamp
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
        ("last_gp_sync", now),
    )
    con.execute(
        "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
        ("last_satcat_sync", now),
    )
    con.commit()

    elapsed = time.time() - t0
    log.info("Full ingestion completed in %.1f s", elapsed)


# ── Incremental update ──────────────────────────────────────────

def ingest_update(con, st):
    """Incremental update: fetch GP records newer than last sync, refresh SATCAT."""
    t0 = time.time()

    # Determine last sync time
    row = con.execute(
        "SELECT value FROM sync_meta WHERE key = 'last_gp_sync'"
    ).fetchone()
    if row is None:
        log.warning("No previous sync found — run --full first.")
        return
    last_sync = row[0]
    log.info("Last GP sync: %s", last_sync)

    # Fetch updated GP records (epoch newer than last sync)
    log.info("Fetching GP updates since %s …", last_sync)
    gp_json = st.gp(
        epoch=op.greater_than(last_sync),
        decay_date=None,
        orderby="norad_cat_id",
        format="json",
    )
    gp_records = json.loads(gp_json) if isinstance(gp_json, str) else gp_json
    log.info("Received %d updated GP records", len(gp_records))

    if gp_records:
        # Move existing records for these objects to history
        norad_ids = [int(r["NORAD_CAT_ID"]) for r in gp_records]
        for i in range(0, len(norad_ids), 500):
            batch = norad_ids[i : i + 500]
            placeholders = ", ".join(["?"] * len(batch))
            con.execute(
                f"INSERT OR IGNORE INTO gp_history SELECT * FROM gp "
                f"WHERE NORAD_CAT_ID IN ({placeholders})",
                batch,
            )
            con.execute(
                f"DELETE FROM gp WHERE NORAD_CAT_ID IN ({placeholders})",
                batch,
            )
        con.commit()
        log.info("Archived %d existing GP records to gp_history", len(norad_ids))

        # Insert new GP records
        rows = [_build_gp_row(r) for r in gp_records]
        sql = f"INSERT OR REPLACE INTO gp ({_gp_col_names()}) VALUES ({_gp_placeholders()})"
        for i in range(0, len(rows), BATCH_SIZE):
            con.executemany(sql, rows[i : i + BATCH_SIZE])
        con.commit()
        log.info("Inserted %d updated GP records", len(rows))

    # Update GP sync timestamp
    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
        ("last_gp_sync", now),
    )

    # Refresh SATCAT (full replace — lightweight, metadata changes rarely)
    log.info("Refreshing SATCAT …")
    sat_json = st.satcat(orderby="norad_cat_id", format="json")
    sat_records = json.loads(sat_json) if isinstance(sat_json, str) else sat_json
    rows = [_build_satcat_row(r) for r in sat_records]
    sql = f"INSERT OR REPLACE INTO satcat ({_satcat_col_names()}) VALUES ({_satcat_placeholders()})"
    for i in range(0, len(rows), BATCH_SIZE):
        con.executemany(sql, rows[i : i + BATCH_SIZE])
    con.commit()
    log.info("Refreshed %d SATCAT records", len(rows))

    con.execute(
        "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
        ("last_satcat_sync", now),
    )
    con.commit()

    elapsed = time.time() - t0
    log.info("Incremental update completed in %.1f s", elapsed)


# ── Status ──────────────────────────────────────────────────────

def show_status(con):
    """Print database statistics."""
    gp_count = con.execute("SELECT COUNT(*) FROM gp").fetchone()[0]
    hist_count = con.execute("SELECT COUNT(*) FROM gp_history").fetchone()[0]
    sat_count = con.execute("SELECT COUNT(*) FROM satcat").fetchone()[0]

    print(f"\n{'='*50}")
    print(f"  Satellite Catalog Database Status")
    print(f"{'='*50}")
    print(f"  GP records (latest):    {gp_count:>8,}")
    print(f"  GP history records:     {hist_count:>8,}")
    print(f"  SATCAT records:         {sat_count:>8,}")

    # Sync times
    for key_label in [("last_gp_sync", "Last GP sync"), ("last_satcat_sync", "Last SATCAT sync")]:
        row = con.execute(
            "SELECT value FROM sync_meta WHERE key = ?", (key_label[0],)
        ).fetchone()
        val = row[0] if row else "never"
        print(f"  {key_label[1]}:       {val}")

    # Object type breakdown
    if gp_count > 0:
        print(f"\n  GP by object type:")
        for row in con.execute(
            "SELECT OBJECT_TYPE, COUNT(*) AS cnt FROM gp GROUP BY OBJECT_TYPE ORDER BY cnt DESC"
        ):
            print(f"    {row[0] or 'NULL':<20} {row[1]:>8,}")

        # Epoch range
        oldest = con.execute("SELECT MIN(EPOCH) FROM gp").fetchone()[0]
        newest = con.execute("SELECT MAX(EPOCH) FROM gp").fetchone()[0]
        print(f"\n  Oldest epoch: {oldest}")
        print(f"  Newest epoch: {newest}")

    if sat_count > 0:
        print(f"\n  SATCAT by object type:")
        for row in con.execute(
            "SELECT OBJECT_TYPE, COUNT(*) AS cnt FROM satcat GROUP BY OBJECT_TYPE ORDER BY cnt DESC"
        ):
            print(f"    {row[0] or 'NULL':<20} {row[1]:>8,}")

    print(f"{'='*50}\n")


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Space-Track catalog ingestion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true", help="Full initial load")
    group.add_argument("--update", action="store_true", help="Incremental update since last sync")
    group.add_argument("--status", action="store_true", help="Show DB stats")
    args = parser.parse_args()

    con = init_db()

    if args.status:
        show_status(con)
        con.close()
        return

    st = SpaceTrackClient(identity=SPACETRACK_IDENTITY, password=SPACETRACK_PASSWORD)
    log.info("Authenticated with Space-Track")

    if args.full:
        ingest_full(con, st)
    elif args.update:
        ingest_update(con, st)

    show_status(con)
    con.close()


if __name__ == "__main__":
    main()
