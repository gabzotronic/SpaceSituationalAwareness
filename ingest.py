"""Space-Track API ingestion into local SQLite database."""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import spacetrack.operators as op
from spacetrack import SpaceTrackClient

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE,
    DB_PATH,
    MANEUVER_THRESHOLDS,
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
    """Create/open DB and apply schema, including idempotent OD-column migration."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    with open(SCHEMA_PATH) as f:
        con.executescript(f.read())

    # Idempotent migration: add Phase-1 OD columns to existing maneuvers tables.
    # ALTER TABLE fails silently if columns already exist.
    for col_def in [
        "CLASSIFICATION  TEXT",
        "KP              REAL",
        "VEL_RESIDUAL_MS REAL",
        "BSTAR_DELTA     REAL",
    ]:
        try:
            con.execute(f"ALTER TABLE maneuvers ADD COLUMN {col_def}")
            con.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

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

        # Run maneuver detection on the objects just updated
        detect_maneuvers(con, norad_ids)

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


# ── Maneuver detection ─────────────────────────────────────────

def detect_maneuvers(con, norad_ids):
    """Run OD-based maneuver detection for recently updated objects.

    Loads Kp once, then calls analyse_maneuvers() per NORAD ID so that
    results are persisted to the maneuvers table with CLASSIFICATION,
    VEL_RESIDUAL_MS, and BSTAR_DELTA populated.
    """
    if not norad_ids:
        return

    from analysis.maneuver_detection import analyse_maneuvers, get_kp_index

    data_dir = Path(__file__).parent / "analysis" / "data"

    # Determine time window from the objects just updated
    placeholders = ", ".join(["?"] * len(norad_ids))
    row = con.execute(
        f"SELECT MIN(EPOCH), MAX(EPOCH) FROM gp_history "
        f"WHERE NORAD_CAT_ID IN ({placeholders})",
        list(norad_ids),
    ).fetchone()
    window_start = (row[0] or "2020-01-01")[:10]
    window_end   = (row[1] or datetime.now(timezone.utc).date().isoformat())[:10]

    # Load Kp once for the entire batch
    try:
        kp_df = get_kp_index(window_start, window_end, data_dir)
    except Exception as exc:
        log.warning("Kp fetch failed (%s) — proceeding without space weather filter", exc)
        kp_df = None

    detected = 0
    for nid in norad_ids:
        try:
            result = analyse_maneuvers(
                norad_id=nid,
                data_dir=data_dir,
                con=con,
                kp_df=kp_df,
            )
            if not result.empty:
                detected += result["likely_maneuver"].sum()
        except Exception as exc:
            log.warning("analyse_maneuvers failed for NORAD %d: %s", nid, exc)

    log.info("Detected %d maneuver event(s) across %d object(s)", detected, len(norad_ids))


def show_maneuvers(con, limit=20):
    """Print recent maneuver detections."""
    rows = con.execute(
        "SELECT m.NORAD_CAT_ID, g.OBJECT_NAME, m.EPOCH_BEFORE, m.EPOCH_AFTER, "
        "m.DELTA_SMA, m.DELTA_INCLINATION, m.VEL_RESIDUAL_MS, m.CLASSIFICATION, m.detected_at "
        "FROM maneuvers m "
        "LEFT JOIN gp g ON m.NORAD_CAT_ID = g.NORAD_CAT_ID "
        "ORDER BY m.EPOCH_AFTER DESC LIMIT ?",
        (limit,),
    ).fetchall()

    total = con.execute("SELECT COUNT(*) FROM maneuvers").fetchone()[0]
    print(f"\n{'='*80}")
    print(f"  Maneuver Detections  (showing {len(rows)} of {total})")
    print(f"{'='*80}")
    if not rows:
        print("  No maneuvers detected yet.")
    else:
        print(f"  {'NORAD':>7}  {'Name':<22} {'Epoch After':<22} {'dSMA km':>8} {'VR m/s':>8} {'Class':<10}")
        print(f"  {'-'*7}  {'-'*22} {'-'*22} {'-'*8} {'-'*8} {'-'*10}")
        for r in rows:
            name  = (r["OBJECT_NAME"] or "")[:22]
            dsma  = f"{r['DELTA_SMA']:.2f}" if r["DELTA_SMA"] is not None else "N/A"
            vr    = f"{r['VEL_RESIDUAL_MS']:.2f}" if r["VEL_RESIDUAL_MS"] is not None else "N/A"
            cls   = r["CLASSIFICATION"] or "legacy"
            print(f"  {r['NORAD_CAT_ID']:>7}  {name:<22} {r['EPOCH_AFTER']:<22} {dsma:>8} {vr:>8} {cls:<10}")
    print(f"{'='*80}\n")


# ── Historical backfill ─────────────────────────────────────────

def backfill_gp_history(con, st, norad_id: int, start: str, end: str):
    """Fetch gp_history from Space-Track for one object over a date range
    and insert into the local gp_history table.

    Args:
        norad_id: NORAD_CAT_ID to backfill.
        start:    Start date string 'YYYY-MM-DD' (inclusive).
        end:      End date string 'YYYY-MM-DD' (inclusive).
    """
    log.info(
        "Fetching gp_history for NORAD %d from %s to %s ...", norad_id, start, end
    )
    raw = st.gp_history(
        norad_cat_id=norad_id,
        epoch=op.inclusive_range(start, end),
        orderby="epoch asc",
        format="json",
    )
    records = json.loads(raw) if isinstance(raw, str) else (raw or [])
    log.info("Received %d records from Space-Track", len(records))

    if not records:
        log.warning("No records returned — check NORAD ID and date range.")
        return

    rows = [_build_gp_row(r) for r in records]
    sql = (
        f"INSERT OR IGNORE INTO gp_history ({_gp_col_names()}) "
        f"VALUES ({_gp_placeholders()})"
    )
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        con.executemany(sql, batch)
        inserted += con.execute("SELECT changes()").fetchone()[0]
    con.commit()
    log.info("Inserted %d new records into gp_history (%d duplicates skipped)",
             inserted, len(rows) - inserted)


# ── Status ──────────────────────────────────────────────────────

def show_status(con):
    """Print database statistics."""
    gp_count = con.execute("SELECT COUNT(*) FROM gp").fetchone()[0]
    hist_count = con.execute("SELECT COUNT(*) FROM gp_history").fetchone()[0]
    sat_count = con.execute("SELECT COUNT(*) FROM satcat").fetchone()[0]
    mnv_count = con.execute("SELECT COUNT(*) FROM maneuvers").fetchone()[0]

    print(f"\n{'='*50}")
    print(f"  Satellite Catalog Database Status")
    print(f"{'='*50}")
    print(f"  GP records (latest):    {gp_count:>8,}")
    print(f"  GP history records:     {hist_count:>8,}")
    print(f"  SATCAT records:         {sat_count:>8,}")
    print(f"  Maneuver detections:    {mnv_count:>8,}")

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
    group.add_argument("--backfill", action="store_true",
                       help="Backfill gp_history for one object over a date range")
    group.add_argument("--status", action="store_true", help="Show DB stats")
    group.add_argument("--maneuvers", action="store_true", help="Show recent maneuver detections")
    _today     = datetime.now(timezone.utc).date()
    _start_def = (_today - timedelta(days=90)).isoformat()
    _end_def   = _today.isoformat()
    parser.add_argument("--norad", type=int,
                        help="NORAD_CAT_ID to backfill (required with --backfill)")
    parser.add_argument("--start", default=_start_def,
                        help=f"Start date for --backfill (YYYY-MM-DD, default: 90 days ago)")
    parser.add_argument("--end", default=_end_def,
                        help=f"End date for --backfill (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    con = init_db()

    if args.status:
        show_status(con)
        con.close()
        return

    if args.maneuvers:
        show_maneuvers(con)
        con.close()
        return

    st = SpaceTrackClient(identity=SPACETRACK_IDENTITY, password=SPACETRACK_PASSWORD)
    log.info("Authenticated with Space-Track")

    if args.full:
        ingest_full(con, st)
    elif args.update:
        ingest_update(con, st)
    elif args.backfill:
        if not args.norad:
            parser.error("--backfill requires --norad NORAD_ID")
        backfill_gp_history(con, st, args.norad, args.start, args.end)

    show_status(con)
    con.close()


if __name__ == "__main__":
    main()
