"""Query helpers for the satellite catalog database."""

from __future__ import annotations

import sqlite3
from typing import Optional

from config import DB_PATH


def _connect():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def get_object(norad_id: int) -> dict | None:
    """Return combined GP + SATCAT data for a single object."""
    con = _connect()
    row = con.execute(
        "SELECT * FROM catalog_view WHERE NORAD_CAT_ID = ?", (norad_id,)
    ).fetchone()
    con.close()
    return dict(row) if row else None


def get_objects_in_altitude_band(min_km: float, max_km: float) -> list[dict]:
    """Return objects whose perigee–apogee range overlaps [min_km, max_km].

    An object overlaps the band if its periapsis < max_km AND apoapsis > min_km.
    """
    con = _connect()
    rows = con.execute(
        "SELECT * FROM gp WHERE PERIAPSIS <= ? AND APOAPSIS >= ? ORDER BY NORAD_CAT_ID",
        (max_km, min_km),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_objects_by_type(object_type: str) -> list[dict]:
    """Filter GP records by object type (PAYLOAD, ROCKET BODY, DEBRIS, etc.)."""
    con = _connect()
    rows = con.execute(
        "SELECT * FROM gp WHERE OBJECT_TYPE = ? ORDER BY NORAD_CAT_ID",
        (object_type,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_catalog_stats() -> dict:
    """Return summary statistics about the catalog."""
    con = _connect()
    stats = {}
    stats["gp_count"] = con.execute("SELECT COUNT(*) FROM gp").fetchone()[0]
    stats["gp_history_count"] = con.execute("SELECT COUNT(*) FROM gp_history").fetchone()[0]
    stats["satcat_count"] = con.execute("SELECT COUNT(*) FROM satcat").fetchone()[0]

    # Breakdown by object type
    stats["gp_by_type"] = {
        row[0]: row[1]
        for row in con.execute(
            "SELECT OBJECT_TYPE, COUNT(*) FROM gp GROUP BY OBJECT_TYPE ORDER BY COUNT(*) DESC"
        )
    }
    stats["satcat_by_country"] = {
        row[0]: row[1]
        for row in con.execute(
            "SELECT COUNTRY, COUNT(*) FROM satcat GROUP BY COUNTRY ORDER BY COUNT(*) DESC LIMIT 20"
        )
    }

    stats["oldest_epoch"] = con.execute("SELECT MIN(EPOCH) FROM gp").fetchone()[0]
    stats["newest_epoch"] = con.execute("SELECT MAX(EPOCH) FROM gp").fetchone()[0]

    con.close()
    return stats


def get_tle(norad_id: int) -> tuple[str, str, str] | None:
    """Return (line0, line1, line2) TLE strings ready for SGP4 propagation."""
    con = _connect()
    row = con.execute(
        "SELECT TLE_LINE0, TLE_LINE1, TLE_LINE2 FROM gp WHERE NORAD_CAT_ID = ?",
        (norad_id,),
    ).fetchone()
    con.close()
    if row is None:
        return None
    return (row["TLE_LINE0"], row["TLE_LINE1"], row["TLE_LINE2"])


if __name__ == "__main__":
    # Quick verification
    import json

    stats = get_catalog_stats()
    print(json.dumps(stats, indent=2, default=str))

    # Try looking up ISS
    iss = get_object(25544)
    if iss:
        print(f"\nISS: {iss['OBJECT_NAME']}, Epoch: {iss['EPOCH']}")
        tle = get_tle(25544)
        if tle:
            print(f"  {tle[0]}")
            print(f"  {tle[1]}")
            print(f"  {tle[2]}")
    else:
        print("\nISS not found — run 'python ingest.py --full' first")
