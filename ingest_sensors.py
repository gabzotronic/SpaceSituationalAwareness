"""Ingest ground-based SSA sensor data from Excel into the sensor table.

Reads groundbased_SSA_infra.xlsx and populates the sensor table, filling in
missing fields (MAX_RANGE_KM, MIN_RANGE_KM, ALT_M, SENSITIVITY) with
reasonable defaults based on sensor type and aperture/remarks.

Usage:
    python ingest_sensors.py              # Load sensors into DB
    python ingest_sensors.py --status     # Show sensor counts
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys

import openpyxl
from pathlib import Path

from config import DB_PATH, SCHEMA_PATH

EXCEL_PATH = Path(__file__).parent / "groundbased_SSA_infra.xlsx"


# ── Defaults estimation ──────────────────────────────────────────────────────

def _extract_aperture_m(remarks: str) -> float | None:
    """Try to pull telescope aperture in metres from remarks like '0.5m', '2.6m'."""
    m = re.search(r'(\d+\.?\d*)\s*m\b', remarks)
    if m:
        val = float(m.group(1))
        if 0.1 <= val <= 10.0:
            return val
    return None


def _normalise_type(raw_type: str) -> str:
    """Map Excel sensor types to canonical DB values."""
    t = raw_type.strip().upper()
    if t == "PHASED ARRAY RADAR":
        return "PHASED_ARRAY_RADAR"
    if t in ("OPTICAL", "RADAR", "LASER"):
        return t
    return t


def estimate_defaults(sensor_type: str, remarks: str) -> dict:
    """Estimate MIN_RANGE_KM, MAX_RANGE_KM, ALT_M, SENSITIVITY from type + remarks."""
    remarks_lower = (remarks or "").lower()
    aperture = _extract_aperture_m(remarks or "")
    st = sensor_type.upper()

    # Altitude: observatories typically 500-2500m, default 500m
    alt_m = 500.0

    if st == "RADAR":
        min_range = 200.0
        if "64m" in remarks_lower or "34m" in remarks_lower:
            max_range = 5000.0
            sensitivity = 0.01   # m² min RCS
        elif "trajectography" in remarks_lower or "satam" in remarks_lower:
            max_range = 1500.0
            sensitivity = 0.1
        elif "600m ring" in remarks_lower:
            max_range = 3000.0
            sensitivity = 0.1
        else:
            max_range = 2000.0
            sensitivity = 0.5

    elif st == "PHASED_ARRAY_RADAR":
        min_range = 200.0
        if "vhf" in remarks_lower:
            # Early-warning class (GRAVES, Russian VHF radars)
            max_range = 6000.0
            sensitivity = 0.1
        elif "l-band" in remarks_lower:
            max_range = 3000.0
            sensitivity = 0.05
        else:
            max_range = 3000.0
            sensitivity = 0.1

    elif st == "LASER":
        min_range = 300.0
        max_range = 40000.0
        sensitivity = 18.0   # limiting magnitude equivalent
        alt_m = 800.0

    else:  # OPTICAL
        min_range = 200.0
        if aperture is not None:
            if aperture >= 2.0:
                max_range = 42000.0
                sensitivity = 21.0
            elif aperture >= 1.0:
                max_range = 40000.0
                sensitivity = 20.0
            elif aperture >= 0.5:
                max_range = 36000.0
                sensitivity = 18.0
            else:
                max_range = 2000.0
                sensitivity = 16.0
        else:
            # Unknown aperture optical — assume mid-range
            max_range = 36000.0
            sensitivity = 18.0
        alt_m = 1000.0

    return {
        "MIN_RANGE_KM": min_range,
        "MAX_RANGE_KM": max_range,
        "ALT_M": alt_m,
        "SENSITIVITY": sensitivity,
    }


# ── Database ─────────────────────────────────────────────────────────────────

def ensure_schema(con: sqlite3.Connection):
    """Create tables if they don't exist."""
    con.executescript(SCHEMA_PATH.read_text())


def ingest_sensors(excel_path: Path = EXCEL_PATH):
    """Read Excel and insert sensor rows into the database."""
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(min_row=1, values_only=True))
    header = [str(h).strip() for h in rows[0]]
    data = rows[1:]

    col = {name: i for i, name in enumerate(header)}

    con = sqlite3.connect(str(DB_PATH))
    ensure_schema(con)

    inserted = 0
    skipped = 0

    for row in data:
        def cell(name):
            idx = col.get(name)
            if idx is None or idx >= len(row):
                return None
            return row[idx]

        name = str(cell("Sensor Name") or "").strip()
        if not name:
            continue

        raw_type = str(cell("Sensor Type") or "OPTICAL").strip()
        sensor_type = _normalise_type(raw_type)
        lat = cell("Lat")
        lon = cell("Lon")
        if lat is None or lon is None:
            skipped += 1
            continue

        lat = float(lat)
        lon = float(lon)
        min_el = float(cell("Elevation Min") or 0)
        max_el = float(cell("Elevation Max") or 90)
        az_min = float(cell("Azimuth Min") or -180)
        az_max = float(cell("Azimuth Max") or 180)
        orientation = float(cell("Orientation") or 0)
        country = str(cell("Country") or "").strip()
        organisation = str(cell("Organisation") or "").strip()
        org_type = str(cell("Org Type") or "").strip()
        remarks = str(cell("Remarks") or "").strip()

        defaults = estimate_defaults(sensor_type, remarks)

        try:
            con.execute(
                """INSERT OR REPLACE INTO sensor
                   (NAME, COUNTRY, ORGANISATION, ORG_TYPE, TYPE,
                    LAT, LON, ALT_M, MIN_EL_DEG, MAX_EL_DEG,
                    AZ_MIN_DEG, AZ_MAX_DEG, ORIENTATION_DEG,
                    MIN_RANGE_KM, MAX_RANGE_KM, SENSITIVITY, REMARKS)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name, country, organisation, org_type, sensor_type,
                    lat, lon, defaults["ALT_M"], min_el, max_el,
                    az_min, az_max, orientation,
                    defaults["MIN_RANGE_KM"], defaults["MAX_RANGE_KM"],
                    defaults["SENSITIVITY"], remarks,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    con.commit()
    con.close()
    wb.close()

    print(f"Sensors ingested: {inserted}  |  Skipped: {skipped}")
    return inserted


def show_status():
    """Print sensor table summary."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    total = con.execute("SELECT COUNT(*) FROM sensor").fetchone()[0]
    print(f"\nSensor table: {total} records\n")

    print("By type:")
    for row in con.execute(
        "SELECT TYPE, COUNT(*) AS cnt FROM sensor GROUP BY TYPE ORDER BY cnt DESC"
    ):
        print(f"  {row['TYPE']:<22} {row['cnt']:>4}")

    print("\nBy country (top 15):")
    for row in con.execute(
        "SELECT COUNTRY, COUNT(*) AS cnt FROM sensor GROUP BY COUNTRY ORDER BY cnt DESC LIMIT 15"
    ):
        print(f"  {row['COUNTRY']:<22} {row['cnt']:>4}")

    print("\nRange capabilities:")
    for row in con.execute(
        "SELECT TYPE, MIN(MIN_RANGE_KM) AS min_r, MAX(MAX_RANGE_KM) AS max_r "
        "FROM sensor GROUP BY TYPE ORDER BY max_r DESC"
    ):
        print(f"  {row['TYPE']:<22}  {row['min_r']:>7.0f} – {row['max_r']:>7.0f} km")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest SSA sensors from Excel.")
    parser.add_argument("--status", action="store_true", help="Show sensor table summary")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        ingest_sensors()
        show_status()


if __name__ == "__main__":
    main()
