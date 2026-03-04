"""
visibility_windows.py -- On-demand sensor-satellite visibility computation (Approach B).

Given a list of sensor names and a time window, returns the list of satellites
visible to those sensors, with pass details (AOS, LOS, max elevation, range)
grouped per satellite.

Steps per sensor:
  1. SQL pre-filter: keep objects whose perigee-apogee band overlaps the
     sensor's [MIN_RANGE_KM, MAX_RANGE_KM] range envelope.
  2. SGP4 propagation at a configurable time step across the requested window.
  3. Geometric visibility check: elevation >= sensor min_el, slant range <=
     sensor max_range.
  4. Collect AOS / LOS / max-elevation passes.

Usage:
    python visibility_windows.py --list-sensors
    python visibility_windows.py --sensors "Space Fence AN/FSY-3,Fylingdales" --hours 6
    python visibility_windows.py --sensor-ids 119,153 --hours 12
    python visibility_windows.py --sensors all --hours 1 --step 60

Dependencies: sgp4>=2.22, numpy
"""
# Performance notes:
#   Vectorised over time steps (sgp4_array + numpy array ops) to eliminate
#   Python loop overhead — dominant cost in the scalar version.
#   Parallelised over candidates with ProcessPoolExecutor (4 workers).

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import numpy as np
from sgp4.api import Satrec, jday
from pathlib import Path

# Add parent directory to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_PATH

# -- Constants ----------------------------------------------------------------

EARTH_RADIUS_KM = 6378.137          # WGS-84 semi-major axis
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_E2 = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2

DEFAULT_STEP_S = 30       # propagation step (seconds)
DEFAULT_HOURS = 24
MIN_PASS_DURATION_S = 10  # ignore passes shorter than this


# -- Database helpers ---------------------------------------------------------

def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def list_sensors() -> list[dict]:
    """Return all sensors from the database."""
    con = _connect()
    rows = con.execute(
        "SELECT SENSOR_ID, NAME, COUNTRY, TYPE, LAT, LON, "
        "MIN_RANGE_KM, MAX_RANGE_KM, MIN_EL_DEG FROM sensor ORDER BY SENSOR_ID"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_sensors_by_names(names: list[str]) -> list[dict]:
    """Fetch sensors by name. Tries exact match first, falls back to substring."""
    con = _connect()
    sensors = []
    for name in names:
        name = name.strip()
        rows = con.execute(
            "SELECT * FROM sensor WHERE UPPER(NAME) = UPPER(?) ORDER BY SENSOR_ID",
            (name,),
        ).fetchall()
        if not rows:
            rows = con.execute(
                "SELECT * FROM sensor WHERE NAME LIKE ? ORDER BY SENSOR_ID",
                (f"%{name}%",),
            ).fetchall()
        sensors.extend(dict(r) for r in rows)
    con.close()
    return sensors


def get_sensors_by_ids(ids: list[int]) -> list[dict]:
    """Fetch sensors by SENSOR_ID."""
    con = _connect()
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        f"SELECT * FROM sensor WHERE SENSOR_ID IN ({placeholders}) ORDER BY SENSOR_ID",
        ids,
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_candidates_for_sensor(sensor: dict) -> list[dict]:
    """SQL pre-filter: objects whose altitude band overlaps the sensor's range envelope."""
    max_alt = sensor["MAX_RANGE_KM"]
    min_alt = max(0, sensor["MIN_RANGE_KM"] * math.sin(math.radians(sensor["MIN_EL_DEG"])))

    con = _connect()
    rows = con.execute(
        """SELECT NORAD_CAT_ID, OBJECT_NAME, OBJECT_TYPE, RCS_SIZE,
                  TLE_LINE1, TLE_LINE2, PERIAPSIS, APOAPSIS, INCLINATION
           FROM gp
           WHERE PERIAPSIS <= ?
             AND APOAPSIS  >= ?
             AND TLE_LINE1 IS NOT NULL
             AND TLE_LINE2 IS NOT NULL
           ORDER BY NORAD_CAT_ID""",
        (max_alt, min_alt),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# -- Coordinate transforms ---------------------------------------------------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS-84 geodetic (lat, lon, alt) to ECEF (km)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    alt_km = alt_m / 1000.0

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = EARTH_RADIUS_KM / math.sqrt(1.0 - EARTH_E2 * sin_lat ** 2)

    x = (N + alt_km) * cos_lat * math.cos(lon)
    y = (N + alt_km) * cos_lat * math.sin(lon)
    z = (N * (1.0 - EARTH_E2) + alt_km) * sin_lat
    return np.array([x, y, z])


def _gmst_rad(jd: float, fr: float) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982)."""
    tut1 = (jd + fr - 2451545.0) / 36525.0
    theta = (67310.54841
             + (876600.0 * 3600.0 + 8640184.812866) * tut1
             + 0.093104 * tut1 ** 2
             - 6.2e-6 * tut1 ** 3)
    return math.radians(theta / 240.0 % 360.0)


def eci_to_ecef(r_eci: tuple, jd: float, fr: float) -> np.ndarray:
    """Rotate ECI position to ECEF using GMST."""
    gst = _gmst_rad(jd, fr)
    cos_g = math.cos(gst)
    sin_g = math.sin(gst)
    x = r_eci[0] * cos_g + r_eci[1] * sin_g
    y = -r_eci[0] * sin_g + r_eci[1] * cos_g
    z = r_eci[2]
    return np.array([x, y, z])


def compute_azel(sat_ecef: np.ndarray, sensor_ecef: np.ndarray,
                 sensor_lat_deg: float, sensor_lon_deg: float) -> Tuple[float, float, float]:
    """Compute azimuth (deg), elevation (deg), and slant range (km) in ENU frame."""
    diff = sat_ecef - sensor_ecef
    slant_range = np.linalg.norm(diff)

    lat = math.radians(sensor_lat_deg)
    lon = math.radians(sensor_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    e = -sin_lon * diff[0] + cos_lon * diff[1]
    n = -sin_lat * cos_lon * diff[0] - sin_lat * sin_lon * diff[1] + cos_lat * diff[2]
    u = cos_lat * cos_lon * diff[0] + cos_lat * sin_lon * diff[1] + sin_lat * diff[2]

    el_deg = math.degrees(math.asin(u / slant_range)) if slant_range > 0 else 0.0
    az_deg = math.degrees(math.atan2(e, n)) % 360.0
    return az_deg, el_deg, slant_range


# -- Vectorised coordinate transforms (all time steps for one satellite) ------

def _gmst_rad_vec(jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Vectorised GMST in radians (IAU 1982). jd, fr: [N]."""
    tut1 = (jd + fr - 2451545.0) / 36525.0
    theta = (67310.54841
             + (876600.0 * 3600.0 + 8640184.812866) * tut1
             + 0.093104 * tut1 ** 2
             - 6.2e-6 * tut1 ** 3)
    return np.radians(theta / 240.0 % 360.0)


def eci_to_ecef_vec(r_eci: np.ndarray, jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Vectorised ECI -> ECEF rotation. r_eci: [N,3], returns [N,3]."""
    gst = _gmst_rad_vec(jd, fr)   # [N]
    cos_g = np.cos(gst)
    sin_g = np.sin(gst)
    x = r_eci[:, 0] * cos_g + r_eci[:, 1] * sin_g
    y = -r_eci[:, 0] * sin_g + r_eci[:, 1] * cos_g
    z = r_eci[:, 2]
    return np.stack([x, y, z], axis=1)   # [N,3]


def compute_azel_vec(
    sat_ecef: np.ndarray,
    sensor_ecef: np.ndarray,
    sensor_lat_deg: float,
    sensor_lon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised az/el/range for all time steps. sat_ecef: [N,3], returns 3×[N]."""
    diff = sat_ecef - sensor_ecef                        # [N,3]
    slant = np.sqrt((diff ** 2).sum(axis=1))             # [N], avoids linalg.norm overhead

    lat = math.radians(sensor_lat_deg)
    lon = math.radians(sensor_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    e = -sin_lon * diff[:, 0] + cos_lon * diff[:, 1]
    n = (-sin_lat * cos_lon * diff[:, 0]
         - sin_lat * sin_lon * diff[:, 1]
         + cos_lat * diff[:, 2])
    u = (cos_lat * cos_lon * diff[:, 0]
         + cos_lat * sin_lon * diff[:, 1]
         + sin_lat * diff[:, 2])

    safe_slant = np.where(slant > 0, slant, 1.0)
    el = np.degrees(np.arcsin(np.clip(u / safe_slant, -1.0, 1.0)))
    el = np.where(slant > 0, el, 0.0)
    az = np.degrees(np.arctan2(e, n)) % 360.0
    return az, el, slant


# -- SGP4 helpers -------------------------------------------------------------

def make_satrec(line1: str, line2: str) -> Optional[Satrec]:
    try:
        return Satrec.twoline2rv(line1, line2)
    except Exception:
        return None


def dt_to_jdfr(dt: datetime) -> Tuple[float, float]:
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond / 1e6,
    )
    return jd, fr


def propagate(sat: Satrec, jd: float, fr: float):
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None
    return r


# -- Visibility pass detection ------------------------------------------------

def find_passes(
    sat: Satrec,
    sensor_ecef: np.ndarray,
    sensor_lat: float,
    sensor_lon: float,
    min_el_deg: float,
    max_range_km: float,
    t_start: datetime,
    t_end: datetime,
    step_s: float,
) -> list[dict]:
    """Find all visibility passes for one satellite from one sensor.

    Returns list of dicts: {aos, los, max_el_deg, max_el_time, min_range_km, duration_s}
    """
    passes = []
    step = timedelta(seconds=step_s)
    t = t_start

    in_pass = False
    pass_aos = None
    pass_max_el = 0.0
    pass_max_el_time = None
    pass_min_range = 1e9

    while t <= t_end:
        jd, fr = dt_to_jdfr(t)
        r_eci = propagate(sat, jd, fr)

        if r_eci is not None:
            r_ecef = eci_to_ecef(r_eci, jd, fr)
            az, el, slant = compute_azel(r_ecef, sensor_ecef, sensor_lat, sensor_lon)
            visible = el >= min_el_deg and slant <= max_range_km
        else:
            visible = False

        if visible:
            if not in_pass:
                in_pass = True
                pass_aos = t
                pass_max_el = el
                pass_max_el_time = t
                pass_min_range = slant
            else:
                if el > pass_max_el:
                    pass_max_el = el
                    pass_max_el_time = t
                if slant < pass_min_range:
                    pass_min_range = slant
        else:
            if in_pass:
                duration_s = (t - pass_aos).total_seconds()
                if duration_s >= MIN_PASS_DURATION_S:
                    passes.append({
                        "aos": pass_aos,
                        "los": t,
                        "max_el_deg": pass_max_el,
                        "max_el_time": pass_max_el_time,
                        "min_range_km": pass_min_range,
                        "duration_s": duration_s,
                    })
                in_pass = False

        t += step

    if in_pass:
        duration_s = (t_end - pass_aos).total_seconds()
        if duration_s >= MIN_PASS_DURATION_S:
            passes.append({
                "aos": pass_aos,
                "los": t_end,
                "max_el_deg": pass_max_el,
                "max_el_time": pass_max_el_time,
                "min_range_km": pass_min_range,
                "duration_s": duration_s,
            })

    return passes


def find_passes_vec(
    sat: Satrec,
    sensor_ecef: np.ndarray,
    sensor_lat: float,
    sensor_lon: float,
    min_el_deg: float,
    max_range_km: float,
    jd_arr: np.ndarray,
    fr_arr: np.ndarray,
    t_start: datetime,
    step_s: float,
) -> list[dict]:
    """Vectorised pass detection: propagates all time steps in one sgp4_array call.

    jd_arr / fr_arr are precomputed once per sensor window (shape [N]).
    """
    e_arr, r_arr, _ = sat.sgp4_array(jd_arr, fr_arr)   # [N], [N,3], [N,3]

    valid = e_arr == 0   # [N] bool
    if not valid.any():
        return []

    # Zero-out failed steps so they fall through as not-visible
    if not valid.all():
        r_arr = r_arr.copy()
        r_arr[~valid] = 0.0

    r_ecef = eci_to_ecef_vec(r_arr, jd_arr, fr_arr)           # [N,3]
    _, el, slant = compute_azel_vec(r_ecef, sensor_ecef, sensor_lat, sensor_lon)

    visible = valid & (el >= min_el_deg) & (slant <= max_range_km)   # [N] bool
    if not visible.any():
        return []

    # Detect rising / falling edges with padded diff
    padded = np.empty(len(visible) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = visible.view(np.int8)
    padded[-1] = 0
    edges = np.diff(padded)                        # [N+1]
    aos_idxs = np.where(edges == 1)[0]             # first visible index of each pass
    los_idxs = np.where(edges == -1)[0]            # first NOT-visible index after pass

    passes = []
    for i_aos, i_los in zip(aos_idxs, los_idxs):
        duration_s = int(i_los - i_aos) * step_s
        if duration_s < MIN_PASS_DURATION_S:
            continue

        seg_el = el[i_aos:i_los]
        seg_sl = slant[i_aos:i_los]
        max_el_idx = int(i_aos) + int(np.argmax(seg_el))

        passes.append({
            "aos":         t_start + timedelta(seconds=int(i_aos) * step_s),
            "los":         t_start + timedelta(seconds=int(i_los) * step_s),
            "max_el_deg":  float(seg_el.max()),
            "max_el_time": t_start + timedelta(seconds=max_el_idx * step_s),
            "min_range_km": float(seg_sl.min()),
            "duration_s":  duration_s,
        })

    return passes


# -- Worker for parallelised candidate processing ----------------------------
# Must be a module-level function so Windows 'spawn' workers can pickle it.

def _worker_chunk(args):
    """Process a chunk of candidates in a worker process.

    args: (candidates, sensor_ecef, lat, lon, min_el, max_range,
           jd_arr, fr_arr, t_start, step_s)
    Returns list of (cand_dict, passes_list).
    """
    (candidates, sensor_ecef, sensor_lat, sensor_lon,
     min_el_deg, max_range_km, jd_arr, fr_arr, t_start, step_s) = args

    results = []
    for cand in candidates:
        sat = make_satrec(cand["TLE_LINE1"], cand["TLE_LINE2"])
        if sat is None:
            continue
        passes = find_passes_vec(
            sat, sensor_ecef, sensor_lat, sensor_lon,
            min_el_deg, max_range_km,
            jd_arr, fr_arr, t_start, step_s,
        )
        if passes:
            results.append((cand, passes))
    return results


# -- Main computation ---------------------------------------------------------

def compute_visibility(
    sensors: list[dict],
    t_start: datetime,
    t_end: datetime,
    step_s: float = DEFAULT_STEP_S,
    n_workers: int = 4,
) -> dict[int, dict]:
    """Compute visibility windows for the given sensors.

    Returns a dict keyed by NORAD_CAT_ID:
        {
            NORAD_CAT_ID: {
                "OBJECT_NAME": str,
                "OBJECT_TYPE": str,
                "RCS_SIZE": str,
                "sensors": {
                    sensor_name: [pass, pass, ...],
                    ...
                }
            }
        }

    Each pass dict: {aos, los, max_el_deg, max_el_time, min_range_km, duration_s}
    """
    # Object-centric accumulator: NORAD_CAT_ID -> {metadata, sensors: {name: [passes]}}
    visible_objects: dict[int, dict] = {}

    # Precompute JD/FR arrays for the full window — shared across all candidates
    n_steps = int((t_end - t_start).total_seconds() / step_s) + 1
    jd0, fr0 = dt_to_jdfr(t_start)
    step_days = step_s / 86400.0
    total_jd = jd0 + fr0 + np.arange(n_steps, dtype=np.float64) * step_days
    jd_arr = np.floor(total_jd)
    fr_arr = total_jd - jd_arr

    for sensor in sensors:
        sensor_ecef = geodetic_to_ecef(sensor["LAT"], sensor["LON"], sensor["ALT_M"])
        min_el = sensor["MIN_EL_DEG"]
        max_range = sensor["MAX_RANGE_KM"]
        sensor_name = sensor["NAME"]

        # Step 1: SQL pre-filter
        candidates = get_candidates_for_sensor(sensor)
        print(
            f"  {sensor_name:<30} | "
            f"type={sensor['TYPE']:<20} | "
            f"range={max_range:.0f} km | "
            f"{len(candidates):>5} candidates"
        )

        # Step 2: parallel SGP4 propagation + geometric visibility
        n_with_passes = 0
        n_actual = min(n_workers, len(candidates))
        chunk_size = math.ceil(len(candidates) / n_actual)
        chunks = [
            candidates[i: i + chunk_size]
            for i in range(0, len(candidates), chunk_size)
        ]
        shared = (
            sensor_ecef, sensor["LAT"], sensor["LON"],
            min_el, max_range, jd_arr, fr_arr, t_start, step_s,
        )
        print(
            f"    {len(candidates):,} candidates -> "
            f"{len(chunks)} chunks x {n_actual} workers",
            flush=True,
        )

        with ProcessPoolExecutor(max_workers=n_actual) as executor:
            futures = [
                executor.submit(_worker_chunk, (chunk,) + shared)
                for chunk in chunks
            ]
            for future in as_completed(futures):
                for cand, passes in future.result():
                    n_with_passes += 1
                    nid = cand["NORAD_CAT_ID"]
                    if nid not in visible_objects:
                        visible_objects[nid] = {
                            "NORAD_CAT_ID": nid,
                            "OBJECT_NAME": cand["OBJECT_NAME"],
                            "OBJECT_TYPE": cand["OBJECT_TYPE"],
                            "RCS_SIZE": cand["RCS_SIZE"],
                            "sensors": {},
                        }
                    visible_objects[nid]["sensors"][sensor_name] = passes

        print(
            f"    {n_with_passes} objects with passes"
            f"                              "
        )

    return visible_objects


# -- Reporting ----------------------------------------------------------------

def print_report(
    visible_objects: dict[int, dict],
    sensor_names: list[str],
    t_start: datetime,
    t_end: datetime,
):
    """Print satellite-centric visibility report."""
    print()
    print("=" * 100)
    print(
        f"  VISIBLE SATELLITES  |  "
        f"{t_start.strftime('%Y-%m-%d %H:%M')} -> {t_end.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    print(f"  Sensors: {', '.join(sensor_names)}")
    print("=" * 100)

    if not visible_objects:
        print("  No satellites visible from the selected sensors in this window.\n")
        return

    # Sort by NORAD_CAT_ID
    sorted_objects = sorted(visible_objects.values(), key=lambda o: o["NORAD_CAT_ID"])

    # Summary table: one row per satellite, columns show which sensors see it
    print(f"\n  {'NORAD':>6}  {'Name':<24}  {'Type':<12}  {'RCS':<6}", end="")
    for sn in sensor_names:
        short = sn[:16]
        print(f"  {short:>16}", end="")
    print()
    print("  " + "-" * (52 + 18 * len(sensor_names)))

    for obj in sorted_objects:
        nm = (obj["OBJECT_NAME"] or "?")[:24]
        ot = (obj["OBJECT_TYPE"] or "?")[:12]
        rcs = (obj["RCS_SIZE"] or "?")[:6]
        print(f"  {obj['NORAD_CAT_ID']:>6}  {nm:<24}  {ot:<12}  {rcs:<6}", end="")
        for sn in sensor_names:
            passes = obj["sensors"].get(sn, [])
            if passes:
                total_dur = sum(p["duration_s"] for p in passes) / 60.0
                best_el = max(p["max_el_deg"] for p in passes)
                print(f"  {len(passes):>2}p {total_dur:>4.0f}m {best_el:>4.0f}el", end="")
            else:
                print(f"  {'--':>16}", end="")
        print()

    # Totals
    n_total = len(sorted_objects)
    n_all_sensors = sum(
        1 for obj in sorted_objects
        if all(sn in obj["sensors"] for sn in sensor_names)
    )
    n_any_sensor = n_total

    print()
    print(f"  Total visible satellites : {n_total}")
    print(f"  Visible by ALL sensors  : {n_all_sensors}")
    for sn in sensor_names:
        n = sum(1 for obj in sorted_objects if sn in obj["sensors"])
        passes = sum(
            len(obj["sensors"].get(sn, []))
            for obj in sorted_objects
        )
        print(f"  {sn:<30}: {n:>5} objects, {passes:>6} passes")
    print()

    # Detailed pass listing (first 50 objects)
    print("-" * 100)
    print("  PASS DETAILS (first 50 objects)")
    print("-" * 100)
    for obj in sorted_objects[:50]:
        nm = obj["OBJECT_NAME"] or "?"
        print(f"\n  {obj['NORAD_CAT_ID']} {nm}  ({obj['OBJECT_TYPE'] or '?'})")
        for sn in sensor_names:
            passes = obj["sensors"].get(sn, [])
            if not passes:
                continue
            for p in passes:
                aos = p["aos"].strftime("%Y-%m-%d %H:%M:%S")
                los = p["los"].strftime("%Y-%m-%d %H:%M:%S")
                dur_m = p["duration_s"] / 60.0
                print(
                    f"    {sn:<30}  AOS {aos}  LOS {los}  "
                    f"MaxEl {p['max_el_deg']:>5.1f}  "
                    f"Dur {dur_m:>5.1f}m  "
                    f"Range {p['min_range_km']:>6.0f}km"
                )

    if len(sorted_objects) > 50:
        print(f"\n  ... {len(sorted_objects) - 50} more objects not shown")
    print()


def print_sensor_list():
    """Print all available sensors."""
    sensors = list_sensors()
    print(f"\nAvailable sensors: {len(sensors)}\n")
    print(f"  {'ID':>4}  {'Name':<32}  {'Country':<14}  {'Type':<20}  "
          f"{'Lat':>7}  {'Lon':>8}  {'Range km':>9}  {'MinEl':>5}")
    print("  " + "-" * 114)
    for s in sensors:
        print(
            f"  {s['SENSOR_ID']:>4}  {s['NAME']:<32}  {s['COUNTRY']:<14}  "
            f"{s['TYPE']:<20}  {s['LAT']:>7.2f}  {s['LON']:>8.2f}  "
            f"{s['MAX_RANGE_KM']:>9.0f}  {s['MIN_EL_DEG']:>5.1f}"
        )
    print()


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute which satellites are visible from selected sensors."
    )
    parser.add_argument(
        "--list-sensors", action="store_true",
        help="List all available sensors and exit",
    )
    parser.add_argument(
        "--sensors", type=str, default=None,
        help='Comma-separated sensor names (exact match, fallback substring), or "all"',
    )
    parser.add_argument(
        "--sensor-ids", type=str, default=None,
        help="Comma-separated sensor IDs (integers)",
    )
    parser.add_argument(
        "--hours", type=float, default=DEFAULT_HOURS,
        help=f"Prediction window in hours from now (default: {DEFAULT_HOURS})",
    )
    parser.add_argument(
        "--step", type=float, default=DEFAULT_STEP_S,
        help=f"Propagation time step in seconds (default: {DEFAULT_STEP_S})",
    )
    parser.add_argument(
        "--min-el", type=float, default=None,
        help="Override minimum elevation for all sensors (degrees)",
    )
    args = parser.parse_args()

    if args.list_sensors:
        print_sensor_list()
        return

    # Resolve sensors
    if args.sensor_ids:
        ids = [int(x.strip()) for x in args.sensor_ids.split(",")]
        sensors = get_sensors_by_ids(ids)
    elif args.sensors:
        if args.sensors.strip().lower() == "all":
            sensors = list_sensors()
            ids = [s["SENSOR_ID"] for s in sensors]
            sensors = get_sensors_by_ids(ids)
        else:
            names = [n.strip() for n in args.sensors.split(",")]
            sensors = get_sensors_by_names(names)
    else:
        print("ERROR: specify --sensors or --sensor-ids (use --list-sensors to see options)")
        sys.exit(1)

    if not sensors:
        print("ERROR: no sensors matched your query.")
        sys.exit(1)

    # Apply min-el override
    if args.min_el is not None:
        for s in sensors:
            s["MIN_EL_DEG"] = args.min_el

    # Deduplicate
    seen = set()
    unique_sensors = []
    for s in sensors:
        if s["SENSOR_ID"] not in seen:
            seen.add(s["SENSOR_ID"])
            unique_sensors.append(s)
    sensors = unique_sensors
    sensor_names = [s["NAME"] for s in sensors]

    t_start = datetime.now(timezone.utc)
    t_end = t_start + timedelta(hours=args.hours)

    print(f"\nVisibility computation")
    print(f"  Sensors : {', '.join(sensor_names)}")
    print(f"  Window  : {t_start.strftime('%Y-%m-%d %H:%M')} -> "
          f"{t_end.strftime('%Y-%m-%d %H:%M')} UTC  ({args.hours}h)")
    print(f"  Step    : {args.step}s\n")

    visible_objects = compute_visibility(sensors, t_start, t_end, step_s=args.step)
    print_report(visible_objects, sensor_names, t_start, t_end)


if __name__ == "__main__":
    main()
