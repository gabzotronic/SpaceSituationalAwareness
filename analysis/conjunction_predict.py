"""
conjunction_predict.py — Basic conjunction prediction for TELEOS-2 (or any primary).

Approach:
  1. Look up the primary satellite by name in the local catalog.
  2. Altitude-band pre-screening: keep objects whose perigee–apogee range
     overlaps the primary's band ± ALTITUDE_PAD_KM.
  3. Coarse SGP4 propagation (COARSE_STEP_S) across the full time window.
  4. When coarse distance < REFINE_TRIGGER_KM, refine at FINE_STEP_S steps
     to find the true Time of Closest Approach (TCA).
  5. Report all events below the warning and red/critical thresholds.

Usage:
    python conjunction_predict.py
    python conjunction_predict.py --days 4 --warn-km 25 --red-km 5
    python conjunction_predict.py --primary "TELEOS-2" --days 2

Dependencies: sgp4>=2.22 (pip install sgp4)

NOTE: SGP4 positional error grows ~1–3 km/day from TLE epoch. All results
      are indicative screening only — not a substitute for high-fidelity CDMs.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Optional

from sgp4.api import Satrec, jday
from pathlib import Path

# Add parent directory to path to import config from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

# ── Tunable parameters ────────────────────────────────────────────────────────
DEFAULT_DAYS       = 30
ALTITUDE_PAD_KM    = 50    # Extend altitude band search by ±50 km
COARSE_STEP_S      = 60    # Seconds between coarse propagation samples
REFINE_TRIGGER_KM  = 100   # Start fine search when distance drops below this
FINE_STEP_S        = 5     # Seconds for fine-grained TCA search
DEFAULT_WARN_KM    = 25    # Yellow/warning threshold
DEFAULT_RED_KM     = 5     # Red/critical threshold


# ── Database helpers ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def find_primary(name: str) -> Optional[dict]:
    """Find a satellite by name (case-insensitive substring match)."""
    con = _connect()
    row = con.execute(
        """SELECT NORAD_CAT_ID, OBJECT_NAME, OBJECT_TYPE, TLE_LINE1, TLE_LINE2,
                  PERIAPSIS, APOAPSIS, INCLINATION, EPOCH
           FROM gp
           WHERE OBJECT_NAME LIKE ? AND TLE_LINE1 IS NOT NULL
           ORDER BY NORAD_CAT_ID LIMIT 1""",
        (f"%{name}%",),
    ).fetchone()
    con.close()
    return dict(row) if row else None


def get_candidates(periapsis: float, apoapsis: float, pad: float, primary_id: int) -> list:
    """Altitude-band pre-screen: objects whose orbit overlaps [peri-pad, apo+pad]."""
    lo = periapsis - pad
    hi = apoapsis + pad
    con = _connect()
    rows = con.execute(
        """SELECT NORAD_CAT_ID, OBJECT_NAME, OBJECT_TYPE, COUNTRY_CODE,
                  TLE_LINE1, TLE_LINE2, PERIAPSIS, APOAPSIS
           FROM gp
           WHERE PERIAPSIS <= ?
             AND APOAPSIS  >= ?
             AND NORAD_CAT_ID != ?
             AND TLE_LINE1 IS NOT NULL
             AND TLE_LINE2 IS NOT NULL
           ORDER BY NORAD_CAT_ID""",
        (hi, lo, primary_id),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── SGP4 helpers ──────────────────────────────────────────────────────────────

def make_satrec(line1: str, line2: str) -> Optional[Satrec]:
    try:
        return Satrec.twoline2rv(line1, line2)
    except Exception:
        return None


def propagate(sat: Satrec, jd: float, fr: float):
    """Return (r, v) in km / km/s. Returns (None, None) on error."""
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return r, v


def dt_to_jdfr(dt: datetime):
    """Convert a UTC datetime to SGP4's (jd, fr) pair."""
    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond / 1e6,
    )
    return jd, fr


def dist_km(r1, r2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(r1, r2)))


def rel_speed_km_s(v1, v2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _gmst_rad(jd: float, fr: float) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982 model)."""
    tut1 = (jd + fr - 2451545.0) / 36525.0
    theta = (67310.54841
             + (876600.0 * 3600.0 + 8640184.812866) * tut1
             + 0.093104 * tut1 ** 2
             - 6.2e-6  * tut1 ** 3)
    return math.radians(theta / 240.0 % 360.0)


def eci_to_latlon(r_eci, jd: float, fr: float):
    """Convert ECI position (km) → (lat_deg, lon_deg) using spherical Earth + GMST."""
    gst = _gmst_rad(jd, fr)
    x =  r_eci[0] * math.cos(gst) + r_eci[1] * math.sin(gst)
    y = -r_eci[0] * math.sin(gst) + r_eci[1] * math.cos(gst)
    z =  r_eci[2]
    r = math.sqrt(x*x + y*y + z*z)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


_OMEGA_EARTH = 7.2921150e-5  # rad/s — Earth's rotation rate


def eci_groundtrack_bearing(r_eci, v_eci, jd: float, fr: float) -> float:
    """
    Groundtrack bearing of a satellite at a given epoch, in radians clockwise
    from North.

    The bearing is derived from the satellite's velocity relative to the
    rotating Earth, projected onto the local East-North plane.

    Derivation:
      v_ecef = R_z(GMST) * v_eci + Omega_e * [ry_ecef, -rx_ecef, 0]
    (the extra term removes Earth's rotation contribution).
    """
    gst = _gmst_rad(jd, fr)
    cos_g, sin_g = math.cos(gst), math.sin(gst)

    # ECI → ECEF position
    rx =  r_eci[0] * cos_g + r_eci[1] * sin_g
    ry = -r_eci[0] * sin_g + r_eci[1] * cos_g
    rz =  r_eci[2]

    # ECEF velocity relative to rotating Earth
    vx =  v_eci[0] * cos_g + v_eci[1] * sin_g + _OMEGA_EARTH * ry
    vy = -v_eci[0] * sin_g + v_eci[1] * cos_g - _OMEGA_EARTH * rx
    vz =  v_eci[2]

    # Spherical geocentric lat/lon of the sub-satellite point
    r_norm = math.sqrt(rx*rx + ry*ry + rz*rz)
    lat_r  = math.asin(rz / r_norm)
    lon_r  = math.atan2(ry, rx)
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)

    # Project ECEF velocity onto local East and North unit vectors
    # East:  e_E = [-sin(lon), cos(lon), 0]
    # North: e_N = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
    v_east  = -sin_lon * vx + cos_lon * vy
    v_north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz

    return math.atan2(v_east, v_north)   # bearing CW from North, radians


# ── Conjunction screening ─────────────────────────────────────────────────────

def _refine_window(
    primary_sat: Satrec,
    cand_sat: Satrec,
    t_start: datetime,
    t_end: datetime,
    warn_km: float,
) -> Optional[dict]:
    """Fine-grained TCA search within a coarse trigger window."""
    step = timedelta(seconds=FINE_STEP_S)
    # Expand window by one coarse step on each side
    t = t_start - timedelta(seconds=COARSE_STEP_S)
    end = t_end + timedelta(seconds=COARSE_STEP_S)

    min_dist = 1e9
    min_t = min_r1 = min_r2 = min_v1 = min_v2 = None

    while t <= end:
        jd, fr = dt_to_jdfr(t)
        r1, v1 = propagate(primary_sat, jd, fr)
        r2, v2 = propagate(cand_sat, jd, fr)
        if r1 is not None and r2 is not None:
            d = dist_km(r1, r2)
            if d < min_dist:
                min_dist = d
                min_t = t
                min_r1, min_r2 = r1, r2
                min_v1, min_v2 = v1, v2
        t += step

    if min_dist < warn_km and min_t is not None:
        return {
            "tca": min_t,
            "miss_dist_km": min_dist,
            "rel_speed_km_s": rel_speed_km_s(min_v1, min_v2),
            "r1_tca": min_r1,   # primary ECI position at TCA (km)
            "r2_tca": min_r2,   # secondary ECI position at TCA (km)
            "v2_tca": min_v2,   # secondary ECI velocity at TCA (km/s) — for groundtrack bearing
        }
    return None


def screen_candidate(
    primary_sat: Satrec,
    cand_sat: Satrec,
    t_start: datetime,
    t_end: datetime,
    warn_km: float,
) -> list:
    """
    Two-pass conjunction screen against one candidate.

    Returns a list of event dicts, each with:
      tca, miss_dist_km, rel_speed_km_s
    Multiple distinct close-approach windows may be returned.
    """
    events = []
    coarse_step = timedelta(seconds=COARSE_STEP_S)

    t = t_start
    in_window = False
    window_t0 = None

    while t <= t_end:
        jd, fr = dt_to_jdfr(t)
        r1, v1 = propagate(primary_sat, jd, fr)
        r2, v2 = propagate(cand_sat, jd, fr)

        if r1 is None or r2 is None:
            t += coarse_step
            continue

        d = dist_km(r1, r2)

        if d < REFINE_TRIGGER_KM:
            if not in_window:
                in_window = True
                window_t0 = t
        else:
            if in_window:
                ev = _refine_window(primary_sat, cand_sat, window_t0, t, warn_km)
                if ev:
                    events.append(ev)
                in_window = False

        t += coarse_step

    # Handle a window still open at end of propagation
    if in_window:
        ev = _refine_window(primary_sat, cand_sat, window_t0, t_end, warn_km)
        if ev:
            events.append(ev)

    return events


# ── Plotting ─────────────────────────────────────────────────────────────────

# Consistent colour per object type (upper-case keys)
_TYPE_COLORS = {
    "PAYLOAD":      "#2196F3",   # blue
    "DEBRIS":       "#F44336",   # red
    "ROCKET BODY":  "#FF9800",   # orange
    "UNKNOWN":      "#9E9E9E",   # grey
}

def _type_color(obj_type: str) -> str:
    return _TYPE_COLORS.get((obj_type or "UNKNOWN").upper(), "#9C27B0")  # purple fallback


def plot_miss_vs_tca(events: list, primary_name: str, warn_km: float, red_km: float):
    """Scatter: miss distance (km) vs TCA, coloured and labelled by object type."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(13, 6))

    # Group by object type so we can build a clean legend
    type_groups: dict = {}
    for ev in events:
        ot = (ev["candidate"].get("OBJECT_TYPE") or "UNKNOWN").upper()
        type_groups.setdefault(ot, []).append(ev)

    for ot, evs in sorted(type_groups.items()):
        xs = [e["tca"] for e in evs]
        ys = [e["miss_dist_km"] for e in evs]
        ax.scatter(xs, ys, color=_type_color(ot), label=ot.title(),
                   s=65, zorder=3, edgecolors="white", linewidths=0.6)
        for ev in evs:
            nm = (ev["candidate"].get("OBJECT_NAME") or "")[:16]
            ax.annotate(
                nm, xy=(ev["tca"], ev["miss_dist_km"]),
                xytext=(5, 4), textcoords="offset points",
                fontsize=6.5, color="#333333", clip_on=True,
            )

    # Threshold reference lines
    ax.axhline(warn_km, color="#FFC107", linewidth=1.4, linestyle="--",
               label=f"Warning  {warn_km} km")
    ax.axhline(red_km,  color="#D32F2F", linewidth=1.4, linestyle="--",
               label=f"Red alert  {red_km} km")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_xlabel("Time of Closest Approach (UTC)", fontsize=11)
    ax.set_ylabel("Miss Distance  (km)", fontsize=11)
    ax.set_title(
        f"Conjunction Events — {primary_name}",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_country_histogram(events: list, primary_name: str):
    """Stacked bar: conjunction events by country code of the conjuncting object,
    stacked by object type."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    counts: dict = defaultdict(lambda: defaultdict(int))
    for ev in events:
        c = ev["candidate"]
        country = ((c.get("COUNTRY_CODE") or "").strip() or "(unknown)")
        ot = (c.get("OBJECT_TYPE") or "UNKNOWN").upper()
        counts[country][ot] += 1

    sorted_countries = sorted(
        counts.items(), key=lambda x: sum(x[1].values()), reverse=True
    )
    countries = [k for k, _ in sorted_countries]

    type_order = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]
    extra = sorted({ot for _, v in sorted_countries for ot in v} - set(type_order))
    all_types = [t for t in type_order if any(counts[c].get(t, 0) for c in countries)] + extra

    fig, ax = plt.subplots(figsize=(max(8, len(countries) * 0.7), 6))
    bottoms = [0] * len(countries)

    for ot in all_types:
        vals = [counts[c].get(ot, 0) for c in countries]
        ax.bar(
            countries, vals, bottom=bottoms,
            label=ot.title(), color=_type_color(ot),
            edgecolor="white", linewidth=0.5,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xlabel("Country Code", fontsize=11)
    ax.set_ylabel("Number of Conjunction Events", fontsize=11)
    ax.set_title(
        f"Conjunction Events — {primary_name} — by Country & Object Type",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── Parallel worker ───────────────────────────────────────────────────────────

def _screen_worker(args):
    """
    Module-level worker for ProcessPoolExecutor.

    Receives plain-Python args (picklable on Windows spawn), builds Satrec
    objects inside the worker, and returns (cand, events_list).
    events_list is None if the TLE failed to parse.
    """
    primary_tle1, primary_tle2, cand, t_now, t_end, warn_km = args
    primary_sat = make_satrec(primary_tle1, primary_tle2)
    cand_sat    = make_satrec(cand["TLE_LINE1"], cand["TLE_LINE2"])
    if primary_sat is None or cand_sat is None:
        return cand, None
    return cand, screen_candidate(primary_sat, cand_sat, t_now, t_end, warn_km)


def _antimeridian_segments(lats: list, lons: list):
    """
    Split parallel lat/lon lists into continuous segments that do not cross
    the ±180° antimeridian.  A crossing is detected when consecutive longitudes
    differ by more than 180°.  Each segment is returned as (lats, lons).
    """
    if not lats:
        return []
    segments = []
    seg_lats, seg_lons = [lats[0]], [lons[0]]
    for i in range(1, len(lats)):
        if abs(lons[i] - lons[i - 1]) > 180.0:
            segments.append((seg_lats, seg_lons))
            seg_lats, seg_lons = [], []
        seg_lats.append(lats[i])
        seg_lons.append(lons[i])
    segments.append((seg_lats, seg_lons))
    return segments


def plot_tca_map(events: list, primary: dict, t_start: datetime, t_end: datetime):
    """
    World map showing:
      • Primary satellite ground track over the analysis window (grey line,
        antimeridian-safe — no cross-plot zipping).
      • Primary sub-satellite point at each TCA, marked 'rx'.
      • Green shaded circle centred on Singapore (1.35°N, 103.8°E), r = 10°.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("cartopy not installed — skipping TCA map  (conda install -c conda-forge cartopy)")
        return None

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    primary_name = primary["OBJECT_NAME"]

    # ── Build ground track ────────────────────────────────────────────────────
    primary_sat = make_satrec(primary["TLE_LINE1"], primary["TLE_LINE2"])
    track_lats, track_lons = [], []
    if primary_sat is not None:
        t = t_start
        step = timedelta(seconds=60)
        while t <= t_end:
            jd, fr = dt_to_jdfr(t)
            r, _ = propagate(primary_sat, jd, fr)
            if r is not None:
                lat, lon = eci_to_latlon(r, jd, fr)
                track_lats.append(lat)
                track_lons.append(lon)
            t += step

    # ── Figure / projection ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()

    # ── Base map ──────────────────────────────────────────────────────────────
    ax.add_feature(cfeature.LAND,      facecolor="#EBEBEB", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#888888",
                   linestyle=":", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # ── Ground track (split at antimeridian) ──────────────────────────────────
    for seg_lats, seg_lons in _antimeridian_segments(track_lats, track_lons):
        ax.plot(seg_lons, seg_lats, color="#5588BB", linewidth=0.7, alpha=0.45,
                transform=ccrs.PlateCarree(), zorder=2)

    # ── Singapore AOR circle (r = 10°, parametric in PlateCarree) ─────────────
    SG_LAT, SG_LON, RADIUS_DEG = 1.35, 103.8, 10.0
    theta = np.linspace(0, 2 * np.pi, 360)
    circ_lons = SG_LON + RADIUS_DEG * np.cos(theta)
    circ_lats = SG_LAT + RADIUS_DEG * np.sin(theta)
    ax.fill(circ_lons, circ_lats, color="green", alpha=0.18,
            transform=ccrs.PlateCarree(), zorder=3)
    ax.plot(circ_lons, circ_lats, color="green", linewidth=1.4,
            transform=ccrs.PlateCarree(), zorder=3)
    ax.plot(SG_LON, SG_LAT, "g+", markersize=8, markeredgewidth=1.5,
            transform=ccrs.PlateCarree(), zorder=3)

    # ── TCA approach arrows ───────────────────────────────────────────────────
    # Collect (lon, lat, u, v) for a single vectorised quiver call.
    # Arrow direction = secondary's groundtrack bearing at TCA.
    # Arrows are normalised to unit vectors; quiver scale controls visual size.
    arr_lons, arr_lats, arr_us, arr_vs = [], [], [], []
    plotted = 0

    for ev in events:
        r1 = ev.get("r1_tca")
        if r1 is None:
            continue
        jd, fr = dt_to_jdfr(ev["tca"])
        lat, lon = eci_to_latlon(r1, jd, fr)

        r2 = ev.get("r2_tca")
        v2 = ev.get("v2_tca")
        if r2 is not None and v2 is not None:
            bearing = eci_groundtrack_bearing(r2, v2, jd, fr)
            arr_us.append(math.sin(bearing))
            arr_vs.append(math.cos(bearing))
        else:
            # No velocity info — point straight up as fallback
            arr_us.append(0.0)
            arr_vs.append(1.0)

        arr_lons.append(lon)
        arr_lats.append(lat)
        plotted += 1

    if arr_lons:
        ax.quiver(
            arr_lons, arr_lats, arr_us, arr_vs,
            color="red",
            transform=ccrs.PlateCarree(),
            scale=120,          # higher → shorter arrows; tune to taste
            width=0.002,        # shaft width as fraction of axes width
            headwidth=4,
            headlength=5,
            headaxislength=4,
            zorder=4,
        )

    ax.set_title(
        f"Ground Track & TCA Positions — {primary_name}  ({plotted} event(s))\n"
        f"Red arrows show secondary groundtrack bearing at TCA  "
        f"| Green circle: Singapore AOR (1.35°N 103.8°E, r = 10°)\n"
        f"Window: {t_start.strftime('%Y-%m-%d %H:%M')} → {t_end.strftime('%Y-%m-%d %H:%M')} UTC",
        fontsize=10, fontweight="bold",
    )

    legend_handles = [
        Line2D([0], [0], color="#5588BB", linewidth=1.5, alpha=0.7,
               label=f"{primary_name} ground track"),
        Line2D([0], [0], color="red", linewidth=0,
               marker=r"$\rightarrow$", markersize=12,
               label="Secondary groundtrack bearing at TCA"),
        Line2D([0], [0], color="green", linewidth=1.4,
               label="Singapore AOR (r = 10°)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return fig


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_table(events: list, label: str, warn_km: float, red_km: float):
    col_w = 75
    hdr = (
        f"{'#':>3}  {'NORAD':>6}  {'Name':<24}  {'Type':<12}"
        f"  {'TCA (UTC)':<20}  {'Miss km':>8}  {'Rel v km/s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, ev in enumerate(events, 1):
        c  = ev["candidate"]
        nm = (c["OBJECT_NAME"] or "UNKNOWN")[:24]
        ot = (c["OBJECT_TYPE"] or "?")[:12]
        tca = ev["tca"].strftime("%Y-%m-%d %H:%M:%S")
        flag = "***" if ev["miss_dist_km"] <= red_km else "---"
        print(
            f"[{flag}] {i:>2}.  {c['NORAD_CAT_ID']:>6}  {nm:<24}  {ot:<12}"
            f"  {tca:<20}  {ev['miss_dist_km']:>8.3f}  {ev['rel_speed_km_s']:>10.3f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Basic conjunction prediction using SGP4 and local SATCAT database."
    )
    parser.add_argument("--primary", default="TELEOS 2",
                        help="Primary satellite name (substring match, default: TELEOS-2)")
    parser.add_argument("--days", type=float, default=DEFAULT_DAYS,
                        help=f"Propagation window in days (default: {DEFAULT_DAYS})")
    parser.add_argument("--warn-km", type=float, default=DEFAULT_WARN_KM,
                        help=f"Warning distance threshold km (default: {DEFAULT_WARN_KM})")
    parser.add_argument("--red-km", type=float, default=DEFAULT_RED_KM,
                        help=f"Red/critical distance threshold km (default: {DEFAULT_RED_KM})")
    parser.add_argument("--plot", dest="plot", action="store_true", default=True,
                        help="Show and save matplotlib plots (default: on)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Disable plots")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes for SGP4 screening (default: 4)")
    args = parser.parse_args()

    # ── Find primary ──────────────────────────────────────────────────────────
    print(f"\nLooking up '{args.primary}' in catalog...")
    primary = find_primary(args.primary)
    if not primary:
        print(
            f"ERROR: '{args.primary}' not found. Check the name or run:\n"
            "  python ingest.py --update"
        )
        sys.exit(1)

    print(f"  Primary : {primary['OBJECT_NAME']}  (NORAD {primary['NORAD_CAT_ID']})")
    print(f"  Orbit   : {primary['PERIAPSIS']:.0f} x {primary['APOAPSIS']:.0f} km  "
          f"inc {primary['INCLINATION']:.2f}°")
    print(f"  TLE epoch: {primary['EPOCH']}")

    primary_sat = make_satrec(primary["TLE_LINE1"], primary["TLE_LINE2"])
    if primary_sat is None:
        print("ERROR: Failed to parse primary TLE.")
        sys.exit(1)

    # ── Candidate altitude-band pre-screen ────────────────────────────────────
    candidates = get_candidates(
        primary["PERIAPSIS"], primary["APOAPSIS"],
        ALTITUDE_PAD_KM, primary["NORAD_CAT_ID"],
    )
    band_lo = primary["PERIAPSIS"] - ALTITUDE_PAD_KM
    band_hi = primary["APOAPSIS"]  + ALTITUDE_PAD_KM
    print(
        f"\nAltitude pre-screen ({band_lo:.0f}–{band_hi:.0f} km): "
        f"{len(candidates):,} candidates pass"
    )

    # ── Propagation window ────────────────────────────────────────────────────
    t_now = datetime.now(timezone.utc)
    t_end = t_now + timedelta(days=args.days)
    print(
        f"Propagation window : {t_now.strftime('%Y-%m-%d %H:%M')} → "
        f"{t_end.strftime('%Y-%m-%d %H:%M')} UTC  ({args.days} days)"
    )
    print(f"Warn threshold     : {args.warn_km} km  |  Red threshold: {args.red_km} km\n")

    # ── Screen all candidates (parallel) ─────────────────────────────────────
    all_events = []
    failed = 0
    total = len(candidates)
    done = 0

    print(f"  Using {args.workers} worker process(es).")

    worker_args = [
        (primary["TLE_LINE1"], primary["TLE_LINE2"], cand, t_now, t_end, args.warn_km)
        for cand in candidates
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_screen_worker, wa): wa for wa in worker_args}
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  Screening {done:>5}/{total}  ({done/total*100:.0f}%)...",
                      end="\r", flush=True)
            try:
                cand, evs = fut.result()
            except Exception:
                failed += 1
                continue
            if evs is None:
                failed += 1
            else:
                for ev in evs:
                    all_events.append({**ev, "candidate": cand})

    print(f"  Screening complete. {failed} TLE parse failures (skipped).           ")

    # ── Sort by miss distance and report ──────────────────────────────────────
    all_events.sort(key=lambda e: e["miss_dist_km"])

    print()
    print("=" * 75)
    print(
        f"  CONJUNCTION REPORT — {primary['OBJECT_NAME']} "
        f"(NORAD {primary['NORAD_CAT_ID']})  |  {len(all_events)} event(s) found"
    )
    print("=" * 75)

    if not all_events:
        print("  No conjunctions detected within the warning threshold.\n")
    else:
        reds  = [e for e in all_events if e["miss_dist_km"] <= args.red_km]
        warns = [e for e in all_events if args.red_km < e["miss_dist_km"] <= args.warn_km]

        if reds:
            print(f"\n[***] RED — {len(reds)} event(s)  (miss distance ≤ {args.red_km} km)\n")
            _print_table(reds, "RED", args.warn_km, args.red_km)

        if warns:
            print(f"\n[---] WARN — {len(warns)} event(s)  ({args.red_km} km < miss ≤ {args.warn_km} km)\n")
            _print_table(warns, "WARN", args.warn_km, args.red_km)

    print()
    print("=" * 75)
    print("Parameters used:")
    print(f"  Altitude pad     : ±{ALTITUDE_PAD_KM} km")
    print(f"  Coarse step      : {COARSE_STEP_S} s")
    print(f"  Fine step        : {FINE_STEP_S} s  (within {REFINE_TRIGGER_KM} km trigger)")
    print("DISCLAIMER: SGP4 positional uncertainty ~1–3 km/day from TLE epoch.")
    print("            Results are indicative only — not a substitute for CDMs.")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found — skipping plots (pip install matplotlib)")
        else:
            saved = []

            if all_events:
                fig1 = plot_miss_vs_tca(
                    all_events, primary["OBJECT_NAME"], args.warn_km, args.red_km
                )
                fig1.savefig("../plots/conjunction_scatter.png", dpi=150)
                saved.append("conjunction_scatter.png")
            else:
                print("(No events to plot for miss-distance chart)")

            fig2 = plot_country_histogram(all_events, primary["OBJECT_NAME"])
            fig2.savefig("../plots/conjunction_countries.png", dpi=150)
            saved.append("conjunction_countries.png")

            if all_events:
                fig3 = plot_tca_map(all_events, primary, t_now, t_end)
                if fig3 is not None:
                    fig3.savefig("../plots/conjunction_tca_map.png", dpi=150)
                    saved.append("conjunction_tca_map.png")

            if saved:
                print("Plots saved: " + ", ".join(saved))

            plt.show()


if __name__ == "__main__":
    main()
