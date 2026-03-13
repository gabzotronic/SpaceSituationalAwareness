"""SSA Sensor Tasking Tool — SEA satellite access windows + CP-SAT optimisation.

Computes visibility windows for three ground-based SSA sensors over 04–06 Mar 2026,
then solves a CP-SAT scheduling problem to minimise average fleet revisit time.
Output: Gantt-style access window plot (selected vs. skipped windows).

Usage:
    conda run -n orbit python analysis/ssa_sensor_tasking.py
"""

from __future__ import annotations

# =============================================================================
# Section 1: Imports + sys.path bootstrap
# =============================================================================
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Bootstrap so 'from config import DB_PATH' works when run from analysis/
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from config import DB_PATH  # noqa: E402

try:
    from skyfield.api import EarthSatellite, load, wgs84
except ImportError:
    sys.exit(
        "skyfield not found. Install with:\n"
        "  conda run -n orbit pip install skyfield"
    )

try:
    from ortools.sat.python import cp_model as _cp_model
    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False

# =============================================================================
# Section 2: Analysis window + constants
# =============================================================================
_T_START = datetime(2026, 3, 5, 0, 0, 0, tzinfo=timezone.utc)
_T_END   = datetime(2026, 3, 8, 0, 0, 0, tzinfo=timezone.utc)

PASS_STEP_S       = 15    # seconds between samples inside a detected pass
SOLVER_TIME_LIMIT = 60.0  # seconds

# =============================================================================
# Section 3: Sensor dataclass + SENSORS list
# =============================================================================

@dataclass
class Sensor:
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_el_deg: float = 5.0
    max_el_deg: float = 90.0
    az_min_deg: float = 0.0
    az_max_deg: float = 360.0
    max_range_km: float = 800.0


SENSORS: list[Sensor] = [
    Sensor("SENSOR-UGA", lat_deg=-1.8934673093749366, lon_deg=29.859583255988728),
    Sensor("SENSOR-SGP", lat_deg=1.2042358604926489,  lon_deg=103.76650425742542),
    Sensor("SENSOR-BRA", lat_deg=-6.5631692448828245, lon_deg=-49.70637661120628),
]

SENSOR_COLORS: dict[str, str] = {
    "SENSOR-UGA": "#1f77b4",
    "SENSOR-SGP": "#ff7f0e",
    "SENSOR-BRA": "#2ca02c",
}

# =============================================================================
# Section 4: Database query
# =============================================================================

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_sing_satellites() -> list[dict]:
    """Return SEA-registered objects with valid TLEs and epoch ≥ 2026-01-01."""
    sql = """
        SELECT NORAD_CAT_ID, OBJECT_NAME, TLE_LINE1, TLE_LINE2
        FROM gp
        WHERE COUNTRY_CODE IN ('SING', 'MALA', 'INDO', 'THAI')
          AND TLE_LINE1 IS NOT NULL AND TLE_LINE2 IS NOT NULL
          AND EPOCH >= '2026-01-01'
    """
    with _connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]

# =============================================================================
# Section 5: Access interval computation (skyfield)
# =============================================================================

def _passes_from_events(times, events) -> list[tuple]:
    """Pair rise(0)/set(2) events from find_events into (t_rise, t_set) windows."""
    passes = []
    pending_rise = None
    for t, ev in zip(times, events):
        if ev == 0:
            pending_rise = t
        elif ev == 2 and pending_rise is not None:
            passes.append((pending_rise, t))
            pending_rise = None
    return passes


def _az_visible(az_deg: float, sensor: Sensor) -> bool:
    lo, hi = sensor.az_min_deg, sensor.az_max_deg
    if lo <= hi:
        return lo <= az_deg <= hi
    return az_deg >= lo or az_deg <= hi


def _check_range_in_pass(
    sat_sf: EarthSatellite,
    observer_sf,
    sensor: Sensor,
    t_rise,
    t_set,
    ts,
) -> list[tuple[datetime, datetime]]:
    """Sample at PASS_STEP_S inside a pass; find sub-intervals within range."""
    from datetime import timedelta

    duration_s = (t_set - t_rise) * 86400.0
    n_steps = max(2, int(duration_s / PASS_STEP_S) + 1)
    t_samples = ts.linspace(t_rise, t_set, n_steps)

    topocentric = (sat_sf - observer_sf).at(t_samples)
    alt, az, dist = topocentric.altaz()

    in_window = np.zeros(n_steps, dtype=bool)
    for i in range(n_steps):
        in_window[i] = (
            dist.km[i] <= sensor.max_range_km
            and _az_visible(az.degrees[i], sensor)
            and alt.degrees[i] >= sensor.min_el_deg
        )

    intervals: list[tuple[datetime, datetime]] = []
    i = 0
    while i < n_steps:
        if in_window[i]:
            j = i
            while j < n_steps and in_window[j]:
                j += 1
            t_s = t_samples[i].utc_datetime()
            t_e = t_samples[j - 1].utc_datetime()
            if t_s == t_e:
                t_e = t_s + timedelta(seconds=PASS_STEP_S)
            intervals.append((t_s, t_e))
            i = j
        else:
            i += 1

    return intervals


def compute_access_intervals(
    sensors: list[Sensor],
    satellites: list[dict],
    t_start: datetime,
    t_end: datetime,
) -> list[tuple[str, int, str, datetime, datetime]]:
    """Compute all (sensor, satellite) access intervals.

    Returns list of (sensor_name, norad_id, object_name, t_aos, t_los).
    """
    ts = load.timescale()
    t0 = ts.from_datetime(t_start)
    t1 = ts.from_datetime(t_end)

    results: list[tuple[str, int, str, datetime, datetime]] = []

    for sensor in sensors:
        observer = wgs84.latlon(
            sensor.lat_deg, sensor.lon_deg, elevation_m=sensor.alt_km * 1000.0
        )
        print(f"\n[{sensor.name}] Processing {len(satellites)} satellites...")

        for sat in satellites:
            norad_id    = sat["NORAD_CAT_ID"]
            object_name = sat["OBJECT_NAME"].strip() if sat["OBJECT_NAME"] else str(norad_id)
            tle1        = sat["TLE_LINE1"].strip()
            tle2        = sat["TLE_LINE2"].strip()

            try:
                sat_sf = EarthSatellite(tle1, tle2, object_name, ts)
                times, events = sat_sf.find_events(
                    observer, t0, t1, altitude_degrees=sensor.min_el_deg
                )
            except Exception as exc:
                print(f"  WARNING: find_events failed for {object_name} ({norad_id}): {exc}")
                continue

            for t_rise, t_set in _passes_from_events(times, events):
                for t_aos, t_los in _check_range_in_pass(sat_sf, observer, sensor, t_rise, t_set, ts):
                    results.append((sensor.name, norad_id, object_name, t_aos, t_los))

    return results

# =============================================================================
# Section 5b: CP-SAT optimisation — minimise average fleet revisit time
# =============================================================================

def optimize_sensor_tasking(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    sensors: list[Sensor],
    t_start: datetime,
    t_end: datetime,
) -> tuple[set[int], str]:
    """Select access windows to minimise average fleet revisit time.

    CP-SAT formulation
    ------------------
    Variables   : x[i] ∈ {0,1}  — 1 if window i is assigned for tracking
    Constraints : AddNoOverlap per sensor (phased-array ⇒ zero slew time,
                  so back-to-back windows are allowed without any gap)
    Objective   : Maximise Σ_i gap_before[i] × x[i]
                  where gap_before[i] = seconds from the AOS of the
                  previous window of the same satellite (or t_start) to
                  the AOS of window i.

    Rationale: a window that arrives after a long coverage gap carries a
    high reward, driving the solver to prioritise stale satellites over
    those recently updated.  This is a linear proxy for minimising the
    sum of inter-track intervals (= average revisit time).

    Returns (selected_indices: set[int], status_str: str).
    """
    if not _ORTOOLS_AVAILABLE:
        print(
            "\nWARNING: ortools not installed — optimisation skipped.\n"
            "  Install with: conda run -n orbit pip install ortools\n"
            "  Returning all windows as 'selected'."
        )
        return set(range(len(intervals))), "SKIPPED"

    n       = len(intervals)
    t0_epoch = int(t_start.timestamp())

    cp      = _cp_model.CpModel()

    # ── Decision variables ────────────────────────────────────────────── #
    x = [cp.new_bool_var(f"x_{i}") for i in range(n)]

    # ── Optional interval variables (for NoOverlap) ───────────────────── #
    iv_vars = []
    for i, (_, _, _, t_aos, t_los) in enumerate(intervals):
        s = int(t_aos.timestamp()) - t0_epoch
        e = int(t_los.timestamp()) - t0_epoch
        d = max(e - s, PASS_STEP_S)
        iv = cp.new_optional_interval_var(s, d, s + d, x[i], f"iv_{i}")
        iv_vars.append(iv)

    # ── NoOverlap per sensor ──────────────────────────────────────────── #
    sensor_ivs: dict[str, list] = defaultdict(list)
    for i, (sensor_name, _, _, _, _) in enumerate(intervals):
        sensor_ivs[sensor_name].append(iv_vars[i])
    for ivs in sensor_ivs.values():
        cp.add_no_overlap(ivs)

    # ── Objective weights ─────────────────────────────────────────────── #
    # gap_before[i] = seconds since the previous access window of the same
    # satellite (chronologically), or since t_start for the first window.
    sat_wins: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (_, norad_id, _, t_aos, _) in enumerate(intervals):
        sat_wins[norad_id].append((int(t_aos.timestamp()), i))

    weights = [0] * n
    for wins in sat_wins.values():
        wins.sort()
        prev_t = t0_epoch
        for t_aos_s, idx in wins:
            weights[idx] = max(t_aos_s - prev_t, 0)
            prev_t = t_aos_s

    # Scale to minutes to keep coefficient magnitudes small
    cp.maximize(sum((weights[i] // 60) * x[i] for i in range(n)))

    # ── Solve ─────────────────────────────────────────────────────────── #
    solver = _cp_model.CpSolver()
    solver.parameters.max_time_in_seconds  = SOLVER_TIME_LIMIT
    solver.parameters.num_search_workers   = 4
    solver.parameters.log_search_progress  = False

    print(
        f"\nRunning CP-SAT optimisation  "
        f"({n} windows, {len(sensors)} sensors, "
        f"{SOLVER_TIME_LIMIT:.0f}s limit)..."
    )
    status     = solver.solve(cp)
    status_str = solver.status_name(status)
    print(f"  Status    : {status_str}")
    print(f"  Objective : {solver.objective_value:.0f} weighted gap-minutes covered")
    print(f"  Wall time : {solver.wall_time:.2f}s")

    selected: set[int] = set()
    if status in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
        for i in range(n):
            if solver.value(x[i]):
                selected.add(i)

    return selected, status_str, weights


def _compute_revisit_stats(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    selected: set[int],
    t_start: datetime,
    t_end: datetime,
) -> dict[int, dict]:
    """Per-satellite revisit statistics from the selected windows.

    For each satellite we build a timeline:
        t_start → [selected AOS times sorted] → t_end
    and compute the gaps between consecutive events.

    Returns dict keyed by norad_id with keys:
        name, n_tracks, gaps_min, avg_gap_min, max_gap_min
    """
    t0 = t_start.timestamp()
    t1 = t_end.timestamp()

    sat_tracks: dict[int, list[float]]  = defaultdict(list)
    sat_names:  dict[int, str]          = {}

    for i, (_, norad_id, obj_name, t_aos, _) in enumerate(intervals):
        sat_names[norad_id] = obj_name
        if i in selected:
            sat_tracks[norad_id].append(t_aos.timestamp())

    all_nids = {norad_id for _, norad_id, _, _, _ in intervals}

    stats: dict[int, dict] = {}
    for nid in all_nids:
        tracks   = sorted(sat_tracks.get(nid, []))
        timeline = [t0] + tracks + [t1]
        gaps_min = [(timeline[k+1] - timeline[k]) / 60.0 for k in range(len(timeline) - 1)]
        stats[nid] = {
            "name":        sat_names[nid],
            "n_tracks":    len(tracks),
            "gaps_min":    gaps_min,
            "avg_gap_min": sum(gaps_min) / len(gaps_min),
            "max_gap_min": max(gaps_min),
        }

    return stats


def print_optimisation_summary(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    selected: set[int],
    sensors: list[Sensor],
    t_start: datetime,
    t_end: datetime,
) -> None:
    """Print per-sensor selection counts and per-satellite revisit statistics."""
    print("\n" + "=" * 70)
    print("OPTIMISED TASKING SCHEDULE")
    print("=" * 70)

    sensor_sel: dict[str, int] = defaultdict(int)
    sensor_tot: dict[str, int] = defaultdict(int)
    for i, (sname, _, _, _, _) in enumerate(intervals):
        sensor_tot[sname] += 1
        if i in selected:
            sensor_sel[sname] += 1

    for s in sensors:
        n_sel = sensor_sel.get(s.name, 0)
        n_tot = sensor_tot.get(s.name, 0)
        skipped = n_tot - n_sel
        print(f"  {s.name}: {n_sel} scheduled, {skipped} skipped  (of {n_tot} available)")

    print(f"\n  Total: {len(selected)} / {len(intervals)} windows selected\n")

    stats = _compute_revisit_stats(intervals, selected, t_start, t_end)

    print(f"  {'Satellite':<28} {'Tracks':>6}  {'Avg gap':>9}  {'Max gap':>9}")
    print("  " + "-" * 58)
    avg_gaps = []
    for nid in sorted(stats.keys()):
        s  = stats[nid]
        lbl = f"{s['name']} ({nid})"[:27]
        print(
            f"  {lbl:<28} {s['n_tracks']:>6}  "
            f"{s['avg_gap_min']:>8.1f}m  {s['max_gap_min']:>8.1f}m"
        )
        avg_gaps.append(s["avg_gap_min"])

    fleet_avg = sum(avg_gaps) / len(avg_gaps) if avg_gaps else 0.0
    print("  " + "-" * 58)
    print(f"  {'FLEET AVERAGE':<28} {'':>6}  {fleet_avg:>8.1f}m")
    print("=" * 70)

def print_conflict_verification(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    selected: set[int],
    t_start: datetime,
) -> None:
    """Sanity check: for every skipped window, compare true staleness of the
    skipped satellite vs the winner, where staleness is measured from the
    last *confirmed-tracked* window (not the last access opportunity).

    A violation occurs when the skipped satellite was actually staler than
    the winner — meaning the optimiser chose the wrong satellite.
    """
    t0 = t_start.timestamp()

    # Build per-satellite sorted list of selected track times (across all sensors)
    sat_selected_tracks: dict[int, list[float]] = defaultdict(list)
    for i, (_, norad_id, _, t_aos, _) in enumerate(intervals):
        if i in selected:
            sat_selected_tracks[norad_id].append(t_aos.timestamp())
    for tracks in sat_selected_tracks.values():
        tracks.sort()

    def true_staleness(norad_id: int, query_time: float) -> float:
        """Seconds since the last confirmed track before query_time, or since t_start."""
        tracks = sat_selected_tracks.get(norad_id, [])
        # Last selected track strictly before this window's AOS
        prior = [t for t in tracks if t < query_time]
        return query_time - (prior[-1] if prior else t0)

    # Pre-index: sensor → list of (t_aos_s, t_los_s, idx)
    sensor_wins: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for i, (sname, _, _, t_aos, t_los) in enumerate(intervals):
        sensor_wins[sname].append((t_aos.timestamp(), t_los.timestamp(), i))

    skipped = [i for i in range(len(intervals)) if i not in selected]
    if not skipped:
        print("\nNo skipped windows — nothing to verify.")
        return

    print("\n" + "=" * 76)
    print("CONFLICT VERIFICATION  (staleness = time since last confirmed track)")
    print("=" * 76)

    any_violation = False

    for i in skipped:
        sname, nid_i, name_i, t_aos_i, t_los_i = intervals[i]
        a_i  = t_aos_i.timestamp()
        l_i  = t_los_i.timestamp()
        s_i  = true_staleness(nid_i, a_i)

        overlapping_selected = [
            j for (a_j, l_j, j) in sensor_wins[sname]
            if j != i and j in selected and a_j < l_i and l_j > a_i
        ]

        t_str = t_aos_i.strftime("%d %b %H:%Mz")
        print(f"\n  [{sname}]  {t_str}")
        print(f"    SKIPPED : {name_i:<28}  staleness={s_i/3600:>5.2f} hr")

        for j in overlapping_selected:
            _, nid_j, name_j, t_aos_j, _ = intervals[j]
            s_j     = true_staleness(nid_j, t_aos_j.timestamp())
            verdict = "OK" if s_j >= s_i else "*** VIOLATION ***"
            if s_j < s_i:
                any_violation = True
            print(f"    WINNER  : {name_j:<28}  staleness={s_j/3600:>5.2f} hr  {verdict}")

        if not overlapping_selected:
            print(f"    (no overlapping selected window found — check pairing)")

    print("\n" + "-" * 76)
    if any_violation:
        print("  RESULT: violations found — solver chose a fresher satellite over a staler one.")
    else:
        print("  RESULT: all conflicts correctly resolved — staler satellite always won.")
    print("=" * 76)
    print("=" * 72)


# =============================================================================
# Section 6: Output — console summaries + Gantt plot
# =============================================================================

def _print_access_summary(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    sensors: list[Sensor],
) -> None:
    print("\n" + "=" * 60)
    print("ACCESS WINDOW SUMMARY  (all feasible windows)")
    print("=" * 60)
    for sensor in sensors:
        ivs       = [iv for iv in intervals if iv[0] == sensor.name]
        total_min = sum((l - a).total_seconds() / 60.0 for _, _, _, a, l in ivs)
        print(f"  {sensor.name}: {len(ivs)} contacts, {total_min:.1f} total minutes")
    print(f"\n  TOTAL: {len(intervals)} contact windows")
    print("=" * 60)


def check_overlapping_windows(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    sensors: list[Sensor],
) -> None:
    """Detect simultaneous contacts at each sensor via sweep-line."""
    by_sensor: dict[str, list] = defaultdict(list)
    for sensor_name, norad_id, object_name, t_aos, t_los in intervals:
        by_sensor[sensor_name].append((t_aos, t_los, norad_id, object_name))

    print("\n" + "=" * 60)
    print("SIMULTANEOUS CONTACT OVERLAP REPORT")
    print("=" * 60)

    for sensor in sensors:
        ivs = by_sensor.get(sensor.name, [])
        if not ivs:
            print(f"\n  {sensor.name}: no contacts")
            continue

        events: list[tuple] = []
        for t_aos, t_los, norad_id, obj_name in ivs:
            events.append((t_aos, +1, norad_id, obj_name))
            events.append((t_los, -1, norad_id, obj_name))
        events.sort(key=lambda e: (e[0], e[1]))

        active: dict[int, str] = {}
        overlap_windows = []
        prev_t: Optional[datetime] = None
        prev_set: Optional[frozenset] = None

        for t, delta, norad_id, obj_name in events:
            if prev_set is not None and len(prev_set) >= 2 and prev_t is not None:
                overlap_windows.append((prev_t, t, prev_set))
            if delta == +1:
                active[norad_id] = obj_name
            else:
                active.pop(norad_id, None)
            prev_t   = t
            prev_set = frozenset(active.items())

        merged: list[list] = []
        for t_s, t_e, sat_set in overlap_windows:
            if merged and merged[-1][2] == sat_set and merged[-1][1] == t_s:
                merged[-1][1] = t_e
            else:
                merged.append([t_s, t_e, sat_set])

        if not merged:
            print(f"\n  {sensor.name}: no simultaneous contacts")
            continue

        print(f"\n  {sensor.name}: {len(merged)} overlap window(s)")
        for t_s, t_e, sat_set in merged:
            names   = sorted(name for _, name in sat_set)
            dur_s   = (t_e - t_s).total_seconds()
            print(
                f"    [{sensor.name}] [{len(sat_set)} sats] "
                f"[{', '.join(names)}]  "
                f"{t_s.strftime('%d %b %H:%Mz')}–{t_e.strftime('%H:%Mz')} "
                f"({dur_s:.0f}s)"
            )

    print("=" * 60)


def plot_gantt(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    sensors: list[Sensor],
    t_start: datetime,
    t_end: datetime,
    selected: Optional[set[int]] = None,
) -> None:
    """Render a Gantt-style access window chart.

    If `selected` is provided, scheduled windows are drawn in sensor colour
    and unscheduled feasible windows are shown as faint gray bars behind them.
    """
    seen: dict[int, str] = {}
    for _, norad_id, object_name, _, _ in intervals:
        seen[norad_id] = object_name
    sat_ids = sorted(seen.keys())
    n_sats  = len(sat_ids)

    fig, ax = plt.subplots(figsize=(16, max(4.0, 0.5 * n_sats + 2.0)))

    optimised = selected is not None

    # ── Pass 1: unselected (opportunity) bars + rx markers ───────────── #
    skipped_legend_added = False
    if optimised:
        for i, (sensor_name, norad_id, _, t_s, t_e) in enumerate(intervals):
            if i not in selected:
                y = sat_ids.index(norad_id)
                ax.barh(y, t_e - t_s, left=t_s, height=0.6,
                        color=SENSOR_COLORS.get(sensor_name, "gray"), alpha=0.5)
                t_mid = t_s + (t_e - t_s) / 2
                label = "Skipped (conflict)" if not skipped_legend_added else "_nolegend_"
                ax.plot(t_mid, y, "rx", markersize=6, markeredgewidth=1.5,
                        label=label, zorder=5)
                skipped_legend_added = True

    # ── Pass 2: selected (scheduled) bars ─────────────────────────────── #
    legend_added: set[str] = set()
    draw_indices = selected if optimised else range(len(intervals))

    for i in draw_indices:
        sensor_name, norad_id, _, t_s, t_e = intervals[i]
        color = SENSOR_COLORS.get(sensor_name, "gray")
        label = sensor_name if sensor_name not in legend_added else "_nolegend_"
        legend_added.add(sensor_name)
        ax.barh(
            sat_ids.index(norad_id),
            t_e - t_s,
            left=t_s,
            height=0.6,
            color=color,
            alpha=0.85,
            label=label,
        )

    if not intervals:
        ax.text(0.5, 0.5, "No access windows found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray")

    # ── Legend extras ─────────────────────────────────────────────────── #

    ax.set_yticks(range(n_sats))
    ax.set_yticklabels([f"{seen[nid]} ({nid})" for nid in sat_ids], fontsize=8)
    ax.set_ylim(-0.5, max(n_sats - 0.5, 0.5))

    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%Mz"))

    ax.set_xlabel("UTC")
    ax.set_ylabel("Satellite")
    title_suffix = " — Optimised Schedule" if optimised else ""
    ax.set_title(
        f"SEA Satellite SSA Sensor Access Windows (SING/MALA/INDO/THAI) "
        f"— 04–06 Mar 2026{title_suffix}"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    if legend_added:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_revisit_time(
    intervals: list[tuple[str, int, str, datetime, datetime]],
    selected: set[int],
    t_start: datetime,
    t_end: datetime,
) -> None:
    """Plot per-satellite revisit time evolution (sawtooth) + fleet average.

    Revisit time at time t = (t - last_track_AOS) for satellites that have
    been tracked at least once.  Before the first track the curve sits at 0.
    Fleet average is computed over satellites that have been imaged at least
    once up to time t (dynamic denominator).
    """
    sat_names:  dict[int, str]        = {}
    sat_tracks: dict[int, list[float]] = defaultdict(list)

    for i, (_, norad_id, obj_name, t_aos, _) in enumerate(intervals):
        sat_names[norad_id] = obj_name
        if i in selected:
            sat_tracks[norad_id].append(t_aos.timestamp())

    tracked_nids = sorted(nid for nid, tr in sat_tracks.items() if tr)
    if not tracked_nids:
        print("No selected tracks — revisit plot skipped.")
        return

    # Dense time axis: 1-minute resolution over the analysis window
    t0    = t_start.timestamp()
    t1    = t_end.timestamp()
    n_pts = int((t1 - t0) / 60) + 1
    t_arr = np.linspace(t0, t1, n_pts)
    t_dt  = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in t_arr])

    n_sats       = len(tracked_nids)
    all_revisit  = np.zeros((n_sats, n_pts))   # minutes since last track
    active_mat   = np.zeros((n_sats, n_pts), dtype=bool)  # True once first track seen

    for k, nid in enumerate(tracked_nids):
        tr_arr = np.array(sorted(sat_tracks[nid]))
        # For each time point, index of last track ≤ t  (-1 = none yet)
        idx = np.searchsorted(tr_arr, t_arr, side="right") - 1
        observed = idx >= 0
        all_revisit[k, observed] = (t_arr[observed] - tr_arr[idx[observed]]) / 3600.0
        active_mat[k] = t_arr >= tr_arr[0]

    # Fleet average: mean over currently-active satellites
    n_active  = active_mat.sum(axis=0)
    fleet_avg = np.where(
        n_active > 0,
        (all_revisit * active_mat).sum(axis=0) / n_active,
        0.0,
    )

    fig, ax = plt.subplots(figsize=(16, 5))

    # Individual satellite sawtooth lines
    for k, nid in enumerate(tracked_nids):
        ax.plot(t_dt, all_revisit[k], color="steelblue", alpha=0.35,
                linewidth=0.8)

    # Fleet average
    ax.plot(t_dt, fleet_avg, color="crimson", linewidth=2.2,
            label=f"Fleet average ({n_sats} sats)", zorder=5)

    # Annotate final fleet average value
    ax.annotate(
        f"{fleet_avg[-1]:.1f} hr",
        xy=(t_dt[-1], fleet_avg[-1]),
        xytext=(-45, 6), textcoords="offset points",
        color="crimson", fontsize=9,
    )

    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%Mz"))
    ax.set_xlabel("UTC")
    ax.set_ylabel("Revisit time (hours)")
    ax.set_title(
        "Per-Satellite Revisit Time — SEA Fleet Optimised Schedule  04–06 Mar 2026"
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Section 7: main()
# =============================================================================

def main() -> None:
    print("Loading SEA satellites from DB...")
    satellites = load_sing_satellites()

    if not satellites:
        print(
            "ERROR: No SEA satellites with valid TLEs (epoch ≥ 2026-01-01) found.\n"
            "Run 'python ingest.py --update' or '--full' to refresh the catalog."
        )
        return

    print(f"  {len(satellites)} satellites loaded:")
    for s in satellites:
        print(f"    {s['NORAD_CAT_ID']:>6}  {s['OBJECT_NAME']}")

    intervals = compute_access_intervals(SENSORS, satellites, _T_START, _T_END)

    _print_access_summary(intervals, SENSORS)
    check_overlapping_windows(intervals, SENSORS)

    selected, status, weights = optimize_sensor_tasking(intervals, SENSORS, _T_START, _T_END)

    if status != "SKIPPED":
        print_optimisation_summary(intervals, selected, SENSORS, _T_START, _T_END)
        print_conflict_verification(intervals, selected, _T_START)

    plot_gantt(intervals, SENSORS, _T_START, _T_END,
               selected=selected if status != "SKIPPED" else None)

    if status != "SKIPPED":
        plot_revisit_time(intervals, selected, _T_START, _T_END)


if __name__ == "__main__":
    main()
