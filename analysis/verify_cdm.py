"""
verify_cdm.py -- Verify SGP4 propagation against a Space-Track CDM.

Given a CDM ID, fetches the conjunction details and historical TLEs for
both objects (epochs closest to TCA), then runs our SGP4 propagation
workflow and compares the result against the CDM ground truth.

Usage:
    python verify_cdm.py 1351309127
    python verify_cdm.py 1351309127 --window-min 10 --fine-step 1

Dependencies: spacetrack, sgp4>=2.22, matplotlib (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPACETRACK_IDENTITY, SPACETRACK_PASSWORD

from conjunction_predict import (
    make_satrec,
    propagate,
    dt_to_jdfr,
    dist_km,
    rel_speed_km_s,
    eci_to_latlon,
)

import spacetrack.operators as op
from spacetrack import SpaceTrackClient


# -- CDM fetch ----------------------------------------------------------------

def fetch_cdm(st: SpaceTrackClient, cdm_id: int) -> Optional[dict]:
    """Fetch a single CDM record by ID."""
    raw = st.cdm_public(cdm_id=cdm_id, format="json")
    recs = json.loads(raw) if isinstance(raw, str) else (raw or [])
    return recs[0] if recs else None


def parse_cdm(cdm: dict) -> dict:
    """Extract the fields we need from a raw cdm_public record."""
    tca_str = cdm["TCA"]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            tca = datetime.strptime(tca_str, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse TCA: {tca_str}")

    miss_m = cdm.get("MIN_RNG")
    pc = cdm.get("PC")

    return {
        "cdm_id": cdm["CDM_ID"],
        "tca": tca,
        "miss_distance_km": float(miss_m) / 1000.0 if miss_m else None,
        "collision_probability": float(pc) if pc and pc != "" else None,
        "sat1_norad": int(cdm["SAT_1_ID"]),
        "sat1_name": cdm.get("SAT_1_NAME", "").strip(),
        "sat1_type": cdm.get("SAT1_OBJECT_TYPE", "").strip(),
        "sat1_rcs": cdm.get("SAT1_RCS", "").strip(),
        "sat2_norad": int(cdm["SAT_2_ID"]),
        "sat2_name": cdm.get("SAT_2_NAME", "").strip(),
        "sat2_type": cdm.get("SAT2_OBJECT_TYPE", "").strip(),
        "sat2_rcs": cdm.get("SAT2_RCS", "").strip(),
    }


# -- Historical TLE fetch ----------------------------------------------------

def fetch_closest_tle(
    st: SpaceTrackClient,
    norad_id: int,
    target_dt: datetime,
    search_days: float = 3.0,
) -> Optional[dict]:
    """
    Fetch the GP record with epoch closest to (and preferably just before)
    target_dt from Space-Track gp_history.

    Returns dict with TLE_LINE1, TLE_LINE2, EPOCH, or None.
    """
    # Search window: a few days before TCA to just after
    dt_lo = (target_dt - timedelta(days=search_days)).strftime("%Y-%m-%d")
    dt_hi = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = st.gp_history(
        norad_cat_id=norad_id,
        epoch=op.inclusive_range(dt_lo, dt_hi),
        orderby="epoch desc",
        limit=20,
        format="json",
    )
    recs = json.loads(raw) if isinstance(raw, str) else (raw or [])

    if not recs:
        # Widen the search
        dt_lo = (target_dt - timedelta(days=search_days * 3)).strftime("%Y-%m-%d")
        raw = st.gp_history(
            norad_cat_id=norad_id,
            epoch=op.inclusive_range(dt_lo, dt_hi),
            orderby="epoch desc",
            limit=20,
            format="json",
        )
        recs = json.loads(raw) if isinstance(raw, str) else (raw or [])

    if not recs:
        return None

    # Parse epochs and pick the closest one that is before (or nearest to) TCA
    def _parse_epoch(r):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(r["EPOCH"], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    candidates = []
    for r in recs:
        ep = _parse_epoch(r)
        if ep and r.get("TLE_LINE1") and r.get("TLE_LINE2"):
            candidates.append((r, ep))

    if not candidates:
        return None

    # Prefer the most recent TLE that is still before TCA
    before_tca = [(r, ep) for r, ep in candidates if ep <= target_dt]
    if before_tca:
        best, best_ep = max(before_tca, key=lambda x: x[1])
    else:
        # All TLEs are after TCA; pick the closest one
        best, best_ep = min(candidates, key=lambda x: abs((x[1] - target_dt).total_seconds()))

    return {
        "TLE_LINE1": best["TLE_LINE1"],
        "TLE_LINE2": best["TLE_LINE2"],
        "EPOCH": best["EPOCH"],
        "epoch_dt": best_ep,
        "PERIAPSIS": best.get("PERIAPSIS"),
        "APOAPSIS": best.get("APOAPSIS"),
        "INCLINATION": best.get("INCLINATION"),
    }


def fetch_tle_series(
    st: SpaceTrackClient,
    norad_id: int,
    target_dt: datetime,
    search_days: float = 5.0,
) -> list[dict]:
    """
    Fetch ALL available TLEs for an object in the window
    [target_dt - search_days, target_dt].

    Returns list of dicts sorted by epoch (oldest first), each with
    TLE_LINE1, TLE_LINE2, EPOCH, epoch_dt.
    """
    dt_lo = (target_dt - timedelta(days=search_days)).strftime("%Y-%m-%d")
    dt_hi = (target_dt + timedelta(hours=1)).strftime("%Y-%m-%d")

    raw = st.gp_history(
        norad_cat_id=norad_id,
        epoch=op.inclusive_range(dt_lo, dt_hi),
        orderby="epoch asc",
        limit=100,
        format="json",
    )
    recs = json.loads(raw) if isinstance(raw, str) else (raw or [])

    results = []
    seen_epochs = set()
    for r in recs:
        if not r.get("TLE_LINE1") or not r.get("TLE_LINE2"):
            continue
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                ep = datetime.strptime(r["EPOCH"], fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            continue
        if ep > target_dt:
            continue
        # Deduplicate on epoch string (Space-Track sometimes returns dupes)
        epoch_key = r["EPOCH"]
        if epoch_key in seen_epochs:
            continue
        seen_epochs.add(epoch_key)
        results.append({
            "TLE_LINE1": r["TLE_LINE1"],
            "TLE_LINE2": r["TLE_LINE2"],
            "EPOCH": r["EPOCH"],
            "epoch_dt": ep,
        })

    return results


# -- SGP4 propagation around TCA ---------------------------------------------

def propagate_pair(
    sat1_tle: dict,
    sat2_tle: dict,
    tca_cdm: datetime,
    window_min: float = 10.0,
    fine_step_s: float = 1.0,
) -> Optional[dict]:
    """
    Propagate both objects around the CDM TCA and find the actual
    closest approach according to SGP4.

    Two passes:
      1. Coarse scan over [-window_min, +window_min] at 5s steps
      2. Fine scan +/- 30s around the coarse minimum at fine_step_s steps

    Returns dict with sgp4_tca, sgp4_miss_km, sgp4_rel_speed_km_s, etc.
    """
    sat1 = make_satrec(sat1_tle["TLE_LINE1"], sat1_tle["TLE_LINE2"])
    sat2 = make_satrec(sat2_tle["TLE_LINE1"], sat2_tle["TLE_LINE2"])
    if sat1 is None or sat2 is None:
        return None

    window = timedelta(minutes=window_min)
    t_start = tca_cdm - window
    t_end = tca_cdm + window

    # -- Coarse pass (5s steps) -----------------------------------------------
    coarse_step = timedelta(seconds=5)
    min_dist = 1e9
    min_t = t_start

    t = t_start
    while t <= t_end:
        jd, fr = dt_to_jdfr(t)
        r1, _ = propagate(sat1, jd, fr)
        r2, _ = propagate(sat2, jd, fr)
        if r1 is not None and r2 is not None:
            d = dist_km(r1, r2)
            if d < min_dist:
                min_dist = d
                min_t = t
        t += coarse_step

    # -- Fine pass (+/- 30s around coarse minimum) ----------------------------
    fine_window = timedelta(seconds=30)
    fine_step = timedelta(seconds=fine_step_s)
    t = min_t - fine_window
    t_end_fine = min_t + fine_window

    best_dist = 1e9
    best_t = min_t
    best_r1 = best_r2 = best_v1 = best_v2 = None

    while t <= t_end_fine:
        jd, fr = dt_to_jdfr(t)
        r1, v1 = propagate(sat1, jd, fr)
        r2, v2 = propagate(sat2, jd, fr)
        if r1 is not None and r2 is not None:
            d = dist_km(r1, r2)
            if d < best_dist:
                best_dist = d
                best_t = t
                best_r1, best_r2 = r1, r2
                best_v1, best_v2 = v1, v2
        t += fine_step

    if best_r1 is None:
        return None

    jd, fr = dt_to_jdfr(best_t)
    lat1, lon1 = eci_to_latlon(best_r1, jd, fr)
    lat2, lon2 = eci_to_latlon(best_r2, jd, fr)

    return {
        "sgp4_tca": best_t,
        "sgp4_miss_km": best_dist,
        "sgp4_rel_speed_km_s": rel_speed_km_s(best_v1, best_v2),
        "sat1_lat": lat1,
        "sat1_lon": lon1,
        "sat2_lat": lat2,
        "sat2_lon": lon2,
    }


# -- Distance profile for plotting -------------------------------------------

def compute_distance_profile(
    sat1_tle: dict,
    sat2_tle: dict,
    tca_cdm: datetime,
    window_min: float = 10.0,
    step_s: float = 5.0,
) -> tuple[list[float], list[float]]:
    """Return (offsets_s, distances_km) for the distance profile plot."""
    sat1 = make_satrec(sat1_tle["TLE_LINE1"], sat1_tle["TLE_LINE2"])
    sat2 = make_satrec(sat2_tle["TLE_LINE1"], sat2_tle["TLE_LINE2"])
    if sat1 is None or sat2 is None:
        return [], []

    offsets = []
    dists = []
    step = timedelta(seconds=step_s)
    window = timedelta(minutes=window_min)
    t = tca_cdm - window

    while t <= tca_cdm + window:
        jd, fr = dt_to_jdfr(t)
        r1, _ = propagate(sat1, jd, fr)
        r2, _ = propagate(sat2, jd, fr)
        if r1 is not None and r2 is not None:
            offsets.append((t - tca_cdm).total_seconds())
            dists.append(dist_km(r1, r2))
        t += step

    return offsets, dists


# -- Reporting ----------------------------------------------------------------

def print_report(cdm_info: dict, sgp4_result: dict, sat1_tle: dict, sat2_tle: dict):
    w = 72
    print()
    print("=" * w)
    print(f"  CDM VERIFICATION REPORT -- CDM ID {cdm_info['cdm_id']}")
    print("=" * w)

    # -- CDM details ----------------------------------------------------------
    print(f"\n  CDM Ground Truth:")
    print(f"    TCA:             {cdm_info['tca'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
    miss = cdm_info["miss_distance_km"]
    print(f"    Miss distance:   {miss * 1000:.1f} m  ({miss:.3f} km)" if miss else "    Miss distance:   N/A")
    pc = cdm_info["collision_probability"]
    print(f"    Pc:              {pc:.2e}" if pc else "    Pc:              N/A")

    print(f"\n  SAT 1:  {cdm_info['sat1_name']:<24}  NORAD {cdm_info['sat1_norad']}"
          f"  ({cdm_info['sat1_type']}, {cdm_info['sat1_rcs']})")
    print(f"  SAT 2:  {cdm_info['sat2_name']:<24}  NORAD {cdm_info['sat2_norad']}"
          f"  ({cdm_info['sat2_type']}, {cdm_info['sat2_rcs']})")

    # -- TLE details ----------------------------------------------------------
    print(f"\n  Historical TLEs used:")
    for label, tle in [("SAT 1", sat1_tle), ("SAT 2", sat2_tle)]:
        age_h = (cdm_info["tca"] - tle["epoch_dt"]).total_seconds() / 3600
        sign = "before" if age_h >= 0 else "after"
        print(f"    {label}:  epoch {tle['EPOCH']}"
              f"  ({abs(age_h):.1f}h {sign} TCA)")

    # -- SGP4 result ----------------------------------------------------------
    print(f"\n  SGP4 Propagation Result:")
    print(f"    TCA:             {sgp4_result['sgp4_tca'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
    print(f"    Miss distance:   {sgp4_result['sgp4_miss_km'] * 1000:.1f} m"
          f"  ({sgp4_result['sgp4_miss_km']:.3f} km)")
    print(f"    Rel speed:       {sgp4_result['sgp4_rel_speed_km_s']:.3f} km/s")
    print(f"    Location:        ({sgp4_result['sat1_lat']:.1f}, {sgp4_result['sat1_lon']:.1f})")

    # -- Comparison -----------------------------------------------------------
    tca_offset = (sgp4_result["sgp4_tca"] - cdm_info["tca"]).total_seconds()
    miss_error = sgp4_result["sgp4_miss_km"] - miss if miss else None

    print(f"\n  Comparison:")
    print(f"    TCA offset:      {tca_offset:+.1f} s")
    if miss_error is not None:
        print(f"    Miss error:      {miss_error * 1000:+.1f} m  ({miss_error:+.3f} km)")
        if miss > 0:
            print(f"    Miss ratio:      {sgp4_result['sgp4_miss_km'] / miss:.1f}x")

    print()
    print("=" * w)
    print()


# -- TLE age sensitivity ------------------------------------------------------

def compute_tle_age_sensitivity(
    st: SpaceTrackClient,
    cdm_info: dict,
    window_min: float = 10.0,
    fine_step_s: float = 1.0,
    search_days: float = 5.0,
) -> list[dict]:
    """
    For every available historical TLE (for both objects) at progressively
    older epochs, run the propagation and record how miss distance and TCA
    error change as TLE age increases.

    Strategy: vary SAT 1's TLE age while keeping SAT 2 at its freshest,
    then vary SAT 2 while keeping SAT 1 at its freshest.  Finally, also
    test with both at the same age bucket where possible.

    Returns list of dicts sorted by max_age_h:
        max_age_h, miss_km, tca_offset_s, miss_error_m, sat1_age_h, sat2_age_h
    """
    tca = cdm_info["tca"]

    print(f"  Fetching TLE series for SAT 1 (NORAD {cdm_info['sat1_norad']}) ...")
    sat1_series = fetch_tle_series(st, cdm_info["sat1_norad"], tca, search_days)
    print(f"    {len(sat1_series)} TLEs found")

    print(f"  Fetching TLE series for SAT 2 (NORAD {cdm_info['sat2_norad']}) ...")
    sat2_series = fetch_tle_series(st, cdm_info["sat2_norad"], tca, search_days)
    print(f"    {len(sat2_series)} TLEs found")

    if not sat1_series or not sat2_series:
        return []

    # Freshest TLEs for each
    sat1_fresh = sat1_series[-1]
    sat2_fresh = sat2_series[-1]

    results = []
    seen_ages = set()

    def _run(s1_tle, s2_tle):
        age1 = (tca - s1_tle["epoch_dt"]).total_seconds() / 3600
        age2 = (tca - s2_tle["epoch_dt"]).total_seconds() / 3600
        max_age = max(age1, age2)
        # Bucket to nearest 0.5h to avoid near-duplicate points
        bucket = round(max_age * 2) / 2
        if bucket in seen_ages:
            return
        seen_ages.add(bucket)

        res = propagate_pair(s1_tle, s2_tle, tca, window_min, fine_step_s)
        if res is None:
            return

        miss = cdm_info["miss_distance_km"]
        tca_off = (res["sgp4_tca"] - tca).total_seconds()
        miss_err = (res["sgp4_miss_km"] - miss) * 1000 if miss else None

        results.append({
            "max_age_h": max_age,
            "sat1_age_h": age1,
            "sat2_age_h": age2,
            "miss_km": res["sgp4_miss_km"],
            "tca_offset_s": tca_off,
            "miss_error_m": miss_err,
        })

    # Vary SAT 1 age, SAT 2 fresh
    for s1 in sat1_series:
        _run(s1, sat2_fresh)

    # Vary SAT 2 age, SAT 1 fresh
    for s2 in sat2_series:
        _run(sat1_fresh, s2)

    # Both at similar ages
    for s1 in sat1_series:
        age1 = (tca - s1["epoch_dt"]).total_seconds() / 3600
        # Find the SAT 2 TLE closest in age
        best_s2 = min(sat2_series,
                      key=lambda s: abs((tca - s["epoch_dt"]).total_seconds() / 3600 - age1))
        _run(s1, best_s2)

    results.sort(key=lambda r: r["max_age_h"])
    return results


# -- Plots --------------------------------------------------------------------

def plot_results(
    offsets: list[float],
    dists: list[float],
    cdm_info: dict,
    sgp4_result: dict,
    age_sensitivity: list[dict],
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots")
        return

    has_age = len(age_sensitivity) > 1
    ncols = 2 if has_age else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # -- Panel 1: Distance profile around TCA ---------------------------------
    ax = axes[0]
    ax.plot(offsets, dists, color="#2196F3", linewidth=1.5, label="SGP4 distance")

    if cdm_info["miss_distance_km"] is not None:
        ax.axvline(0, color="#F44336", linewidth=1, linestyle="--", alpha=0.7)
        ax.axhline(cdm_info["miss_distance_km"], color="#F44336", linewidth=1,
                   linestyle=":", alpha=0.7,
                   label=f"CDM miss: {cdm_info['miss_distance_km'] * 1000:.0f} m")

    sgp4_offset = (sgp4_result["sgp4_tca"] - cdm_info["tca"]).total_seconds()
    ax.plot(sgp4_offset, sgp4_result["sgp4_miss_km"], "o", color="#4CAF50",
            markersize=10, zorder=5,
            label=f"SGP4 TCA: {sgp4_result['sgp4_miss_km'] * 1000:.0f} m")

    ax.set_xlabel("Time offset from CDM TCA (seconds)")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Distance Profile Around TCA", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # -- Panel 2: TLE age sensitivity -----------------------------------------
    if has_age:
        ax = axes[1]
        ages = [r["max_age_h"] for r in age_sensitivity]
        misses_m = [r["miss_km"] * 1000 for r in age_sensitivity]

        ax.plot(ages, misses_m, "o-", color="#2196F3", markersize=5,
                linewidth=1.2, label="SGP4 miss distance")

        if cdm_info["miss_distance_km"] is not None:
            cdm_m = cdm_info["miss_distance_km"] * 1000
            ax.axhline(cdm_m, color="#F44336", linewidth=1, linestyle="--",
                       alpha=0.7, label=f"CDM miss: {cdm_m:.0f} m")

        ax.set_xlabel("Max TLE age (hours before TCA)")
        ax.set_ylabel("SGP4 miss distance (m)")
        ax.set_title("TLE Age Sensitivity", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        f"CDM {cdm_info['cdm_id']}:  {cdm_info['sat1_name']}  vs  {cdm_info['sat2_name']}\n"
        f"TCA {cdm_info['tca'].strftime('%Y-%m-%d %H:%M:%S')} UTC",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_path = Path(__file__).parent.parent / "plots" / f"verify_cdm_{cdm_info['cdm_id']}.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {out_path.name}")
    plt.show()


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify SGP4 propagation against a single Space-Track CDM."
    )
    parser.add_argument("cdm_id", type=int, help="CDM ID to verify against")
    parser.add_argument("--window-min", type=float, default=10.0,
                        help="Propagation window +/- minutes around TCA (default: 10)")
    parser.add_argument("--fine-step", type=float, default=1.0,
                        help="Fine-pass step size in seconds (default: 1.0)")
    parser.add_argument("--tle-search-days", type=float, default=3.0,
                        help="How many days before TCA to search for TLEs (default: 3)")
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    # -- Fetch CDM ------------------------------------------------------------
    print(f"\nFetching CDM {args.cdm_id} from Space-Track ...")
    st = SpaceTrackClient(identity=SPACETRACK_IDENTITY, password=SPACETRACK_PASSWORD)
    raw_cdm = fetch_cdm(st, args.cdm_id)
    if not raw_cdm:
        print(f"ERROR: CDM {args.cdm_id} not found.")
        sys.exit(1)

    cdm_info = parse_cdm(raw_cdm)
    print(f"  TCA:   {cdm_info['tca'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  SAT 1: {cdm_info['sat1_name']} (NORAD {cdm_info['sat1_norad']})")
    print(f"  SAT 2: {cdm_info['sat2_name']} (NORAD {cdm_info['sat2_norad']})")
    miss = cdm_info["miss_distance_km"]
    print(f"  Miss:  {miss * 1000:.0f} m" if miss else "  Miss:  N/A")

    # -- Fetch historical TLEs ------------------------------------------------
    print(f"\nFetching historical TLEs (within {args.tle_search_days} days of TCA) ...")

    sat1_tle = fetch_closest_tle(st, cdm_info["sat1_norad"], cdm_info["tca"], args.tle_search_days)
    if sat1_tle is None:
        print(f"ERROR: No TLE found for SAT 1 (NORAD {cdm_info['sat1_norad']}) near TCA.")
        sys.exit(1)
    age1 = (cdm_info["tca"] - sat1_tle["epoch_dt"]).total_seconds() / 3600
    print(f"  SAT 1: epoch {sat1_tle['EPOCH']}  ({abs(age1):.1f}h {'before' if age1 >= 0 else 'after'} TCA)")

    sat2_tle = fetch_closest_tle(st, cdm_info["sat2_norad"], cdm_info["tca"], args.tle_search_days)
    if sat2_tle is None:
        print(f"ERROR: No TLE found for SAT 2 (NORAD {cdm_info['sat2_norad']}) near TCA.")
        sys.exit(1)
    age2 = (cdm_info["tca"] - sat2_tle["epoch_dt"]).total_seconds() / 3600
    print(f"  SAT 2: epoch {sat2_tle['EPOCH']}  ({abs(age2):.1f}h {'before' if age2 >= 0 else 'after'} TCA)")

    # -- Propagate ------------------------------------------------------------
    print(f"\nPropagating +/- {args.window_min} min around TCA (fine step: {args.fine_step}s) ...")
    sgp4_result = propagate_pair(
        sat1_tle, sat2_tle, cdm_info["tca"],
        window_min=args.window_min, fine_step_s=args.fine_step,
    )
    if sgp4_result is None:
        print("ERROR: SGP4 propagation failed for one or both TLEs.")
        sys.exit(1)

    # -- Report ---------------------------------------------------------------
    print_report(cdm_info, sgp4_result, sat1_tle, sat2_tle)

    # -- TLE age sensitivity --------------------------------------------------
    print("Running TLE age sensitivity analysis ...")
    age_sensitivity = compute_tle_age_sensitivity(
        st, cdm_info,
        window_min=args.window_min,
        fine_step_s=args.fine_step,
        search_days=args.tle_search_days,
    )
    if age_sensitivity:
        print(f"  {len(age_sensitivity)} data points computed")
        print(f"\n  {'Age (h)':>8}  {'Miss (m)':>9}  {'TCA off (s)':>11}  {'Miss err (m)':>12}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*12}")
        for r in age_sensitivity:
            me = f"{r['miss_error_m']:+.0f}" if r["miss_error_m"] is not None else "N/A"
            print(f"  {r['max_age_h']:>8.1f}  {r['miss_km']*1000:>9.0f}"
                  f"  {r['tca_offset_s']:>+11.1f}  {me:>12}")
        print()

    # -- Plots ----------------------------------------------------------------
    if args.plot:
        offsets, dists = compute_distance_profile(
            sat1_tle, sat2_tle, cdm_info["tca"],
            window_min=args.window_min, step_s=5.0,
        )
        if offsets:
            plot_results(offsets, dists, cdm_info, sgp4_result, age_sensitivity)


if __name__ == "__main__":
    main()
