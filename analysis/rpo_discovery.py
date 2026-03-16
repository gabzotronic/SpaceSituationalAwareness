"""
RPO Discovery — Historical RPO detection from TLE data (GEO or LEO).

Screens object pairs through a 4-tier pipeline:
  Tier 1: Orbital plane matching (inclination + RAAN filter)
  Tier 2: SMA convergence episode detection (time-series)
  Tier 3: SGP4 proximity verification
  Tier 4: RPO causal-chain scoring (maneuver, directed approach, station-keeping)

Usage:
    # GEO belt (default)
    python analysis/rpo_discovery.py --backfill --top 30

    # SSO LEO band (400-600 km, inc 95-100 deg)
    python analysis/rpo_discovery.py --alt-min 400 --alt-max 600 \\
        --inc-min 95 --inc-max 100 --delta-inc 1.0 --delta-raan 5.0 \\
        --conv-sma 20 --prox-km 20 --step-hours 0.25 --backfill
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

# ── Repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, SPACETRACK_IDENTITY, SPACETRACK_PASSWORD

# Reuse from rpo_monitoring
from analysis.rpo_monitoring import _propagate_to, _sgp4_state, detect_maneuvers

# ── Configuration ──────────────────────────────────────────────────────────────
@dataclass
class DiscoveryConfig:
    # Time window
    window_start: str = "2016-01-01"
    window_end: str = "2025-12-31"
    # Altitude band (km) — GEO default; set to e.g. 400/600 for LEO
    alt_min: float = 35000
    alt_max: float = 37000
    # Inclination band filter (deg) — None means no bound
    inc_min: Optional[float] = None
    inc_max: Optional[float] = None
    # Tier 1: orbital plane matching
    delta_inc_max: float = 5.0         # deg — max inclination difference
    delta_raan_max: Optional[float] = None  # deg — max RAAN difference (None = skip)
    # Tier 2: SMA convergence
    convergence_sma_km: float = 50.0   # km — convergence threshold
    convergence_min_days: int = 7      # minimum episode duration
    # Tier 3: SGP4 proximity
    proximity_km: float = 200.0        # km — proximity threshold
    coarse_step_hours: float = 1.0     # propagation step size

    @property
    def label(self) -> str:
        if self.alt_min >= 35000:
            return "GEO belt"
        return f"LEO {self.alt_min:.0f}–{self.alt_max:.0f} km"


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 0 — GEO backfill
# ══════════════════════════════════════════════════════════════════════════════

def _band_norad_ids(con: sqlite3.Connection, cfg: DiscoveryConfig) -> List[int]:
    """Return NORAD_CAT_IDs for all objects in the configured altitude/inc band."""
    sql = ("SELECT NORAD_CAT_ID FROM gp "
           "WHERE PERIAPSIS >= ? AND APOAPSIS <= ?")
    params: list = [cfg.alt_min, cfg.alt_max]
    if cfg.inc_min is not None:
        sql += " AND INCLINATION >= ?"
        params.append(cfg.inc_min)
    if cfg.inc_max is not None:
        sql += " AND INCLINATION <= ?"
        params.append(cfg.inc_max)
    sql += " ORDER BY NORAD_CAT_ID"
    rows = con.execute(sql, params).fetchall()
    return [r[0] for r in rows]


def backfill_geo(cfg: DiscoveryConfig) -> None:
    """Backfill gp_history for all GEO objects over the analysis window.

    Calls ingest.backfill_gp_history() per object, skipping those that
    already have records in the window.
    """
    from spacetrack import SpaceTrackClient
    from ingest import backfill_gp_history

    st = SpaceTrackClient(SPACETRACK_IDENTITY, SPACETRACK_PASSWORD)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    ids = _band_norad_ids(con, cfg)
    print(f"Backfill [{cfg.label}]: {len(ids)} objects, "
          f"window {cfg.window_start} to {cfg.window_end}", flush=True)

    skipped = 0
    fetched = 0
    for idx, norad_id in enumerate(ids, 1):
        # Skip if already backfilled
        count = con.execute(
            "SELECT COUNT(*) FROM gp_history "
            "WHERE NORAD_CAT_ID = ? AND EPOCH >= ? AND EPOCH <= ?",
            (norad_id, cfg.window_start, cfg.window_end),
        ).fetchone()[0]
        if count > 0:
            skipped += 1
            if idx % 200 == 0:
                print(f"  [{idx}/{len(ids)}] skipped {skipped} (already backfilled), "
                      f"fetched {fetched}", flush=True)
            continue

        fetched += 1
        print(f"  [{idx}/{len(ids)}] Fetching NORAD {norad_id} ... "
              f"(fetched {fetched}, skipped {skipped})", flush=True)

        try:
            backfill_gp_history(con, st, norad_id, cfg.window_start, cfg.window_end)
        except Exception as exc:
            print(f"  [{idx}/{len(ids)}] NORAD {norad_id} — ERROR: {exc}", flush=True)

    con.close()
    print(f"GEO backfill complete. Fetched {fetched}, skipped {skipped}.", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_sma(cfg: DiscoveryConfig) -> Dict[int, pd.DataFrame]:
    """Load SMA time series for all objects in the configured band from gp_history.

    Returns {norad_id: DataFrame(columns=[epoch, sma])} where epoch is
    datetime (UTC) and sma is in km.
    """
    con = sqlite3.connect(DB_PATH)

    # Build the inner subquery with optional inclination filter
    inner = "SELECT NORAD_CAT_ID FROM gp WHERE PERIAPSIS >= ? AND APOAPSIS <= ?"
    inner_params: list = [cfg.alt_min, cfg.alt_max]
    if cfg.inc_min is not None:
        inner += " AND INCLINATION >= ?"
        inner_params.append(cfg.inc_min)
    if cfg.inc_max is not None:
        inner += " AND INCLINATION <= ?"
        inner_params.append(cfg.inc_max)

    sql = f"""
        SELECT NORAD_CAT_ID, EPOCH, SEMIMAJOR_AXIS
        FROM gp_history
        WHERE NORAD_CAT_ID IN ({inner})
          AND EPOCH >= ? AND EPOCH <= ?
        ORDER BY NORAD_CAT_ID, EPOCH
    """
    df = pd.read_sql_query(
        sql, con,
        params=inner_params + [cfg.window_start, cfg.window_end],
    )
    con.close()

    if df.empty:
        print(f"WARNING: No gp_history data found for {cfg.label}. Run --backfill first.")
        return {}

    df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)
    df = df.dropna(subset=["SEMIMAJOR_AXIS"])

    sma_dict: Dict[int, pd.DataFrame] = {}
    for norad_id, grp in df.groupby("NORAD_CAT_ID"):
        sma_dict[int(norad_id)] = (
            grp[["EPOCH", "SEMIMAJOR_AXIS"]]
            .rename(columns={"EPOCH": "epoch", "SEMIMAJOR_AXIS": "sma"})
            .reset_index(drop=True)
        )

    print(f"Loaded SMA time series for {len(sma_dict)} objects [{cfg.label}] "
          f"({len(df)} total records)")
    return sma_dict


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 1 — Orbital plane matching
# ══════════════════════════════════════════════════════════════════════════════

def orbital_plane_pairs(cfg: DiscoveryConfig) -> pd.DataFrame:
    """Return object pairs with compatible orbital planes.

    Filters by altitude band, optional inclination band, inclination
    difference, and optionally RAAN difference (with wraparound).
    """
    con = sqlite3.connect(DB_PATH)

    # Build altitude + inclination band clause
    band_clause = "PERIAPSIS >= ? AND APOAPSIS <= ?"
    band_params: list = [cfg.alt_min, cfg.alt_max]
    if cfg.inc_min is not None:
        band_clause += " AND INCLINATION >= ?"
        band_params.append(cfg.inc_min)
    if cfg.inc_max is not None:
        band_clause += " AND INCLINATION <= ?"
        band_params.append(cfg.inc_max)

    raan_clause = ""
    if cfg.delta_raan_max is not None:
        # Handle RAAN wraparound (0°/360° boundary)
        raan_clause = (
            f" AND MIN(ABS(a.RA_OF_ASC_NODE - b.RA_OF_ASC_NODE), "
            f"360.0 - ABS(a.RA_OF_ASC_NODE - b.RA_OF_ASC_NODE)) < {cfg.delta_raan_max}"
        )

    sql = f"""
        SELECT a.NORAD_CAT_ID AS id_a, b.NORAD_CAT_ID AS id_b,
               a.OBJECT_NAME AS name_a, b.OBJECT_NAME AS name_b,
               a.INCLINATION AS inc_a, b.INCLINATION AS inc_b,
               a.RA_OF_ASC_NODE AS raan_a, b.RA_OF_ASC_NODE AS raan_b,
               ABS(a.INCLINATION - b.INCLINATION) AS delta_inc
        FROM gp a JOIN gp b ON a.NORAD_CAT_ID < b.NORAD_CAT_ID
        WHERE ({band_clause.replace('PERIAPSIS', 'a.PERIAPSIS')
                            .replace('APOAPSIS', 'a.APOAPSIS')
                            .replace('INCLINATION', 'a.INCLINATION')})
          AND ({band_clause.replace('PERIAPSIS', 'b.PERIAPSIS')
                            .replace('APOAPSIS', 'b.APOAPSIS')
                            .replace('INCLINATION', 'b.INCLINATION')})
          AND ABS(a.INCLINATION - b.INCLINATION) < ?
          {raan_clause}
    """
    params = band_params + band_params + [cfg.delta_inc_max]
    df = pd.read_sql_query(sql, con, params=params)
    con.close()

    raan_note = (f", delta_RAAN < {cfg.delta_raan_max}°"
                 if cfg.delta_raan_max is not None else "")
    print(f"Tier 1: {len(df)} candidate pairs "
          f"(delta_inc < {cfg.delta_inc_max}°{raan_note})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 2 — SMA convergence episodes
# ══════════════════════════════════════════════════════════════════════════════

def _find_episodes_for_pair(
    ts_a: pd.DataFrame,
    ts_b: pd.DataFrame,
    threshold_km: float,
    min_days: int,
) -> List[dict]:
    """Find time windows where two objects' SMAs converge.

    Resamples both SMA time series to a common daily grid, computes
    |delta_SMA|, and finds contiguous runs below threshold_km lasting
    at least min_days.

    Returns list of episode dicts with keys:
        start, end, duration_days, min_delta_sma, approach_rate_km_per_day
    """
    if len(ts_a) < 2 or len(ts_b) < 2:
        return []

    # Deduplicate epochs (keep last), then resample to daily grid
    ts_a = ts_a.drop_duplicates(subset="epoch", keep="last").set_index("epoch")
    ts_b = ts_b.drop_duplicates(subset="epoch", keep="last").set_index("epoch")
    ts_a = ts_a.resample("1D").nearest()
    ts_b = ts_b.resample("1D").nearest()

    # Align to common date range
    common = ts_a.index.intersection(ts_b.index)
    if len(common) < min_days:
        return []

    sma_a = ts_a.loc[common, "sma"].values
    sma_b = ts_b.loc[common, "sma"].values
    delta = np.abs(sma_a - sma_b)
    dates = common

    # Fix #2: skip pairs already co-located at the start of the window.
    # If delta_SMA is below threshold on day 0, this is permanent co-location,
    # not an approach event — discard the pair entirely.
    if delta[0] < threshold_km:
        return []

    # Find contiguous runs below threshold
    below = delta < threshold_km
    episodes = []
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            duration = (dates[j - 1] - dates[i]).days + 1
            if duration >= min_days:
                # Approach rate: delta_SMA change in the 30 days before episode
                pre_start = max(0, i - 30)
                if pre_start < i and i > 0:
                    approach_rate = (delta[pre_start] - delta[i]) / max(i - pre_start, 1)
                else:
                    approach_rate = 0.0

                episodes.append({
                    "start": dates[i],
                    "end": dates[j - 1],
                    "duration_days": duration,
                    "min_delta_sma": float(np.nanmin(delta[i:j])),
                    "approach_rate_km_per_day": float(approach_rate),
                    "pre_episode_delta_sma": float(delta[pre_start]),
                })
            i = j
        else:
            i += 1

    return episodes


def find_convergence_episodes(
    sma_dict: Dict[int, pd.DataFrame],
    pairs: pd.DataFrame,
    cfg: DiscoveryConfig,
) -> pd.DataFrame:
    """Screen all Tier 1 pairs for SMA convergence episodes.

    Returns a DataFrame with one row per episode, including pair IDs,
    names, and episode metadata.
    """
    results = []
    n_pairs = len(pairs)

    for idx, row in pairs.iterrows():
        id_a, id_b = int(row["id_a"]), int(row["id_b"])
        if id_a not in sma_dict or id_b not in sma_dict:
            continue

        episodes = _find_episodes_for_pair(
            sma_dict[id_a], sma_dict[id_b],
            cfg.convergence_sma_km, cfg.convergence_min_days,
        )
        for ep in episodes:
            results.append({
                "id_a": id_a,
                "id_b": id_b,
                "name_a": row["name_a"],
                "name_b": row["name_b"],
                **ep,
            })

        if (idx + 1) % 10000 == 0:
            print(f"  Tier 2 progress: {idx + 1}/{n_pairs} pairs screened, "
                  f"{len(results)} episodes found so far")

    df = pd.DataFrame(results)
    if df.empty:
        print("Tier 2: No convergence episodes found.")
        return pd.DataFrame()

    df = df.sort_values("min_delta_sma").reset_index(drop=True)
    print(f"Tier 2: {len(df)} convergence episodes from {df[['id_a','id_b']].drop_duplicates().shape[0]} unique pairs")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 3 — SGP4 proximity verification
# ══════════════════════════════════════════════════════════════════════════════

def _load_tles_for_window(
    norad_id: int, start: datetime, end: datetime,
) -> pd.DataFrame:
    """Load TLE history for one object from gp_history over a time window."""
    con = sqlite3.connect(DB_PATH)
    sql = """
        SELECT EPOCH, TLE_LINE1, TLE_LINE2, SEMIMAJOR_AXIS
        FROM gp_history
        WHERE NORAD_CAT_ID = ? AND EPOCH >= ? AND EPOCH <= ?
        ORDER BY EPOCH
    """
    df = pd.read_sql_query(
        sql, con,
        params=(norad_id, start.isoformat(), end.isoformat()),
    )
    con.close()
    if not df.empty:
        df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)
    return df


def _best_tle_at(tles: pd.DataFrame, t: datetime) -> Optional[Tuple[str, str]]:
    """Return the TLE closest in time to t, preferring the one just before."""
    if tles.empty:
        return None
    diffs = (tles["EPOCH"] - pd.Timestamp(t)).abs()
    idx = diffs.idxmin()
    row = tles.loc[idx]
    tle1, tle2 = row["TLE_LINE1"], row["TLE_LINE2"]
    if not isinstance(tle1, str) or not isinstance(tle2, str):
        return None
    return tle1, tle2


def verify_proximity(
    episodes: pd.DataFrame,
    cfg: DiscoveryConfig,
) -> pd.DataFrame:
    """Verify physical proximity via SGP4 propagation for each episode.

    Filters to episodes where min 3D separation < proximity_km.
    """
    if episodes.empty:
        return pd.DataFrame()

    results = []
    step = timedelta(hours=cfg.coarse_step_hours)

    for idx, ep in episodes.iterrows():
        id_a, id_b = int(ep["id_a"]), int(ep["id_b"])
        ep_start = pd.Timestamp(ep["start"]).to_pydatetime().replace(tzinfo=timezone.utc)
        ep_end = pd.Timestamp(ep["end"]).to_pydatetime().replace(tzinfo=timezone.utc)

        # Extend window slightly for TLE selection
        margin = timedelta(days=5)
        tles_a = _load_tles_for_window(id_a, ep_start - margin, ep_end + margin)
        tles_b = _load_tles_for_window(id_b, ep_start - margin, ep_end + margin)

        if tles_a.empty or tles_b.empty:
            continue

        # Fix #4: TLE quality gate — skip objects with sparse tracking.
        # Avg gap > 5 days means TLE noise will dominate maneuver detection.
        def _avg_gap_days(tles: pd.DataFrame) -> float:
            epochs = pd.to_datetime(tles["EPOCH"], utc=True).sort_values()
            if len(epochs) < 2:
                return float("inf")
            gaps = epochs.diff().dropna().dt.total_seconds() / 86400.0
            return float(gaps.mean())

        if _avg_gap_days(tles_a) > 5.0 or _avg_gap_days(tles_b) > 5.0:
            continue

        min_sep = float("inf")
        time_of_min = None
        hours_below_100 = 0
        hours_below_50 = 0
        total_steps = 0

        t = ep_start
        while t <= ep_end:
            tle_pair_a = _best_tle_at(tles_a, t)
            tle_pair_b = _best_tle_at(tles_b, t)
            if tle_pair_a is None or tle_pair_b is None:
                t += step
                continue

            r_a, _ = _propagate_to(tle_pair_a[0], tle_pair_a[1], t)
            r_b, _ = _propagate_to(tle_pair_b[0], tle_pair_b[1], t)
            if r_a is None or r_b is None:
                t += step
                continue

            sep = np.linalg.norm(np.array(r_a) - np.array(r_b))
            total_steps += 1

            if sep < min_sep:
                min_sep = sep
                time_of_min = t

            if sep < 100.0:
                hours_below_100 += cfg.coarse_step_hours
            if sep < 50.0:
                hours_below_50 += cfg.coarse_step_hours

            t += step

        if min_sep < cfg.proximity_km:
            results.append({
                **ep.to_dict(),
                "min_sep_km": float(min_sep),
                "time_of_min_sep": time_of_min.isoformat() if time_of_min else None,
                "hours_below_100km": hours_below_100,
                "hours_below_50km": hours_below_50,
            })

        if (idx + 1) % 20 == 0:
            print(f"  Tier 3 progress: {idx + 1}/{len(episodes)} episodes verified, "
                  f"{len(results)} passed proximity filter")

    df = pd.DataFrame(results)
    if df.empty:
        print("Tier 3: No episodes passed proximity verification.")
        return pd.DataFrame()

    df = df.sort_values("min_sep_km").reset_index(drop=True)
    print(f"Tier 3: {len(df)} episodes with min separation < {cfg.proximity_km} km")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Tier 4 — RPO causal-chain scoring
# ══════════════════════════════════════════════════════════════════════════════

def _load_tle_history(norad_id: int, start: str, end: str) -> pd.DataFrame:
    """Load full TLE history for maneuver detection."""
    con = sqlite3.connect(DB_PATH)
    sql = """
        SELECT NORAD_CAT_ID, OBJECT_NAME, EPOCH,
               SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION,
               TLE_LINE1, TLE_LINE2
        FROM gp_history
        WHERE NORAD_CAT_ID = ? AND EPOCH >= ? AND EPOCH <= ?
        ORDER BY EPOCH
    """
    df = pd.read_sql_query(sql, con, params=(norad_id, start, end))
    con.close()
    if not df.empty:
        df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)
    return df


def score_rpo(
    episodes: pd.DataFrame,
    cfg: DiscoveryConfig,
) -> pd.DataFrame:
    """Compute RPO causal-chain scores for proximity-verified episodes.

    Sub-scores:
        s_maneuver   — chaser maneuvered during/before the episode
        s_directed   — chaser SMA moved toward target SMA
        s_sustained  — proximity duration (longer = higher)
        s_stationkeep — chaser maneuvers during the proximity window
        s_asymmetry  — one object maneuvers much more than the other
    """
    if episodes.empty:
        return pd.DataFrame()

    SUSTAINED_CAP_DAYS = 90.0  # normalisation cap for s_sustained
    # Weights for composite score
    W_MANEUVER = 0.25
    W_DIRECTED = 0.20
    W_SUSTAINED = 0.20
    W_STATIONKEEP = 0.20
    W_ASYMMETRY = 0.15

    results = []
    for idx, ep in episodes.iterrows():
        id_a, id_b = int(ep["id_a"]), int(ep["id_b"])
        ep_start = pd.Timestamp(ep["start"])
        ep_end = pd.Timestamp(ep["end"])

        # Window for maneuver detection: 60 days before episode to episode end
        pre_margin = timedelta(days=60)
        man_start = (ep_start - pre_margin).strftime("%Y-%m-%d")
        man_end = ep_end.strftime("%Y-%m-%d")

        hist_a = _load_tle_history(id_a, man_start, man_end)
        hist_b = _load_tle_history(id_b, man_start, man_end)

        name_a = str(ep.get("name_a", f"NORAD {id_a}"))
        name_b = str(ep.get("name_b", f"NORAD {id_b}"))

        man_a = detect_maneuvers(hist_a, name_a) if len(hist_a) >= 3 else pd.DataFrame()
        man_b = detect_maneuvers(hist_b, name_b) if len(hist_b) >= 3 else pd.DataFrame()

        n_man_a = int(man_a["is_maneuver"].sum()) if "is_maneuver" in man_a.columns else 0
        n_man_b = int(man_b["is_maneuver"].sum()) if "is_maneuver" in man_b.columns else 0

        # Assign chaser/target: the one with more maneuvers is the chaser
        if n_man_a >= n_man_b:
            chaser_id, target_id = id_a, id_b
            chaser_name, target_name = name_a, name_b
            n_chaser_man, n_target_man = n_man_a, n_man_b
            hist_chaser, hist_target = hist_a, hist_b
            man_chaser = man_a
        else:
            chaser_id, target_id = id_b, id_a
            chaser_name, target_name = name_b, name_a
            n_chaser_man, n_target_man = n_man_b, n_man_a
            hist_chaser, hist_target = hist_b, hist_a
            man_chaser = man_b

        # s_maneuver: did the chaser maneuver at all?
        s_maneuver = min(n_chaser_man / 3.0, 1.0)

        # s_directed: did chaser SMA move toward target SMA?
        s_directed = 0.0
        if len(hist_chaser) >= 2 and len(hist_target) >= 2:
            sma_chaser_start = hist_chaser["SEMIMAJOR_AXIS"].iloc[0]
            sma_chaser_mid = hist_chaser["SEMIMAJOR_AXIS"].median()
            sma_target_mid = hist_target["SEMIMAJOR_AXIS"].median()
            if sma_chaser_start is not None and sma_target_mid is not None:
                initial_gap = abs(sma_chaser_start - sma_target_mid)
                mid_gap = abs(sma_chaser_mid - sma_target_mid)
                if initial_gap > 0:
                    closure = (initial_gap - mid_gap) / initial_gap
                    s_directed = max(0.0, min(closure, 1.0))

        # s_sustained: duration normalised
        duration_days = float(ep.get("duration_days", 0))
        s_sustained = min(duration_days / SUSTAINED_CAP_DAYS, 1.0)

        # s_stationkeep: maneuvers during the proximity window
        s_stationkeep = 0.0
        if "is_maneuver" in man_chaser.columns and "epoch_after" in man_chaser.columns:
            during = man_chaser[
                man_chaser["is_maneuver"]
                & (man_chaser["epoch_after"] >= ep_start)
                & (man_chaser["epoch_after"] <= ep_end)
            ]
            s_stationkeep = min(len(during) / 3.0, 1.0)

        # s_asymmetry: one maneuvers a lot more than the other
        total_man = n_chaser_man + n_target_man
        if total_man > 0:
            s_asymmetry = (n_chaser_man - n_target_man) / total_man
        else:
            s_asymmetry = 0.0

        # Composite
        rpo_score = (
            W_MANEUVER * s_maneuver
            + W_DIRECTED * s_directed
            + W_SUSTAINED * s_sustained
            + W_STATIONKEEP * s_stationkeep
            + W_ASYMMETRY * s_asymmetry
        )

        results.append({
            "chaser_id": chaser_id,
            "chaser_name": chaser_name,
            "target_id": target_id,
            "target_name": target_name,
            "episode_start": ep_start.isoformat(),
            "episode_end": ep_end.isoformat(),
            "duration_days": duration_days,
            "min_delta_sma": ep.get("min_delta_sma"),
            "min_sep_km": ep.get("min_sep_km"),
            "time_of_min_sep": ep.get("time_of_min_sep"),
            "hours_below_100km": ep.get("hours_below_100km"),
            "hours_below_50km": ep.get("hours_below_50km"),
            "n_chaser_maneuvers": n_chaser_man,
            "n_target_maneuvers": n_target_man,
            "s_maneuver": round(s_maneuver, 3),
            "s_directed": round(s_directed, 3),
            "s_sustained": round(s_sustained, 3),
            "s_stationkeep": round(s_stationkeep, 3),
            "s_asymmetry": round(s_asymmetry, 3),
            "rpo_score": round(rpo_score, 3),
        })

        if (idx + 1) % 10 == 0:
            print(f"  Tier 4 progress: {idx + 1}/{len(episodes)} episodes scored")

    df = pd.DataFrame(results)
    if df.empty:
        print("Tier 4: No episodes scored.")
        return pd.DataFrame()

    df = df.sort_values("rpo_score", ascending=False).reset_index(drop=True)
    print(f"Tier 4: {len(df)} scored episodes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_discovery(cfg: DiscoveryConfig) -> pd.DataFrame:
    """Run the full 4-tier RPO discovery pipeline."""
    print(f"\n{'='*80}")
    print(f"  RPO Discovery — {cfg.label}, {cfg.window_start} to {cfg.window_end}")
    print(f"{'='*80}\n")

    # Load SMA data
    sma_dict = load_all_sma(cfg)
    if not sma_dict:
        return pd.DataFrame()

    # Tier 1
    pairs = orbital_plane_pairs(cfg)
    if pairs.empty:
        return pd.DataFrame()

    # Tier 2
    episodes = find_convergence_episodes(sma_dict, pairs, cfg)
    if episodes.empty:
        return pd.DataFrame()

    # Tier 3
    verified = verify_proximity(episodes, cfg)
    if verified.empty:
        return pd.DataFrame()

    # Tier 4
    scored = score_rpo(verified, cfg)
    return scored


def print_results(results: pd.DataFrame, top_n: int = 20) -> None:
    """Pretty-print the top RPO discovery results."""
    if results.empty:
        print("\nNo RPO episodes discovered.")
        return

    show = results.head(top_n)
    print(f"\n{'='*110}")
    print(f"  Top {min(top_n, len(results))} RPO episodes (of {len(results)} total)")
    print(f"{'='*110}")
    print(f"  {'Score':>5}  {'Chaser':<25} {'Target':<25} {'Start':<12} {'End':<12} "
          f"{'Days':>5} {'MinSep':>7} {'Chsr#':>5} {'Tgt#':>4}")
    print(f"  {'-'*5}  {'-'*25} {'-'*25} {'-'*12} {'-'*12} "
          f"{'-'*5} {'-'*7} {'-'*5} {'-'*4}")

    for _, r in show.iterrows():
        chaser = f"{r['chaser_name'][:20]} ({r['chaser_id']})"
        target = f"{r['target_name'][:20]} ({r['target_id']})"
        start = str(r["episode_start"])[:10]
        end = str(r["episode_end"])[:10]
        sep = f"{r['min_sep_km']:.1f}" if pd.notna(r.get("min_sep_km")) else "N/A"
        print(f"  {r['rpo_score']:>5.3f}  {chaser:<25} {target:<25} {start:<12} {end:<12} "
              f"{r['duration_days']:>5.0f} {sep:>7} {r['n_chaser_maneuvers']:>5} {r['n_target_maneuvers']:>4}")

    print(f"{'='*110}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Discover historical RPO activity from TLE data (GEO or LEO).",
    )
    # Time window
    parser.add_argument("--start", default="2016-01-01",
                        help="Window start date (default: 2016-01-01)")
    parser.add_argument("--end", default="2025-12-31",
                        help="Window end date (default: 2025-12-31)")
    # Altitude / inclination band
    parser.add_argument("--alt-min", type=float, default=35000,
                        help="Min periapsis altitude km (default: 35000 = GEO)")
    parser.add_argument("--alt-max", type=float, default=37000,
                        help="Max apoapsis altitude km (default: 37000 = GEO)")
    parser.add_argument("--inc-min", type=float, default=None,
                        help="Min inclination deg filter (optional)")
    parser.add_argument("--inc-max", type=float, default=None,
                        help="Max inclination deg filter (optional)")
    # Tier 1
    parser.add_argument("--delta-inc", type=float, default=5.0,
                        help="Tier 1: max inclination difference deg (default: 5.0)")
    parser.add_argument("--delta-raan", type=float, default=None,
                        help="Tier 1: max RAAN difference deg (default: off; recommend 5.0 for LEO)")
    # Tier 2
    parser.add_argument("--conv-sma", type=float, default=50.0,
                        help="Tier 2: SMA convergence threshold km (default: 50; use 20 for LEO)")
    parser.add_argument("--conv-days", type=int, default=7,
                        help="Tier 2: min convergence duration days (default: 7)")
    # Tier 3
    parser.add_argument("--prox-km", type=float, default=200.0,
                        help="Tier 3: proximity threshold km (default: 200; use 20 for LEO)")
    parser.add_argument("--step-hours", type=float, default=1.0,
                        help="Tier 3: propagation step hours (default: 1.0; use 0.25 for LEO)")
    # Output
    parser.add_argument("--backfill", action="store_true",
                        help="Run Space-Track backfill for the configured band before analysis")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    parser.add_argument("--top", type=int, default=20,
                        help="Print top N results (default: 20)")
    args = parser.parse_args()

    cfg = DiscoveryConfig(
        window_start=args.start,
        window_end=args.end,
        alt_min=args.alt_min,
        alt_max=args.alt_max,
        inc_min=args.inc_min,
        inc_max=args.inc_max,
        delta_inc_max=args.delta_inc,
        delta_raan_max=args.delta_raan,
        convergence_sma_km=args.conv_sma,
        convergence_min_days=args.conv_days,
        proximity_km=args.prox_km,
        coarse_step_hours=args.step_hours,
    )

    if args.backfill:
        backfill_geo(cfg)

    results = run_discovery(cfg)

    print_results(results, args.top)

    if args.csv and not results.empty:
        results.to_csv(args.csv, index=False)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
