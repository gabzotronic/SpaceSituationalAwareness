"""
maneuver_detection.py — Detect orbital maneuvers from GP element history.

Approach (OD-based, phases 3-7 of MANEUVER_DETECTION_MERGE_PLAN.md):
  1. Pull historical GP records (incl. BSTAR, TLE lines) from gp_history.
  2. Pre-filter TLEs with |B*| > OD_BSTAR_NOISE_MAX (likely poor fit).
  3. For each consecutive epoch pair (gap ≤ OD_MAX_GAP_DAYS):
       a. SGP4 state at epoch₁ (mean → osculating)
       b. RK4 integrate epoch₁ → epoch₂ with J2 + US Std Atm drag
       c. SGP4 state at epoch₂ for actual state
       d. Compute position and velocity residuals
  4. Adaptive MAD-based threshold on gap-normalised velocity residual.
  5. B* step-change as secondary signal.
  6. Cross-check Kp for confirmed vs uncertain classification.
  7. Persist to maneuvers table; generate 7-panel plot on demand.

Usage:
    python analysis/maneuver_detection.py
    python analysis/maneuver_detection.py --norad 58316
    python analysis/maneuver_detection.py --norad 25544 --plot
    python analysis/maneuver_detection.py --no-plot

Dependencies: numpy, pandas, matplotlib, requests, sgp4
Run under: conda run -n orbit python analysis/maneuver_detection.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH,
    MIN_HISTORY_DAYS,
    OD_BSTAR_DELTA_THRESH,
    OD_BSTAR_NOISE_MAX,
    OD_KP_THRESHOLD,
    OD_MAX_GAP_DAYS,
    OD_SIGMA_MULTIPLIER,
)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_NORAD = 66748
MU_KM3_S2     = 398600.4418   # Earth gravitational parameter km³/s²
RE_KM         = 6378.137      # Earth equatorial radius km
J2            = 1.08262668e-3  # second zonal harmonic
OMEGA_EARTH   = 7.2921150e-5  # Earth rotation rate rad/s


# ── 1. DB queries + B* parsing ────────────────────────────────────────────────

def _parse_bstar_tle1(tle1: str) -> float:
    """
    Parse B* drag term from TLE line 1 columns 53:61 (0-indexed, 8-char field).

    TLE compact format: SMMMMM±EE
        S       = sign character (' ' for positive, '-' for negative)
        MMMMM   = 5-digit mantissa (implicit decimal: 0.MMMMM)
        ±EE     = signed 2-digit exponent

    Result is in 1/Earth-radius (SGP4 native units), same as the GP JSON
    BSTAR column.  This parser does NOT strip before checking length so
    positive values (leading space) are handled correctly.

    Note: the reference script (orbit_analysis_od.py _parse_bstar) strips
    first, reducing positive values to 7 chars and returning 0.0 — a bug
    that causes all positive B* to be silently zeroed.
    """
    try:
        s = tle1[53:61]           # exactly 8 chars, no strip
        if len(s) == 8:
            sign     = -1.0 if s[0] == '-' else 1.0
            mantissa = float(s[1:6]) * 1e-5
            exp      = int(s[6:8])
            return sign * mantissa * (10 ** exp)
        return float(s.strip())
    except Exception:
        return 0.0


def get_gp_history(
    norad_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Pull historical GP element sets for one object from gp_history.

    Returns DataFrame with columns:
        EPOCH, SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION,
        RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY,
        BSTAR, BSTAR_TLE, TLE_LINE1, TLE_LINE2

    BSTAR_TLE is parsed directly from TLE_LINE1 chars 53:61 using a correct
    8-char parser, independent of the GP JSON BSTAR column.
    EPOCH is parsed to datetime.

    Optional start_date / end_date ('YYYY-MM-DD') clip the returned window.
    """
    con = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT EPOCH, SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION,
               RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY,
               BSTAR, TLE_LINE1, TLE_LINE2
        FROM gp_history
        WHERE NORAD_CAT_ID = ?
    """
    params = [norad_id]
    if start_date:
        query += " AND EPOCH >= ?"
        params.append(start_date)
    if end_date:
        query += " AND EPOCH <= ?"
        params.append(end_date + "T23:59:59")
    query += " ORDER BY EPOCH ASC"
    rows = con.execute(query, params).fetchall()
    con.close()

    cols = [
        "EPOCH", "SEMIMAJOR_AXIS", "ECCENTRICITY", "INCLINATION",
        "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
        "BSTAR", "TLE_LINE1", "TLE_LINE2",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["EPOCH"] = pd.to_datetime(df["EPOCH"])
        # Parse B* from TLE_LINE1 text — authoritative source, correct units
        df["BSTAR_TLE"] = df["TLE_LINE1"].apply(
            lambda t: _parse_bstar_tle1(t) if isinstance(t, str) else float("nan")
        )
        # Deduplicate on EPOCH: Space-Track occasionally issues revised element
        # sets with the same epoch timestamp but a new GP_ID. Keep only the last
        # (highest GP_ID) per epoch — the most recently updated elements.
        # The DB query is already ORDER BY EPOCH ASC, so the last occurrence of
        # each epoch is the most recently inserted (highest GP_ID).
        n_before = len(df)
        df = df.drop_duplicates(subset=["EPOCH"], keep="last").reset_index(drop=True)
        if len(df) < n_before:
            print(f"  Deduplicated {n_before - len(df)} revised-element-set duplicates "
                  f"({n_before} → {len(df)} records)")
    return df


def get_object_name(norad_id: int) -> Optional[str]:
    """Return OBJECT_NAME from gp table, or None if not found."""
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT OBJECT_NAME FROM gp WHERE NORAD_CAT_ID = ?", (norad_id,)
    ).fetchone()
    con.close()
    return row[0] if row else None


# ── 2. SGP4 + RK4 propagator ──────────────────────────────────────────────────

def _sgp4_state(tle1: str, tle2: str, dt_sec: float) -> Optional[np.ndarray]:
    """
    Propagate TLE by dt_sec seconds using SGP4.

    Returns 6-element state vector [x,y,z km, vx,vy,vz km/s] in TEME frame,
    or None if SGP4 errors out.
    """
    try:
        from sgp4.api import Satrec, WGS84
        sat = Satrec.twoline2rv(tle1, tle2, WGS84)
        # SGP4 epoch from TLE
        epoch_jd = sat.jdsatepoch + sat.jdsatepochF
        target_jd = epoch_jd + dt_sec / 86400.0
        jd_whole = int(target_jd)
        jd_frac  = target_jd - jd_whole
        e, r, v = sat.sgp4(jd_whole, jd_frac)
        if e != 0:
            return None
        return np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
    except Exception:
        return None


def _atmo_density(alt_km: float) -> float:
    """
    US Standard Atmosphere 1976 — piecewise exponential density (kg/m³).
    """
    # Table: [base_alt_km, scale_height_km, ref_density_kg_m3]
    layers = [
        (0,    8.44,  1.225),
        (25,   6.49,  3.899e-2),
        (30,   6.75,  1.774e-2),
        (40,   7.58,  3.972e-3),
        (50,   8.55,  1.057e-3),
        (60,   7.71,  3.206e-4),
        (70,   6.59,  8.770e-5),
        (80,   5.98,  1.905e-5),
        (90,   5.57,  3.396e-6),
        (100,  5.23,  5.297e-7),
        (110,  5.63,  9.661e-8),
        (120,  6.34,  2.438e-8),
        (130,  7.43,  8.484e-9),
        (140,  8.82,  3.845e-9),
        (150, 11.51,  2.070e-9),
        (180, 15.67,  5.464e-10),
        (200, 18.29,  2.789e-10),
        (250, 22.26,  7.248e-11),
        (300, 29.74,  2.418e-11),
        (350, 37.11,  9.158e-12),
        (400, 45.45,  3.725e-12),
        (450, 53.64,  1.585e-12),
        (500, 60.92,  6.967e-13),
        (600, 63.14,  1.454e-13),
        (700, 58.52,  3.614e-14),
        (800, 53.30,  1.170e-14),
        (900, 53.55,  5.245e-15),
        (1000, 60.78, 3.019e-15),
    ]
    if alt_km < 0:
        return layers[0][2]
    base, scale, rho0 = layers[0]
    for i in range(1, len(layers)):
        if alt_km < layers[i][0]:
            break
        base, scale, rho0 = layers[i]
    return rho0 * np.exp(-(alt_km - base) / scale)


def _bstar_to_drag(bstar: float) -> float:
    """
    Convert B* (1/earth-radii) to ballistic coefficient used in drag EOM.

    B* = (C_D * A) / (2 * m) * rho_0  [1/ER]
    drag_coeff = B* * 2 / rho_0_ref   [m²/kg] — absorbed into EOM as B*
    We return B* converted to SI units: 1/m
    1 Earth radius = 6378137 m → B* [1/m] = B* [1/ER] / 6378137
    """
    return bstar / RE_KM / 1000.0  # 1/m


def _derivatives(t: float, state: np.ndarray, bstar_si: float) -> np.ndarray:
    """
    Equations of motion: two-body + J2 oblateness + atmospheric drag.

    state = [x, y, z (km), vx, vy, vz (km/s)]
    Returns d/dt(state) in same units.
    """
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x*x + y*y + z*z)
    r_m = r * 1000.0  # km → m

    # Two-body
    a_2body = -MU_KM3_S2 / r**3 * np.array([x, y, z])

    # J2 perturbation
    factor = 1.5 * J2 * MU_KM3_S2 * RE_KM**2 / r**5
    z_r2   = (z / r)**2
    a_j2   = factor * np.array([
        x * (5*z_r2 - 1),
        y * (5*z_r2 - 1),
        z * (5*z_r2 - 3),
    ])

    # Atmospheric drag
    alt_km  = r - RE_KM
    rho     = _atmo_density(alt_km)         # kg/m³
    v_rel   = np.array([vx, vy, vz])        # approx inertial ≈ relative (ignore Earth rotation)
    v_mag   = np.linalg.norm(v_rel) * 1000  # km/s → m/s
    # drag acceleration: a_drag = -B* * rho * v²  (in m/s²)
    # convert back to km/s²
    a_drag_ms2 = -bstar_si * rho * v_mag**2 * (v_rel / np.linalg.norm(v_rel))
    a_drag     = a_drag_ms2 / 1000.0  # m/s² → km/s²

    a_total = a_2body + a_j2 + a_drag
    return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])


def propagate_od(tle1: str, tle2: str, bstar: float, dt_sec: float) -> Optional[np.ndarray]:
    """
    Propagate TLE forward by dt_sec seconds using SGP4 (initial state) + RK4
    integration with J2 + atmospheric drag.

    Returns 6-element state [x,y,z km, vx,vy,vz km/s] or None on error.
    """
    state0 = _sgp4_state(tle1, tle2, 0.0)
    if state0 is None:
        return None

    bstar_si = _bstar_to_drag(bstar)
    state    = state0.copy()

    # Adaptive RK4: use 60 s steps, smaller near epoch for accuracy
    step     = min(60.0, dt_sec / 10.0) if dt_sec > 0 else 60.0
    t        = 0.0

    while t < dt_sec:
        h = min(step, dt_sec - t)
        k1 = _derivatives(t,       state,           bstar_si)
        k2 = _derivatives(t + h/2, state + h/2*k1,  bstar_si)
        k3 = _derivatives(t + h/2, state + h/2*k2,  bstar_si)
        k4 = _derivatives(t + h,   state + h*k3,    bstar_si)
        state = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h

    return state


# ── 3. Space weather ──────────────────────────────────────────────────────────

def _parse_kp_text(text: str, file_type: str = "nowcast") -> pd.DataFrame:
    """Parse raw GFZ Potsdam Kp text into a DataFrame."""
    records = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])

            if file_type == "nowcast":
                hour = float(parts[3])
                kp   = float(parts[7])
                if kp < 0:
                    continue
                records.append({
                    "datetime": pd.Timestamp(year, month, day, int(hour), int((hour % 1) * 60)),
                    "kp": kp,
                })
            elif file_type == "daily":
                for hour_idx in range(8):
                    kp = float(parts[7 + hour_idx])
                    if kp < 0:
                        continue
                    records.append({
                        "datetime": pd.Timestamp(year, month, day, hour_idx * 3),
                        "kp": kp,
                    })

        except (ValueError, IndexError):
            continue

    return pd.DataFrame(records)


def _fetch_kp_from_source(since_year: Optional[int] = None):
    """Download Kp index from GFZ Potsdam and return (DataFrame, raw_text)."""
    if since_year is not None and since_year >= 2024:
        url       = "https://kp.gfz-potsdam.de/app/files/Kp_ap_nowcast.txt"
        file_type = "nowcast"
        print("  Downloading Kp nowcast file...")
    else:
        url       = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
        file_type = "daily"
        print("  Downloading full Kp historical file (this may take a moment)...")

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return _parse_kp_text(response.text, file_type=file_type), response.text


def get_kp_index(start_date: str, end_date: str, data_dir: Path) -> pd.DataFrame:
    """
    Return Kp geomagnetic index for a date range, using a local cache if fresh.
    Downloads from GFZ Potsdam on first run or when stale (> 1 day old).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "kp_index.csv"
    txt_path = data_dir / "kp_index.txt"

    if csv_path.exists():
        kp_df    = pd.read_csv(str(csv_path), parse_dates=["datetime"])
        latest   = kp_df["datetime"].max()
        age_days = (datetime.now() - latest.to_pydatetime().replace(tzinfo=None)).days

        if kp_df["kp"].max() > 9 or kp_df["kp"].min() < 0:
            print("WARNING: Local Kp file has invalid values. Re-downloading...")
            csv_path.unlink()
            if txt_path.exists():
                txt_path.unlink()
            kp_df, raw_txt = _fetch_kp_from_source(since_year=None)
            txt_path.write_text(raw_txt)
            kp_df.to_csv(str(csv_path), index=False)
            print(f"Saved {csv_path} ({len(kp_df):,} records, up to {kp_df['datetime'].max()})")

        elif age_days <= 1:
            print(f"Loaded local Kp file (latest: {latest})")
        else:
            print(f"Local Kp file is {age_days} days old. Fetching recent data from GFZ Potsdam...")
            new_df, new_txt = _fetch_kp_from_source(since_year=latest.year)

            with open(str(txt_path), "a") as f:
                f.write(new_txt)

            kp_df = (
                pd.concat([kp_df[kp_df["datetime"] < new_df["datetime"].min()], new_df])
                .drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            kp_df.to_csv(str(csv_path), index=False)
            print(f"Updated {csv_path} (now covers up to {kp_df['datetime'].max()})")
    else:
        print("No local Kp file found. Downloading full historical file from GFZ Potsdam...")
        kp_df, raw_txt = _fetch_kp_from_source(since_year=None)
        txt_path.write_text(raw_txt)
        kp_df.to_csv(str(csv_path), index=False)
        print(f"Saved {csv_path} ({len(kp_df):,} records, up to {kp_df['datetime'].max()})")

    return kp_df[
        (kp_df["datetime"] >= start_date) & (kp_df["datetime"] <= end_date)
    ].reset_index(drop=True)


def _get_kp_for_epoch(epoch: pd.Timestamp, kp_df: pd.DataFrame) -> float:
    """Return the Kp value for the 3-hour interval containing epoch."""
    mask  = (kp_df["datetime"] <= epoch) & (kp_df["datetime"] + pd.Timedelta(hours=3) > epoch)
    match = kp_df[mask]
    if not match.empty:
        return match.iloc[0]["kp"]
    idx = (kp_df["datetime"] - epoch).abs().argmin()
    return kp_df.iloc[idx]["kp"]


# ── 4. Maneuver analysis ──────────────────────────────────────────────────────

def analyse_maneuvers(
    norad_id: int,
    data_dir: Path,
    con: Optional[sqlite3.Connection] = None,
    kp_df: Optional[pd.DataFrame] = None,
    sigma_multiplier: float = OD_SIGMA_MULTIPLIER,
    kp_threshold: float = OD_KP_THRESHOLD,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Detect maneuvers for a single object using SGP4+RK4+J2+drag propagation.

    Args:
        norad_id:         NORAD_CAT_ID to analyse.
        data_dir:         Directory for Kp index cache.
        con:              Optional open SQLite connection for write-back.
                          If None, results are not persisted to the DB.
        kp_df:            Pre-loaded Kp DataFrame (pass from fleet loop).
                          If None, fetched internally.
        sigma_multiplier: Adaptive threshold = median + N × MAD.
        kp_threshold:     Kp above which space weather is 'disturbed'.

    Returns:
        DataFrame with per-pair signals and classification columns.
    """
    df = get_gp_history(norad_id, start_date=start_date, end_date=end_date)

    if df.empty:
        print(f"No gp_history records found for NORAD {norad_id}.")
        print("Run: python ingest.py --backfill --norad", norad_id)
        return pd.DataFrame()

    # Phase 0 coverage check
    coverage_days = (df["EPOCH"].max() - df["EPOCH"].min()).days
    if coverage_days < MIN_HISTORY_DAYS:
        print(f"WARNING: Only {coverage_days} days of gp_history found for NORAD {norad_id}.")
        print(f"Run: python ingest.py --backfill --norad {norad_id}")
        print("Proceeding with available data — detection quality will be reduced.")

    print(f"Loaded {len(df)} GP history records "
          f"({df['EPOCH'].min().date()} -> {df['EPOCH'].max().date()})")

    # Phase 4a — B* noise pre-filter (adaptive)
    # Use BSTAR_TLE (parsed from TLE text) as the authoritative source.
    # If the fixed threshold would remove >50% of TLEs, relax it to the
    # 95th percentile of |B*| so at least ~95% of TLEs are retained.
    n_before   = len(df)
    bstar_col  = "BSTAR_TLE" if "BSTAR_TLE" in df.columns else "BSTAR"
    bstar_abs  = df[bstar_col].abs()
    fixed_keep = (bstar_abs <= OD_BSTAR_NOISE_MAX).sum()
    if fixed_keep < n_before * 0.5:
        relaxed_thresh = float(np.percentile(bstar_abs.dropna(), 95))
        print(f"  B* noise filter: fixed threshold {OD_BSTAR_NOISE_MAX:.0e} would keep only "
              f"{fixed_keep}/{n_before} TLEs (<50%). Relaxing to 95th-pct = {relaxed_thresh:.2e}")
        noise_thresh = relaxed_thresh
    else:
        noise_thresh = OD_BSTAR_NOISE_MAX

    df = df[df[bstar_col].notna() & (bstar_abs <= noise_thresh)].reset_index(drop=True)
    n_filtered = n_before - len(df)
    if n_filtered:
        print(f"  Filtered {n_filtered} TLEs with |B*| > {noise_thresh:.2e} (noise)")

    if len(df) < 2:
        print("Insufficient clean TLEs after B* filter.")
        return pd.DataFrame()

    # Check TLE availability
    has_tles = df["TLE_LINE1"].notna().any() and df["TLE_LINE2"].notna().any()
    if not has_tles:
        print("WARNING: TLE_LINE1/TLE_LINE2 missing — cannot run OD propagator.")
        return pd.DataFrame()

    # ── Phase 4b — Per-pair computation ──────────────────────────────────────
    results = []
    n_pairs = 0
    n_skipped_gap = 0
    n_skipped_sgp4 = 0

    for i in range(len(df) - 1):
        row_now  = df.iloc[i]
        row_next = df.iloc[i + 1]

        gap_sec  = (row_next["EPOCH"] - row_now["EPOCH"]).total_seconds()
        gap_days = gap_sec / 86400.0

        if gap_days < 1e-4 or gap_days > OD_MAX_GAP_DAYS:
            n_skipped_gap += 1
            continue

        if pd.isna(row_now["TLE_LINE1"]) or pd.isna(row_now["TLE_LINE2"]):
            continue
        if pd.isna(row_next["TLE_LINE1"]) or pd.isna(row_next["TLE_LINE2"]):
            continue

        tle1_now  = str(row_now["TLE_LINE1"])
        tle2_now  = str(row_now["TLE_LINE2"])
        tle1_next = str(row_next["TLE_LINE1"])
        tle2_next = str(row_next["TLE_LINE2"])

        # Propagated state: epoch₁ → epoch₂ via OD propagator
        state_pred = propagate_od(tle1_now, tle2_now, float(row_now[bstar_col]), gap_sec)
        # Actual state at epoch₂: SGP4 at dt=0
        state_act  = _sgp4_state(tle1_next, tle2_next, 0.0)

        if state_pred is None or state_act is None:
            n_skipped_sgp4 += 1
            continue

        pos_residual_km = float(np.linalg.norm(state_pred[:3] - state_act[:3]))
        vel_residual_ms = float(np.linalg.norm(state_pred[3:] - state_act[3:]) * 1000)
        vel_res_per_day = vel_residual_ms / max(gap_days, 1e-6)

        bstar_now  = float(row_now[bstar_col])
        bstar_next = float(row_next[bstar_col])
        # Only compute delta when both values are genuinely non-zero.
        # A zero B* means the TLE fitter assigned no drag (deliberate or
        # data gap) — delta between zero and non-zero is not a maneuver signal.
        if bstar_now != 0.0 and bstar_next != 0.0:
            delta_bstar = abs(bstar_next - bstar_now)
        else:
            delta_bstar = 0.0

        results.append({
            "epoch_from":      row_now["EPOCH"],
            "epoch_to":        row_next["EPOCH"],
            "gap_days":        round(gap_days, 3),
            "pos_residual_km": round(pos_residual_km, 3),
            "vel_residual_ms": round(vel_residual_ms, 4),
            "vel_res_per_day": round(vel_res_per_day, 4),
            "bstar":           bstar_next,
            "delta_bstar":     round(delta_bstar, 8),
            "delta_sma":       round(float(row_next["SEMIMAJOR_AXIS"] - row_now["SEMIMAJOR_AXIS"]), 3),
            "delta_ecc":       round(float(row_next["ECCENTRICITY"]   - row_now["ECCENTRICITY"]),   6),
            "delta_inc":       round(float(row_next["INCLINATION"]    - row_now["INCLINATION"]),    4),
        })
        n_pairs += 1

    if n_skipped_gap:
        print(f"  Skipped {n_skipped_gap} pairs with gap > {OD_MAX_GAP_DAYS} days")
    if n_skipped_sgp4:
        print(f"  Skipped {n_skipped_sgp4} pairs due to SGP4 errors")

    if not results:
        print("No valid epoch pairs found.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # ── Phase 4c — Adaptive MAD thresholds ───────────────────────────────────
    # VR/day threshold
    vrpd      = result_df["vel_res_per_day"].values
    finite    = np.isfinite(vrpd)
    median_vr = float(np.median(vrpd[finite]))
    mad_vr    = float(np.median(np.abs(vrpd[finite] - median_vr)))
    threshold = median_vr + sigma_multiplier * mad_vr
    result_df["vr_threshold"] = threshold

    # B* delta adaptive threshold — same MAD logic.
    # Only include pairs where both B* values were non-zero (delta > 0 is meaningful).
    db_vals  = result_df["delta_bstar"].values
    db_nonz  = db_vals[db_vals > 0]
    if len(db_nonz) >= 10:
        median_db       = float(np.median(db_nonz))
        mad_db          = float(np.median(np.abs(db_nonz - median_db)))
        bstar_threshold = median_db + sigma_multiplier * mad_db
    else:
        # Fallback to fixed threshold when not enough non-zero pairs
        bstar_threshold = OD_BSTAR_DELTA_THRESH
    result_df["bstar_threshold"] = bstar_threshold

    # ── Phase 5 — Kp lookup ───────────────────────────────────────────────────
    start = result_df["epoch_from"].min().strftime("%Y-%m-%d")
    end   = result_df["epoch_to"].max().strftime("%Y-%m-%d")

    if kp_df is None:
        kp_df = get_kp_index(start, end, data_dir)

    if kp_df.empty:
        print("WARNING: No Kp data for this time range. All maneuvers marked uncertain.")
        result_df["kp"] = float("nan")
    else:
        result_df["kp"] = result_df["epoch_from"].apply(
            lambda ep: _get_kp_for_epoch(ep, kp_df)
        )

    # ── Phase 4d — Classification ─────────────────────────────────────────────
    result_df["bad_space_weather"]  = result_df["kp"] >= kp_threshold
    od_flag    = np.isfinite(result_df["vel_res_per_day"]) & (result_df["vel_res_per_day"] > threshold)
    bstar_flag = result_df["delta_bstar"] > bstar_threshold
    likely     = od_flag | bstar_flag
    result_df["od_flag"]            = od_flag
    result_df["bstar_flag"]         = bstar_flag
    result_df["likely_maneuver"]    = likely
    result_df["confirmed_maneuver"] = likely & ~result_df["bad_space_weather"]
    result_df["uncertain_maneuver"] = likely &  result_df["bad_space_weather"]
    result_df["classification"]     = result_df.apply(
        lambda r: "confirmed" if r["confirmed_maneuver"]
                  else ("uncertain" if r["uncertain_maneuver"] else None),
        axis=1,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Maneuver Analysis: NORAD {norad_id} ──────────────────────────────────")
    print(f"  Epoch pairs analysed:      {len(result_df)}")
    print(f"  VR/day threshold:          {threshold:.4f} m/s/day  "
          f"(median={median_vr:.4f}, MAD={mad_vr:.4f}, N={sigma_multiplier})")
    print(f"  B* delta threshold:        {bstar_threshold:.2e}  (adaptive MAD)")
    print(f"  Kp threshold:              {kp_threshold}")
    print(f"  OD-flagged:                {od_flag.sum()}")
    print(f"  B*-flagged:                {bstar_flag.sum()}")
    print(f"  Confirmed maneuvers:       {result_df['confirmed_maneuver'].sum()}")
    print(f"  Uncertain (bad weather):   {result_df['uncertain_maneuver'].sum()}")

    def _print_rows(rows: pd.DataFrame, label: str) -> None:
        for _, r in rows.iterrows():
            kp_str = f"{r['kp']:.1f}" if pd.notna(r["kp"]) else "N/A"
            flags  = []
            if r["od_flag"]:
                flags.append("OD")
            if r["bstar_flag"]:
                flags.append("B*")
            print(f"  {r['epoch_from']}  ->  {r['epoch_to']}  "
                  f"(gap: {r['gap_days']:.3f}d)  "
                  f"VR: {r['vel_residual_ms']:.2f} m/s  "
                  f"dSMA: {r['delta_sma']:+.3f} km  "
                  f"dB*: {r['delta_bstar']:.2e}  "
                  f"Kp: {kp_str}  [{label}|{'+'.join(flags)}]")

    if result_df["confirmed_maneuver"].any():
        print("\n── Confirmed Maneuver Epochs ────────────────────────────────────────")
        _print_rows(result_df[result_df["confirmed_maneuver"]], "CONFIRMED")

    if result_df["uncertain_maneuver"].any():
        print("\n── Uncertain Epochs (possible maneuver, disturbed space weather) ─────")
        _print_rows(result_df[result_df["uncertain_maneuver"]], "UNCERTAIN")

    # ── Phase 6a — DB write-back ──────────────────────────────────────────────
    if con is not None:
        _write_maneuvers(con, norad_id, result_df)

    return result_df


def _write_maneuvers(con: sqlite3.Connection, norad_id: int, result_df: pd.DataFrame) -> None:
    """Persist confirmed/uncertain maneuver detections to the maneuvers table."""
    to_write = result_df[result_df["likely_maneuver"]]
    if to_write.empty:
        return

    rows = []
    for _, r in to_write.iterrows():
        rows.append((
            norad_id,
            r["epoch_from"].isoformat(),
            r["epoch_to"].isoformat(),
            r["delta_sma"],
            r["delta_ecc"],
            r["delta_inc"],
            None,   # DELTA_RAAN — not tracked in OD mode (informational only)
            None,   # DELTA_PERIOD
            None,   # DELTA_APOAPSIS
            None,   # DELTA_PERIAPSIS
            r["classification"],
            float(r["kp"]) if pd.notna(r["kp"]) else None,
            r["vel_residual_ms"],
            r["delta_bstar"],
        ))

    con.executemany(
        """INSERT OR IGNORE INTO maneuvers
           (NORAD_CAT_ID, EPOCH_BEFORE, EPOCH_AFTER,
            DELTA_SMA, DELTA_ECCENTRICITY, DELTA_INCLINATION,
            DELTA_RAAN, DELTA_PERIOD, DELTA_APOAPSIS, DELTA_PERIAPSIS,
            CLASSIFICATION, KP, VEL_RESIDUAL_MS, BSTAR_DELTA)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()


# ── 5. Plot ───────────────────────────────────────────────────────────────────

def plot_maneuvers(
    result_df: pd.DataFrame,
    norad_id: int,
    object_name: Optional[str],
    kp_threshold: float,
    plot_dir: Path,
) -> None:
    """7-panel maneuver detection plot."""
    title_name = f"{object_name} (NORAD {norad_id})" if object_name else f"NORAD {norad_id}"
    confirmed  = result_df[result_df["confirmed_maneuver"]]
    uncertain  = result_df[result_df["uncertain_maneuver"]]
    x          = result_df["epoch_to"]
    threshold  = result_df["vr_threshold"].iloc[0] if "vr_threshold" in result_df.columns else None

    fig, axes = plt.subplots(7, 1, figsize=(14, 20), sharex=True)

    def _mark(ax: plt.Axes, col: str) -> None:
        if not confirmed.empty:
            ax.scatter(confirmed["epoch_to"], confirmed[col],
                       color="red", zorder=5, s=40, label="Confirmed")
        if not uncertain.empty:
            ax.scatter(uncertain["epoch_to"], uncertain[col],
                       color="orange", zorder=5, marker="^", s=40, label="Uncertain")

    # Panel 1: gap-normalised velocity residual (log scale)
    axes[0].semilogy(x, result_df["vel_res_per_day"], marker="o", markersize=3, color="steelblue")
    if threshold is not None:
        axes[0].axhline(threshold, color="red", linestyle="--", linewidth=1,
                        label=f"Threshold ({threshold:.2f})")
    _mark(axes[0], "vel_res_per_day")
    axes[0].set_ylabel("VR/day (m/s/day)")
    axes[0].set_title(f"Maneuver Detection — {title_name}")
    axes[0].legend(fontsize=7, loc="upper right")

    # Panel 2: raw velocity residual
    axes[1].plot(x, result_df["vel_residual_ms"], marker="o", markersize=3, color="teal")
    _mark(axes[1], "vel_residual_ms")
    axes[1].set_ylabel("Vel residual (m/s)")
    axes[1].legend(fontsize=7, loc="upper right")

    # Panel 3: B* history
    axes[2].plot(x, result_df["bstar"], marker="o", markersize=3, color="purple")
    axes[2].set_ylabel("B* (1/ER)")

    # Panel 4: ΔB* step-changes
    bstar_thresh_plot = result_df["bstar_threshold"].iloc[0] if "bstar_threshold" in result_df.columns else OD_BSTAR_DELTA_THRESH
    axes[3].plot(x, result_df["delta_bstar"], marker="o", markersize=3, color="darkviolet")
    axes[3].axhline(bstar_thresh_plot, color="red", linestyle=":", linewidth=1,
                    label=f"Threshold ({bstar_thresh_plot:.2e})")
    axes[3].legend(fontsize=7, loc="upper right")
    axes[3].set_ylabel("ΔB*")

    # Panel 5: ΔSMA
    axes[4].plot(x, result_df["delta_sma"], marker="o", markersize=3, color="darkorange")
    axes[4].axhline(0, color="gray", linestyle="--")
    _mark(axes[4], "delta_sma")
    axes[4].set_ylabel("dSMA (km)")
    axes[4].legend(fontsize=7, loc="upper right")

    # Panel 6: ΔEccentricity
    axes[5].plot(x, result_df["delta_ecc"], marker="o", markersize=3, color="green")
    axes[5].axhline(0, color="gray", linestyle="--")
    _mark(axes[5], "delta_ecc")
    axes[5].set_ylabel("dEccentricity")
    axes[5].legend(fontsize=7, loc="upper right")

    # Panel 7: Kp index
    if result_df["kp"].notna().any():
        axes[6].plot(x, result_df["kp"], marker="o", markersize=3, color="indigo")
        axes[6].axhline(kp_threshold, color="red", linestyle="--", linewidth=1,
                        label=f"Kp threshold ({kp_threshold})")
        axes[6].fill_between(
            x, result_df["kp"], kp_threshold,
            where=result_df["kp"] >= kp_threshold,
            color="red", alpha=0.2, label="Disturbed",
        )
        axes[6].legend(fontsize=7, loc="upper right")
    else:
        axes[6].text(0.5, 0.5, "Kp data unavailable",
                     transform=axes[6].transAxes, ha="center", va="center", color="gray")
    axes[6].set_ylabel("Kp")
    axes[6].set_xlabel("Epoch")

    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"maneuver_detection_{norad_id}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect orbital maneuvers from GP element history in satcat.db",
    )
    parser.add_argument(
        "--norad", type=int, default=DEFAULT_NORAD,
        help=f"NORAD_CAT_ID to analyse (default: {DEFAULT_NORAD})",
    )
    parser.add_argument(
        "--sigma", type=float, default=OD_SIGMA_MULTIPLIER,
        help=f"MAD multiplier for adaptive threshold (default: {OD_SIGMA_MULTIPLIER})",
    )
    parser.add_argument(
        "--kp", type=float, default=OD_KP_THRESHOLD,
        help=f"Kp threshold for disturbed space weather (default: {OD_KP_THRESHOLD})",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (default: earliest available in gp_history)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: latest available in gp_history)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate 7-panel detection plot",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (default behaviour — use --plot to enable)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    plot_dir = Path(__file__).parent.parent / "plots"

    object_name = get_object_name(args.norad)
    if object_name:
        print(f"Analysing: {object_name} (NORAD {args.norad})")
    else:
        print(f"Analysing: NORAD {args.norad} (not found in current gp table)")

    result_df = analyse_maneuvers(
        norad_id=args.norad,
        data_dir=data_dir,
        sigma_multiplier=args.sigma,
        kp_threshold=args.kp,
        start_date=args.start,
        end_date=args.end,
    )

    if result_df.empty:
        return

    print("\n" + result_df[[
        "epoch_from", "epoch_to", "vel_residual_ms", "vel_res_per_day",
        "delta_bstar", "delta_sma", "kp",
        "confirmed_maneuver", "uncertain_maneuver",
    ]].to_string(index=False))

    if args.plot and not args.no_plot:
        plot_maneuvers(
            result_df,
            norad_id=args.norad,
            object_name=object_name,
            kp_threshold=args.kp,
            plot_dir=plot_dir,
        )


if __name__ == "__main__":
    main()
