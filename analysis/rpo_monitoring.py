"""
rpo_monitoring.py — RPO monitoring analysis.

Configurable for any chaser/target rendezvous and proximity operation (RPO).
Edit the MISSION config at the bottom to analyse a different mission.

This script:
  1. Backfills gp_history for both objects over the mission window.
  2. Detects chaser maneuver epochs via SGP4 velocity residuals.
  3. Runs same detection on target as a noise-floor reference.
  4. Computes inter-satellite separation over the window at 5-min resolution.
  5. Produces two annotated figures:
       plots/rpo_overview.png   — separation, altitude, TLE cadence
       plots/rpo_maneuvers.png  — velocity residuals, DELTA_SMA

Usage (from repo root):
    conda run -n orbit python analysis/rpo_monitoring.py

Dependencies: sgp4, numpy, pandas, matplotlib, spacetrack, python-dotenv
"""

from __future__ import annotations

import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from sgp4.api import Satrec, jday

# ── Repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, SPACETRACK_IDENTITY, SPACETRACK_PASSWORD


# ── Mission configuration ────────────────────────────────────────────────────

@dataclass
class MissionConfig:
    chaser_name: str
    chaser_norad: int
    target_name: str
    target_norad: int
    window_start: str          # "YYYY-MM-DD"
    window_end: str            # "YYYY-MM-DD"
    phases: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.chaser_name} / {self.target_name}"

    @property
    def t_start(self) -> datetime:
        return datetime.strptime(self.window_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    @property
    def t_end(self) -> datetime:
        dt = datetime.strptime(self.window_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.replace(hour=23, minute=59, second=59)

    def date_range_label(self) -> str:
        t0 = self.t_start
        t1 = datetime.strptime(self.window_end, "%Y-%m-%d")
        return f"{t0.strftime('%b')}–{t1.strftime('%b %Y')}"

    def phase_for_time(self, t: datetime) -> Optional[str]:
        for name, start, end in self.phases:
            t0 = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            t1 = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if t0 <= t <= t1:
                return name
        return None


# ── Detection / sampling parameters ──────────────────────────────────────────

# 5-minute sampling for separation time history
SEP_STEP_MINUTES = 5
# Refinement threshold: switch to 10-second steps when separation < 1 km
REFINE_THRESHOLD_KM = 1.0
REFINE_STEP_SECONDS = 10

# Maneuver detection: adaptive MAD threshold multiplier
SIGMA_MULTIPLIER = 5.0
MAX_GAP_DAYS = 10.0        # skip pairs with epoch gap > this

# ── Ingest helpers (re-implemented to avoid circular imports) ─────────────────

_GP_COLS = [
    "GP_ID", "NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_ID", "EPOCH",
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH",
    "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
    "OBJECT_TYPE", "RCS_SIZE", "COUNTRY_CODE", "LAUNCH_DATE", "SITE",
    "DECAY_DATE", "FILE", "GP_ID",
    "TLE_LINE0", "TLE_LINE1", "TLE_LINE2",
]

_FLOAT_COLS = {
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS",
}
_INT_COLS = {
    "GP_ID", "NORAD_CAT_ID", "EPHEMERIS_TYPE", "ELEMENT_SET_NO",
    "REV_AT_EPOCH", "FILE", "APOGEE", "PERIGEE",
}


def _coerce(val, col):
    if val is None or val == "":
        return None
    if col in _FLOAT_COLS:
        return float(val)
    if col in _INT_COLS:
        return int(val)
    return val


def _build_gp_row(rec):
    return tuple(_coerce(rec.get(col), col) for col in _GP_COLS)


def _gp_col_names():
    cols = list(_GP_COLS)
    cols[cols.index("GP_ID", 1)] = "GP_ID_ORIG"
    return ", ".join(cols)


def _gp_placeholders():
    return ", ".join(["?"] * len(_GP_COLS))


# ── DB connection ─────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


# ── Step 1: Backfill gp_history ───────────────────────────────────────────────

def _check_existing(norad_id: int, start: str, end: str) -> int:
    """Count gp_history rows for an object in the window."""
    con = _connect()
    count = con.execute(
        "SELECT COUNT(*) FROM gp_history WHERE NORAD_CAT_ID = ? AND EPOCH >= ? AND EPOCH <= ?",
        (norad_id, start, end + "T23:59:59"),
    ).fetchone()[0]
    con.close()
    return count


def backfill_if_needed(norad_id: int, label: str, cfg: MissionConfig):
    """Trigger Space-Track backfill only if gp_history is empty for this window."""
    existing = _check_existing(norad_id, cfg.window_start, cfg.window_end)
    if existing > 0:
        print(f"  {label} ({norad_id}): {existing} TLEs already in DB — skipping backfill.")
        return

    print(f"  Backfilling {label} ({norad_id}): {cfg.window_start} to {cfg.window_end} ...")
    st = SpaceTrackClient(
        identity=SPACETRACK_IDENTITY,
        password=SPACETRACK_PASSWORD,
    )
    import json
    raw = st.gp_history(
        norad_cat_id=norad_id,
        epoch=op.inclusive_range(cfg.window_start, cfg.window_end),
        orderby="epoch asc",
        format="json",
    )
    records = json.loads(raw) if isinstance(raw, str) else (raw or [])
    print(f"  Received {len(records)} records from Space-Track.")

    if not records:
        print(f"  WARNING: No records returned for {label} ({norad_id}).")
        return

    rows = [_build_gp_row(r) for r in records]
    sql = (
        f"INSERT OR IGNORE INTO gp_history ({_gp_col_names()}) "
        f"VALUES ({_gp_placeholders()})"
    )
    con = _connect()
    BATCH = 5000
    inserted = 0
    for i in range(0, len(rows), BATCH):
        con.executemany(sql, rows[i : i + BATCH])
        inserted += con.execute("SELECT changes()").fetchone()[0]
    con.commit()
    con.close()
    print(f"  Inserted {inserted} new records ({len(rows) - inserted} duplicates skipped).")


# ── Step 2: Load historical TLEs ─────────────────────────────────────────────

def load_tle_history(norad_id: int, cfg: MissionConfig) -> pd.DataFrame:
    """
    Load gp_history for one object over the mission window.
    Returns DataFrame sorted by EPOCH with deduplication on exact epoch.
    """
    con = _connect()
    rows = con.execute(
        """
        SELECT NORAD_CAT_ID, EPOCH, TLE_LINE1, TLE_LINE2,
               SEMIMAJOR_AXIS, APOAPSIS, PERIAPSIS, BSTAR
        FROM gp_history
        WHERE NORAD_CAT_ID = ?
          AND EPOCH >= ?
          AND EPOCH <= ?
        ORDER BY EPOCH ASC
        """,
        (norad_id, cfg.window_start, cfg.window_end + "T23:59:59"),
    ).fetchall()
    con.close()

    cols = ["NORAD_CAT_ID", "EPOCH", "TLE_LINE1", "TLE_LINE2",
            "SEMIMAJOR_AXIS", "APOAPSIS", "PERIAPSIS", "BSTAR"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["EPOCH"] = pd.to_datetime(df["EPOCH"])
    # Deduplicate on EPOCH — keep last (most recently inserted)
    n_before = len(df)
    df = df.drop_duplicates(subset=["EPOCH"], keep="last").reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Deduplicated {n_before - len(df)} duplicate epochs "
              f"({n_before} → {len(df)} records)")
    return df


# ── Step 3 & secondary: Maneuver detection ───────────────────────────────────

def _sgp4_state(tle1: str, tle2: str, dt_sec: float) -> Optional[np.ndarray]:
    """
    Propagate TLE by dt_sec seconds using SGP4.
    Returns 6-element state [x,y,z km, vx,vy,vz km/s] in TEME, or None on error.
    """
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
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


def detect_maneuvers(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Detect maneuver signatures in a TLE history DataFrame.

    For each consecutive pair (i, i+1):
      - Propagate TLE_i forward to epoch_{i+1}
      - Compare predicted vs actual velocity
      - Flag as maneuver if residual exceeds adaptive MAD threshold

    Returns DataFrame with columns:
        epoch_after, vel_residual_ms, delta_sma, threshold, is_maneuver
    """
    results = []
    n_skipped_gap = 0
    n_skipped_sgp4 = 0

    for i in range(len(df) - 1):
        row_i  = df.iloc[i]
        row_j  = df.iloc[i + 1]

        t_i = row_i["EPOCH"]
        t_j = row_j["EPOCH"]
        gap_days = (t_j - t_i).total_seconds() / 86400.0

        if gap_days > MAX_GAP_DAYS or gap_days <= 0:
            n_skipped_gap += 1
            continue

        dt_sec = gap_days * 86400.0

        # Predicted state at t_j (propagated from TLE_i)
        tle1_i = row_i["TLE_LINE1"]
        tle2_i = row_i["TLE_LINE2"]
        if not isinstance(tle1_i, str) or not isinstance(tle2_i, str):
            n_skipped_sgp4 += 1
            continue
        pred = _sgp4_state(tle1_i, tle2_i, dt_sec)
        if pred is None:
            n_skipped_sgp4 += 1
            continue

        # Actual state at t_j (zero-propagation of TLE_{i+1})
        tle1_j = row_j["TLE_LINE1"]
        tle2_j = row_j["TLE_LINE2"]
        if not isinstance(tle1_j, str) or not isinstance(tle2_j, str):
            n_skipped_sgp4 += 1
            continue
        actual = _sgp4_state(tle1_j, tle2_j, 0.0)
        if actual is None:
            n_skipped_sgp4 += 1
            continue

        vel_res_ms = float(np.linalg.norm(pred[3:6] - actual[3:6]) * 1000.0)

        sma_i = row_i["SEMIMAJOR_AXIS"]
        sma_j = row_j["SEMIMAJOR_AXIS"]
        delta_sma = float(sma_j - sma_i) if (sma_i and sma_j) else float("nan")

        results.append({
            "epoch_after":      t_j,
            "vel_residual_ms":  vel_res_ms,
            "delta_sma":        delta_sma,
        })

    if n_skipped_gap:
        print(f"  [{label}] Skipped {n_skipped_gap} pairs with gap > {MAX_GAP_DAYS} days")
    if n_skipped_sgp4:
        print(f"  [{label}] Skipped {n_skipped_sgp4} pairs due to SGP4 errors")

    if not results:
        return pd.DataFrame(columns=["epoch_after", "vel_residual_ms", "delta_sma",
                                     "threshold", "is_maneuver"])

    res_df = pd.DataFrame(results)

    # Adaptive MAD threshold
    vr = res_df["vel_residual_ms"].values
    finite = np.isfinite(vr)
    median_vr = float(np.median(vr[finite]))
    mad_vr    = float(np.median(np.abs(vr[finite] - median_vr)))
    threshold = median_vr + SIGMA_MULTIPLIER * mad_vr

    res_df["threshold"]   = threshold
    res_df["is_maneuver"] = res_df["vel_residual_ms"] > threshold

    return res_df


# ── Step 4: Separation distance time history ─────────────────────────────────

def _most_recent_tle(df: pd.DataFrame, t: datetime):
    """Return (tle1, tle2) from the most recent epoch in df whose EPOCH <= t."""
    mask = df["EPOCH"] <= t
    if not mask.any():
        return None, None
    row = df[mask].iloc[-1]
    return row["TLE_LINE1"], row["TLE_LINE2"]


def _propagate_to(tle1: str, tle2: str, t: datetime):
    """Propagate TLE to datetime t; return (r_km, v_km_s) or (None, None)."""
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                      t.second + t.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            return None, None
        return r, v
    except Exception:
        return None, None


def compute_separation(df_chaser: pd.DataFrame, df_target: pd.DataFrame,
                       cfg: MissionConfig):
    """
    Build separation time history at SEP_STEP_MINUTES intervals.
    Refines to 10-second resolution in neighbourhood of REFINE_THRESHOLD_KM.

    Returns DataFrame with columns: time, sep_km
    """
    t_start = cfg.t_start
    t_end   = cfg.t_end
    step    = timedelta(minutes=SEP_STEP_MINUTES)

    records = []
    t = t_start

    # Ensure EPOCH is timezone-aware for comparison
    if df_chaser["EPOCH"].dt.tz is None:
        df_chaser = df_chaser.copy()
        df_chaser["EPOCH"] = df_chaser["EPOCH"].dt.tz_localize("UTC")
    if df_target["EPOCH"].dt.tz is None:
        df_target = df_target.copy()
        df_target["EPOCH"] = df_target["EPOCH"].dt.tz_localize("UTC")

    total_steps = int((t_end - t_start) / step) + 1
    report_every = max(1, total_steps // 20)  # progress every 5%
    step_count = 0

    print(f"  Computing separation time history ({total_steps} coarse steps) ...")

    while t <= t_end:
        tle1_a, tle2_a = _most_recent_tle(df_chaser, t)
        tle1_h, tle2_h = _most_recent_tle(df_target, t)

        if tle1_a and tle1_h:
            r_a, _ = _propagate_to(tle1_a, tle2_a, t)
            r_h, _ = _propagate_to(tle1_h, tle2_h, t)
            if r_a and r_h:
                sep = math.sqrt(sum((a - b)**2 for a, b in zip(r_a, r_h)))
                records.append({"time": t, "sep_km": sep})

                # Refine if below threshold
                if sep < REFINE_THRESHOLD_KM:
                    fine_step = timedelta(seconds=REFINE_STEP_SECONDS)
                    t_back = t - step + fine_step
                    while t_back < t:
                        tle1_a2, tle2_a2 = _most_recent_tle(df_chaser, t_back)
                        tle1_h2, tle2_h2 = _most_recent_tle(df_target, t_back)
                        if tle1_a2 and tle1_h2:
                            r_a2, _ = _propagate_to(tle1_a2, tle2_a2, t_back)
                            r_h2, _ = _propagate_to(tle1_h2, tle2_h2, t_back)
                            if r_a2 and r_h2:
                                s2 = math.sqrt(sum((a - b)**2 for a, b in zip(r_a2, r_h2)))
                                records.append({"time": t_back, "sep_km": s2})
                        t_back += fine_step

        t += step
        step_count += 1
        if step_count % report_every == 0:
            pct = 100 * step_count / total_steps
            print(f"    ... {pct:.0f}% ({t.strftime('%Y-%m-%d')})")

    sep_df = pd.DataFrame(records).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return sep_df


# ── Phase shading helper ─────────────────────────────────────────────────────

# Colours cycling for phase bands
_PHASE_COLOURS = ["#dcf7d7", "#d7eef7"]


def shade_phases(ax, phases, label_y=0.97):
    """
    Draw alternating grey/white background bands for mission phases.
    Places phase labels at the top of the axes.
    """
    for i, (name, start, end) in enumerate(phases):
        t0 = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        t1 = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        colour = _PHASE_COLOURS[i % 2]
        ax.axvspan(t0, t1, facecolor=colour, alpha=0.6, zorder=0)
        mid = t0 + (t1 - t0) / 2
        ax.text(
            mid, label_y, name,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=6.5, color="#555555",
            rotation=0, clip_on=True,
        )


# ── Figure 1: RPO Overview ────────────────────────────────────────────────────

def plot_rpo_overview(sep_df, df_chaser, df_target, mnv_chaser, tca_time, tca_sep,
                      phase_min_seps, out_path, cfg: MissionConfig):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    ax_sep, ax_alt, ax_rug = axes

    t_start = cfg.t_start
    t_end   = cfg.t_end

    # ── Panel A: Separation distance ─────────────────────────────────────────
    shade_phases(ax_sep, cfg.phases)
    ax_sep.semilogy(sep_df["time"], sep_df["sep_km"], color="steelblue", lw=0.8,
                    label="Separation (km)", zorder=2)

    # TCA marker
    if tca_time is not None:
        ax_sep.axvline(tca_time, color="red", ls="--", lw=1.2, zorder=3)
        ax_sep.annotate(
            f"TCA: {tca_time.strftime('%Y-%m-%d %H:%M')} UTC\nD = {tca_sep:.2f} km",
            xy=(tca_time, tca_sep),
            xytext=(20, 20), textcoords="offset points",
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    # Per-phase minimum annotations
    for phase_name, min_t, min_d in phase_min_seps:
        ax_sep.annotate(
            f"{phase_name[:1]}: {min_d:.1f} km",
            xy=(min_t, min_d),
            xytext=(0, -18), textcoords="offset points",
            ha="center", fontsize=7, color="grey",
        )

    # Maneuver epoch lines
    if mnv_chaser is not None and not mnv_chaser.empty:
        mnv_detected = mnv_chaser[mnv_chaser["is_maneuver"]]
        for _, mrow in mnv_detected.iterrows():
            ax_sep.axvline(mrow["epoch_after"], color="darkorange", ls="--",
                           lw=0.6, alpha=0.7, zorder=2)

    ax_sep.set_ylabel("Separation (km)", fontsize=9)
    ax_sep.set_title(f"{cfg.title} RPO Overview ({cfg.date_range_label()})", fontsize=12)
    ax_sep.legend(loc="upper right", fontsize=8)
    ax_sep.set_xlim(t_start, t_end)
    ax_sep.grid(True, which="both", ls=":", alpha=0.4, zorder=1)

    # ── Panel B: Altitude ─────────────────────────────────────────────────────
    shade_phases(ax_alt, cfg.phases)

    if not df_chaser.empty and "APOAPSIS" in df_chaser.columns:
        alt_a = (df_chaser["APOAPSIS"] + df_chaser["PERIAPSIS"]) / 2
        ax_alt.plot(df_chaser["EPOCH"], alt_a, color="steelblue", lw=0.9,
                    marker=".", ms=2, label=f"{cfg.chaser_name} mean alt", zorder=2)
    if not df_target.empty and "APOAPSIS" in df_target.columns:
        alt_h = (df_target["APOAPSIS"] + df_target["PERIAPSIS"]) / 2
        ax_alt.plot(df_target["EPOCH"], alt_h, color="tomato", lw=0.9,
                    marker=".", ms=2, label=f"{cfg.target_name} mean alt", zorder=2)

    ax_alt.set_ylabel("Mean Altitude (km)", fontsize=9)
    ax_alt.legend(loc="upper right", fontsize=8)
    ax_alt.grid(True, ls=":", alpha=0.4, zorder=1)

    # ── Panel C: TLE cadence rug plot ─────────────────────────────────────────
    shade_phases(ax_rug, cfg.phases)

    if not df_chaser.empty:
        epochs_a = pd.to_datetime(df_chaser["EPOCH"])
        ax_rug.vlines(epochs_a, ymin=0.55, ymax=0.95,
                      linewidths=0.6, colors="steelblue",
                      label=f"{cfg.chaser_name} TLE epochs")
    if not df_target.empty:
        epochs_h = pd.to_datetime(df_target["EPOCH"])
        ax_rug.vlines(epochs_h, ymin=0.05, ymax=0.45,
                      linewidths=0.6, colors="tomato",
                      label=f"{cfg.target_name} TLE epochs")
    ax_rug.set_ylim(0, 1)
    ax_rug.set_yticks([0.25, 0.75])
    ax_rug.set_yticklabels([cfg.target_name, cfg.chaser_name], fontsize=8)
    ax_rug.set_ylabel("TLE Cadence", fontsize=9)
    ax_rug.legend(loc="upper right", fontsize=8)
    ax_rug.grid(True, axis="x", ls=":", alpha=0.4, zorder=1)

    # Shared x-axis formatting (auto-adapts when zooming)
    locator = mdates.AutoDateLocator()
    ax_rug.xaxis.set_major_locator(locator)
    ax_rug.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax_rug.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.show()


# ── Figure 2: Maneuver Signatures ────────────────────────────────────────────

def plot_maneuver_signatures(mnv_chaser, mnv_target, df_chaser, out_path,
                             cfg: MissionConfig):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    ax_vr_a, ax_vr_h, ax_sma = axes

    t_start = cfg.t_start
    t_end   = cfg.t_end

    # ── Panel A: Chaser velocity residual ────────────────────────────────────
    shade_phases(ax_vr_a, cfg.phases)
    if not mnv_chaser.empty:
        normal   = mnv_chaser[~mnv_chaser["is_maneuver"]]
        detected = mnv_chaser[ mnv_chaser["is_maneuver"]]
        ax_vr_a.scatter(normal["epoch_after"],   normal["vel_residual_ms"],
                        color="steelblue", s=12, alpha=0.7, zorder=2, label="Normal")
        ax_vr_a.scatter(detected["epoch_after"], detected["vel_residual_ms"],
                        color="red", s=25, zorder=3, label="Detected maneuver")
        if "threshold" in mnv_chaser.columns:
            threshold = mnv_chaser["threshold"].iloc[0]
            ax_vr_a.axhline(threshold, color="darkorange", ls="--", lw=1.2,
                            label=f"MAD threshold ({threshold:.2f} m/s)")
    ax_vr_a.set_ylabel("Velocity Residual (m/s)", fontsize=9)
    ax_vr_a.set_title(f"{cfg.title} Maneuver Signatures", fontsize=12)
    ax_vr_a.legend(loc="upper right", fontsize=8)
    ax_vr_a.set_xlim(t_start, t_end)
    ax_vr_a.grid(True, ls=":", alpha=0.4, zorder=1)

    # ── Panel B: Target velocity residual (noise floor) ──────────────────────
    shade_phases(ax_vr_h, cfg.phases)
    if not mnv_target.empty:
        ax_vr_h.scatter(mnv_target["epoch_after"], mnv_target["vel_residual_ms"],
                        color="grey", s=10, alpha=0.7, zorder=2,
                        label=f"{cfg.target_name} (noise floor)")
        p95 = float(np.nanpercentile(mnv_target["vel_residual_ms"].values, 95))
        ax_vr_h.axhline(p95, color="dimgrey", ls=":", lw=1.0,
                        label=f"95th percentile ({p95:.2f} m/s)")
    ax_vr_h.set_ylabel("Velocity Residual (m/s)", fontsize=9)
    ax_vr_h.legend(loc="upper right", fontsize=8)
    ax_vr_h.grid(True, ls=":", alpha=0.4, zorder=1)

    # Match y-axis scale to Panel A for direct comparison
    if not mnv_chaser.empty and not mnv_target.empty:
        ymax = max(
            mnv_chaser["vel_residual_ms"].max() if not mnv_chaser.empty else 0,
            mnv_target["vel_residual_ms"].max()  if not mnv_target.empty else 0,
        )
        ax_vr_a.set_ylim(bottom=0, top=ymax * 1.1)
        ax_vr_h.set_ylim(bottom=0, top=ymax * 1.1)

    # ── Panel C: Chaser ΔSMA ─────────────────────────────────────────────────
    shade_phases(ax_sma, cfg.phases)
    if not mnv_chaser.empty and "delta_sma" in mnv_chaser.columns:
        dsma = mnv_chaser["delta_sma"].values
        finite_mask = np.isfinite(dsma)
        times_sma = mnv_chaser["epoch_after"].values[finite_mask]
        dsma_finite = dsma[finite_mask]
        colours = ["green" if d >= 0 else "red" for d in dsma_finite]
        ax_sma.bar(times_sma, dsma_finite, color=colours, width=1.5, alpha=0.7, zorder=2)
        ax_sma.axhline(0, color="black", lw=0.6)
    ax_sma.set_ylabel("ΔSMA (km)", fontsize=9)
    ax_sma.legend(
        handles=[
            mpatches.Patch(color="green", alpha=0.7, label="Orbit raise"),
            mpatches.Patch(color="red",   alpha=0.7, label="Orbit lower"),
        ],
        loc="upper right", fontsize=8,
    )
    ax_sma.grid(True, ls=":", alpha=0.4, zorder=1)

    locator = mdates.AutoDateLocator()
    ax_sma.xaxis.set_major_locator(locator)
    ax_sma.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.setp(ax_sma.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.show()


# ── Step 6: Console Summary ───────────────────────────────────────────────────

def print_summary(df_chaser, df_target, mnv_chaser, mnv_target,
                  sep_df, tca_time, tca_sep, phase_min_seps,
                  cfg: MissionConfig):
    print()
    print("=" * 70)
    print(f"  {cfg.title} RPO ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Mission window: {cfg.window_start}  →  {cfg.window_end}")
    print()
    print(f"  TLE counts over mission window:")
    print(f"    {cfg.chaser_name}  ({cfg.chaser_norad}): {len(df_chaser):>5} TLEs")
    print(f"    {cfg.target_name} ({cfg.target_norad}): {len(df_target):>5} TLEs")
    print()

    if not mnv_chaser.empty:
        n_det = int(mnv_chaser["is_maneuver"].sum())
        print(f"  {cfg.chaser_name} detected maneuver epochs: {n_det}")
        if n_det > 0:
            det = mnv_chaser[mnv_chaser["is_maneuver"]]
            phase_counts: dict = {}
            for _, row in det.iterrows():
                t = row["epoch_after"]
                if hasattr(t, "to_pydatetime"):
                    t = t.to_pydatetime()
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                phase = cfg.phase_for_time(t) or "Gap/unphased"
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            for ph, cnt in sorted(phase_counts.items()):
                print(f"      {ph}: {cnt}")
    print()

    print("  Per-phase minimum separation:")
    for phase_name, min_t, min_d in phase_min_seps:
        print(f"    {phase_name:<30}  {min_d:>8.2f} km  "
              f"  ({min_t.strftime('%Y-%m-%d %H:%M') if min_t else 'N/A'})")
    print()

    if tca_time is not None:
        print(f"  Global TCA: {tca_time.strftime('%Y-%m-%d %H:%M')} UTC"
              f"  —  separation: {tca_sep:.2f} km")
    print()

    if not mnv_target.empty:
        p95 = float(np.nanpercentile(mnv_target["vel_residual_ms"].values, 95))
        print(f"  {cfg.target_name} velocity residual 95th percentile: {p95:.2f} m/s  (noise floor)")
    print("=" * 70)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: MissionConfig):
    analysis_dir = Path(__file__).parent

    # ── Step 1: Backfill gp_history ──────────────────────────────────────────
    print("\n[Step 1] Checking / backfilling gp_history ...")
    backfill_if_needed(cfg.chaser_norad, cfg.chaser_name, cfg)
    backfill_if_needed(cfg.target_norad, cfg.target_name, cfg)

    # ── Step 2: Load TLE histories ───────────────────────────────────────────
    print("\n[Step 2] Loading TLE histories from DB ...")
    df_chaser = load_tle_history(cfg.chaser_norad, cfg)
    df_target = load_tle_history(cfg.target_norad, cfg)

    print(f"  {cfg.chaser_name}  ({cfg.chaser_norad}): {len(df_chaser)} TLEs loaded")
    print(f"  {cfg.target_name} ({cfg.target_norad}): {len(df_target)} TLEs loaded")

    if df_chaser.empty:
        print(f"  ERROR: No {cfg.chaser_name} TLEs found. Check backfill and DB.")
        return
    if df_target.empty:
        print(f"  ERROR: No {cfg.target_name} TLEs found. Check backfill and DB.")
        return

    # ── Step 3: Maneuver detection ───────────────────────────────────────────
    print("\n[Step 3] Detecting maneuvers ...")
    print(f"  Processing {cfg.chaser_name} ...")
    mnv_chaser = detect_maneuvers(df_chaser, cfg.chaser_name)
    print(f"  Processing {cfg.target_name} (noise floor reference) ...")
    mnv_target = detect_maneuvers(df_target, cfg.target_name)

    if not mnv_chaser.empty:
        n_det = int(mnv_chaser["is_maneuver"].sum())
        thresh = mnv_chaser["threshold"].iloc[0] if "threshold" in mnv_chaser.columns else float("nan")
        print(f"  {cfg.chaser_name}: {n_det} maneuver events detected "
              f"(threshold = {thresh:.2f} m/s, "
              f"out of {len(mnv_chaser)} TLE pairs)")
    if not mnv_target.empty:
        p95 = float(np.nanpercentile(mnv_target["vel_residual_ms"].values, 95))
        print(f"  {cfg.target_name}: 95th-percentile vel residual = {p95:.2f} m/s (noise floor)")

    # ── Step 4: Separation time history ──────────────────────────────────────
    print("\n[Step 4] Computing separation time history ...")
    sep_df = compute_separation(df_chaser, df_target, cfg)

    # Global TCA
    tca_time = tca_sep = None
    if not sep_df.empty:
        idx_min = sep_df["sep_km"].idxmin()
        tca_time = sep_df.loc[idx_min, "time"]
        tca_sep  = float(sep_df.loc[idx_min, "sep_km"])
        if hasattr(tca_time, "to_pydatetime"):
            tca_time = tca_time.to_pydatetime()
        if tca_time.tzinfo is None:
            tca_time = tca_time.replace(tzinfo=timezone.utc)
        print(f"  Global TCA: {tca_time.strftime('%Y-%m-%d %H:%M')} UTC  —  {tca_sep:.2f} km")

    # Per-phase minimum separations
    phase_min_seps = []
    if not sep_df.empty:
        sep_times_raw = pd.to_datetime(sep_df["time"])
        if sep_times_raw.dt.tz is None:
            sep_times = sep_times_raw.dt.tz_localize("UTC")
        else:
            sep_times = sep_times_raw
        for phase_name, start, end in cfg.phases:
            t0 = pd.Timestamp(start, tz="UTC")
            t1 = pd.Timestamp(end,   tz="UTC")
            mask = (sep_times >= t0) & (sep_times <= t1)
            if mask.any():
                sub = sep_df[mask.values]
                idx = sub["sep_km"].idxmin()
                min_t = sub.loc[idx, "time"]
                min_d = float(sub.loc[idx, "sep_km"])
                if hasattr(min_t, "to_pydatetime"):
                    min_t = min_t.to_pydatetime()
                if min_t.tzinfo is None:
                    min_t = min_t.replace(tzinfo=timezone.utc)
                phase_min_seps.append((phase_name, min_t, min_d))
            else:
                phase_min_seps.append((phase_name, None, float("inf")))

    # ── Step 5: Plots ─────────────────────────────────────────────────────────
    print("\n[Step 5] Generating plots ...")

    plots_dir = analysis_dir.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_overview  = plots_dir / "rpo_overview.png"
    out_maneuvers = plots_dir / "rpo_maneuvers.png"

    plot_rpo_overview(
        sep_df, df_chaser, df_target, mnv_chaser,
        tca_time, tca_sep, phase_min_seps, out_overview, cfg,
    )
    plot_maneuver_signatures(mnv_chaser, mnv_target, df_chaser, out_maneuvers, cfg)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print_summary(df_chaser, df_target, mnv_chaser, mnv_target,
                  sep_df, tca_time, tca_sep, phase_min_seps, cfg)


# ── Mission definitions ──────────────────────────────────────────────────────

ADRASJ_MISSION = MissionConfig(
    chaser_name  = "ADRAS-J",
    chaser_norad = 58992,
    target_name  = "H-IIA R/B",
    target_norad = 33500,
    window_start = "2024-02-18",
    window_end   = "2024-11-30",
    phases = [
        ("1 — Launch, IOT, Rendezvous",             "2024-02-18", "2024-04-09"),
        ("2 — Far Proximity (>100m)",               "2024-04-09", "2024-04-17"),
        ("3 — Near Proximity",                      "2024-04-17", "2024-06-19"),
        ("4 — Fly-around",                          "2024-06-19", "2024-07-17"),
        ("5 — Final Approach",                      "2024-08-13", "2024-11-30"),
    ],
)

LDPE3A_SJ23_MISSION = MissionConfig(
    chaser_name  = "LDPE-3A",
    chaser_norad = 55264,
    target_name  = "SJ-23",
    target_norad = 55131,
    window_start = "2024-09-15",
    window_end   = "2024-11-30",
    phases = [
        ("1 — Pre-approach",           "2024-09-15", "2024-10-28"),
        ("2 — Maneuver & Convergence", "2024-10-29", "2024-11-03"),
        ("3 — Close Approach",         "2024-11-04", "2024-11-05"),
        ("4 — Trailing",               "2024-11-06", "2024-11-09"),
        ("5 — Station-keeping",        "2024-11-09", "2024-11-30"),
    ],
)

if __name__ == "__main__":
    main(LDPE3A_SJ23_MISSION)
