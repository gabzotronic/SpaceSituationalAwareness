"""
maneuver_detection.py — Detect orbital maneuvers from GP element history.

Approach:
  1. Pull historical GP records for a satellite from gp_history.
  2. For each consecutive epoch pair, propagate forward using a simple
     Kepler model and compare against the next actual state.
  3. Flag pairs where ΔSMA or ΔEccentricity exceed thresholds.
  4. Cross-check against the geomagnetic Kp index (GFZ Potsdam) to
     distinguish true maneuvers from space-weather-driven element drift.

Data availability note:
  gp_history is populated by `python ingest.py --update` cycles. The
  analysis depth depends on how long updates have been running — unlike
  CelesTrak which serves years of history, this table only covers epochs
  ingested since the DB was initialised.

Usage:
    python analysis/maneuver_detection.py
    python analysis/maneuver_detection.py --norad 58316
    python analysis/maneuver_detection.py --norad 66748 --delta-sma 0.5 --kp 4.0
    python analysis/maneuver_detection.py --no-plot

Dependencies: numpy, pandas, matplotlib, requests
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
from config import DB_PATH


# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_NORAD        = 66748
DEFAULT_DELTA_SMA    = 1.0    # km
DEFAULT_DELTA_ECC    = 5e-5
DEFAULT_KP_THRESHOLD = 5.0


# ── 1. DB query ───────────────────────────────────────────────────────────────

def get_gp_history(norad_id: int) -> pd.DataFrame:
    """
    Pull historical GP element sets for one object from gp_history, oldest first.

    Returns a DataFrame with columns:
        EPOCH, SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION,
        RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY
    EPOCH is parsed to datetime.
    """
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        """
        SELECT EPOCH, SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION,
               RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY
        FROM gp_history
        WHERE NORAD_CAT_ID = ?
        ORDER BY EPOCH ASC
        """,
        (norad_id,),
    ).fetchall()
    con.close()

    cols = [
        "EPOCH", "SEMIMAJOR_AXIS", "ECCENTRICITY", "INCLINATION",
        "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["EPOCH"] = pd.to_datetime(df["EPOCH"])
    return df


def get_object_name(norad_id: int) -> Optional[str]:
    """Return OBJECT_NAME from gp table, or None if not found."""
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT OBJECT_NAME FROM gp WHERE NORAD_CAT_ID = ?", (norad_id,)
    ).fetchone()
    con.close()
    return row[0] if row else None


# ── 2. Kepler propagator ──────────────────────────────────────────────────────

def kepler_position(
    sma_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    mean_anom_deg: float,
) -> np.ndarray:
    """Compute ECI position vector from Keplerian elements."""
    inc  = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    M    = np.radians(mean_anom_deg)

    # Solve Kepler's equation for eccentric anomaly
    E = M
    for _ in range(100):
        E = M + ecc * np.sin(E)

    nu = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2),
    )
    r = sma_km * (1 - ecc * np.cos(E))

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)
    cos_inc,  sin_inc  = np.cos(inc),  np.sin(inc)

    x = (cos_raan * cos_argp - sin_raan * sin_argp * cos_inc) * x_orb + \
        (-cos_raan * sin_argp - sin_raan * cos_argp * cos_inc) * y_orb
    y = (sin_raan * cos_argp + cos_raan * sin_argp * cos_inc) * x_orb + \
        (-sin_raan * sin_argp + cos_raan * cos_argp * cos_inc) * y_orb
    z = (sin_argp * sin_inc) * x_orb + (cos_argp * sin_inc) * y_orb

    return np.array([x, y, z])


def propagate_kepler(row: pd.Series, dt_sec: float) -> np.ndarray:
    """Propagate a single GP row forward by dt_sec seconds using two-body Kepler."""
    mu      = 398600.4418  # km^3/s^2
    n_rad_s = np.sqrt(mu / row["SEMIMAJOR_AXIS"] ** 3)
    M_prop  = np.degrees(np.radians(row["MEAN_ANOMALY"]) + n_rad_s * dt_sec) % 360

    return kepler_position(
        row["SEMIMAJOR_AXIS"], row["ECCENTRICITY"], row["INCLINATION"],
        row["RA_OF_ASC_NODE"], row["ARG_OF_PERICENTER"], M_prop,
    )


# ── 3. Space weather ──────────────────────────────────────────────────────────

def _parse_kp_text(text: str, file_type: str = "nowcast") -> pd.DataFrame:
    """
    Parse raw GFZ Potsdam Kp text into a DataFrame.

    'nowcast' — Kp_ap_nowcast.txt: one row per 3hr interval, Kp at index 7.
    'daily'   — Kp_ap_Ap_SN_F107_since_1932.txt: one row per day, Kp at indices 7-14.
    """
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
    # Fallback: nearest interval if epoch is outside Kp data range
    idx = (kp_df["datetime"] - epoch).abs().argmin()
    return kp_df.iloc[idx]["kp"]


# ── 4. Maneuver analysis ──────────────────────────────────────────────────────

def analyse_maneuvers(
    norad_id: int,
    data_dir: Path,
    delta_sma_threshold: float = DEFAULT_DELTA_SMA,
    delta_ecc_threshold: float = DEFAULT_DELTA_ECC,
    kp_threshold: float = DEFAULT_KP_THRESHOLD,
) -> pd.DataFrame:
    """
    Detect maneuvers for a single object using its GP element history.

    Compares consecutive epoch pairs from gp_history, flags significant
    changes in SMA or eccentricity, and cross-checks against the Kp
    geomagnetic index to distinguish maneuvers from space weather effects.

    Args:
        norad_id:             NORAD_CAT_ID of the object
        data_dir:             Directory for Kp index cache
        delta_sma_threshold:  SMA change to flag a maneuver (km, default 1.0)
        delta_ecc_threshold:  Eccentricity change to flag a maneuver (default 5e-5)
        kp_threshold:         Kp above which space weather is considered disturbed

    Returns:
        DataFrame with epoch pairs, deltas, Kp values, and maneuver flags.
        Empty DataFrame if insufficient history exists in gp_history.
    """
    df = get_gp_history(norad_id)

    if df.empty:
        print(f"No gp_history records found for NORAD {norad_id}.")
        print("Run `python ingest.py --update` to accumulate historical element sets.")
        return pd.DataFrame()

    print(f"Loaded {len(df)} GP history records "
          f"({df['EPOCH'].min().date()} -> {df['EPOCH'].max().date()})")

    if df["MEAN_ANOMALY"].isna().all():
        print("MEAN_ANOMALY not available — approximating as 0.0 (epoch ~= perigee pass)")
        df["MEAN_ANOMALY"] = 0.0

    missing = [
        c for c in ["EPOCH", "SEMIMAJOR_AXIS", "ECCENTRICITY", "INCLINATION",
                    "RA_OF_ASC_NODE", "ARG_OF_PERICENTER"]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in gp_history: {missing}")

    # ── Compute epoch pair deltas ─────────────────────────────────────────────
    results = []
    for i in range(len(df) - 1):
        row_now  = df.iloc[i]
        row_next = df.iloc[i + 1]

        gap_sec  = (row_next["EPOCH"] - row_now["EPOCH"]).total_seconds()
        gap_days = gap_sec / 86400

        if gap_days > 10:  # skip data gaps
            continue

        pos_predicted = propagate_kepler(row_now, gap_sec)
        pos_actual    = kepler_position(
            row_next["SEMIMAJOR_AXIS"], row_next["ECCENTRICITY"],
            row_next["INCLINATION"],    row_next["RA_OF_ASC_NODE"],
            row_next["ARG_OF_PERICENTER"], row_next["MEAN_ANOMALY"],
        )

        results.append({
            "epoch_from": row_now["EPOCH"],
            "epoch_to":   row_next["EPOCH"],
            "gap_days":   round(gap_days, 3),
            "error_km":   round(float(np.linalg.norm(pos_predicted - pos_actual)), 3),
            "delta_sma":  round(row_next["SEMIMAJOR_AXIS"] - row_now["SEMIMAJOR_AXIS"], 3),
            "delta_ecc":  round(row_next["ECCENTRICITY"]   - row_now["ECCENTRICITY"],   6),
            "delta_inc":  round(row_next["INCLINATION"]    - row_now["INCLINATION"],    4),
        })

    if not results:
        print("No consecutive epoch pairs found within 10-day gaps.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # ── Space weather Kp lookup ───────────────────────────────────────────────
    start = result_df["epoch_from"].min().strftime("%Y-%m-%d")
    end   = result_df["epoch_to"].max().strftime("%Y-%m-%d")
    kp_df = get_kp_index(start, end, data_dir)

    if kp_df.empty:
        print("WARNING: No Kp data for this time range. All maneuvers marked uncertain.")
        result_df["kp"] = float("nan")
    else:
        result_df["kp"] = result_df["epoch_from"].apply(
            lambda ep: _get_kp_for_epoch(ep, kp_df)
        )

    # ── Maneuver classification ───────────────────────────────────────────────
    result_df["bad_space_weather"]  = result_df["kp"] >= kp_threshold
    result_df["likely_maneuver"]    = (
        (result_df["delta_sma"].abs() > delta_sma_threshold) |
        (result_df["delta_ecc"].abs() > delta_ecc_threshold)
    )
    result_df["confirmed_maneuver"] = result_df["likely_maneuver"] & ~result_df["bad_space_weather"]
    result_df["uncertain_maneuver"] = result_df["likely_maneuver"] &  result_df["bad_space_weather"]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Maneuver Analysis: NORAD {norad_id} ──────────────────────────────────")
    print(f"  Epoch pairs analysed:      {len(result_df)}")
    print(f"  delta_sma threshold:       {delta_sma_threshold} km")
    print(f"  delta_ecc threshold:       {delta_ecc_threshold}")
    print(f"  Kp threshold:              {kp_threshold}")
    print(f"  Likely maneuvers:          {result_df['likely_maneuver'].sum()}")
    print(f"  Confirmed maneuvers:       {result_df['confirmed_maneuver'].sum()}")
    print(f"  Uncertain (bad weather):   {result_df['uncertain_maneuver'].sum()}")

    def _print_rows(rows: pd.DataFrame, label: str) -> None:
        for _, r in rows.iterrows():
            kp_str = f"{r['kp']:.1f}" if pd.notna(r["kp"]) else "N/A"
            print(f"  {r['epoch_from']}  ->  {r['epoch_to']}  "
                  f"(gap: {r['gap_days']:.3f} days)  "
                  f"dSMA: {r['delta_sma']:+.3f} km  "
                  f"dEcc: {r['delta_ecc']:+.2e}  "
                  f"Kp: {kp_str}  [{label}]")

    if result_df["confirmed_maneuver"].any():
        print("\n── Confirmed Maneuver Epochs ────────────────────────────────────────")
        _print_rows(result_df[result_df["confirmed_maneuver"]], "CONFIRMED")

    if result_df["uncertain_maneuver"].any():
        print("\n── Uncertain Epochs (possible maneuver, disturbed space weather) ─────")
        _print_rows(result_df[result_df["uncertain_maneuver"]], "UNCERTAIN")

    return result_df


# ── 5. Plotting ───────────────────────────────────────────────────────────────

def plot_maneuvers(
    result_df: pd.DataFrame,
    norad_id: int,
    object_name: Optional[str],
    delta_sma_threshold: float,
    delta_ecc_threshold: float,
    kp_threshold: float,
    plot_dir: Path,
) -> None:
    """4-panel maneuver detection plot: ΔSMA, ΔEcc, position error, Kp."""
    title_name = f"{object_name} (NORAD {norad_id})" if object_name else f"NORAD {norad_id}"

    confirmed = result_df[result_df["confirmed_maneuver"]]
    uncertain = result_df[result_df["uncertain_maneuver"]]

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

    # Panel 1: ΔSMA
    axes[0].plot(result_df["epoch_to"], result_df["delta_sma"],
                 marker="o", markersize=3, color="darkorange")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].axhline( delta_sma_threshold, color="red", linestyle=":", linewidth=1,
                     label=f"±{delta_sma_threshold} km")
    axes[0].axhline(-delta_sma_threshold, color="red", linestyle=":", linewidth=1)
    if not confirmed.empty:
        axes[0].scatter(confirmed["epoch_to"], confirmed["delta_sma"],
                        color="red", zorder=5, label="Confirmed maneuver")
    if not uncertain.empty:
        axes[0].scatter(uncertain["epoch_to"], uncertain["delta_sma"],
                        color="orange", zorder=5, marker="^", label="Uncertain (bad weather)")
    axes[0].set_ylabel("dSMA (km)")
    axes[0].set_title(f"Maneuver Detection — {title_name}")
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel 2: ΔEccentricity
    axes[1].plot(result_df["epoch_to"], result_df["delta_ecc"],
                 marker="o", markersize=3, color="green")
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].axhline( delta_ecc_threshold, color="red", linestyle=":", linewidth=1,
                     label=f"±{delta_ecc_threshold}")
    axes[1].axhline(-delta_ecc_threshold, color="red", linestyle=":", linewidth=1)
    if not confirmed.empty:
        axes[1].scatter(confirmed["epoch_to"], confirmed["delta_ecc"],
                        color="red", zorder=5, label="Confirmed maneuver")
    if not uncertain.empty:
        axes[1].scatter(uncertain["epoch_to"], uncertain["delta_ecc"],
                        color="orange", zorder=5, marker="^", label="Uncertain (bad weather)")
    axes[1].set_ylabel("dEccentricity")
    axes[1].legend(fontsize=8, loc="upper right")

    # Panel 3: Kepler prediction error
    axes[2].plot(result_df["epoch_to"], result_df["error_km"],
                 marker="o", markersize=3, color="steelblue")
    if not confirmed.empty:
        axes[2].scatter(confirmed["epoch_to"], confirmed["error_km"],
                        color="red", zorder=5, label="Confirmed maneuver")
    if not uncertain.empty:
        axes[2].scatter(uncertain["epoch_to"], uncertain["error_km"],
                        color="orange", zorder=5, marker="^", label="Uncertain (bad weather)")
    axes[2].set_ylabel("Kepler prediction error (km)")
    axes[2].legend(fontsize=8, loc="upper right")

    # Panel 4: Kp index
    if result_df["kp"].notna().any():
        axes[3].plot(result_df["epoch_to"], result_df["kp"],
                     marker="o", markersize=3, color="purple")
        axes[3].axhline(kp_threshold, color="red", linestyle="--", linewidth=1,
                        label=f"Kp threshold ({kp_threshold})")
        axes[3].fill_between(
            result_df["epoch_to"], result_df["kp"], kp_threshold,
            where=result_df["kp"] >= kp_threshold,
            color="red", alpha=0.2, label="Disturbed",
        )
        axes[3].legend(fontsize=8, loc="upper right")
    else:
        axes[3].text(0.5, 0.5, "Kp data unavailable",
                     transform=axes[3].transAxes, ha="center", va="center", color="gray")
    axes[3].set_ylabel("Kp")
    axes[3].set_xlabel("Epoch")

    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"maneuver_detection_{norad_id}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.show()
    # plt.close(fig)


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
        "--delta-sma", type=float, default=DEFAULT_DELTA_SMA,
        help=f"SMA change threshold in km (default: {DEFAULT_DELTA_SMA})",
    )
    parser.add_argument(
        "--delta-ecc", type=float, default=DEFAULT_DELTA_ECC,
        help=f"Eccentricity change threshold (default: {DEFAULT_DELTA_ECC})",
    )
    parser.add_argument(
        "--kp", type=float, default=DEFAULT_KP_THRESHOLD,
        help=f"Kp threshold for disturbed space weather (default: {DEFAULT_KP_THRESHOLD})",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation",
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
        delta_sma_threshold=args.delta_sma,
        delta_ecc_threshold=args.delta_ecc,
        kp_threshold=args.kp,
    )

    if result_df.empty:
        return

    print("\n" + result_df[[
        "epoch_from", "epoch_to", "delta_sma", "delta_ecc",
        "kp", "confirmed_maneuver", "uncertain_maneuver",
    ]].to_string(index=False))

    if not args.no_plot:
        plot_maneuvers(
            result_df,
            norad_id=args.norad,
            object_name=object_name,
            delta_sma_threshold=args.delta_sma,
            delta_ecc_threshold=args.delta_ecc,
            kp_threshold=args.kp,
            plot_dir=plot_dir,
        )


if __name__ == "__main__":
    main()
