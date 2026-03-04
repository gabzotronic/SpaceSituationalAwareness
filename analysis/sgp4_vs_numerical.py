"""
sgp4_vs_numerical.py -- Compare SGP4 (analytical) vs high-fidelity numerical
propagation accuracy for SSA conjunction assessment.

Propagates from one TLE epoch to the next using both methods and compares
against the "truth" state from the destination TLE.  Uses Teleos 2
(NORAD 56310) TLEs from 2026-03-03 as real-world test data.

Usage:
    conda run -n orbit python analysis/sgp4_vs_numerical.py
    conda run -n orbit python analysis/sgp4_vs_numerical.py --norad 25544
    conda run -n orbit python analysis/sgp4_vs_numerical.py --no-plot

Dependencies: orekit-jpype (with bundled JVM), spacetrack, numpy, matplotlib
"""

from __future__ import annotations

# Orekit must be initialised before any Java imports
import orekit_jpype
orekit_jpype.initVM()
from orekit_jpype.pyhelpers import setup_orekit_data
setup_orekit_data()

# Orekit Java imports
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import IERSConventions, Constants
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# Python imports
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPACETRACK_IDENTITY, SPACETRACK_PASSWORD

import spacetrack.operators as op
from spacetrack import SpaceTrackClient


# ── Constants ────────────────────────────────────────────────────────────────

EARTH_MU = Constants.WGS84_EARTH_MU          # m^3/s^2
EARTH_RE_M = Constants.WGS84_EARTH_EQUATORIAL_RADIUS  # m


# ── TLE retrieval ────────────────────────────────────────────────────────────

def _parse_epoch(epoch_str: str) -> Optional[datetime]:
    """Parse Space-Track epoch string to UTC datetime."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(epoch_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_tle_pairs(
    norad_id: int,
    dt_start: str,
    dt_end: str,
) -> List[Tuple[datetime, str, str]]:
    """
    Fetch gp_history TLEs from Space-Track for *norad_id* between
    *dt_start* and *dt_end* (ISO date strings).

    Returns deduplicated list of (epoch_dt, tle_line1, tle_line2),
    sorted by epoch.
    """
    st = SpaceTrackClient(SPACETRACK_IDENTITY, SPACETRACK_PASSWORD)
    raw = st.gp_history(
        norad_cat_id=norad_id,
        epoch=op.inclusive_range(dt_start, dt_end),
        orderby="epoch asc",
        format="json",
    )
    recs = json.loads(raw) if isinstance(raw, str) else (raw or [])

    # Deduplicate by epoch (keep first occurrence)
    seen_epochs = set()
    tles = []
    for r in recs:
        ep = _parse_epoch(r.get("EPOCH", ""))
        l1, l2 = r.get("TLE_LINE1"), r.get("TLE_LINE2")
        if ep is None or not l1 or not l2:
            continue
        epoch_key = ep.strftime("%Y-%m-%dT%H:%M:%S")
        if epoch_key in seen_epochs:
            continue
        seen_epochs.add(epoch_key)
        tles.append((ep, l1, l2))

    tles.sort(key=lambda t: t[0])
    return tles


# ── Orekit helpers ───────────────────────────────────────────────────────────

def _datetime_to_absolute(dt: datetime) -> AbsoluteDate:
    """Convert Python UTC datetime to Orekit AbsoluteDate."""
    utc = TimeScalesFactory.getUTC()
    return AbsoluteDate(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, float(dt.second + dt.microsecond / 1e6),
        utc,
    )


def _pv_to_numpy(pv) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (pos_m, vel_m_s) numpy arrays from Orekit PVCoordinates."""
    p = pv.getPosition()
    v = pv.getVelocity()
    return (
        np.array([p.getX(), p.getY(), p.getZ()]),
        np.array([v.getX(), v.getY(), v.getZ()]),
    )


def _build_numerical_propagator(
    initial_state: SpacecraftState,
    cross_section: float = 10.0,
    cd: float = 2.2,
    mass_kg: float = 500.0,
) -> NumericalPropagator:
    """
    Configure a high-fidelity NumericalPropagator with:
      - 70×70 gravity field
      - Harris-Priester atmospheric drag
      - Sun & Moon third-body perturbations
      - Relativistic corrections
    """
    integrator = DormandPrince853Integrator(0.001, 60.0, 1.0, 1e-6)
    prop = NumericalPropagator(integrator)

    state_with_mass = initial_state.withMass(float(mass_kg))
    prop.resetInitialState(state_with_mass)

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # Gravity 70×70
    gravity_field = GravityFieldFactory.getNormalizedProvider(70, 70)
    prop.addForceModel(
        HolmesFeatherstoneAttractionModel(itrf, gravity_field)
    )

    # Atmospheric drag — Harris-Priester
    sun = CelestialBodyFactory.getSun()
    earth_ellipsoid = OneAxisEllipsoid(
        EARTH_RE_M,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )
    atmosphere = HarrisPriester(sun, earth_ellipsoid)
    prop.addForceModel(DragForce(atmosphere, IsotropicDrag(cross_section, cd)))

    # Third-body: Sun + Moon
    prop.addForceModel(ThirdBodyAttraction(sun))
    prop.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))

    # Relativity
    try:
        from org.orekit.forces.gravity import Relativity
        prop.addForceModel(Relativity(EARTH_MU))
    except Exception:
        pass

    return prop


# ── RTN decomposition ───────────────────────────────────────────────────────

def _rtn_decompose(
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    delta_pos: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Decompose *delta_pos* into RTN components relative to the reference
    orbit defined by *pos_ref* / *vel_ref*.

    Returns (radial, along_track, cross_track) in the same units as input.
    """
    r_hat = pos_ref / np.linalg.norm(pos_ref)
    h = np.cross(pos_ref, vel_ref)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)

    return (
        float(np.dot(delta_pos, r_hat)),
        float(np.dot(delta_pos, t_hat)),
        float(np.dot(delta_pos, n_hat)),
    )


# ── Propagation loop ────────────────────────────────────────────────────────

def propagate_pair(
    line1_a: str,
    line2_a: str,
    line1_b: str,
    line2_b: str,
    step_s: float = 60.0,
) -> dict:
    """
    Propagate from TLE_A epoch to TLE_B epoch using both SGP4 and numerical,
    compare against TLE_B's state at its own epoch (truth).

    Returns dict with arrays of elapsed times, errors, RTN components, etc.
    """
    eme2000 = FramesFactory.getEME2000()

    # Parse TLEs
    tle_a = TLE(line1_a, line2_a)
    tle_b = TLE(line1_b, line2_b)

    epoch_a = tle_a.getDate()
    epoch_b = tle_b.getDate()
    duration_s = epoch_b.durationFrom(epoch_a)

    # ── Truth: TLE_B propagated at its own epoch ─────────────────────────
    sgp4_b = TLEPropagator.selectExtrapolator(tle_b)
    truth_state = sgp4_b.propagate(epoch_b)
    truth_pos, truth_vel = _pv_to_numpy(
        truth_state.getPVCoordinates(eme2000)
    )

    # ── SGP4 propagator from TLE_A ──────────────────────────────────────
    sgp4_a = TLEPropagator.selectExtrapolator(tle_a)

    # ── Numerical propagator from TLE_A's SGP4 state at epoch_a ─────────
    init_state_sgp4 = sgp4_a.propagate(epoch_a)
    # Convert to EME2000 for the numerical propagator
    init_orbit_eme = init_state_sgp4.getOrbit()
    init_sc_state = SpacecraftState(init_orbit_eme)

    num_prop = _build_numerical_propagator(init_sc_state)

    # ── Propagate at step_s intervals ────────────────────────────────────
    n_steps = int(duration_s / step_s) + 1
    elapsed = np.zeros(n_steps)
    sgp4_err_3d = np.zeros(n_steps)
    num_err_3d = np.zeros(n_steps)
    diverge_3d = np.zeros(n_steps)       # SGP4-vs-numerical divergence
    sgp4_rtn = np.zeros((n_steps, 3))
    num_rtn = np.zeros((n_steps, 3))
    diverge_rtn = np.zeros((n_steps, 3))

    for i in range(n_steps):
        t = min(i * step_s, duration_s)
        elapsed[i] = t
        target_date = epoch_a.shiftedBy(t)

        # SGP4
        sgp4_state = sgp4_a.propagate(target_date)
        sgp4_pos, sgp4_vel = _pv_to_numpy(
            sgp4_state.getPVCoordinates(eme2000)
        )

        # Numerical
        num_state = num_prop.propagate(target_date)
        num_pos, _ = _pv_to_numpy(num_state.getPVCoordinates(eme2000))

        # Errors vs truth (TLE_B at its own epoch)
        sgp4_delta = sgp4_pos - truth_pos
        num_delta = num_pos - truth_pos

        sgp4_err_3d[i] = np.linalg.norm(sgp4_delta)
        num_err_3d[i] = np.linalg.norm(num_delta)

        sgp4_rtn[i] = _rtn_decompose(truth_pos, truth_vel, sgp4_delta)
        num_rtn[i] = _rtn_decompose(truth_pos, truth_vel, num_delta)

        # SGP4-vs-numerical divergence (the key metric)
        div_delta = num_pos - sgp4_pos
        diverge_3d[i] = np.linalg.norm(div_delta)
        diverge_rtn[i] = _rtn_decompose(sgp4_pos, sgp4_vel, div_delta)

    return {
        "elapsed_s": elapsed,
        "elapsed_h": elapsed / 3600.0,
        "duration_h": duration_s / 3600.0,
        "sgp4_err_3d_m": sgp4_err_3d,
        "num_err_3d_m": num_err_3d,
        "diverge_3d_m": diverge_3d,
        "sgp4_rtn_m": sgp4_rtn,
        "num_rtn_m": num_rtn,
        "diverge_rtn_m": diverge_rtn,
        "sgp4_final_m": float(sgp4_err_3d[-1]),
        "num_final_m": float(num_err_3d[-1]),
        "diverge_final_m": float(diverge_3d[-1]),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

# Material Design colours
CLR_SGP4 = "#F44336"       # red
CLR_NUM = "#2196F3"        # blue
CLR_RADIAL = "#4CAF50"     # green
CLR_ALONG = "#FF9800"      # orange
CLR_CROSS = "#9C27B0"      # purple


def plot_rtn_errors(results: List[dict], plot_dir: Path) -> None:
    """Figure 1 — RTN divergence between SGP4 and numerical (one row per pair)."""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(
        "SGP4 vs Numerical Divergence — RTN Components",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for idx, res in enumerate(results):
        hours = res["elapsed_h"]
        dur = res["duration_h"]
        rtn = res["diverge_rtn_m"]

        ax = axes[idx, 0]
        ax.plot(hours, rtn[:, 0], color=CLR_RADIAL, linewidth=1.2, label="Radial")
        ax.plot(hours, rtn[:, 1], color=CLR_ALONG, linewidth=1.2, label="Along-track")
        ax.plot(hours, rtn[:, 2], color=CLR_CROSS, linewidth=1.2, label="Cross-track")
        ax.set_ylabel("Divergence (m)")
        ax.set_xlabel("Elapsed time (h)")
        ax.set_title(f"Pair {idx + 1}  ({dur:.1f} h)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = plot_dir / "sgp4_vs_numerical_rtn.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_3d_error(results: List[dict], plot_dir: Path) -> None:
    """Figure 2 — SGP4-vs-numerical 3D divergence + truth errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: propagator divergence (the meaningful metric)
    for idx, res in enumerate(results):
        hours = res["elapsed_h"]
        dur = res["duration_h"]
        alpha = 0.5 + 0.5 * idx / max(len(results) - 1, 1)
        ax1.plot(
            hours, res["diverge_3d_m"] / 1000.0,
            linewidth=1.5, alpha=alpha,
            label=f"Pair {idx + 1} ({dur:.1f} h)",
        )

    ax1.set_xlabel("Elapsed time (hours)", fontsize=11)
    ax1.set_ylabel("SGP4 vs Numerical divergence (km)", fontsize=11)
    ax1.set_title(
        "Propagator Divergence Over Time",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right panel: both vs TLE_B truth (shows TLE fit noise dominates)
    for idx, res in enumerate(results):
        hours = res["elapsed_h"]
        dur = res["duration_h"]
        alpha = 0.5 + 0.5 * idx / max(len(results) - 1, 1)
        ax2.plot(
            hours, res["sgp4_err_3d_m"] / 1000.0,
            linestyle="--", linewidth=1.2, color=CLR_SGP4, alpha=alpha,
            label=f"SGP4 — P{idx + 1}" if idx == 0 else f"_SGP4 P{idx + 1}",
        )
        ax2.plot(
            hours, res["num_err_3d_m"] / 1000.0,
            linestyle="-", linewidth=1.2, color=CLR_NUM, alpha=alpha,
            label=f"Num — P{idx + 1}" if idx == 0 else f"_Num P{idx + 1}",
        )

    # Custom legend for just SGP4 vs Numerical
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=CLR_SGP4, linestyle="--", label="SGP4"),
        Line2D([0], [0], color=CLR_NUM, linestyle="-", label="Numerical"),
    ]
    ax2.legend(handles=legend_elements, fontsize=10, loc="upper left")
    ax2.set_xlabel("Elapsed time (hours)", fontsize=11)
    ax2.set_ylabel("Error vs TLE_B truth (km)", fontsize=11)
    ax2.set_title(
        "Both Propagators vs Next-TLE Truth",
        fontsize=13, fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = plot_dir / "sgp4_vs_numerical_3d.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_summary_bars(results: List[dict], plot_dir: Path) -> None:
    """Figure 3 — Bar chart: TLE-truth errors + propagator divergence."""
    n = len(results)
    x = np.arange(n)
    width = 0.25

    sgp4_final = [r["sgp4_final_m"] for r in results]
    num_final = [r["num_final_m"] for r in results]
    div_final = [r["diverge_final_m"] for r in results]
    durations = [r["duration_h"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, 3 * n), 6))

    # Left: errors vs truth (large, dominated by TLE fit noise)
    b1 = ax1.bar(x - width / 2, [v / 1000 for v in sgp4_final], width,
                 color=CLR_SGP4, label="SGP4")
    b2 = ax1.bar(x + width / 2, [v / 1000 for v in num_final], width,
                 color=CLR_NUM, label="Numerical")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=7)
    labels = [f"P{i + 1}\n({durations[i]:.1f}h)" for i in range(n)]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Error vs next TLE (km)", fontsize=11)
    ax1.set_title("Error vs TLE_B Truth", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: propagator divergence (the actionable metric)
    bars = ax2.bar(x, div_final, width * 2, color="#4CAF50")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{h:.0f} m", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("SGP4 vs Numerical divergence (m)", fontsize=11)
    ax2.set_title("Propagator Divergence at TLE_B Epoch", fontsize=12,
                  fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = plot_dir / "sgp4_vs_numerical_summary.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Console report ───────────────────────────────────────────────────────────

def print_summary(results: List[dict]) -> None:
    """Print a comparison table to stdout."""
    print("\n" + "=" * 80)
    print("  SGP4 vs NUMERICAL PROPAGATION — SUMMARY")
    print("=" * 80)
    print(
        f"  {'Pair':>4}  {'Interval':>8}  "
        f"{'SGP4→truth':>11}  {'Num→truth':>10}  "
        f"{'Divergence':>11}  {'Div rate':>10}"
    )
    print(
        f"  {'':>4}  {'(hours)':>8}  "
        f"{'(km)':>11}  {'(km)':>10}  "
        f"{'(m)':>11}  {'(m/h)':>10}"
    )
    print("-" * 80)
    for idx, res in enumerate(results):
        dur = res["duration_h"]
        s_err = res["sgp4_final_m"]
        n_err = res["num_final_m"]
        div_m = res["diverge_final_m"]
        rate = div_m / dur if dur > 0 else 0
        print(
            f"  {idx + 1:>4}  {dur:>8.2f}  "
            f"{s_err / 1000:>11.1f}  {n_err / 1000:>10.1f}  "
            f"{div_m:>11.1f}  {rate:>10.1f}"
        )
    print("=" * 80)
    print(
        "\n  Note: 'truth' errors are ~400 km due to TLE-to-TLE fitting noise."
    )
    print(
        "  The 'divergence' column shows how much SGP4 and numerical disagree —"
    )
    print(
        "  this is the operationally meaningful metric for conjunction screening."
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SGP4 vs numerical propagation accuracy",
    )
    parser.add_argument(
        "--norad", type=int, default=56310,
        help="NORAD catalog ID (default: 56310 — Teleos 2)",
    )
    parser.add_argument(
        "--start", default="2026-03-02T20:00:00",
        help="Search window start (ISO, default: 2026-03-02T20:00)",
    )
    parser.add_argument(
        "--end", default="2026-03-04T00:00:00",
        help="Search window end (ISO, default: 2026-03-04T00:00)",
    )
    parser.add_argument(
        "--step", type=float, default=60.0,
        help="Propagation time step in seconds (default: 60)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    # ── Fetch TLEs ───────────────────────────────────────────────────────
    print(f"Fetching gp_history for NORAD {args.norad}  "
          f"[{args.start} → {args.end}] ...")
    tles = fetch_tle_pairs(args.norad, args.start, args.end)
    print(f"  Found {len(tles)} unique TLEs")

    if len(tles) < 2:
        print("Need at least 2 TLEs to form a pair. Exiting.")
        sys.exit(1)

    for i, (ep, l1, l2) in enumerate(tles):
        print(f"    TLE {i + 1}: {ep.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ── Propagate each consecutive pair ──────────────────────────────────
    results = []
    n_pairs = len(tles) - 1
    for i in range(n_pairs):
        ep_a, l1_a, l2_a = tles[i]
        ep_b, l1_b, l2_b = tles[i + 1]
        gap_h = (ep_b - ep_a).total_seconds() / 3600.0
        print(f"\nPair {i + 1}/{n_pairs}: "
              f"{ep_a.strftime('%H:%M')} → {ep_b.strftime('%H:%M')}  "
              f"({gap_h:.1f} h)")

        res = propagate_pair(l1_a, l2_a, l1_b, l2_b, step_s=args.step)
        results.append(res)

        print(f"  SGP4 vs truth:         {res['sgp4_final_m'] / 1000:.1f} km")
        print(f"  Numerical vs truth:    {res['num_final_m'] / 1000:.1f} km")
        print(f"  SGP4↔Numerical div:    {res['diverge_final_m']:.1f} m")

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(results)

    # ── Plots ────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_dir = Path(__file__).parent.parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        print("\nGenerating plots ...")
        plot_rtn_errors(results, plot_dir)
        plot_3d_error(results, plot_dir)
        plot_summary_bars(results, plot_dir)
        print("Done.")


if __name__ == "__main__":
    main()
