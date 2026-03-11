"""
sgp4_vs_numerical.py -- Compare SGP4 (analytical) vs high-fidelity numerical
propagation accuracy using Sentinel-1A Precise Orbit Ephemerides as ground truth.

Workflow:
  1. Load a single POEORB file (cm-level osculating state vectors, 10s, ITRF)
  2. Initialise numerical propagator from the first POE state (ITRF -> EME2000)
  3. Fetch closest TLE, initialise SGP4 propagator (TLEPropagator)
  4. Propagate both forward over the POE validity window
  5. Compare both against POE truth at each timestep (all in ITRF)

The numerical propagator gets a true osculating initial state, while SGP4
uses its native mean-element TLE -- a fair comparison of each method's
natural operating mode.

Usage:
    conda run -n orbit python analysis/sgp4_vs_numerical.py
    conda run -n orbit python analysis/sgp4_vs_numerical.py --eof path/to/file.EOF
    conda run -n orbit python analysis/sgp4_vs_numerical.py --hours 12 --no-plot

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
from org.orekit.orbits import CartesianOrbit
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.frames import FramesFactory, Transform
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import IERSConventions, Constants, PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# Python imports
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPACETRACK_IDENTITY, SPACETRACK_PASSWORD

import spacetrack.operators as op
from spacetrack import SpaceTrackClient


# ── Constants ────────────────────────────────────────────────────────────────

SENTINEL_1A_NORAD = 39634
EARTH_MU = Constants.WGS84_EARTH_MU
EARTH_RE_M = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

# Sentinel-1A physical parameters (ESA documentation)
SENTINEL_1A_MASS_KG = 2300.0
SENTINEL_1A_AREA_M2 = 25.0     # approximate cross-section (large SAR antenna)
SENTINEL_1A_CD = 2.2


# ── EOF (POEORB) parser ─────────────────────────────────────────────────────

def parse_eof(path: Path) -> Dict:
    """
    Parse an ESA Precise Orbit Ephemeris (.EOF) file.

    Returns dict with:
        'utc':  np.array of datetime objects (UTC)
        'pos':  np.array shape (N, 3), ITRF position in metres
        'vel':  np.array shape (N, 3), ITRF velocity in m/s
    """
    tree = ET.parse(str(path))
    root = tree.getroot()

    osvs = root.findall(".//{*}OSV") or root.findall(".//OSV")

    utc_list = []
    pos_list = []
    vel_list = []

    for osv in osvs:
        utc_el = osv.find("{*}UTC") or osv.find("UTC")
        utc_str = utc_el.text.replace("UTC=", "")
        dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S.%f").replace(
            tzinfo=timezone.utc
        )
        utc_list.append(dt)

        x = float((osv.find("{*}X") or osv.find("X")).text)
        y = float((osv.find("{*}Y") or osv.find("Y")).text)
        z = float((osv.find("{*}Z") or osv.find("Z")).text)
        vx = float((osv.find("{*}VX") or osv.find("VX")).text)
        vy = float((osv.find("{*}VY") or osv.find("VY")).text)
        vz = float((osv.find("{*}VZ") or osv.find("VZ")).text)

        pos_list.append([x, y, z])
        vel_list.append([vx, vy, vz])

    return {
        "utc": np.array(utc_list, dtype=object),
        "pos": np.array(pos_list),
        "vel": np.array(vel_list),
    }


# ── TLE retrieval ────────────────────────────────────────────────────────────

def _parse_epoch(epoch_str: str) -> Optional[datetime]:
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(epoch_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def fetch_tles(
    norad_id: int, dt_start: str, dt_end: str,
) -> List[Tuple[datetime, str, str]]:
    """Fetch gp_history TLEs from Space-Track, deduplicated and sorted."""
    st = SpaceTrackClient(SPACETRACK_IDENTITY, SPACETRACK_PASSWORD)
    raw = st.gp_history(
        norad_cat_id=norad_id,
        epoch=op.inclusive_range(dt_start, dt_end),
        orderby="epoch asc",
        format="json",
    )
    recs = json.loads(raw) if isinstance(raw, str) else (raw or [])

    seen = set()
    tles = []
    for r in recs:
        ep = _parse_epoch(r.get("EPOCH", ""))
        l1, l2 = r.get("TLE_LINE1"), r.get("TLE_LINE2")
        if ep is None or not l1 or not l2:
            continue
        key = ep.strftime("%Y-%m-%dT%H:%M:%S")
        if key in seen:
            continue
        seen.add(key)
        tles.append((ep, l1, l2))
    tles.sort(key=lambda t: t[0])
    return tles


def pick_best_tle(
    tles: List[Tuple[datetime, str, str]], target_dt: datetime,
) -> Tuple[datetime, str, str]:
    """Pick the TLE with epoch closest to (and preferably just before) target_dt."""
    before = [(ep, l1, l2) for ep, l1, l2 in tles if ep <= target_dt]
    if before:
        return max(before, key=lambda t: t[0])
    return min(tles, key=lambda t: abs((t[0] - target_dt).total_seconds()))


# ── Orekit helpers ───────────────────────────────────────────────────────────

def _datetime_to_absolute(dt: datetime) -> AbsoluteDate:
    utc = TimeScalesFactory.getUTC()
    return AbsoluteDate(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, float(dt.second + dt.microsecond / 1e6),
        utc,
    )


def _pv_to_numpy(pv) -> Tuple[np.ndarray, np.ndarray]:
    p = pv.getPosition()
    v = pv.getVelocity()
    return (
        np.array([p.getX(), p.getY(), p.getZ()]),
        np.array([v.getX(), v.getY(), v.getZ()]),
    )


def _make_initial_state_from_poe(
    pos_itrf: np.ndarray,
    vel_itrf: np.ndarray,
    epoch_ad: AbsoluteDate,
) -> SpacecraftState:
    """
    Create an Orekit SpacecraftState from a POE state vector (ITRF).
    Converts ITRF osculating pos/vel to EME2000 for the numerical propagator.
    """
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    eme2000 = FramesFactory.getEME2000()

    # Build PVCoordinates in ITRF
    pos_vec = Vector3D(float(pos_itrf[0]), float(pos_itrf[1]), float(pos_itrf[2]))
    vel_vec = Vector3D(float(vel_itrf[0]), float(vel_itrf[1]), float(vel_itrf[2]))
    pv_itrf = PVCoordinates(pos_vec, vel_vec)

    # Transform ITRF -> EME2000 at this epoch
    itrf_to_eme = itrf.getTransformTo(eme2000, epoch_ad)
    pv_eme = itrf_to_eme.transformPVCoordinates(pv_itrf)

    # Create CartesianOrbit in EME2000
    orbit = CartesianOrbit(pv_eme, eme2000, epoch_ad, EARTH_MU)
    state = SpacecraftState(orbit)
    return state.withMass(float(SENTINEL_1A_MASS_KG))


def _build_numerical_propagator(
    initial_state: SpacecraftState,
    cross_section: float = SENTINEL_1A_AREA_M2,
    cd: float = SENTINEL_1A_CD,
) -> NumericalPropagator:
    """
    High-fidelity NumericalPropagator:
      - 70x70 gravity, Harris-Priester drag, Sun+Moon, relativity
    """
    integrator = DormandPrince853Integrator(0.001, 60.0, 1.0, 1e-6)
    prop = NumericalPropagator(integrator)
    prop.resetInitialState(initial_state)

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # Gravity 70x70
    gravity_field = GravityFieldFactory.getNormalizedProvider(70, 70)
    prop.addForceModel(
        HolmesFeatherstoneAttractionModel(itrf, gravity_field)
    )

    # Atmospheric drag
    sun = CelestialBodyFactory.getSun()
    earth = OneAxisEllipsoid(
        EARTH_RE_M, Constants.WGS84_EARTH_FLATTENING, itrf,
    )
    atmosphere = HarrisPriester(sun, earth)
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
    pos_ref: np.ndarray, vel_ref: np.ndarray, delta_pos: np.ndarray,
) -> Tuple[float, float, float]:
    r_hat = pos_ref / np.linalg.norm(pos_ref)
    h = np.cross(pos_ref, vel_ref)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return (
        float(np.dot(delta_pos, r_hat)),
        float(np.dot(delta_pos, t_hat)),
        float(np.dot(delta_pos, n_hat)),
    )


# ── Core comparison ──────────────────────────────────────────────────────────

def compare_propagators(
    poe_data: Dict,
    tle_line1: str,
    tle_line2: str,
    tle_epoch: datetime,
    max_hours: float = 24.0,
    step_s: float = 60.0,
) -> Dict:
    """
    Compare SGP4 and numerical propagation against POE truth.

    - Numerical propagator: initialised from first POE state (true osculating)
    - SGP4 propagator: initialised from closest TLE (mean elements)
    - Both compared against POE states in ITRF at each timestep
    """
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # ── POE initial state -> Numerical propagator ─────────────────────────
    poe_start = poe_data["utc"][0]
    poe_end = poe_data["utc"][-1]
    epoch_ad = _datetime_to_absolute(poe_start)

    print(f"  Numerical IC: POE state at {poe_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    init_state = _make_initial_state_from_poe(
        poe_data["pos"][0], poe_data["vel"][0], epoch_ad,
    )
    num_prop = _build_numerical_propagator(init_state)

    # ── TLE -> SGP4 propagator ────────────────────────────────────────────
    tle = TLE(tle_line1, tle_line2)
    sgp4_prop = TLEPropagator.selectExtrapolator(tle)
    tle_age_h = abs((poe_start - tle_epoch).total_seconds()) / 3600.0
    print(f"  SGP4 IC:      TLE epoch {tle_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}"
          f"  (age: {tle_age_h:.1f} h)")

    # ── Propagation window ───────────────────────────────────────────────
    prop_end = min(poe_start + timedelta(hours=max_hours), poe_end)
    duration_s = (prop_end - poe_start).total_seconds()

    # Pre-compute POE timestamps for interpolation
    poe_ts = np.array([dt.timestamp() for dt in poe_data["utc"]])

    n_steps = int(duration_s / step_s) + 1
    elapsed_h = np.zeros(n_steps)
    sgp4_err_3d = np.zeros(n_steps)
    num_err_3d = np.zeros(n_steps)
    diverge_3d = np.zeros(n_steps)
    sgp4_rtn = np.zeros((n_steps, 3))
    num_rtn = np.zeros((n_steps, 3))
    diverge_rtn = np.zeros((n_steps, 3))

    for i in range(n_steps):
        t = min(i * step_s, duration_s)
        current_dt = poe_start + timedelta(seconds=t)
        elapsed_h[i] = t / 3600.0
        target_ad = _datetime_to_absolute(current_dt)

        # ── POE truth (ITRF) -- linear interpolation ─────────────────────
        ts = current_dt.timestamp()
        idx = np.searchsorted(poe_ts, ts) - 1
        idx = max(0, min(idx, len(poe_ts) - 2))
        frac = (ts - poe_ts[idx]) / (poe_ts[idx + 1] - poe_ts[idx])
        truth_pos = (
            (1 - frac) * poe_data["pos"][idx] + frac * poe_data["pos"][idx + 1]
        )
        truth_vel = (
            (1 - frac) * poe_data["vel"][idx] + frac * poe_data["vel"][idx + 1]
        )

        # ── SGP4 -> ITRF ─────────────────────────────────────────────────
        sgp4_state = sgp4_prop.propagate(target_ad)
        sgp4_pos, sgp4_vel = _pv_to_numpy(sgp4_state.getPVCoordinates(itrf))

        # ── Numerical -> ITRF ────────────────────────────────────────────
        num_state = num_prop.propagate(target_ad)
        num_pos, num_vel = _pv_to_numpy(num_state.getPVCoordinates(itrf))

        # ── Errors vs POE truth ──────────────────────────────────────────
        sgp4_delta = sgp4_pos - truth_pos
        num_delta = num_pos - truth_pos

        sgp4_err_3d[i] = np.linalg.norm(sgp4_delta)
        num_err_3d[i] = np.linalg.norm(num_delta)

        sgp4_rtn[i] = _rtn_decompose(truth_pos, truth_vel, sgp4_delta)
        num_rtn[i] = _rtn_decompose(truth_pos, truth_vel, num_delta)

        # ── SGP4 vs Numerical divergence ─────────────────────────────────
        div_delta = num_pos - sgp4_pos
        diverge_3d[i] = np.linalg.norm(div_delta)
        diverge_rtn[i] = _rtn_decompose(sgp4_pos, sgp4_vel, div_delta)

    return {
        "poe_start": poe_start,
        "poe_end": prop_end,
        "tle_epoch": tle_epoch,
        "tle_age_h": tle_age_h,
        "elapsed_h": elapsed_h,
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

CLR_SGP4 = "#F44336"       # red
CLR_NUM = "#2196F3"        # blue
CLR_RADIAL = "#4CAF50"     # green
CLR_ALONG = "#FF9800"      # orange
CLR_CROSS = "#9C27B0"      # purple


def plot_rtn_vs_truth(result: Dict, plot_dir: Path) -> None:
    """Figure 1 -- RTN error components vs POE truth for both propagators."""
    hours = result["elapsed_h"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "Position Error vs POE Truth -- RTN Components",
        fontsize=14, fontweight="bold",
    )

    for ax, label, rtn in [
        (ax1, "SGP4 (TLEPropagator)", result["sgp4_rtn_m"]),
        (ax2, "Numerical (70x70 gravity + drag + 3rd body)", result["num_rtn_m"]),
    ]:
        ax.plot(hours, rtn[:, 0], color=CLR_RADIAL, linewidth=1.2, label="Radial")
        ax.plot(hours, rtn[:, 1], color=CLR_ALONG, linewidth=1.2, label="Along-track")
        ax.plot(hours, rtn[:, 2], color=CLR_CROSS, linewidth=1.2, label="Cross-track")
        ax.set_ylabel("Error (m)")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    ax2.set_xlabel("Elapsed time from POE start (hours)")

    fig.tight_layout()
    out = plot_dir / "sgp4_vs_numerical_rtn.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_3d_error(result: Dict, plot_dir: Path) -> None:
    """Figure 2 -- 3D position error vs POE truth + propagator divergence."""
    hours = result["elapsed_h"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax1.plot(hours, result["sgp4_err_3d_m"], color=CLR_SGP4,
             linewidth=1.5, label="SGP4")
    ax1.plot(hours, result["num_err_3d_m"], color=CLR_NUM,
             linewidth=1.5, label="Numerical")
    ax1.set_ylabel("3D position error vs POE (m)", fontsize=11)
    ax1.set_title(
        "Position Error vs Precise Orbit (POE Truth)",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours, result["diverge_3d_m"], color="#4CAF50", linewidth=1.5)
    ax2.set_ylabel("SGP4 vs Numerical divergence (m)", fontsize=11)
    ax2.set_xlabel("Elapsed time from POE start (hours)", fontsize=11)
    ax2.set_title(
        "Propagator Divergence (SGP4 vs Numerical)",
        fontsize=13, fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = plot_dir / "sgp4_vs_numerical_3d.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_summary(result: Dict, plot_dir: Path) -> None:
    """Figure 3 -- Error growth comparison + improvement ratio."""
    hours = result["elapsed_h"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: log-scale error growth
    mask = (result["sgp4_err_3d_m"] > 0) & (result["num_err_3d_m"] > 0)
    ax1.semilogy(hours[mask], result["sgp4_err_3d_m"][mask] / 1000.0,
                 color=CLR_SGP4, linewidth=1.5, label="SGP4")
    ax1.semilogy(hours[mask], result["num_err_3d_m"][mask] / 1000.0,
                 color=CLR_NUM, linewidth=1.5, label="Numerical")
    ax1.set_xlabel("Elapsed time (hours)", fontsize=11)
    ax1.set_ylabel("3D error vs POE truth (km, log)", fontsize=11)
    ax1.set_title("Error Growth (log scale)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: improvement ratio (log scale, skip first few steps where num~0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = result["sgp4_err_3d_m"] / result["num_err_3d_m"]
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    # Skip first few minutes where numerical error is near-zero
    skip_mask = hours >= 0.5
    ax2.semilogy(hours[skip_mask], ratio[skip_mask], color="#4CAF50", linewidth=1.5)
    ax2.axhline(1.0, color="#9E9E9E", linestyle="--", alpha=0.7,
                label="Parity (ratio = 1)")
    ax2.set_xlabel("Elapsed time (hours)", fontsize=11)
    ax2.set_ylabel("SGP4 error / Numerical error", fontsize=11)
    ax2.set_title(
        "Numerical Improvement Factor (>1 = numerical wins)",
        fontsize=12, fontweight="bold",
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = plot_dir / "sgp4_vs_numerical_summary.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ── Console report ───────────────────────────────────────────────────────────

def print_summary(result: Dict) -> None:
    hours = result["elapsed_h"]
    milestones = [0.5, 1, 2, 4, 6, 8, 12, 18, 24]

    print("\n" + "=" * 78)
    print("  SGP4 vs NUMERICAL -- vs POE TRUTH (Sentinel-1A)")
    print("=" * 78)
    print(f"  POE start:   {result['poe_start'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  TLE epoch:   {result['tle_epoch'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
          f"  (age: {result['tle_age_h']:.1f} h at POE start)")
    print(f"  Duration:    {result['duration_h']:.1f} h")
    print()
    print(f"  Numerical IC: first POE state (osculating, ITRF -> EME2000)")
    print(f"  SGP4 IC:      closest TLE (mean elements)")
    print()
    print(f"  {'Hours':>6}  {'SGP4 err':>12}  {'Num err':>12}  "
          f"{'Ratio':>8}  {'Divergence':>12}")
    print("-" * 78)

    for mh in milestones:
        idx = np.searchsorted(hours, mh)
        if idx >= len(hours):
            break
        se = result["sgp4_err_3d_m"][idx]
        ne = result["num_err_3d_m"][idx]
        dv = result["diverge_3d_m"][idx]
        ratio = se / ne if ne > 0 else float("inf")
        print(
            f"  {mh:>5.1f}h  {se:>10.1f} m  {ne:>10.1f} m  "
            f"{ratio:>7.1f}x  {dv:>10.1f} m"
        )

    # Final
    se = result["sgp4_final_m"]
    ne = result["num_final_m"]
    dv = result["diverge_final_m"]
    ratio = se / ne if ne > 0 else float("inf")
    dur = result["duration_h"]
    print("-" * 78)
    print(
        f"  {dur:>5.1f}h  {se:>10.1f} m  {ne:>10.1f} m  "
        f"{ratio:>7.1f}x  {dv:>10.1f} m   (final)"
    )
    print("=" * 78)
    if ratio > 1:
        print(f"\n  Numerical propagator is {ratio:.1f}x more accurate than SGP4"
              f" at {dur:.1f} h.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    default_eof_dir = Path(__file__).parent / "data" / "sentinel1_poeorb"
    # Use the first .EOF file by default
    default_eof_files = sorted(default_eof_dir.glob("*.EOF")) if default_eof_dir.exists() else []
    default_eof = str(default_eof_files[0]) if default_eof_files else None

    parser = argparse.ArgumentParser(
        description="Compare SGP4 vs numerical propagation against POE truth",
    )
    parser.add_argument(
        "--eof", default=default_eof,
        help="Path to a single .EOF file (default: first in data/sentinel1_poeorb/)",
    )
    parser.add_argument(
        "--hours", type=float, default=24.0,
        help="Max propagation duration in hours (default: 24)",
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

    if not args.eof:
        print("No .EOF file found. Provide --eof path/to/file.EOF")
        sys.exit(1)

    eof_path = Path(args.eof)

    # ── Load single EOF ──────────────────────────────────────────────────
    print(f"Loading POE: {eof_path.name}")
    poe = parse_eof(eof_path)
    print(f"  Coverage: {poe['utc'][0].strftime('%Y-%m-%d %H:%M')} -> "
          f"{poe['utc'][-1].strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  {len(poe['utc'])} state vectors at 10s intervals")

    # ── Fetch TLEs ───────────────────────────────────────────────────────
    poe_start = poe["utc"][0]
    poe_end = poe["utc"][-1]
    search_start = (poe_start - timedelta(days=1)).strftime("%Y-%m-%d")
    search_end = (poe_start + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\nFetching TLEs for Sentinel-1A (NORAD {SENTINEL_1A_NORAD}) ...")
    tles = fetch_tles(SENTINEL_1A_NORAD, search_start, search_end)
    print(f"  Found {len(tles)} unique TLEs")
    for ep, _, _ in tles:
        marker = " <-" if ep == pick_best_tle(tles, poe_start)[0] else ""
        print(f"    {ep.strftime('%Y-%m-%d %H:%M:%S UTC')}{marker}")

    if not tles:
        print("No TLEs found. Exiting.")
        sys.exit(1)

    tle_epoch, tle_l1, tle_l2 = pick_best_tle(tles, poe_start)

    # ── Run comparison ───────────────────────────────────────────────────
    print(f"\nPropagating for up to {args.hours:.0f} h ...")
    result = compare_propagators(
        poe, tle_l1, tle_l2, tle_epoch,
        max_hours=args.hours, step_s=args.step,
    )

    print_summary(result)

    # ── Plots ────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_dir = Path(__file__).parent.parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        print("\nGenerating plots ...")
        plot_rtn_vs_truth(result, plot_dir)
        plot_3d_error(result, plot_dir)
        plot_summary(result, plot_dir)
        print("Done.")


if __name__ == "__main__":
    main()
