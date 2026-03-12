# Maneuver Detection Report — UMBRA-10 (NORAD 60547)

**Analysis date:** 2026-03-12
**Data coverage:** 2024-08-29 → 2026-03-11 (≈ 560 days)
**Script:** `analysis/maneuver_detection.py`

---

## 1. Pipeline Processing

### 1.1 Data ingestion

The pipeline loaded GP history from `satcat.db` for NORAD 60547:

| Step | Count |
|---|---|
| Raw records fetched | 1,518 |
| Revised-element-set duplicates removed | 229 |
| Records after deduplication | 1,289 |

Space-Track occasionally issues multiple GP records with the same epoch timestamp (revised fits). The pipeline keeps only the last (most recently issued) element set per epoch. 229 such duplicates were found here — a relatively high rate (~15%) suggesting active TLE refinement, consistent with an actively maneuvering object.

### 1.2 B* noise filter

The fixed `OD_BSTAR_NOISE_MAX` threshold of `1e-03` 1/ER would have retained only 330 of 1,289 TLEs (<50%), triggering the adaptive fallback. The threshold was relaxed to the **95th percentile of |B*|** across the dataset:

```
Adaptive threshold: 2.91e-03  (fixed: 1.00e-03)
Records removed:    65
Records retained:   1,224
```

The high B* values on UMBRA-10 reflect a combination of its low orbit (higher drag), active maneuvering (drag regime shifts post-burn), and the SAR satellite's likely high area-to-mass ratio.

### 1.3 Epoch pair computation

From the 1,224 retained TLEs:

| Item | Count |
|---|---|
| Consecutive pairs analysed | 1,154 |
| Pairs skipped (gap > 10 days) | 69 |

For each pair, the pipeline:
1. Takes SGP4 state at epoch₁ (dt=0) as initial conditions
2. Integrates forward Δt seconds with RK4 + J2 + US Standard Atmosphere 1976 drag using B* from epoch₁'s TLE
3. Compares predicted state to the SGP4 state at epoch₂ (dt=0)
4. Records the scalar velocity residual magnitude and normalises by gap length → **VR/day (m/s/day)**

### 1.4 Adaptive MAD thresholds

Thresholds are computed from the distribution of signals across **all** pairs in the analysis window:

| Threshold | Value | Derivation |
|---|---|---|
| VR/day | **12.99 m/s/day** | median(5.245) + 5.0 × MAD(1.549) |
| ΔB* | **3.73e-04** | median + 5 × MAD over non-zero pairs |
| Kp (space weather) | **5.0** | Fixed config parameter |

The median VR/day of 5.25 m/s/day represents the baseline propagation noise floor for this satellite. The MAD of 1.55 m/s/day is tight, indicating a stable noise floor — the threshold at 5σ sits well above background.

### 1.5 Classification summary

| Category | Count |
|---|---|
| OD-flagged (VR/day > threshold) | 31 |
| B*-flagged (ΔB* > threshold) | 75 |
| **Confirmed maneuvers** (flagged & Kp < 5.0) | **79** |
| Uncertain (flagged & Kp ≥ 5.0) | 20 |

Note: confirmed + uncertain > OD-flagged + B*-flagged because some pairs are flagged by **both** signals simultaneously (OD+B*), so are counted once in each flag total but once in the confirmed/uncertain total.

---

## 2. Results

### 2.1 Overall detection rate

79 confirmed + 20 uncertain = **99 flagged intervals** over 560 days ≈ **one detection every ~5.7 days**. This is a very high cadence, consistent with UMBRA-10's known operational profile as an active SAR imaging satellite in a low Earth orbit requiring frequent station-keeping.

### 2.2 Detection mode breakdown

Most confirmed detections are driven by **B* alone** (ΔB* > threshold, VR/day below threshold). This indicates the majority of detected maneuvers are small burns — insufficient to produce a clearly anomalous propagation residual, but large enough to shift the TLE fitter's drag coefficient estimate. The OD residual catches the larger, higher-confidence burns.

From the confirmed set:

| Flag mode | Count | Interpretation |
|---|---|---|
| OD only | ~15 | Large ΔV — physics residual clearly anomalous |
| B* only | ~55 | Small burn — drag regime shifted, residual below threshold |
| OD + B* | ~9 | High-confidence: both signals fire together |

### 2.3 Notable high-residual events

The largest OD velocity residuals in the confirmed set indicate significant multi-day maneuver sequences:

| Period | VR (m/s) | Gap (days) | dSMA (km) | Notes |
|---|---|---|---|---|
| 2024-10-06 → 2024-10-09 | 147.9 | 2.81 | −0.678 | Largest single-pair residual |
| 2024-10-03 → 2024-10-06 | 139.8 | 2.95 | −0.634 | Sequential with above — sustained deorbit activity |
| 2024-10-26 → 2024-11-02 | 700.2 | 6.62 | −1.475 | Highest residual in dataset; 6.6-day gap amplifies accumulation |
| 2024-11-08 → 2024-11-11 | 116.0 | 2.61 | −0.575 | — |
| 2024-09-11 → 2024-09-14 | 123.8 | 2.95 | −0.540 | — |

The 700 m/s residual on 2024-10-26 → 2024-11-02 is the most extreme and warrants care in interpretation: a 6.6-day gap means the physics model has accumulated substantial error even without a maneuver. The residual should be interpreted as "the orbit changed significantly over this period" rather than a precise ΔV estimate.

### 2.4 Orbit altitude trend visible in plot

Panel 5 (dSMA) shows a strong, persistent **negative bias** throughout the analysis window — virtually all dSMA values are negative. This is the expected signature of a satellite actively lowering its orbit through repeated drag-makeup burns followed by deliberate deorbit-rate adjustments, or simply natural decay with infrequent raise maneuvers. The cumulative altitude loss visible across ~560 days is substantial.

### 2.5 Concentrated maneuver period: Sep–Nov 2024

The early months of the dataset (Aug–Nov 2024) show a markedly higher density of both OD-flagged and B*-flagged detections compared to 2025 onwards. This is visible in panels 1 and 2 of the plot as elevated baseline VR values with frequent confirmed (red) markers. Two possible explanations:

1. **Orbit insertion/commissioning:** UMBRA-10 was undergoing initial orbit phasing maneuvers shortly after launch, resulting in more frequent and larger burns
2. **Data density:** the early period has denser TLE coverage (more updates per day), increasing the sensitivity of the B* threshold to small burns

### 2.6 Uncertain detections and space weather

20 intervals were flagged but downgraded to *uncertain* due to Kp ≥ 5.0. Notable uncertain epochs:

| Epoch | VR (m/s) | Kp | Notes |
|---|---|---|---|
| 2024-10-10 → 2024-10-12 | 57.5 | **8.3** | Severe geomagnetic storm — high confidence this is weather |
| 2025-11-12 → 2025-11-13 | 35.0 | **8.7** | Extreme storm event |
| 2025-11-13 → 2025-11-14 | 25.2 | 7.3 | Consecutive storm day — likely same event |
| 2026-01-20 → 2026-01-21 | 73.4 | 6.7 | Moderate storm; OD+B* both fired |

The 2024-10-10 and 2025-11-12/13 events with Kp > 8 are very likely false positives — geomagnetic storms dramatically increase thermospheric density, inflating drag beyond what the US Standard Atmosphere model captures, producing large propagation errors unrelated to any burn.

The 2026-01-20 event (Kp 6.7, VR 73.4 m/s, OD+B* both fired) is more ambiguous — the residual is large relative to the storm level, and the B* signal also fired. This could be a genuine maneuver executed during disturbed conditions.

### 2.7 B* signal characteristics

Panel 3 shows B* fluctuating in the range ~0.0005–0.003 with substantial variance throughout the mission. This is consistent with a low-altitude satellite (higher drag sensitivity) and active maneuvering. The ΔB* threshold of 3.73e-04 is modest relative to the absolute B* values, explaining why B*-only detections dominate — the fitter frequently reassigns the drag coefficient at each update cycle.

---

## 3. Caveats

- **VR/day is a scalar magnitude** — direction of burn (prograde, retrograde, radial) is not recovered. dSMA provides a weak proxy for burn direction but is confounded by natural perturbations over the gap period.
- **Large-gap pairs** (e.g., the 6.6-day gap in Oct 2024) produce residuals that are dominated by propagation model error accumulation, not burn magnitude. Treat residuals from gaps > 3 days as qualitative indicators only.
- **B*-only detections** may include false positives from rapid drag environment changes (e.g., satellite attitude changes, solar flux variation) that shift the TLE fitter's drag estimate without a burn occurring.
- **Adaptive thresholds are self-calibrating** to this satellite's own noise floor. If UMBRA-10 maneuvers so frequently that maneuver pairs constitute a large fraction of the dataset, the median and MAD will be biased upward and some smaller maneuvers may be missed. The very high detection rate here (99 events / 560 days) suggests this could be relevant.
