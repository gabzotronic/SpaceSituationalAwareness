# Maneuver Detection Merge Plan

Merge the OD-based propagation and detection approach from `orbit_analysis_od.py`
(https://github.com/futureproofbear/SpaceSituationalAnalysis) into our
`analysis/maneuver_detection.py`, replacing the Kepler-only propagator and
fixed-threshold detection with a physically accurate SGP4+RK4+J2+drag pipeline
and adaptive MAD-based thresholding.

---

## Background and motivation

| Limitation in current `maneuver_detection.py` | OD approach fix |
|---|---|
| TLE mean elements treated as osculating Keplerian (wrong) | SGP4 at t=0 correctly converts mean → osculating state |
| Two-body only, no J2 or drag | RK4 integration with J2 + US Std Atm 1976 drag |
| No per-epoch drag calibration | B* from each TLE epoch used directly |
| Fixed ΔSMA/ΔEcc thresholds — must be hand-tuned | Adaptive threshold: median + N×MAD of velocity residuals |
| Residuals not normalised for gap length | Gap-normalised: vel_res_per_day = vel_residual / gap_days |
| No B* step-change signal | ΔB* used as secondary maneuver indicator |
| Kp loaded per-satellite (slow for fleet) | Kp loaded once, passed in as parameter |
| Results not written to `maneuvers` table | analyse_maneuvers() persists to DB |
| ingest.py uses simpler, separate detector | Replaced by call to upgraded analyse_maneuvers() |

---

## Pre-conditions

- `BSTAR`, `TLE_LINE1`, `TLE_LINE2` columns confirmed present in `gp_history` (verify via schema)
- `sgp4` library available in active environment (`conda run -n orbit` satisfies this)
- Space-Track credentials in `.env` (`SPACETRACK_IDENTITY`, `SPACETRACK_PASSWORD`)

---

## Phase 0 — gp_history backfill (ad-hoc, per NORAD ID)

**Goal:** Ensure the target object has sufficient TLE history in `gp_history` before
detection runs. Backfill is triggered per NORAD ID, on-demand — not fleet-wide.

**Current state:** `backfill_gp_history()` and `--backfill` CLI flag already exist in
`ingest.py` but use a hardcoded date range (`2025-12-18` to `2026-03-10`).

**Changes to `ingest.py`:**

1. Replace hardcoded dates with dynamic defaults: `--start` defaults to 90 days before
   today, `--end` defaults to today
2. Add progress logging: rows inserted vs duplicates skipped

**Changes to `analysis/maneuver_detection.py`:**

Add a coverage check at the start of `analyse_maneuvers()`. If `gp_history` has fewer
than `MIN_HISTORY_DAYS` (default 30) of coverage for the requested NORAD ID, print a
warning and prompt the user to run the backfill:

```
WARNING: Only 4 days of gp_history found for NORAD 58316.
Run: python ingest.py --backfill --norad 58316
Proceeding with available data — detection quality will be reduced.
```

The script continues with whatever history exists rather than aborting, so short-history
runs are still possible. The warning makes the limitation explicit.

**Verification:**
- `python ingest.py --backfill --norad 25544` (ISS) with no explicit dates defaults to
  90-day window and completes without error
- `SELECT COUNT(*), MIN(EPOCH), MAX(EPOCH) FROM gp_history WHERE NORAD_CAT_ID = 25544`
  returns ≥ 90 days of coverage after backfill
- `BSTAR`, `TLE_LINE1`, `TLE_LINE2` are non-null for returned rows
- Running `maneuver_detection.py --norad 58316` on a freshly initialised DB prints the
  coverage warning before proceeding

---

## Phase 1 — Schema migration

**Goal:** Add columns to `maneuvers` table to carry OD-derived signals and classification.

**Changes to `schema.sql`:**

```sql
ALTER TABLE maneuvers ADD COLUMN CLASSIFICATION  TEXT;  -- 'confirmed' | 'uncertain'
ALTER TABLE maneuvers ADD COLUMN KP              REAL;
ALTER TABLE maneuvers ADD COLUMN VEL_RESIDUAL_MS REAL;
ALTER TABLE maneuvers ADD COLUMN BSTAR_DELTA     REAL;
```

Wrap each `ALTER TABLE` in a guard so schema.sql is idempotent on re-run:

```sql
-- Example guard pattern
CREATE TABLE IF NOT EXISTS _migration_log (statement TEXT PRIMARY KEY);
INSERT OR IGNORE INTO _migration_log VALUES ('maneuvers_add_classification');
-- Run ALTER only when the INSERT above was not ignored
```

Or handle via `try/except` in the Python migration helper in `ingest.py`.

**Verification:**
- `.schema maneuvers` in sqlite3 CLI shows all four new columns
- Re-running schema initialisation does not error or duplicate columns
- `INSERT INTO maneuvers (..., CLASSIFICATION, KP) VALUES (..., 'confirmed', 3.2)` succeeds

---

## Phase 2 — config.py: replace detection thresholds

**Goal:** Replace element-diff thresholds with OD detection parameters.

**Changes to `config.py`:**

```python
# OD-based maneuver detection parameters
OD_SIGMA_MULTIPLIER    = 5.0    # threshold = median + N × MAD of vel residuals
OD_KP_THRESHOLD        = 5.0    # Kp above which space weather is 'disturbed'
OD_MAX_GAP_DAYS        = 10.0   # skip epoch pairs with larger gaps
OD_BSTAR_NOISE_MAX     = 1e-3   # skip TLEs with |B*| above this (poor fit)
OD_BSTAR_DELTA_THRESH  = 5e-5   # B* step-change secondary signal threshold
```

`MANEUVER_THRESHOLDS` is retained until Phase 6 removes the legacy `detect_maneuvers()`
from `ingest.py`.

**Verification:**
- `from config import OD_SIGMA_MULTIPLIER` succeeds in both `ingest.py` and
  `analysis/maneuver_detection.py` contexts
- No import errors from existing code that still references `MANEUVER_THRESHOLDS`

---

## Phase 3 — Propagator upgrade

**Goal:** Replace `kepler_position()` and `propagate_kepler()` with the SGP4+RK4+J2+drag
propagator from `orbit_analysis_od.py`.

**Changes to `analysis/maneuver_detection.py`:**

**3a.** Extend `get_gp_history()` SELECT to include `BSTAR`, `TLE_LINE1`, `TLE_LINE2`.

**3b.** Remove `kepler_position()` and `propagate_kepler()`.

**3c.** Port in from `orbit_analysis_od.py` (minimal adaptation — remove Space-Track
fetch logic, keep physics):

| Function | Purpose |
|---|---|
| `_sgp4_state(tle1, tle2, dt_sec)` | SGP4 at arbitrary t; dt=0 gives osculating state at epoch |
| `_atmo_density(alt_km)` | US Standard Atmosphere 1976, piecewise exponential |
| `_bstar_to_drag(bstar)` | Convert B* (1/ER) to physical drag coefficient (m²/kg) |
| `_derivatives(t, state, drag_coeff)` | EOM: two-body + J2 + atmospheric drag |
| `propagate_od(tle1, tle2, bstar, dt_sec)` | RK4 integration wrapping the above |

**3d.** Credential handling: `config.py` import already calls `load_dotenv()` so
`os.environ` is populated. No additional change required.

**Verification:**
- Propagate ISS (NORAD 25544) forward 1 orbit (~92 min) and confirm position error
  vs SGP4 at target epoch is < 1 km (two-body Kepler would give > 5 km)
- Unit test: `propagate_od` with dt=0 returns the same state as `_sgp4_state` at epoch
- No `ImportError` for `sgp4` — run under `conda run -n orbit`

---

## Phase 4 — Detection signal overhaul

**Goal:** Replace element-diff flagging with velocity residual + adaptive MAD threshold
+ B* secondary signal.

**Changes to `analyse_maneuvers()` in `analysis/maneuver_detection.py`:**

**4a. B* noise pre-filter** — before building consecutive pairs:
```python
df = df[df["BSTAR"].abs() <= OD_BSTAR_NOISE_MAX]
```

**4b. Per-pair computation** — replaces current loop body:
1. `propagate_od()` from epoch₁ → epoch₂ using TLE₁
2. `_sgp4_state()` at epoch₂ with dt=0 for actual state
3. Compute `pos_residual_km`, `vel_residual_ms`
4. Compute `vel_res_per_day = vel_residual_ms / gap_days`
5. Record `delta_sma`, `delta_ecc`, `delta_inc` as informational (not triggers)
6. Record `bstar` and `delta_bstar = abs(bstar_now - bstar_prev)`

**4c. Adaptive threshold** — after all pairs are computed:
```python
median_vr = np.median(vel_res_per_day)
mad_vr    = np.median(np.abs(vel_res_per_day - median_vr))
threshold = median_vr + sigma_multiplier * mad_vr
```

**4d. Classification flags:**
```python
od_flag    = vel_res_per_day > threshold
bstar_flag = delta_bstar > OD_BSTAR_DELTA_THRESH
likely     = od_flag | bstar_flag
confirmed  = likely & ~bad_space_weather
uncertain  = likely &  bad_space_weather
```

**Verification:**
- ISS (25544) shows confirmed maneuver detections on known maneuver dates
  (cross-reference against public NASA maneuver records or reboost logs)
- A debris object with no maneuvering capability (e.g. a spent rocket body) shows
  zero confirmed maneuvers over the same period
- `od_flag` and `bstar_flag` can independently trigger — verify at least one case
  where only `bstar_flag` fires (B* jump with marginal velocity residual)

---

## Phase 5 — Kp decoupling

**Goal:** Load Kp once per fleet run, pass as parameter to `analyse_maneuvers()`.

**Changes to `analysis/maneuver_detection.py`:**

Update `analyse_maneuvers()` signature:

```python
def analyse_maneuvers(
    norad_id: int,
    kp_df: pd.DataFrame | None = None,   # pre-loaded; fetched internally if None
    data_dir: Path = ...,
    sigma_multiplier: float = OD_SIGMA_MULTIPLIER,
    kp_threshold: float = OD_KP_THRESHOLD,
) -> pd.DataFrame:
```

When `kp_df` is `None`, function falls back to calling `get_kp_index()` internally
(preserves existing single-satellite CLI behaviour).

**Verification:**
- Single-satellite CLI call (`python analysis/maneuver_detection.py --norad 25544`)
  still works without passing `kp_df`
- Fleet loop calling `analyse_maneuvers()` 10 times with pre-loaded `kp_df` makes
  exactly 1 network/cache read for Kp data (confirmed via log output)

---

## Phase 6 — DB write-back and ingest.py integration

**Goal:** Persist detection results to `maneuvers` table; replace legacy
`detect_maneuvers()` in `ingest.py`.

**6a. Write-back in `analyse_maneuvers()`:**

```python
con.execute(
    """INSERT OR IGNORE INTO maneuvers
       (NORAD_CAT_ID, EPOCH_BEFORE, EPOCH_AFTER,
        DELTA_SMA, DELTA_ECCENTRICITY, DELTA_INCLINATION,
        DELTA_PERIOD, DELTA_APOAPSIS, DELTA_PERIAPSIS,
        CLASSIFICATION, KP, VEL_RESIDUAL_MS, BSTAR_DELTA)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (row values...)
)
```

`UNIQUE(NORAD_CAT_ID, EPOCH_BEFORE, EPOCH_AFTER)` handles deduplication on re-runs.
Only rows where `confirmed` or `uncertain` is True are written.

**6b. Replace `detect_maneuvers()` in `ingest.py`:**

```python
from analysis.maneuver_detection import analyse_maneuvers, get_kp_index

def detect_maneuvers(con, norad_ids):
    window_start = ...   # min EPOCH of current update batch
    window_end   = ...   # max EPOCH of current update batch
    kp_df = get_kp_index(window_start, window_end, DATA_DIR)
    for nid in norad_ids:
        analyse_maneuvers(nid, con=con, kp_df=kp_df)
```

`MANEUVER_THRESHOLDS` and the old `_ELEMENT_MAP` / loop in `ingest.py` are removed.

**Verification:**
- Run `python ingest.py --update` end-to-end; confirm `maneuvers` table is populated
  with `CLASSIFICATION` values (`'confirmed'` or `'uncertain'`) — no NULL classification rows
- `python ingest.py --maneuvers` displays results correctly
- Re-running `--update` with no new GP data does not create duplicate rows in `maneuvers`
- `SELECT COUNT(*) FROM maneuvers WHERE CLASSIFICATION IS NULL` returns 0

---

## Phase 7 — Plot update

**Goal:** Replace 4-panel plot with 7-panel layout matching `orbit_analysis_od.py`.

**Panel layout:**

| Panel | Signal | Scale |
|---|---|---|
| 1 | Gap-normalised velocity residual with adaptive threshold line | log |
| 2 | Raw velocity residual (m/s) | linear |
| 3 | B* history | linear |
| 4 | ΔB* step-changes with threshold bounds | linear |
| 5 | ΔSMA (km) | linear |
| 6 | ΔEccentricity | linear |
| 7 | Kp index with disturbed region fill | linear |

Confirmed maneuvers: red circles. Uncertain maneuvers: orange triangles.
Plot generation remains off in the monitoring path; on-demand via `--plot` flag.

**Verification:**
- `python analysis/maneuver_detection.py --norad 25544` produces a 7-panel PNG
  in `plots/`
- Adaptive threshold line is visible in panel 1 and correctly excludes non-maneuver
  baseline noise
- At least one confirmed/uncertain event is marked on panels where the signal fired

---

## Gating verification: output comparison against orbit_analysis_od.py

Before merging to main, run both scripts on the same NORAD ID and time window and
compare outputs. The OD script pulls directly from Space-Track so it may have denser
TLE history; results should agree on maneuver classification for the overlapping epochs.

**Test object:** ISS (NORAD 25544) — frequent, well-documented maneuvers.

**Procedure:**

1. Run `orbit_analysis_od.py` with ISS credentials → capture `confirmed_maneuver` dates
2. Ensure `gp_history` covers the same window via backfill (Phase 0)
3. Run upgraded `maneuver_detection.py --norad 25544` over same window
4. Compare epoch pairs flagged as confirmed in both outputs

**Acceptance criteria:**

| Metric | Threshold |
|---|---|
| Confirmed maneuver epoch agreement | ≥ 80% overlap on flagged epoch pairs |
| False positives in our version (flagged but not in OD output) | ≤ 2× OD false positive rate |
| Zero confirmed detections on inert debris object (e.g. NORAD 20580) | Must hold in both |
| Position error at t+1hr (propagator unit test) | < 2 km vs SGP4 reference |
| Velocity residual at t+1hr | < 5 m/s vs SGP4 reference |

Differences are expected where `gp_history` is sparser than Space-Track's full archive.
Flag those gaps as a known limitation, not a failure.

---

## What is NOT changing

- `get_kp_index()` core logic and GFZ Potsdam source
- `gp_history` as the authoritative data source (no live Space-Track pull per analysis run)
- `confirmed` / `uncertain` classification semantics
- `query.py` helpers: `get_maneuvers()`, `get_maneuvering_objects()`, `get_maneuver_history()`
- `--maneuvers` CLI flag in `ingest.py`
- 10-day gap filter

---

## Phase execution order

```
Phase 0 (backfill)
    │
Phase 1 (schema)  ←─── Phase 2 (config)
    │                       │
    └──────────┬────────────┘
               │
           Phase 3 (propagator)
               │
           Phase 4 (detection signals)
               │
           Phase 5 (Kp decouple)
               │
           Phase 6 (write-back + ingest integration)
               │
           Phase 7 (plot)
               │
           Gating verification
```

Phases 0, 1, and 2 are independent and can proceed in parallel.
Phases 3–7 must be sequential.
