"""
compare_tle_epochs_60547.py
----------------------------
Compares TLE epoch lists for NORAD 60547 between:
  Source 1: Reference TLE text file (3-line format)
  Source 2: SQLite gp_history table

Run with: conda run -n orbit python analysis/compare_tle_epochs_60547.py
"""

import sqlite3
import re
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────
TLE_FILE = (
    r"C:\Users\gabri\OneDrive - Adroitly Consulting Private Limited"
    r"\ADROITLY\OSTIN\SSA\Analysis\SpaceSituationalAnalysis\data\tle_history_60547.txt"
)
DB_PATH = (
    r"C:\Users\gabri\OneDrive - Adroitly Consulting Private Limited"
    r"\ADROITLY\OSTIN\SSA\Analysis\SATCAT\satcat.db"
)

TOLERANCE_SEC = 1  # match tolerance in seconds


# ── TLE epoch parser ───────────────────────────────────────────────────────────
def parse_tle_epoch(line1: str) -> datetime:
    """Parse epoch datetime from TLE line1 columns 18-32 (YYDDD.DDDDDDDD)."""
    epoch_str = line1[18:32].strip()
    year = int(epoch_str[:2])
    year += 2000 if year < 57 else 1900
    day_frac = float(epoch_str[2:])
    return datetime(year, 1, 1) + timedelta(days=day_frac - 1)


# ── Source 1: Parse reference TLE text file ────────────────────────────────────
print("=" * 70)
print("SOURCE 1: Reference TLE text file")
print(f"  {TLE_FILE}")
print("=" * 70)

with open(TLE_FILE, "r") as f:
    raw = f.read()

lines = [l.strip() for l in raw.splitlines() if l.strip()]
file_tles = []
i = 0
while i < len(lines):
    # 3-line format: name, line1, line2
    if (
        not lines[i].startswith("1 ")
        and not lines[i].startswith("2 ")
        and i + 2 < len(lines)
        and lines[i + 1].startswith("1 ")
        and lines[i + 2].startswith("2 ")
    ):
        name  = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        i += 3
    # 2-line format: line1, line2
    elif lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
        line1 = lines[i]
        line2 = lines[i + 1]
        name  = f"CATNR {line1[2:7].strip()}"
        i += 2
    else:
        i += 1
        continue

    try:
        epoch = parse_tle_epoch(line1)
        file_tles.append({"name": name, "line1": line1, "line2": line2, "epoch": epoch})
    except Exception as e:
        print(f"  [WARN] Failed to parse epoch from: {line1[:30]}... — {e}")

file_epochs = [t["epoch"] for t in file_tles]
file_epoch_set = sorted(set(file_epochs))

print(f"  Total TLE records parsed    : {len(file_tles)}")
print(f"  Unique epochs               : {len(file_epoch_set)}")

# Duplicate epochs in file
from collections import Counter
file_epoch_counts = Counter(file_epochs)
file_dups = {e: c for e, c in file_epoch_counts.items() if c > 1}
if file_dups:
    print(f"  Duplicate epochs ({len(file_dups)} unique epochs appear >1 time):")
    for ep, cnt in sorted(file_dups.items()):
        print(f"    {ep.strftime('%Y-%m-%dT%H:%M:%S.%f')}  x{cnt}")
else:
    print("  No duplicate epochs in reference TLE file.")

if file_epochs:
    print(f"  Earliest epoch: {min(file_epochs).strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"  Latest epoch  : {max(file_epochs).strftime('%Y-%m-%dT%H:%M:%S')}")


# ── Source 2: Query SQLite gp_history ─────────────────────────────────────────
print()
print("=" * 70)
print("SOURCE 2: SQLite gp_history table")
print(f"  {DB_PATH}")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.execute(
    "SELECT EPOCH, GP_ID, TLE_LINE1 FROM gp_history WHERE NORAD_CAT_ID = 60547 ORDER BY EPOCH ASC"
)
rows = cur.fetchall()
conn.close()

db_records = []
for epoch_str, gp_id, tle_line1 in rows:
    # EPOCH is stored as ISO 8601 string in the DB
    # Try parsing with and without fractional seconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            ep = datetime.strptime(epoch_str, fmt)
            break
        except ValueError:
            ep = None
    if ep is None:
        print(f"  [WARN] Cannot parse DB epoch: {epoch_str!r}")
        continue
    db_records.append({"epoch": ep, "gp_id": gp_id, "tle_line1": tle_line1})

db_epochs = [r["epoch"] for r in db_records]
print(f"  Total rows in gp_history    : {len(db_records)}")
print(f"  Unique epochs               : {len(set(db_epochs))}")

# Duplicate GP_IDs
gp_id_counts = Counter(r["gp_id"] for r in db_records)
gp_id_dups = {g: c for g, c in gp_id_counts.items() if c > 1}
if gp_id_dups:
    print(f"  Duplicate GP_IDs ({len(gp_id_dups)}):")
    for gp_id, cnt in sorted(gp_id_dups.items()):
        print(f"    GP_ID={gp_id}  x{cnt}")
else:
    print("  No duplicate GP_IDs in gp_history for NORAD 60547.")

if db_epochs:
    print(f"  Earliest epoch: {min(db_epochs).strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"  Latest epoch  : {max(db_epochs).strftime('%Y-%m-%dT%H:%M:%S')}")


# ── Epoch matching with tolerance ─────────────────────────────────────────────
print()
print("=" * 70)
print(f"COMPARISON  (tolerance = {TOLERANCE_SEC}s)")
print("=" * 70)

tol = timedelta(seconds=TOLERANCE_SEC)

def epochs_match(e1: datetime, e2: datetime) -> bool:
    return abs((e1 - e2).total_seconds()) <= TOLERANCE_SEC

# For each file epoch, find if there is a matching DB epoch
file_matched   = []
file_unmatched = []
for fe in file_epochs:
    if any(epochs_match(fe, de) for de in db_epochs):
        file_matched.append(fe)
    else:
        file_unmatched.append(fe)

# For each DB epoch, find if there is a matching file epoch
db_matched   = []
db_unmatched = []
for de in db_epochs:
    if any(epochs_match(de, fe) for fe in file_epochs):
        db_matched.append(de)
    else:
        db_unmatched.append(de)

print(f"  File TLEs total             : {len(file_epochs)}")
print(f"  DB records total            : {len(db_epochs)}")
print(f"  File epochs matched in DB   : {len(file_matched)}")
print(f"  Overlap (unique epochs)     : {len(set(file_matched))}")
print()
print(f"  *** Missing from DB ({len(file_unmatched)} records in file not in DB) ***")
if file_unmatched:
    for ep in sorted(set(file_unmatched)):
        cnt = file_unmatched.count(ep)
        suffix = f" (x{cnt})" if cnt > 1 else ""
        print(f"    {ep.strftime('%Y-%m-%dT%H:%M:%S.%f')}{suffix}")
else:
    print("    None — all file epochs are present in DB.")

print()
print(f"  *** Missing from reference file ({len(db_unmatched)} DB records not in file) ***")
if db_unmatched:
    for ep in sorted(set(db_unmatched)):
        cnt = db_unmatched.count(ep)
        suffix = f" (x{cnt})" if cnt > 1 else ""
        print(f"    {ep.strftime('%Y-%m-%dT%H:%M:%S.%f')}{suffix}")
else:
    print("    None — all DB epochs are present in reference file.")


# ── API endpoint comparison ────────────────────────────────────────────────────
print()
print("=" * 70)
print("API ENDPOINT COMPARISON")
print("=" * 70)
print()
print("Reference script (orbit_analysis_od.py) — _download_tle_history:")
print("  Class  : gp_history")
print("  URL    : https://www.space-track.org/basicspacedata/query/class/gp_history")
print("           /NORAD_CAT_ID/{catnr}/orderby/TLE_LINE1 ASC/format/tle")
print("  Format : format/tle  → raw 3-line TLE text")
print("  Library: requests (direct HTTP, manual session/cookie)")
print()
print("Our ingest.py — via spacetrack library:")
print("  Class  : gp_history  (same class)")
print("  Format : format=json (JSON records, not raw TLE text)")
print("  Library: spacetrack v1.3.1 (handles rate-limiting, auth automatically)")
print()
print("Key difference: SAME Space-Track class (gp_history), but different")
print("  response format — orbit_analysis_od.py fetches raw TLE text (format/tle),")
print("  while ingest.py fetches JSON objects and stores them as structured DB rows.")
print("  Both should yield the same set of element epochs for the same NORAD ID.")
print()
print("Done.")
