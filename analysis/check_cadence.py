"""Quick diagnostic: TLE cadence in gp_history for GEO objects."""
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

con = sqlite3.connect(DB_PATH)

total_geo = con.execute(
    "SELECT COUNT(*) FROM gp WHERE PERIAPSIS >= 35000 AND APOAPSIS <= 37000"
).fetchone()[0]

with_data = con.execute("""
    SELECT COUNT(DISTINCT NORAD_CAT_ID) FROM gp_history
    WHERE NORAD_CAT_ID IN (SELECT NORAD_CAT_ID FROM gp WHERE PERIAPSIS >= 35000 AND APOAPSIS <= 37000)
      AND EPOCH >= '2016-01-01' AND EPOCH <= '2025-12-31'
""").fetchone()[0]

print(f"GEO objects total : {total_geo}")
print(f"Objects with data : {with_data}")
print(f"Objects missing   : {total_geo - with_data}")

rows = con.execute("""
    SELECT NORAD_CAT_ID, COUNT(*) AS cnt
    FROM gp_history
    WHERE NORAD_CAT_ID IN (SELECT NORAD_CAT_ID FROM gp WHERE PERIAPSIS >= 35000 AND APOAPSIS <= 37000)
      AND EPOCH >= '2016-01-01' AND EPOCH <= '2025-12-31'
    GROUP BY NORAD_CAT_ID
    ORDER BY cnt
""").fetchall()

counts = [r[1] for r in rows]
print(f"\nRecords per object:")
print(f"  min    = {min(counts)}")
print(f"  median = {statistics.median(counts):.0f}")
print(f"  mean   = {statistics.mean(counts):.0f}")
print(f"  max    = {max(counts)}")

print(f"\nDistribution:")
buckets = [(1, 10), (11, 50), (51, 200), (201, 500), (501, 2000), (2001, 9999)]
for lo, hi in buckets:
    n = sum(1 for c in counts if lo <= c <= hi)
    bar = "#" * (n // 5)
    print(f"  {lo:>5}-{hi:<5}: {n:>4} objects  {bar}")

# Objects with very few records — likely API rate-limit casualties
sparse = [(r[0], r[1]) for r in rows if r[1] < 10]
print(f"\nObjects with <10 records (possible API limit): {len(sparse)}")
for norad, cnt in sparse[:20]:
    name = con.execute(
        "SELECT OBJECT_NAME FROM gp WHERE NORAD_CAT_ID = ?", (norad,)
    ).fetchone()
    name = name[0] if name else "?"
    print(f"  NORAD {norad:>7}  {name:<30}  {cnt} records")

# Also check: what's the avg gap between consecutive TLEs for a sample object
sample_norad = rows[len(rows) // 2][0]  # median-count object
epochs = con.execute("""
    SELECT EPOCH FROM gp_history
    WHERE NORAD_CAT_ID = ? AND EPOCH >= '2016-01-01' AND EPOCH <= '2025-12-31'
    ORDER BY EPOCH
""", (sample_norad,)).fetchall()
if len(epochs) >= 2:
    from datetime import datetime
    dts = [datetime.fromisoformat(e[0].replace('Z','')) for e in epochs]
    gaps = [(dts[i+1]-dts[i]).total_seconds()/86400 for i in range(len(dts)-1)]
    print(f"\nSample object NORAD {sample_norad}: {len(epochs)} TLEs, "
          f"avg gap {statistics.mean(gaps):.1f} days, "
          f"max gap {max(gaps):.1f} days")

con.close()
