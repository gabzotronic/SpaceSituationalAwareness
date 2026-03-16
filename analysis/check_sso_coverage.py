"""Check SSO gp_history coverage for 2020-2022 RPO discovery window."""
import sqlite3, sys, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

con = sqlite3.connect(DB_PATH)

total_sso = con.execute("""
    SELECT COUNT(*) FROM gp
    WHERE PERIAPSIS >= 400 AND APOAPSIS <= 600
      AND INCLINATION BETWEEN 95 AND 100
""").fetchone()[0]

with_data = con.execute("""
    SELECT COUNT(DISTINCT h.NORAD_CAT_ID) FROM gp_history h
    JOIN gp g ON h.NORAD_CAT_ID = g.NORAD_CAT_ID
    WHERE g.PERIAPSIS >= 400 AND g.APOAPSIS <= 600
      AND g.INCLINATION BETWEEN 95 AND 100
      AND h.EPOCH >= '2020-01-01' AND h.EPOCH <= '2022-12-31'
""").fetchone()[0]

total_records = con.execute("""
    SELECT COUNT(*) FROM gp_history h
    JOIN gp g ON h.NORAD_CAT_ID = g.NORAD_CAT_ID
    WHERE g.PERIAPSIS >= 400 AND g.APOAPSIS <= 600
      AND g.INCLINATION BETWEEN 95 AND 100
      AND h.EPOCH >= '2020-01-01' AND h.EPOCH <= '2022-12-31'
""").fetchone()[0]

print(f"SSO objects in gp (400-600km, inc 95-100):  {total_sso}")
print(f"Objects with 2020-2022 gp_history data:     {with_data}")
print(f"Missing:                                     {total_sso - with_data}")
print(f"Total records in window:                     {total_records:,}")

rows = con.execute("""
    SELECT h.NORAD_CAT_ID, COUNT(*) as cnt
    FROM gp_history h
    JOIN gp g ON h.NORAD_CAT_ID = g.NORAD_CAT_ID
    WHERE g.PERIAPSIS >= 400 AND g.APOAPSIS <= 600
      AND g.INCLINATION BETWEEN 95 AND 100
      AND h.EPOCH >= '2020-01-01' AND h.EPOCH <= '2022-12-31'
    GROUP BY h.NORAD_CAT_ID
    ORDER BY cnt
""").fetchall()

if rows:
    counts = [r[1] for r in rows]
    print(f"\nRecords per object: min={min(counts)}, median={statistics.median(counts):.0f}, max={max(counts)}")
    buckets = [(1, 10), (11, 50), (51, 200), (201, 500), (501, 2000), (2001, 9999)]
    print("Distribution:")
    for lo, hi in buckets:
        n = sum(1 for c in counts if lo <= c <= hi)
        print(f"  {lo:>5}-{hi:<5}: {n:>4} objects")

# Known RPO objects
known = [
    ("Cosmos 2542", 44876),
    ("Cosmos 2543", 44878),
    ("USA-245",     37348),
    ("Cosmos 2558", 52994),
    ("USA-326",     49044),
]
print("\nKnown RPO objects:")
for name, norad in known:
    cnt = con.execute(
        "SELECT COUNT(*) FROM gp_history WHERE NORAD_CAT_ID=? AND EPOCH >= '2020-01-01' AND EPOCH <= '2022-12-31'",
        (norad,)
    ).fetchone()[0]
    gp_row = con.execute(
        "SELECT PERIAPSIS, APOAPSIS, INCLINATION, OBJECT_NAME FROM gp WHERE NORAD_CAT_ID=?", (norad,)
    ).fetchone()
    orbit = f"perigee={gp_row[0]:.0f} apogee={gp_row[1]:.0f} inc={gp_row[2]:.1f}" if gp_row else "NOT IN GP"
    obj_name = gp_row[3] if gp_row else "?"
    print(f"  {name} ({norad}) [{obj_name}]: {cnt} records | {orbit}")

con.close()
