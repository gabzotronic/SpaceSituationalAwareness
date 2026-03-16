"""Check current backfill progress for SSO band."""
import sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

con = sqlite3.connect(DB_PATH)

total_sso = con.execute("""
    SELECT COUNT(*) FROM gp
    WHERE PERIAPSIS >= 400 AND APOAPSIS <= 600
      AND INCLINATION BETWEEN 95 AND 100
""").fetchone()[0]

with_any = con.execute("""
    SELECT COUNT(DISTINCT h.NORAD_CAT_ID) FROM gp_history h
    JOIN gp g ON h.NORAD_CAT_ID = g.NORAD_CAT_ID
    WHERE g.PERIAPSIS >= 400 AND g.APOAPSIS <= 600
      AND g.INCLINATION BETWEEN 95 AND 100
""").fetchone()[0]

total_records = con.execute("""
    SELECT COUNT(*) FROM gp_history h
    JOIN gp g ON h.NORAD_CAT_ID = g.NORAD_CAT_ID
    WHERE g.PERIAPSIS >= 400 AND g.APOAPSIS <= 600
      AND g.INCLINATION BETWEEN 95 AND 100
""").fetchone()[0]

# Most recently ingested records
recent = con.execute("""
    SELECT MAX(h.ingested_at) FROM gp_history h
    JOIN gp g ON h.NORAD_CAT_ID = g.NORAD_CAT_ID
    WHERE g.PERIAPSIS >= 400 AND g.APOAPSIS <= 600
      AND g.INCLINATION BETWEEN 95 AND 100
""").fetchone()[0]

print(f"SSO objects total:     {total_sso}")
print(f"Objects with any data: {with_any}  ({with_any/total_sso*100:.1f}%)")
print(f"Objects remaining:     {total_sso - with_any}")
print(f"Total records:         {total_records:,}")
print(f"Last ingestion:        {recent}")

con.close()
