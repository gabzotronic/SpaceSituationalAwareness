"""Check orbital parameters for ADRAS-J and H-IIA R/B."""
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

con = sqlite3.connect(DB_PATH)

for name, norad in [("ADRAS-J", 58992), ("H-IIA R/B", 33500)]:
    row = con.execute("""
        SELECT OBJECT_NAME, PERIAPSIS, APOAPSIS, INCLINATION, RA_OF_ASC_NODE, SEMIMAJOR_AXIS
        FROM gp WHERE NORAD_CAT_ID = ?
    """, (norad,)).fetchone()
    print(f"{name} ({norad}): {row}")

    cnt = con.execute("""
        SELECT COUNT(*), MIN(EPOCH), MAX(EPOCH) FROM gp_history
        WHERE NORAD_CAT_ID = ? AND EPOCH >= '2024-01-01' AND EPOCH <= '2024-12-31'
    """, (norad,)).fetchone()
    print(f"  gp_history 2024: {cnt[0]} records  ({cnt[1]} → {cnt[2]})")

con.close()
