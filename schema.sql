-- Satellite Catalog Database Schema

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Latest GP element sets (one row per object)
CREATE TABLE IF NOT EXISTS gp (
    GP_ID               INTEGER PRIMARY KEY,
    NORAD_CAT_ID        INTEGER NOT NULL UNIQUE,
    OBJECT_NAME         TEXT,
    OBJECT_ID           TEXT,
    EPOCH               TEXT,
    MEAN_MOTION         REAL,
    ECCENTRICITY        REAL,
    INCLINATION         REAL,
    RA_OF_ASC_NODE      REAL,
    ARG_OF_PERICENTER   REAL,
    MEAN_ANOMALY        REAL,
    EPHEMERIS_TYPE      INTEGER,
    CLASSIFICATION_TYPE TEXT,
    ELEMENT_SET_NO      INTEGER,
    REV_AT_EPOCH        INTEGER,
    BSTAR               REAL,
    MEAN_MOTION_DOT     REAL,
    MEAN_MOTION_DDOT    REAL,
    SEMIMAJOR_AXIS      REAL,
    PERIOD              REAL,
    APOAPSIS            REAL,
    PERIAPSIS           REAL,
    OBJECT_TYPE         TEXT,
    RCS_SIZE            TEXT,
    COUNTRY_CODE        TEXT,
    LAUNCH_DATE         TEXT,
    SITE                TEXT,
    DECAY_DATE          TEXT,
    FILE                INTEGER,
    GP_ID_ORIG          INTEGER,
    TLE_LINE0           TEXT,
    TLE_LINE1           TEXT,
    TLE_LINE2           TEXT,
    ingested_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Historical GP element sets (no unique constraint on NORAD_CAT_ID)
CREATE TABLE IF NOT EXISTS gp_history (
    GP_ID               INTEGER PRIMARY KEY,
    NORAD_CAT_ID        INTEGER NOT NULL,
    OBJECT_NAME         TEXT,
    OBJECT_ID           TEXT,
    EPOCH               TEXT,
    MEAN_MOTION         REAL,
    ECCENTRICITY        REAL,
    INCLINATION         REAL,
    RA_OF_ASC_NODE      REAL,
    ARG_OF_PERICENTER   REAL,
    MEAN_ANOMALY        REAL,
    EPHEMERIS_TYPE      INTEGER,
    CLASSIFICATION_TYPE TEXT,
    ELEMENT_SET_NO      INTEGER,
    REV_AT_EPOCH        INTEGER,
    BSTAR               REAL,
    MEAN_MOTION_DOT     REAL,
    MEAN_MOTION_DDOT    REAL,
    SEMIMAJOR_AXIS      REAL,
    PERIOD              REAL,
    APOAPSIS            REAL,
    PERIAPSIS           REAL,
    OBJECT_TYPE         TEXT,
    RCS_SIZE            TEXT,
    COUNTRY_CODE        TEXT,
    LAUNCH_DATE         TEXT,
    SITE                TEXT,
    DECAY_DATE          TEXT,
    FILE                INTEGER,
    GP_ID_ORIG          INTEGER,
    TLE_LINE0           TEXT,
    TLE_LINE1           TEXT,
    TLE_LINE2           TEXT,
    ingested_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Satellite catalog metadata
CREATE TABLE IF NOT EXISTS satcat (
    NORAD_CAT_ID    INTEGER PRIMARY KEY,
    SATNAME         TEXT,
    OBJECT_NAME     TEXT,
    OBJECT_ID       TEXT,
    INTLDES         TEXT,
    OBJECT_TYPE     TEXT,
    COUNTRY         TEXT,
    LAUNCH          TEXT,
    SITE            TEXT,
    DECAY           TEXT,
    PERIOD          REAL,
    INCLINATION     REAL,
    APOGEE          INTEGER,
    PERIGEE         INTEGER,
    RCS_SIZE        TEXT,
    CURRENT         TEXT,
    COMMENT         TEXT,
    FILE            INTEGER,
    LAUNCH_YEAR     INTEGER,
    LAUNCH_NUM      INTEGER,
    LAUNCH_PIECE    TEXT,
    RCSVALUE        REAL,
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Sync metadata
CREATE TABLE IF NOT EXISTS sync_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- Indexes for conjunction prediction queries
CREATE INDEX IF NOT EXISTS idx_gp_norad       ON gp (NORAD_CAT_ID);
CREATE INDEX IF NOT EXISTS idx_gp_epoch       ON gp (EPOCH);
CREATE INDEX IF NOT EXISTS idx_gp_altitude    ON gp (APOAPSIS, PERIAPSIS);
CREATE INDEX IF NOT EXISTS idx_gp_hist_norad  ON gp_history (NORAD_CAT_ID);
CREATE INDEX IF NOT EXISTS idx_gp_hist_epoch  ON gp_history (EPOCH);

CREATE INDEX IF NOT EXISTS idx_satcat_type    ON satcat (OBJECT_TYPE);
CREATE INDEX IF NOT EXISTS idx_satcat_country ON satcat (COUNTRY);
CREATE INDEX IF NOT EXISTS idx_satcat_current ON satcat (CURRENT);

-- Convenience view joining GP and SATCAT
CREATE VIEW IF NOT EXISTS catalog_view AS
SELECT
    g.NORAD_CAT_ID,
    g.OBJECT_NAME,
    g.OBJECT_ID,
    g.EPOCH,
    g.MEAN_MOTION,
    g.ECCENTRICITY,
    g.INCLINATION   AS GP_INCLINATION,
    g.RA_OF_ASC_NODE,
    g.ARG_OF_PERICENTER,
    g.MEAN_ANOMALY,
    g.BSTAR,
    g.SEMIMAJOR_AXIS,
    g.PERIOD        AS GP_PERIOD,
    g.APOAPSIS,
    g.PERIAPSIS,
    g.OBJECT_TYPE   AS GP_OBJECT_TYPE,
    g.RCS_SIZE      AS GP_RCS_SIZE,
    g.COUNTRY_CODE,
    g.DECAY_DATE,
    g.TLE_LINE0,
    g.TLE_LINE1,
    g.TLE_LINE2,
    s.SATNAME,
    s.INTLDES,
    s.OBJECT_TYPE   AS SAT_OBJECT_TYPE,
    s.COUNTRY,
    s.LAUNCH,
    s.SITE,
    s.DECAY,
    s.APOGEE,
    s.PERIGEE,
    s.RCS_SIZE      AS SAT_RCS_SIZE,
    s.CURRENT,
    g.ingested_at   AS gp_ingested_at,
    s.ingested_at   AS sat_ingested_at
FROM gp g
LEFT JOIN satcat s ON g.NORAD_CAT_ID = s.NORAD_CAT_ID;
