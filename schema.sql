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

-- Detected maneuvers (derived from consecutive GP epoch comparisons)
CREATE TABLE IF NOT EXISTS maneuvers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    NORAD_CAT_ID        INTEGER NOT NULL,
    EPOCH_BEFORE        TEXT NOT NULL,
    EPOCH_AFTER         TEXT NOT NULL,
    DELTA_SMA           REAL,    -- km
    DELTA_ECCENTRICITY  REAL,
    DELTA_INCLINATION   REAL,    -- degrees
    DELTA_RAAN          REAL,    -- degrees
    DELTA_PERIOD        REAL,    -- minutes
    DELTA_APOAPSIS      REAL,    -- km
    DELTA_PERIAPSIS     REAL,    -- km
    CLASSIFICATION      TEXT,    -- 'confirmed' | 'uncertain'
    KP                  REAL,
    VEL_RESIDUAL_MS     REAL,
    BSTAR_DELTA         REAL,
    detected_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(NORAD_CAT_ID, EPOCH_BEFORE, EPOCH_AFTER)
);

-- Idempotent migration: add OD columns to existing maneuvers tables
-- (safe to re-run; errors from duplicate columns are ignored in Python)
-- Migration is handled via try/except in ingest.py init_db().

-- Ground-based SSA sensor network
CREATE TABLE IF NOT EXISTS sensor (
    SENSOR_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
    NAME            TEXT NOT NULL,
    COUNTRY         TEXT,
    ORGANISATION    TEXT,
    ORG_TYPE        TEXT,           -- Civil, Military, Research
    TYPE            TEXT NOT NULL,  -- OPTICAL, RADAR, PHASED_ARRAY_RADAR, LASER
    LAT             REAL NOT NULL,  -- degrees
    LON             REAL NOT NULL,  -- degrees
    ALT_M           REAL NOT NULL DEFAULT 0,  -- metres above WGS84 ellipsoid
    MIN_EL_DEG      REAL NOT NULL DEFAULT 10, -- minimum elevation (horizon mask)
    MAX_EL_DEG      REAL NOT NULL DEFAULT 90,
    AZ_MIN_DEG      REAL DEFAULT -180,
    AZ_MAX_DEG      REAL DEFAULT 180,
    ORIENTATION_DEG REAL DEFAULT 0,
    MIN_RANGE_KM    REAL NOT NULL DEFAULT 200,
    MAX_RANGE_KM    REAL NOT NULL,            -- sensor-dependent detection limit
    SENSITIVITY     REAL,          -- min RCS (m²) for radar; limiting mag for optical/laser
    REMARKS         TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(NAME, LAT, LON)
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

CREATE INDEX IF NOT EXISTS idx_sensor_type    ON sensor (TYPE);
CREATE INDEX IF NOT EXISTS idx_sensor_country ON sensor (COUNTRY);

CREATE INDEX IF NOT EXISTS idx_maneuvers_norad ON maneuvers (NORAD_CAT_ID);
CREATE INDEX IF NOT EXISTS idx_maneuvers_epoch ON maneuvers (EPOCH_AFTER);

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
