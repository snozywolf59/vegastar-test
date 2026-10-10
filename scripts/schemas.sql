CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
-- ---------- vessels ----------
CREATE TABLE IF NOT EXISTS vessels (
    vessel_id             TEXT PRIMARY KEY,
    mmsi                  TEXT,
    imo                   TEXT,
    callsign              TEXT,
    shipname              TEXT,
    flag_code             TEXT,
    flag                  TEXT,
    ship_type_summary     TEXT,
    ship_type_detail_name TEXT,
    length_m              DOUBLE PRECISION,
    width_m               DOUBLE PRECISION,
    dwt                   DOUBLE PRECISION,
    grt                   DOUBLE PRECISION,
    year_built            DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_vessels_mmsi      ON vessels (mmsi);
CREATE INDEX IF NOT EXISTS idx_vessels_imo       ON vessels (imo);
CREATE INDEX IF NOT EXISTS idx_vessels_type      ON vessels (ship_type_summary);
CREATE INDEX IF NOT EXISTS idx_vessels_flag_code ON vessels (flag_code);

-- ---------- ais_positions ----------
CREATE TABLE IF NOT EXISTS ais_positions (
    position_id  BIGSERIAL PRIMARY KEY,
    vessel_id    TEXT NOT NULL,
    mmsi         TEXT,
    event_ts     TIMESTAMPTZ NOT NULL,
    lat          DOUBLE PRECISION,
    lon          DOUBLE PRECISION,
    speed_knots  DOUBLE PRECISION,
    course_deg   DOUBLE PRECISION,
    heading_deg  DOUBLE PRECISION,      -- 511 = unknown
    nav_status   TEXT,
    reported_dest TEXT,
    draught_m    DOUBLE PRECISION,
    geom         geometry(Point, 4326)
                 GENERATED ALWAYS AS (
                     CASE WHEN lat IS NOT NULL AND lon IS NOT NULL
                          THEN ST_SetSRID(ST_MakePoint(lon, lat), 4326) END
                 ) STORED
);
CREATE INDEX IF NOT EXISTS idx_pos_geom_gist  ON ais_positions USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_pos_vessel_ts  ON ais_positions (vessel_id, event_ts);
CREATE INDEX IF NOT EXISTS idx_pos_ts         ON ais_positions (event_ts);

-- ---------- dark_gaps ----------
CREATE TABLE IF NOT EXISTS dark_gaps (
    gap_id               TEXT PRIMARY KEY,
    vessel_id            TEXT NOT NULL,
    mmsi                 TEXT,
    gap_start_ts         TIMESTAMPTZ,
    gap_end_ts           TIMESTAMPTZ,
    gap_duration_seconds DOUBLE PRECISION,
    distance_nm          DOUBLE PRECISION,
    implied_speed_knots  DOUBLE PRECISION,
    start_lat            DOUBLE PRECISION,
    start_lon            DOUBLE PRECISION,
    end_lat              DOUBLE PRECISION,
    end_lon              DOUBLE PRECISION,
    start_geom geometry(Point, 4326)
        GENERATED ALWAYS AS (
            CASE WHEN start_lat IS NOT NULL AND start_lon IS NOT NULL
                 THEN ST_SetSRID(ST_MakePoint(start_lon, start_lat), 4326) END
        ) STORED,
    end_geom geometry(Point, 4326)
        GENERATED ALWAYS AS (
            CASE WHEN end_lat IS NOT NULL AND end_lon IS NOT NULL
                 THEN ST_SetSRID(ST_MakePoint(end_lon, end_lat), 4326) END
        ) STORED,
    gap_line geometry(LineString, 4326)
        GENERATED ALWAYS AS (
            CASE WHEN start_lat IS NOT NULL AND start_lon IS NOT NULL
                  AND end_lat   IS NOT NULL AND end_lon   IS NOT NULL
                  AND (start_lat, start_lon) IS DISTINCT FROM (end_lat, end_lon)
                 THEN ST_SetSRID(ST_MakeLine(ST_MakePoint(start_lon, start_lat),
                                             ST_MakePoint(end_lon,   end_lat)), 4326) END
        ) STORED
);
CREATE INDEX IF NOT EXISTS idx_gaps_vessel_ts  ON dark_gaps (vessel_id, gap_start_ts);
CREATE INDEX IF NOT EXISTS idx_gaps_start_ts   ON dark_gaps (gap_start_ts);
CREATE INDEX IF NOT EXISTS idx_gaps_start_geom ON dark_gaps USING GIST (start_geom);
CREATE INDEX IF NOT EXISTS idx_gaps_end_geom   ON dark_gaps USING GIST (end_geom);
CREATE INDEX IF NOT EXISTS idx_gaps_line       ON dark_gaps USING GIST (gap_line);

-- ---------- ownership ----------
-- company_name_norm: chuẩn hoá biến thể tên công ty
-- (bỏ dấu câu, bỏ hậu tố CO/LTD/LIMITED/INC/..., gộp khoảng trắng).
CREATE TABLE IF NOT EXISTS ownership (
    ownership_id      BIGSERIAL PRIMARY KEY,
    vessel_id         TEXT NOT NULL,
    role              TEXT NOT NULL,
    company_name      TEXT,
    company_country   TEXT,
    start_date        DATE,
    company_name_norm TEXT GENERATED ALWAYS AS (
        NULLIF(btrim(regexp_replace(
            regexp_replace(
                regexp_replace(upper(coalesce(company_name, '')), '[^A-Z0-9 ]', ' ', 'g'),
                '\m(CO|COMPANY|LTD|LIMITED|INC|CORP|CORPORATION|LLC|PTE|SDN|BHD|JSC|SA|AS|GMBH)\M',
                ' ', 'g'),
            '\s+', ' ', 'g')), '')
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_own_vessel     ON ownership (vessel_id);
CREATE INDEX IF NOT EXISTS idx_own_role       ON ownership (role);
CREATE INDEX IF NOT EXISTS idx_own_company    ON ownership (company_name_norm);
CREATE INDEX IF NOT EXISTS idx_own_country    ON ownership (company_country);


-- conversation
CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'closed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    message_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL
        REFERENCES sessions(session_id) ON DELETE CASCADE,

    sequence_no BIGINT NOT NULL,
    role TEXT NOT NULL
        CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT,

    --tool call
    tool_name TEXT,
    tool_call_id TEXT,
    tool_calls JSONB,

    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (session_id, sequence_no)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_sequence
    ON messages (session_id, sequence_no);

CREATE TABLE IF NOT EXISTS session_memories (
    memory_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL
        REFERENCES sessions(session_id) ON DELETE CASCADE,

    memory_key TEXT NOT NULL,
    memory_value JSONB NOT NULL,
    embedding vector,
    source_message_id BIGINT
        REFERENCES messages(message_id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (session_id, memory_key)
);

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS idx_vessels_shipname_trgm
ON vessels USING GIN (shipname gin_trgm_ops);
