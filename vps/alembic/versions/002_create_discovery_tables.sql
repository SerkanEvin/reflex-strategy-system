-- Auto-Discovery System
-- Tracks automatically discovered locations with confidence scoring

-- Drop if exists for clean development
DROP TABLE IF EXISTS discovered_locations CASCADE;

-- Create discovered locations table
CREATE TABLE discovered_locations (
    id SERIAL PRIMARY KEY,
    location_id VARCHAR(100) UNIQUE NOT NULL,
    location_type VARCHAR(50) NOT NULL,
    label VARCHAR(100),
    description TEXT,
    position GEOMETRY(Point, 4326) NOT NULL,
    discovered_at TIMESTAMP NOT NULL DEFAULT now(),
    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    visits INTEGER DEFAULT 1,
    last_visited TIMESTAMP,
    features JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,

    CONSTRAINT location_type_check CHECK (location_type IN ('market', 'vendor', 'healer', 'spawn', 'portal', 'dungeon', 'resource', 'safe_zone', 'poi'))
);

-- Spatial index for proximity queries
CREATE INDEX idx_discovered_position ON discovered_locations USING GIST (position);
CREATE INDEX idx_discovered_type ON discovered_locations(location_type);
CREATE INDEX idx_discovered_confidence ON discovered_locations(confidence DESC);
CREATE INDEX idx_discovered_confirmed ON discovered_locations(confirmed) WHERE confirmed = TRUE;
CREATE INDEX idx_discovered_visited ON discovered_locations(last_visited DESC);

-- Function to get or create discovered location
CREATE OR REPLACE FUNCTION get_or_create_location(
    p_location_id VARCHAR,
    p_type VARCHAR,
    p_label VARCHAR DEFAULT NULL,
    p_position GEOMETRY(Point),
    p_features JSONB DEFAULT NULL,
    p_confidence DOUBLE PRECISION DEFAULT 0.5
) RETURNS INTEGER AS $$
DECLARE
    location_id INTEGER;
    matched_id INTEGER;
BEGIN
    -- Check for existing location within tolerance distance
    -- This handles re-discovery of same location
    SELECT id INTO matched_id
    FROM discovered_locations
    WHERE location_id = p_location_id
    LIMIT 1;

    IF matched_id IS NOT NULL THEN
        -- Update existing
        UPDATE discovered_locations
        SET
            visits = visits + 1,
            last_visited = now(),
            confidence = LEAST(1.0, confidence + 0.05),  -- Increase confidence on revisit
            features = CASE
                WHEN p_features IS NOT NULL
                THEN features || p_features
                ELSE features
            END,
            confirmed = confirmed OR (confidence >= 0.8)
        WHERE id = matched_id
        RETURNING id INTO location_id;

        RETURN location_id;
    END IF;

    -- Create new location
    INSERT INTO discovered_locations (
        location_id,
        location_type,
        label,
        position,
        features,
        confidence,
        confirmed
    ) VALUES (
        p_location_id,
        p_type,
        p_label,
        p_position,
        COALESCE(p_features, '{}'::jsonb),
        p_confidence,
        p_confidence >= 0.8
    )
    RETURNING id INTO location_id;

    RETURN location_id;
END;
$$ LANGUAGE plpgsql;

-- Function to find nearby locations
CREATE OR REPLACE FUNCTION find_nearby_locations(
    p_position GEOMETRY(Point),
    p_location_type VARCHAR DEFAULT NULL,
    p_radius DOUBLE PRECISION DEFAULT 100.0,
    p_min_confidence DOUBLE PRECISION DEFAULT 0.6
) RETURNS TABLE(
    id INTEGER,
    location_id VARCHAR,
    location_type VARCHAR,
    label VARCHAR,
    position GEOMETRY(Point),
    distance DOUBLE PRECISION,
    confidence DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.location_id,
        d.location_type,
        d.label,
        d.position,
        ST_Distance(p_position, d.position) / 100.0 AS distance,  -- Convert to meters (approx)
        d.confidence
    FROM discovered_locations d
    WHERE ST_DWithin(d.position, p_position, p_radius)
      AND (p_location_type IS NULL OR d.location_type = p_location_type)
      AND d.confidence >= p_min_confidence
    ORDER BY ST_Distance(p_position, d.position)
    LIMIT 5;
END;
$$ LANGUAGE plpgsql;

-- Function to find locations of a specific type
CREATE OR REPLACE FUNCTION find_locations_by_type(
    p_type VARCHAR,
    p_limit INTEGER DEFAULT 10,
    p_confirmed_only BOOLEAN DEFAULT FALSE
) RETURNS TABLE(
    id INTEGER,
    location_id VARCHAR,
    label VARCHAR,
    position GEOMETRY(Point),
    confidence DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.location_id,
        d.label,
        d.position,
        d.confidence
    FROM discovered_locations d
    WHERE d.location_type = p_type
      AND (NOT p_confirmed_only OR d.confirmed = TRUE)
    ORDER BY d.confidence DESC, d.visits DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get discovery statistics
CREATE OR REPLACE FUNCTION get_discovery_stats()
RETURNS TABLE(
    total_locations INTEGER,
    confirmed_locations INTEGER,
    by_type JSONB
) AS $$
DECLARE
    type_stats JSONB;
BEGIN
    -- Build type statistics
    SELECT jsonb_object_agg(type, count)
    INTO type_stats
    FROM (
        SELECT jsonb_build_object(
            'count', COUNT(*),
            'confirmed', COUNT(*) FILTER (WHERE confirmed = TRUE)
        ) AS type_data,
        location_type AS type
        FROM discovered_locations
        GROUP BY location_type
    ) sub;

    RETURN QUERY
    SELECT
        COUNT(*) AS total_locations,
        COUNT(*) FILTER (WHERE confirmed = TRUE) AS confirmed_locations,
        type_stats
    FROM discovered_locations;
END;
$$ LANGUAGE plpgsql;

-- Function to apply confidence decay over time
CREATE OR REPLACE FUNCTION decay_confidence()
RETURNS INTEGER AS $$
DECLARE
    affected_rows INTEGER;
BEGIN
    -- Reduce confidence by 0.01 per day for locations not visited in 7 days
    UPDATE discovered_locations
    SET confidence = GREATEST(0.3, confidence - 0.01 * EXTRACT(DAY FROM (now() - last_visited)))
    WHERE last_visited < now() - INTERVAL '7 days'
      AND confidence > 0.3;

    -- Remove locations with very low confidence that haven't been visited recently
    DELETE FROM discovered_locations
    WHERE confidence < 0.3
      AND last_visited < now() - INTERVAL '30 days';

    GET DIAGNOSTICS affected_rows = ROW_COUNT;
    RETURN affected_rows;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE discovered_locations IS 'Automatically discovered locations with confidence tracking';
COMMENT ON COLUMN discovered_locations.location_id IS 'Unique identifier (e.g., vendor_001, market_central)';
COMMENT ON COLUMN discovered_locations.confidence IS 'Confidence score (0.0-1.0), 0.8+ = confirmed';
COMMENT ON COLUMN discovered_locations.features IS 'JSONB: detected features (objects, UI elements, etc)';
COMMENT ON COLUMN discovered_locations.metadata IS 'JSONB: additional metadata from game';

-- Enable the functions for the database
-- Uncomment to run table creation
-- SELECT * FROM get_exploration_stats();
-- SELECT * FROM get_discovery_stats();
