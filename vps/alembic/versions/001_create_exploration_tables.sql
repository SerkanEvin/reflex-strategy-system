-- Exploration Grid System
-- Tracks visited areas on a grid system for exploration tracking

-- Drop if exists for clean development
DROP TABLE IF EXISTS exploration_grid CASCADE;

-- Create exploration grid table
CREATE TABLE exploration_grid (
    id SERIAL PRIMARY KEY,
    grid_x INTEGER NOT NULL,
    grid_y INTEGER NOT NULL,
    visited BOOLEAN NOT NULL DEFAULT FALSE,
    explored BOOLEAN NOT NULL DEFAULT FALSE,  -- Fog of war: visible but not visited
    first_visit TIMESTAMP,
    last_visit TIMESTAMP,
    visits INTEGER DEFAULT 0,
    discoveries JSONB DEFAULT '{}'::jsonb,
    position GEOMETRY(Point, 4326) NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),

    CONSTRAINT unique_grid UNIQUE (grid_x, grid_y)
);

-- Indexes for fast queries
CREATE INDEX idx_exploration_visited ON exploration_grid(grid_x, grid_y) WHERE visited = FALSE;
CREATE INDEX idx_exploration_explored ON exploration_grid(grid_x, grid_y) WHERE explored = TRUE AND visited = FALSE;
CREATE INDEX idx_exploration_position ON exploration_grid USING GIST (position);
CREATE INDEX idx_exploration_last_visit ON exploration_grid(last_visit DESC);
CREATE INDEX idx_exploration_visits ON exploration_grid(visits DESC);

-- Update timestamp trigger
CREATE TRIGGER update_exploration_updated_at
    BEFORE UPDATE ON exploration_grid
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to get or create exploration grid cell
CREATE OR REPLACE FUNCTION get_or_create_grid_cell(
    p_grid_x INTEGER,
    p_grid_y INTEGER,
    p_position GEOMETRY(Point)
) RETURNS INTEGER AS $$
DECLARE
    cell_id INTEGER;
BEGIN
    -- Try to find existing cell
    SELECT id INTO cell_id
    FROM exploration_grid
    WHERE grid_x = p_grid_x AND grid_y = p_grid_y;

    -- If not found, create new
    IF cell_id IS NULL THEN
        INSERT INTO exploration_grid (grid_x, grid_y, position)
        VALUES (p_grid_x, p_grid_y, p_position)
        RETURNING id INTO cell_id;
    END IF;

    RETURN cell_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark grid cell as visited
CREATE OR REPLACE FUNCTION mark_visited(
    p_grid_x INTEGER,
    p_grid_y INTEGER,
    p_discoveries JSONB DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    cell_id INTEGER;
BEGIN
    UPDATE exploration_grid
    SET
        visited = TRUE,
        explored = TRUE,
        visits = visits + 1,
        last_visit = now(),
        first_visit = COALESCE(first_visit, now()),
        discoveries = CASE
            WHEN p_discoveries IS NOT NULL
            THEN discoveries || p_discoveries
            ELSE discoveries
        END
    WHERE grid_x = p_grid_x AND grid_y = p_grid_y
    RETURNING id INTO cell_id;

    RETURN cell_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark neighboring cells as explored (fog of war)
CREATE OR REPLACE FUNCTION mark_neighbors_explored(
    p_center_x INTEGER,
    p_center_y INTEGER,
    p_radius INTEGER DEFAULT 1
) RETURNS INTEGER AS $$
DECLARE
    distance INTEGER;
BEGIN
    UPDATE exploration_grid
    SET explored = TRUE
    WHERE (
        ABS(grid_x - p_center_x) <= p_radius
        AND ABS(grid_y - p_center_y) <= p_radius
    )
    AND visited = FALSE;

    GET DIAGNOSTICS distance = ROW_COUNT;
    RETURN distance;
END;
$$ LANGUAGE plpgsql;

-- Function to get exploration statistics
CREATE OR REPLACE FUNCTION get_exploration_stats()
RETURNS TABLE(
    total_cells INTEGER,
    visited_cells INTEGER,
    explored_cells INTEGER,
    visited_percent NUMERIC,
    explored_percent NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) AS total_cells,
        COUNT(*) FILTER (WHERE visited = TRUE) AS visited_cells,
        COUNT(*) FILTER (WHERE explored = TRUE) AS explored_cells,
        ROUND(
            (COUNT(*) FILTER (WHERE visited = TRUE)::NUMERIC / NULLIF(COUNT(*), 0)) * 100,
            2
        ) AS visited_percent,
        ROUND(
            (COUNT(*) FILTER (WHERE explored = TRUE)::NUMERIC / NULLIF(COUNT(*), 0)) * 100,
            2
        ) AS explored_percent
    FROM exploration_grid;
END;
$$ LANGUAGE plpgsql;

-- Uncomment to enable grid cell auto-creation on queries
-- This trigger automatically creates grid cells when queried but not found
-- CREATE TRIGGER autocreate_grid_cell
--     BEFORE SELECT ON exploration_grid
--     FOR EACH ROW
--     WHEN NEW IS NULL
--     EXECUTE FUNCTION autocreate_if_needed();

COMMENT ON TABLE exploration_grid IS 'Grid-based exploration tracking system';
COMMENT ON COLUMN exploration_grid.grid_x IS 'Grid X coordinate (floor(x / grid_size))';
COMMENT ON COLUMN exploration_grid.grid_y IS 'Grid Y coordinate (floor(y / grid_size))';
COMMENT ON COLUMN exploration_grid.visited IS 'Has player physically visited this cell?';
COMMENT ON COLUMN exploration_grid.explored IS 'Is cell visible (fog of war removed)?';
COMMENT ON COLUMN exploration_grid.discoveries IS 'JSONB: counts/types of objects found in this cell';
