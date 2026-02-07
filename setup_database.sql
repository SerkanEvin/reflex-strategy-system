-- PostgreSQL Database Setup for Reflex-Strategy System
-- Run this on your PostgreSQL server to set up the database

-- Create database
CREATE DATABASE reflex_strategy;

\c reflex_strategy

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create spatial_memory table
CREATE TABLE IF NOT EXISTS spatial_memory (
    id SERIAL PRIMARY KEY,
    object_type VARCHAR(50) NOT NULL,
    label VARCHAR(100) NOT NULL,
    position GEOMETRY(POINTZ, 4326) NOT NULL,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for spatial queries
CREATE INDEX IF NOT EXISTS idx_spatial_memory_position
    ON spatial_memory USING GIST (position);

CREATE INDEX IF NOT EXISTS idx_spatial_memory_type_label
    ON spatial_memory (object_type, label);

CREATE INDEX IF NOT EXISTS idx_spatial_memory_last_seen
    ON spatial_memory (last_seen DESC);

-- Create player_positions table
CREATE TABLE IF NOT EXISTS player_positions (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    position GEOMETRY(POINTZ, 4326) NOT NULL,
    health_percent FLOAT,
    is_in_combat BOOLEAN,
    detected_objects JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_player_positions_session
    ON player_positions (session_id);

CREATE INDEX IF NOT EXISTS idx_player_positions_timestamp
    ON player_positions (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_player_positions_position
    ON player_positions USING GIST (position);

-- Create strategy_logs table
CREATE TABLE IF NOT EXISTS strategy_logs (
    id SERIAL PRIMARY KEY,
    policy_id INTEGER,
    action VARCHAR(100) NOT NULL,
    reasoning TEXT,
    priority INTEGER,
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_strategy_logs_action
    ON strategy_logs (action);

CREATE INDEX IF NOT EXISTS idx_strategy_logs_created_at
    ON strategy_logs (created_at DESC);

-- Create system_logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    session_id UUID,
    client_id VARCHAR(100),
    log_level VARCHAR(20) NOT NULL,
    message TEXT,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp
    ON system_logs (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_logs_session
    ON system_logs (session_id);

-- Create a trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_spatial_memory_updated_at
    BEFORE UPDATE ON spatial_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;

-- Insert some initial spatial memory entries (examples)
INSERT INTO spatial_memory (object_type, label, position, last_seen, confidence, metadata)
VALUES
    ('market', 'central_market', ST_SetSRID(ST_MakePoint(1000.0, 500.0, 0.0), 4326), NOW(), 1.0, '{"type": "market", "vendors": ["weapons", "armor", "potions"]}'::jsonb),
    ('spawn', 'starting_area', ST_SetSRID(ST_MakePoint(0.0, 0.0, 0.0), 4326), NOW(), 1.0, '{"type": "safe_zone", "healer": true}'::jsonb),
    ('vendor', 'potion_vendor', ST_SetSRID(ST_MakePoint(1050.0, 520.0, 0.0), 4326), NOW(), 0.9, '{"sells": ["healing_potion", "mana_potion"]}'::jsonb);

-- Verify the setup
SELECT 'Database setup complete!' as status;
SELECT 'spatial_memory' as table_name, COUNT(*) as row_count FROM spatial_memory
UNION ALL
SELECT 'player_positions' as table_name, COUNT(*) as row_count FROM player_positions
UNION ALL
SELECT 'strategy_logs' as table_name, COUNT(*) as row_count FROM strategy_logs
UNION ALL
SELECT 'system_logs' as table_name, COUNT(*) as row_count FROM system_logs;
