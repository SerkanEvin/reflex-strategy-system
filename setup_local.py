#!/usr/bin/env python
"""
Database Setup Script for Reflex-Strategy System
Sets up PostgreSQL database on local or remote server.
"""
import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    import asyncpg
except ImportError:
    print("Error: asyncpg not installed. Run: pip install asyncpg")
    sys.exit(1)


async def setup_database(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    create_db: bool = True,
    setup_postgis: bool = True
):
    """
    Set up the Reflex-Strategy database.

    Args:
        host: PostgreSQL host.
        port: PostgreSQL port.
        database: Database name.
        user: Database user.
        password: Database password.
        create_db: Create database if it doesn't exist.
        setup_postgis: Set up PostGIS extension and tables.
    """
    print(f"[SETUP] Connecting to PostgreSQL at {host}:{port}...")

    # Connect to postgres database first (for creating new database)
    postgres_dsn = f"postgresql://{user}:{password}@{host}:{port}/postgres"

    try:
        # Check/Create database
        if create_db:
            conn = await asyncpg.connect(postgres_dsn)

            # Check if database exists
            db_exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )

            if db_exists:
                print(f"[SETUP] Database '{database}' already exists")
            else:
                print(f"[SETUP] Creating database '{database}'...")
                await conn.execute(f'CREATE DATABASE "{database}"')
                print(f"[SETUP] Database '{database}' created")

            await conn.close()

        # Connect to target database
        target_dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        conn = await asyncpg.connect(target_dsn)
        print(f"[SETUP] Connected to database '{database}'")

        if setup_postgis:
            # Enable PostGIS extension
            print("[SETUP] Enabling PostGIS extension...")
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS 'uuid-ossp'")
                print("[SETUP] PostGIS extension enabled")
            except Exception as e:
                print(f"[SETUP] Warning: Could not enable PostGIS: {e}")
                print("[SETUP] Spatial features may not work properly")

            # Create tables
            print("[SETUP] Creating tables...")

            # spatial_memory table
            await conn.execute("""
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
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spatial_memory_position
                    ON spatial_memory USING GIST (position)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spatial_memory_type_label
                    ON spatial_memory (object_type, label)
            """)

            # player_positions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS player_positions (
                    id SERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    position GEOMETRY(POINTZ, 4326) NOT NULL,
                    health_percent FLOAT,
                    is_in_combat BOOLEAN,
                    detected_objects JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_player_positions_session
                    ON player_positions (session_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_player_positions_position
                    ON player_positions USING GIST (position)
            """)

            # strategy_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_logs (
                    id SERIAL PRIMARY KEY,
                    policy_id INTEGER,
                    action VARCHAR(100) NOT NULL,
                    reasoning TEXT,
                    priority INTEGER,
                    context JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # system_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    session_id UUID,
                    client_id VARCHAR(100),
                    log_level VARCHAR(20) NOT NULL,
                    message TEXT,
                    metadata JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

            # Create trigger for updated_at
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            """)

            await conn.execute("""
               DROP TRIGGER IF EXISTS update_spatial_memory_updated_at ON spatial_memory
            """)
            await conn.execute("""
                CREATE TRIGGER update_spatial_memory_updated_at
                    BEFORE UPDATE ON spatial_memory
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """)

            print("[SETUP] Tables created successfully")

            # Insert sample data
            print("[SETUP] Inserting sample spatial memory entries...")
            sample_entries = [
                ("market", "central_market", 1000.0, 500.0, 0.0),
                ("spawn", "starting_area", 0.0, 0.0, 0.0),
                ("vendor", "potion_vendor", 1050.0, 520.0, 0.0),
                ("vendor", "weaponsmith", 1100.0, 480.0, 0.0),
                ("healer", "clinic", 1020.0, 550.0, 0.0),
            ]

            for obj_type, label, x, y, z in sample_entries:
                await conn.execute("""
                    INSERT INTO spatial_memory (object_type, label, position, last_seen, confidence, metadata)
                    VALUES ($1, $2, ST_SetSRID(ST_MakePoint($3, $4, $5), 4326), NOW(), 1.0,
                            '{}'::jsonb)
                    ON CONFLICT DO NOTHING
                """, obj_type, label, x, y, z)

            print("[SETUP] Sample data inserted")

        # Verify setup
        print("\n[SETUP] Verification:")

        tables = await conn.fetch("""
            SELECT table_name, row_count
            FROM (
                SELECT 'spatial_memory' as table_name, COUNT(*) as row_count FROM spatial_memory
                UNION ALL
                SELECT 'player_positions' as table_name, COUNT(*) as row_count FROM player_positions
                UNION ALL
                SELECT 'strategy_logs' as table_name, COUNT(*) as row_count FROM strategy_logs
                UNION ALL
                SELECT 'system_logs' as table_name, COUNT(*) as row_count FROM system_logs
            ) t
        """)

        for row in tables:
            print(f"[SETUP]   {row['table_name']}: {row['row_count']} rows")

        await conn.close()
        print("\n[SETUP] Database setup complete!")

    except Exception as e:
        print(f"[SETUP] Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Set up Reflex-Strategy database")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", default="reflex_strategy", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--skip-create-db", action="store_true", help="Skip creating database")
    parser.add_argument("--skip-postgis", action="store_true", help="Skip PostGIS setup")

    args = parser.parse_args()

    asyncio.run(setup_database(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        create_db=not args.skip_create_db,
        setup_postgis=not args.skip_postgis
    ))


if __name__ == "__main__":
    main()
