"""
Spatial Memory Database Module (VPS)
Manages spatial data using PostgreSQL with PostGIS extension.
Stores and queries positions of markets, spawns, resources, etc.
"""
import asyncpg
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict
import json
import subprocess

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import SpatialMemoryEntry


class SpatialDatabase:
    """
    PostgreSQL database handler for spatial memory.
    Uses PostGIS for geospatial queries.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "reflex_strategy",
        user: str = "postgres",
        password: str = None,
        pool_min: int = 5,
        pool_max: int = 20
    ):
        """
        Initialize database connection.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            pool_min: Minimum connection pool size.
            pool_max: Maximum connection pool size.
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_min = pool_min
        self.pool_max = pool_max

        self.pool: Optional[asyncpg.Pool] = None
        self._use_docker_fallback = False
        self._cached_data = None  # Will load lazily

    def _docker_query(self, query: str) -> str:
        """Execute query via docker exec."""
        try:
            result = subprocess.run([
                'docker', 'exec', 'supabase-db',
                'psql', '-U', 'postgres', '-d', self.database,
                '-t', '-c', query
            ], capture_output=True, text=True, timeout=30)
            return result.stdout
        except Exception as e:
            print(f"[DATABASE] Docker query failed: {e}")
            return ""

    def _load_initial_data(self) -> Dict[str, List[Dict]]:
        """Load spatial memory data from database via docker exec."""
        data = {"markets": [], "spawns": [], "vendors": [], "healers": [], "safe_zones": []}

        try:
            result = self._docker_query(
                "SELECT id, object_type, label, ST_X(position) as x, ST_Y(position) as y, "
                "ST_Z(position) as z, confidence FROM spatial_memory;"
            )

            lines = result.strip().split('\n')
            for line in lines:
                if line and line.strip() and not line.startswith('|') and 'id' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 7:
                        try:
                            entry = {
                                'id': int(parts[0]),
                                'type': parts[1],
                                'label': parts[2],
                                'x': float(parts[3]),
                                'y': float(parts[4]),
                                'z': float(parts[5]) if parts[5] and parts[5] not in ('\\N', '') else 0.0,
                                'confidence': float(parts[6])
                            }

                            if entry['type'] == 'market':
                                data['markets'].append(entry)
                            elif entry['type'] == 'spawn':
                                data['spawns'].append(entry)
                            elif entry['type'] == 'vendor':
                                data['vendors'].append(entry)
                            elif entry['type'] == 'healer':
                                data['healers'].append(entry)
                            elif entry['type'] == 'safe_zone':
                                data['safe_zones'].append(entry)
                        except (ValueError, IndexError):
                            continue

            print(f"[DATABASE] Loaded {sum(len(v) for v in data.values())} entries")
        except Exception as e:
            print(f"[DATABASE] Failed to load initial data: {e}")

        return data

    def _get_cached_data(self) -> Dict[str, List[Dict]]:
        """Get cached data, loading if necessary."""
        if self._cached_data is None:
            self._cached_data = self._load_initial_data()
        return self._cached_data

    async def connect(self):
        """
        Establish connection pool to PostgreSQL.
        Uses multiple connection methods as fallbacks.
        """
        # For Supabase Docker setup, use container exec as primary method
        print(f"[DATABASE] Attempting to connect to PostgreSQL...")

        # Try multiple connection methods
        methods = [
            ("localhost", self.port),  # Original
            ("127.0.0.1", 5432),  # Localhost direct
            ("127.0.0.1", 6543),  # Supabase pooler
        ]

        for host, port in methods:
            dsn = f"postgresql://{self.user}:{self.password}@{host}:{port}/{self.database}"

            try:
                self.pool = await asyncpg.create_pool(
                    dsn,
                    min_size=self.pool_min,
                    max_size=self.pool_max,
                    command_timeout=30
                )
                print(f"[DATABASE] Connected to PostgreSQL at {host}:{port}")
                return
            except Exception as e:
                print(f"[DATABASE] Connection to {host}:{port} failed: {type(e).__name__}")
                continue

        # If all methods fail, use docker exec fallback
        print("[DATABASE] All connection methods failed, using docker exec wrapper")
        self._use_docker_fallback = True

    async def close(self):
        """
        Close connection pool.
        """
        if self.pool:
            await self.pool.close()
            print("[DATABASE] Connection closed")

    async def initialize_schema(self):
        """
        Initialize database schema with required tables and PostGIS extension.
        """
        if self.pool is None:
            print("[DATABASE] Using fallback mode - skipping schema initialization")
            return

        async with self.pool.acquire() as conn:
            # Enable PostGIS extension
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS postgis;
            """)

            # Create spatial_memory table
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
                );

                CREATE INDEX IF NOT EXISTS idx_spatial_memory_position
                    ON spatial_memory USING GIST (position);

                CREATE INDEX IF NOT EXISTS idx_spatial_memory_type
                    ON spatial_memory (object_type);

                CREATE INDEX IF NOT EXISTS idx_spatial_memory_last_seen
                    ON spatial_memory (last_seen);
            """)

            # Create player_positions table for tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS player_positions (
                    id SERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    position GEOMETRY(POINTZ, 4326) NOT NULL,
                    health_percent FLOAT,
                    is_in_combat BOOLEAN,
                    detected_objects JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_player_positions_session
                    ON player_positions (session_id);

                CREATE INDEX IF NOT EXISTS idx_player_positions_timestamp
                    ON player_positions (timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_player_positions_position
                    ON player_positions USING GIST (position);
            """)

            # Create strategy_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_logs (
                    id SERIAL PRIMARY KEY,
                    policy_id INTEGER,
                    action VARCHAR(100) NOT NULL,
                    reasoning TEXT,
                    priority INTEGER,
                    context JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            print("[DATABASE] Schema initialized")

    async def upsert_entry(self, entry: SpatialMemoryEntry) -> int:
        """
        Insert or update a spatial memory entry.

        Args:
            entry: SpatialMemoryEntry to upsert.

        Returns:
            ID of inserted/updated entry.
        """
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None

        # Convert position to PostGIS point
        pos_ewkt = f"POINTZ({entry.position['x']} {entry.position['y']} {entry.position['z']})"

        async with self.pool.acquire() as conn:
            # Try to update existing entry by type and label
            result = await conn.execute("""
                INSERT INTO spatial_memory (
                    object_type, label, position, last_seen, confidence, metadata, updated_at
                ) VALUES ($1, $2, ST_GeomFromEWKT($3), $4, $5, $6, NOW())
                ON CONFLICT (object_type, label) DO UPDATE
                SET position = EXCLUDED.position,
                    last_seen = EXCLUDED.last_seen,
                    confidence = EXCLUDED.confidence,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
            """, entry.object_type, entry.label, pos_ewkt, entry.last_seen, entry.confidence, metadata_json)

            return int(result.split()[-1])

    async def get_entry_by_label(
        self,
        object_type: str,
        label: str
    ) -> Optional[SpatialMemoryEntry]:
        """
        Retrieve a spatial entry by type and label.

        Args:
            object_type: Type of object.
            label: Object label.

        Returns:
            SpatialMemoryEntry or None.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, object_type, label, ST_X(position) as x,
                       ST_Y(position) as y, ST_Z(position) as z,
                       last_seen, confidence, metadata
                FROM spatial_memory
                WHERE object_type = $1 AND label = $2
            """, object_type, label)

            if row:
                return SpatialMemoryEntry(
                    entry_id=row['id'],
                    object_type=row['object_type'],
                    label=row['label'],
                    position={'x': row['x'], 'y': row['y'], 'z': row['z']},
                    last_seen=row['last_seen'].timestamp(),
                    confidence=row['confidence'],
                    metadata=row['metadata']
                )
            return None

    async def find_nearest(
        self,
        position: Dict[str, float],
        object_type: str,
        limit: int = 5,
        max_distance: Optional[float] = None
    ) -> List[Tuple[SpatialMemoryEntry, float]]:
        """
        Find nearest entries of a given type to a position.

        Args:
            position: Reference position {x, y, z}.
            object_type: Type of object to search for.
            limit: Maximum number of results.
            max_distance: Maximum distance to search (optional).

        Returns:
            List of (SpatialMemoryEntry, distance) tuples.
        """
        # Use fallback if database pool not available
        if self._use_docker_fallback or self.pool is None:
            return self._find_nearest_fallback(position, object_type, limit, max_distance)

        pos_ewkt = f"POINTZ({position['x']} {position['y']} {position['z']})"

        query = """
            SELECT id, object_type, label,
                   ST_X(position) as x, ST_Y(position) as y, ST_Z(position) as z,
                   last_seen, confidence, metadata,
                   ST_Distance(position, $1::geometry) as distance
            FROM spatial_memory
            WHERE object_type = $2
        """
        params = [pos_ewkt, object_type]

        if max_distance:
            query += " AND ST_Distance(position, $1::geometry) <= $3"
            params.append(max_distance)

        query += " ORDER BY distance ASC LIMIT $4"
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            entry = SpatialMemoryEntry(
                entry_id=row['id'],
                object_type=row['object_type'],
                label=row['label'],
                position={'x': row['x'], 'y': row['y'], 'z': row['z']},
                last_seen=row['last_seen'].timestamp(),
                confidence=row['confidence'],
                metadata=row['metadata']
            )
            results.append((entry, row['distance']))

        return results

    def _find_nearest_fallback(
        self,
        position: Dict[str, float],
        object_type: str,
        limit: int = 5,
        max_distance: Optional[float] = None
    ) -> List[Tuple[SpatialMemoryEntry, float]]:
        """Fallback method using cached data from docker."""
        import time

        data = self._get_cached_data()
        results = []

        # Map object_type to cache key
        type_map = {
            'market': 'markets',
            'spawn': 'spawns',
            'vendor': 'vendors',
            'healer': 'healers',
            'safe_zone': 'safe_zones'
        }
        cache_key = type_map.get(object_type)

        if cache_key:
            entries = data.get(cache_key, [])
            for entry in entries:
                # Calculate Euclidean distance
                dx = entry['x'] - position['x']
                dy = entry['y'] - position['y']
                dz = entry['z'] - position['z']
                distance = (dx**2 + dy**2 + dz**2) ** 0.5

                if max_distance is None or distance <= max_distance:
                    spatial_entry = SpatialMemoryEntry(
                        entry_id=entry['id'],
                        object_type=entry['type'],
                        label=entry['label'],
                        position={'x': entry['x'], 'y': entry['y'], 'z': entry['z']},
                        last_seen=time.time(),
                        confidence=entry['confidence'],
                        metadata={}
                    )
                    results.append((spatial_entry, distance))

            # Sort by distance and limit
            results.sort(key=lambda x: x[1])
            results = results[:limit]

        return results

    async def log_player_position(
        self,
        session_id: str,
        position: Dict[str, float],
        health_percent: float,
        is_in_combat: bool,
        detected_objects: List[Dict]
    ):
        """
        Log player position for tracking and analysis.

        Args:
            session_id: Unique session identifier.
            position: Player position {x, y, z}.
            health_percent: Player health percentage.
            is_in_combat: Whether player is in combat.
            detected_objects: List of detected objects.

        Note:
            If database pool is not available, falls back to docker exec.
        """
        pos_ewkt = f"POINTZ({position['x']} {position['y']} {position['z']})"
        detected_objects_json = json.dumps(detected_objects).replace("'", "''")

        # Check if pool is available
        if not self.pool:
            # Fallback using docker exec
            print("[DATABASE] Using fallback for player position logging")

            query = f"""
                INSERT INTO player_positions (
                    session_id, position, health_percent, is_in_combat, detected_objects
                ) VALUES ('{session_id}', ST_GeomFromEWKT('{pos_ewkt}'), {health_percent}, {str(is_in_combat).lower()}, '{detected_objects_json}'::jsonb);
            """
            self._docker_query(query)
            return

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO player_positions (
                    session_id, position, health_percent, is_in_combat, detected_objects
                ) VALUES ($1, ST_GeomFromEWKT($2), $3, $4, $5)
            """, session_id, pos_ewkt, health_percent, is_in_combat, json.dumps(detected_objects))

    async def get_recent_positions(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent player positions.

        Args:
            session_id: Session identifier.
            limit: Number of positions to retrieve.

        Returns:
            List of position records.

        Note:
            If database pool is not available, returns empty list.
        """
        # Check if pool is available
        if not self.pool:
            print("[DATABASE] Using fallback for recent positions (returning empty)")
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    ST_X(position) as x, ST_Y(position) as y, ST_Z(position) as z,
                    health_percent, is_in_combat, detected_objects, timestamp
                FROM player_positions
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, session_id, limit)

        return [
            {
                'position': {'x': row['x'], 'y': row['y'], 'z': row['z']},
                'health_percent': row['health_percent'],
                'is_in_combat': row['is_in_combat'],
                'detected_objects': row['detected_objects'],
                'timestamp': row['timestamp'].timestamp()
            }
            for row in rows
        ]

    async def log_strategy(
        self,
        policy_id: Optional[int],
        action: str,
        reasoning: str,
        priority: int,
        context: Dict[str, Any]
    ) -> int:
        """
        Log a strategy decision.

        Args:
            policy_id: Associated policy ID.
            action: Action taken.
            reasoning: Reasoning behind the action.
            priority: Priority of the action.
            context: Context dictionary.

        Returns:
            ID of logged strategy.

        Note:
            If database pool is not available, falls back to docker exec
            and returns a timestamp-based ID for tracking.
        """
        # Check if pool is available
        if not self.pool:
            # Fallback using docker exec
            import time
            print("[DATABASE] Using fallback for strategy logging")
            timestamp = int(time.time())
            context_json = json.dumps(context).replace("'", "''")

            query = f"""
                INSERT INTO strategy_logs (policy_id, action, reasoning, priority, context)
                VALUES ({policy_id if policy_id else 'NULL'}, '{action}', '{reasoning}', {priority}, '{context_json}'::jsonb)
                RETURNING id;
            """
            result = self._docker_query(query)
            if result.strip():
                return int(result.strip().split()[0])
            return timestamp

        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                INSERT INTO strategy_logs (policy_id, action, reasoning, priority, context)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, policy_id, action, reasoning, priority, json.dumps(context))

            return int(result.split()[-1])

    async def cleanup_old_entries(self, days: int = 30):
        """
        Remove old spatial memory entries.

        Args:
            days: Remove entries older than this many days.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM spatial_memory
                WHERE last_seen < NOW() - INTERVAL '1 day' * $1
            """, days)

            print(f"[DATABASE] Cleaned up {result.split()[-1]} old entries")

    async def get_all_entries_by_type(
        self,
        object_type: str
    ) -> List[SpatialMemoryEntry]:
        """
        Get all entries of a specific type.

        Args:
            object_type: Type of object.

        Returns:
            List of SpatialMemoryEntry objects.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, object_type, label,
                       ST_X(position) as x, ST_Y(position) as y, ST_Z(position) as z,
                       last_seen, confidence, metadata
                FROM spatial_memory
                WHERE object_type = $1
                ORDER BY last_seen DESC
            """, object_type)

        return [
            SpatialMemoryEntry(
                entry_id=row['id'],
                object_type=row['object_type'],
                label=row['label'],
                position={'x': row['x'], 'y': row['y'], 'z': row['z']},
                last_seen=row['last_seen'].timestamp(),
                confidence=row['confidence'],
                metadata=row['metadata']
            )
            for row in rows
        ]

    async def statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics.
        """
        async with self.pool.acquire() as conn:
            spatial_count = await conn.fetchval("""
                SELECT COUNT(*) FROM spatial_memory
            """)

            player_positions_count = await conn.fetchval("""
                SELECT COUNT(*) FROM player_positions
            """)

            active_sessions = await conn.fetchval("""
                SELECT COUNT(DISTINCT session_id) FROM player_positions
                    WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)

            by_type = await conn.fetch("""
                SELECT object_type, COUNT(*) as count
                FROM spatial_memory
                GROUP BY object_type
            """)

        return {
            'spatial_entries': spatial_count,
            'total_player_positions': player_positions_count,
            'active_sessions': active_sessions,
            'entries_by_type': {row['object_type']: row['count'] for row in by_type}
        }
