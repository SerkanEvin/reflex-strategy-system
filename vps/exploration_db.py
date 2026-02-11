"""
Exploration Tracking Queries (VPS)
Database layer for exploration and location discovery system.
"""
import subprocess
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExplorationQueries:
    """
    Database queries for exploration tracking.
    Uses PostGIS for spatial operations.
    """

    def __init__(self, fallback_table: str = "reflex_strategy"):
        """
        Initialize exploration queries.

        Args:
            fallback_table: Database name for docker fallback.
        """
        self.fallback_table = fallback_table

    def _docker_query(self, query: str) -> str:
        """Execute query via docker exec."""
        try:
            result = subprocess.run([
                'docker', 'exec', 'supabase-db',
                'psql', '-U', 'postgres', '-d', self.fallback_table,
                '-t', '-c', query
            ], capture_output=True, text=True, timeout=30)
            return result.stdout
        except Exception as e:
            logger.warning(f"[DATABASE] Docker query failed: {e}")
            return ""

    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        Get exploration statistics from database.

        Returns:
            Dictionary with exploration status.
        """
        result = self._docker_query(
            "SELECT * FROM get_exploration_stats() LIMIT 1;"
        )

        if not result or 'visited_percent' not in result.lower():
            # Fallback to simple query if function not created
            result = self._docker_query(
                "SELECT COUNT(*) FROM exploration_grid;"
            )
            total = int(result.strip()) if result.strip().isdigit() else 0
            return {
                "total_cells_seen": total,
                "visited_percent": 0.0,
                "explored_percent": 0.0
            }

        # Parse result (simplified)
        lines = result.strip().split('\n')
        for line in lines:
            if not line or '|' in line or line.startswith('--'):
                continue
            try:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) == 5:
                    return {
                        "total_cells_seen": int(parts[0]) if parts[0].isdigit() else 0,
                        "visited_cells": int(parts[1]) if parts[1].isdigit() else 0,
                        "explored_cells": int(parts[2]) if parts[2].isdigit() else 0,
                        "visited_percent": float(parts[3]) if parts[3] else 0.0,
                        "explored_percent": float(parts[4]) if parts[4] else 0.0
                    }
            except:
                continue

        return {
            "total_cells_seen": 0,
            "visited_percent": 0.0,
            "explored_percent": 0.0
        }

    def get_nearby_unexplored(
        self,
        x: float,
        y: float,
        radius: float = 500.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get nearby unexplored grid cells.

        Args:
            x: X coordinate.
            y: Y coordinate.
            radius: Search radius in units.
            limit: Maximum results.

        Returns:
            List of unexplored cells.
        """
        result = self._docker_query(
            f"SELECT grid_x, grid_y, "
            f"ST_X(position) as x, ST_Y(position) as y "
            f"FROM exploration_grid "
            f"WHERE visited = false "
            f"AND ST_DWithin(position, ST_MakePoint({x}, {y}), {radius}) "
            f"LIMIT {limit};"
        )

        unexplored = []
        for line in result.strip().split('\n'):
            if not line or '|' in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                try:
                    unexplored.append({
                        "grid_x": int(parts[0]) if parts[0].isdigit() else 0,
                        "grid_y": int(parts[1]) if parts[1].isdigit() else 0,
                        "x": float(parts[2]),
                        "y": float(parts[3])
                    })
                except:
                    continue

        return unexplored

    def add_discovered_location(
        self,
        location_id: str,
        location_type: str,
        label: str,
        x: float,
        y: float,
        confidence: float = 0.7,
        confirmed: bool = False
    ) -> bool:
        """
        Add a discovered location to the database.

        Args:
            location_id: Unique location identifier.
            location_type: Type of location.
            label: Location name.
            x: X coordinate.
            y: Y coordinate.
            confidence: Confidence score.
            confirmed: Whether location is confirmed.

        Returns:
            True if successful.
        """
        try:
            result = self._docker_query(
                f"INSERT INTO discovered_locations "
                f"(location_id, location_type, label, position, confidence, confirmed) "
                f"VALUES ('{location_id}', '{location_type}', '{label}', "
                f"ST_MakePoint({x}, {y}), {confidence}, {confirmed}) "
                f"ON CONFLICT (location_id) DO UPDATE SET "
                f"visits = discovered_locations.visits + 1, "
                f"last_visited = now(), "
                f"confidence = LEAST(1.0, discovered_locations.confidence + 0.05), "
                f"confirmed = discovered_locations.confirmed OR ({int(confirmed)});"
            )
            return True
        except Exception as e:
            logger.error(f"[DATABASE] Failed to add location: {e}")
            return False

    def get_nearby_locations(
        self,
        x: float,
        y: float,
        location_type: Optional[str] = None,
        radius: float = 100.0,
        min_confidence: float = 0.6,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get nearby discovered locations.

        Args:
            x: X coordinate.
            y: Y coordinate.
            location_type: Filter by type (optional).
            radius: Search radius.
            min_confidence: Minimum confidence threshold.
            limit: Maximum results.

        Returns:
            List of location dictionaries.
        """
        type_filter = f"'{location_type}'" if location_type else "NULL"

        if location_type:
            query = (
                f"SELECT id, location_id, location_type, label, "
                f"ST_X(position) as x, ST_Y(position) as y, "
                f"confidence, confirmed FROM discovered_locations "
                f"WHERE location_type = '{location_type}' "
                f"AND confidence >= {min_confidence} "
                f"AND ST_DWithin(position, ST_MakePoint({x}, {y}), {radius}) "
                f"ORDER BY ST_Distance(position, ST_MakePoint({x}, {y})) "
                f"LIMIT {limit};"
            )
        else:
            query = (
                f"SELECT id, location_id, location_type, label, "
                f"ST_X(position) as x, ST_Y(position) as y, "
                f"confidence, confirmed FROM discovered_locations "
                f"WHERE confidence >= {min_confidence} "
                f"AND ST_DWithin(position, ST_MakePoint({x}, {y}), {radius}) "
                f"ORDER BY ST_Distance(position, ST_MakePoint({x}, {y})) "
                f"LIMIT {limit};"
            )

        result = self._docker_query(query)

        locations = []
        for line in result.strip().split('\n'):
            if not line or '|' in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    locations.append({
                        "id": int(parts[0]) if parts[0].isdigit() else 0,
                        "location_id": parts[1],
                        "location_type": parts[2],
                        "label": parts[3],
                        "x": float(parts[4]),
                        "y": float(parts[5]),
                        "confidence": float(parts[6]),
                        "confirmed": parts[7].lower() == 't' if len(parts) > 7 else False
                    })
                except:
                    continue

        return locations

    def get_all_locations(
        self,
        location_type: Optional[str] = None,
        confirmed_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all discovered locations.

        Args:
            location_type: Filter by type.
            confirmed_only: Only confirmed locations.
            limit: Maximum results.

        Returns:
            List of location dictionaries.
        """
        result = self._docker_query(
            f"SELECT id, location_id, location_type, label, "
            f"ST_X(position) as x, ST_Y(position) as y, "
            f"confidence, confirmed, visits FROM discovered_locations "
            f"ORDER BY confidence DESC, visits DESC "
            f"LIMIT {limit};"
        )

        locations = []
        for line in result.strip().split('\n'):
            if not line or '|' in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    locations.append({
                        "id": int(parts[0]) if parts[0].isdigit() else 0,
                        "location_id": parts[1],
                        "location_type": parts[2],
                        "label": parts[3],
                        "x": float(parts[4]),
                        "y": float(parts[5]),
                        "confidence": float(parts[6]),
                        "confirmed": parts[7].lower() == 't' if len(parts) > 7 else False
                    })
                except:
                    continue

        return locations
