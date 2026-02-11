"""
Exploration Tracker Module (Local)
Tracks player exploration using a grid system.
Marks visited areas, tracks discoveries, and calculates exploration percentage.
"""
import math
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridCell:
    """A single grid cell in the exploration system."""
    grid_x: int
    grid_y: int
    position: Tuple[float, float]  # (x, y) center of cell
    visited: bool
    explored: bool
    visits: int
    first_visit: float  # unix timestamp
    last_visit: float
    discoveries: Dict[str, int]  # counts by type


class ExplorationTracker:
    """
    Tracks exploration using a grid-based system.
    Marks cells as visited when player moves through them.
    """

    DEFAULT_GRID_SIZE = 50.0  # Grid cells are 50x50 units
    FOG_OF_WAR_RADIUS = 1  # Number of cells to reveal as explored (not visited)

    def __init__(
        self,
        grid_size: float = DEFAULT_GRID_SIZE,
        fog_radius: int = FOG_OF_WAR_RADIUS,
        max_cached_cells: int = 10000
    ):
        """
        Initialize exploration tracker.

        Args:
            grid_size: Size of each grid cell in game units.
            fog_radius: Number of neighboring cells to mark as explored.
            max_cached_cells: Maximum cells to cache in memory.
        """
        self.grid_size = grid_size
        self.fog_radius = fog_radius
        self.max_cached_cells = max_cached_cells

        # In-memory cache of visited cells (grid_x, grid_y) -> GridCell
        self._cells_cache: Dict[Tuple[int, int], GridCell] = {}

        # Statistics
        self.total_cells_seen = 0
        self.total_cells_visited = 0

    def position_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert map position to grid coordinates.

        Args:
            x: Map X coordinate.
            y: Map Y coordinate.

        Returns:
            (grid_x, grid_y) tuple.
        """
        grid_x = int(math.floor(x / self.grid_size))
        grid_y = int(math.floor(y / self.grid_size))
        return grid_x, grid_y

    def grid_to_position(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to center position.

        Args:
            grid_x: Grid X coordinate.
            grid_y: Grid Y coordinate.

        Returns:
            (x, y) center of grid cell.
        """
        x = (grid_x + 0.5) * self.grid_size
        y = (grid_y + 0.5) * self.grid_size
        return x, y

    def update_position(
        self,
        position: Optional[Dict[str, float]],
        detections: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update player position and mark cell as visited.

        Args:
            position: Dict with 'x', 'y', 'z' keys (or None).
            detections: List of detection dictionaries.

        Returns:
            Update result with grid info.
        """
        if not position or 'x' not in position or 'y' not in position:
            return {
                "success": False,
                "error": "Invalid position"
            }

        x = position['x']
        y = position['y']
        grid_x, grid_y = self.position_to_grid(x, y)
        cell_center = self.grid_to_position(grid_x, grid_y)

        # Parse discoveries
        discoveries = self._parse_discoveries(detections) if detections else {}

        # Get or create cell
        cell = self._cells_cache.get((grid_x, grid_y))

        if cell is None:
            # New cell
            now = time.time()
            cell = GridCell(
                grid_x=grid_x,
                grid_y=grid_y,
                position=(cell_center[0], cell_center[1]),
                visited=True,
                explored=True,
                visits=1,
                first_visit=now,
                last_visit=now,
                discoveries=discoveries
            )
            self._cells_cache[(grid_x, grid_y)] = cell
            self.total_cells_seen += 1

            logger.info(f"[EXPLORATION] New cell visited: ({grid_x}, {grid_y})")
            return {
                "success": True,
                "new_cell": True,
                "grid": (grid_x, grid_y),
                "position": cell_center,
                "discoveries": discoveries
            }
        else:
            # Update existing cell
            now = time.time()
            cell.visited = True
            cell.explored = True
            cell.visits += 1
            cell.last_visit = now

            # Merge discoveries
            for dtype, count in discoveries.items():
                cell.discoveries[dtype] = cell.discoveries.get(dtype, 0) + count

            self.total_cells_visited += 1

            return {
                "success": True,
                "new_cell": False,
                "grid": (grid_x, grid_y),
                "visits": cell.visits,
                "discoveries": cell.discoveries
            }

    def mark_neighbor_cells_explored(self, center_x: int, center_y: int) -> int:
        """
        Mark neighboring cells as explored (fog of war).

        Args:
            center_x: Grid X coordinate of center cell.
            center_y: Grid Y coordinate of center cell.

        Returns:
            Number of newly explored cells.
        """
        new_explored = 0

        for dx in range(-self.fog_radius, self.fog_radius + 1):
            for dy in range(-self.fog_radius, self.fog_radius + 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = center_x + dx, center_y + dy
                cell = self._cells_cache.get((nx, ny))

                if cell is None:
                    # Create and mark as explored (not visited)
                    cell_center = self.grid_to_position(nx, ny)
                    cell = GridCell(
                        grid_x=nx,
                        grid_y=ny,
                        position=(cell_center[0], cell_center[1]),
                        visited=False,
                        explored=True,
                        visits=0,
                        first_visit=0,
                        last_visit=0,
                        discoveries={}
                    )
                    self._cells_cache[(nx, ny)] = cell
                    self.total_cells_seen += 1
                    new_explored += 1
                elif not cell.explored:
                    cell.explored = True
                    new_explored += 1

        if new_explored > 0:
            logger.info(f"[EXPLORATION] Explored {new_explored} neighboring cells")

        return new_explored

    def get_cell(self, grid_x: int, grid_y: int) -> Optional[GridCell]:
        """Get a specific grid cell."""
        return self._cells_cache.get((grid_x, grid_y))

    def get_nearby_unexplored(
        self,
        center_x: int,
        center_y: int,
        radius: int = 5
    ) -> List[GridCell]:
        """
        Find unexplored cells near a position.

        Args:
            center_x: Grid X coordinate.
            center_y: Grid Y coordinate.
            radius: Search radius in cells.

        Returns:
            List of unexplored cells.
        """
        unexplored = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = center_x + dx, center_y + dy
                cell = self._cells_cache.get((nx, ny))

                if cell is not None and not cell.visited:
                    unexplored.append(cell)

        # Sort by distance (closest first)
        unexplored.sort(key=lambda c: math.hypot(c.grid_x - center_x, c.grid_y - center_y))
        return unexplored

    def get_exploration_status(self) -> Dict[str, Any]:
        """
        Get overall exploration status.

        Returns:
            Dictionary with exploration stats.
        """
        visited_count = sum(1 for c in self._cells_cache.values() if c.visited)
        explored_count = sum(1 for c in self._cells_cache.values() if c.explored)
        total_count = len(self._cells_cache)

        # Calculate estimated map size (based on seen cells)
        if total_count > 0:
            x_coords = [c.grid_x for c in self._cells_cache.values()]
            y_coords = [c.grid_y for c in self._cells_cache.values()]

            if len(x_coords) > 1 and len(y_coords) > 1:
                width = max(x_coords) - min(x_coords) + 1
                height = max(y_coords) - min(y_coords) + 1
                estimated_total = width * height
            else:
                estimated_total = total_count
        else:
            estimated_total = 1

        visited_percent = (visited_count / estimated_total) * 100 if estimated_total > 0 else 0
        explored_percent = (explored_count / estimated_total) * 100 if estimated_total > 0 else 0

        # Aggregate discoveries
        total_discoveries = {}
        for cell in self._cells_cache.values():
            for dtype, count in cell.discoveries.items():
                total_discoveries[dtype] = total_discoveries.get(dtype, 0) + count

        return {
            "total_cells_seen": total_count,
            "estimated_total_cells": estimated_total,
            "visited_cells": visited_count,
            "explored_cells": explored_count,
            "visited_percent": round(visited_percent, 2),
            "explored_percent": round(explored_percent, 2),
            "total_discoveries": total_discoveries,
            "grid_size": self.grid_size
        }

    def _parse_discoveries(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Parse detections into discovery counts."""
        discoveries = {}

        for det in detections:
            dtype = det.get("detection_type", "unknown")
            discoveries[dtype] = discoveries.get(dtype, 0) + 1

        return discoveries

    def export_data(self) -> Dict[str, Any]:
        """Export all exploration data for saving."""
        return {
            "grid_size": self.grid_size,
            "cells": [c.__dict__ for c in self._cells_cache.values()],
            "statistics": self.get_exploration_status()
        }

    def import_data(self, data: Dict[str, Any]) -> None:
        """Import exploration data from saved state."""
        self.grid_size = data.get("grid_size", self.DEFAULT_GRID_SIZE)

        for cell_data in data.get("cells", []):
            cell = GridCell(
                grid_x=cell_data["grid_x"],
                grid_y=cell_data["grid_y"],
                position=tuple(cell_data["position"]),
                visited=cell_data["visited"],
                explored=cell_data["explored"],
                visits=cell_data["visits"],
                first_visit=cell_data["first_visit"],
                last_visit=cell_data["last_visit"],
                discoveries=cell_data["discoveries"]
            )
            self._cells_cache[(cell.grid_x, cell.grid_y)] = cell

        # Reset statistics
        self.total_cells_seen = len(self._cells_cache)
        self.total_cells_visited = sum(c.visits for c in self._cells_cache.values())

        logger.info(f"[EXPLORATION] Imported {len(self._cells_cache)} cells")

    def clear(self) -> None:
        """Clear all exploration data."""
        self._cells_cache.clear()
        self.total_cells_seen = 0
        self.total_cells_visited = 0
