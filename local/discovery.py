"""
Auto-Discovery Module (Local)
Automatically detects, identifies, and catalogs new locations.
Uses vision data and spatial reasoning to discover POIs.
"""
import time
import hashlib
import math
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LocationType(Enum):
    """Types of locations that can be auto-discovered."""
    MARKET = "market"
    VENDOR = "vendor"
    HEALER = "healer"
    SPAWN = "spawn"
    PORTAL = "portal"
    DUNGEON = "dungeon"
    RESOURCE = "resource"
    SAFE_ZONE = "safe_zone"
    POI = "poi"              # Point of Interest


@dataclass
class DiscoveredLocation:
    """A discovered location."""
    location_id: str
    location_type: LocationType
    label: str
    position: Tuple[float, float, float]  # (x, y, z)
    discovered_at: float  # unix timestamp
    confirmed: bool
    confidence: float
    visits: int
    last_visited: float
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location_id": self.location_id,
            "location_type": self.location_type.value,
            "label": self.label,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "discovered_at": self.discovered_at,
            "confirmed": self.confirmed,
            "confidence": self.confidence,
            "visits": self.visits,
            "last_visited": self.last_visited,
            "features": self.features
        }


class LocationPatterns:
    """
    Pattern database for recognizing different location types.
    Matches visual and contextual cues to location types.
    """

    VENDOR_PATTERNS = [
        "merchant", "shop", "trading post", "trader", "vendor",
        "store", "market stall", "crafting"
    ]

    MARKET_PATTERNS = [
        "market", "bazaar", "auction", "marketplace", "exchange"
    ]

    HEALER_PATTERNS = [
        "clinic", "healer", "hospital", "temple", "shrine",
        "priest", "monastery", "apothecary", "pharmacy"
    ]

    SPAWN_PATTERNS = [
        "starting area", "spawn point", "hub", "sanctuary"
    ]

    PORTAL_PATTERNS = [
        "portal", "teleport", "gateway", "waypoint", "portal stone"
    ]

    RESOURCE_PATTERNS = [
        "mine", "quarry", "forest", "herb garden", "fishing spot"
    ]

    SAFE_ZONE_PATTERNS = [
        "safe zone", "town", "village", "inn", "rest area",
        "city", "settlement", "stronghold"
    ]

    @classmethod
    def identify_type(cls, labels: List[str]) -> Optional[LocationType]:
        """
        Identify location type from detected labels.

        Args:
            labels: List of text labels detected.

        Returns:
            LocationType if matched, None otherwise.
        """
        if not labels:
            return None

        label_text = " ".join([l.lower() for l in labels])

        # Check patterns in priority order
        for patterns, loc_type in [
            (cls.MARKET_PATTERNS, LocationType.MARKET),
            (cls.VENDOR_PATTERNS, LocationType.VENDOR),
            (cls.HEALER_PATTERNS, LocationType.HEALER),
            (cls.PORTAL_PATTERNS, LocationType.PORTAL),
            (cls.SAFE_ZONE_PATTERNS, LocationType.SAFE_ZONE),
            (cls.RESOURCE_PATTERNS, LocationType.RESOURCE),
            (cls.SPAWN_PATTERNS, LocationType.SPAWN)
        ]:
            for pattern in patterns:
                if pattern in label_text:
                    return loc_type

        return LocationType.POI

    @classmethod
    def get_confidence(cls, location_type: LocationType, detections: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on detection quality.

        Args:
            location_type: Type of location.
            detections: List of detections.

        Returns:
            Confidence score (0.0 - 1.0).
        """
        if not detections:
            return 0.5

        base_confidence = 0.6

        # High confidence indicators
        for det in detections:
            conf = det.get("confidence", 0)

            # High confidence detection
            if conf >= 0.9:
                base_confidence += 0.2

            # Enemy at vendor (confirms it's safe area)
            if location_type in [LocationType.VENDOR, LocationType.MARKET]:
                if det.get("detection_type") == "friendly":
                    base_confidence += 0.1

        # Multiple vendors in area = likely market
        if location_type == LocationType.MARKET:
            vendor_count = sum(1 for d in detections if d.get("detection_type") == "vendor")
            if vendor_count >= 3:
                base_confidence += 0.2
            elif vendor_count >= 2:
                base_confidence += 0.1

        # Clamp to [0, 1]
        return min(1.0, max(0.0, base_confidence))


class LocationMatcher:
    """
    Matches new detections with known locations.
    Handles proximity matching and confidence updates.
    """

    def __init__(
        self,
        position_tolerance: float = 50.0,
        min_confidence_to_match: float = 0.6
    ):
        """
        Initialize location matcher.

        Args:
            position_tolerance: Maximum distance to match existing location (units).
            min_confidence_to_match: Minimum detection confidence to attempt match.
        """
        self.position_tolerance = position_tolerance
        self.min_confidence_to_match = min_confidence_to_match

        # Map location_id -> DiscoveredLocation
        self._known_locations: Dict[str, DiscoveredLocation] = {}

        # Spatial index: (grid_x, grid_y) -> [location_ids]
        self._spatial_index: Dict[Tuple[int, int], List[str]] = {}

    def find_nearby_match(
        self,
        position: Tuple[float, float, float],
        location_type: LocationType,
        label: Optional[str] = None
    ) -> Optional[DiscoveredLocation]:
        """
        Find nearby known location that matches.

        Args:
            position: (x, y, z) position.
            location_type: Type to match.
            label: Optional label to match.

        Returns:
            Matching DiscoveredLocation or None.
        """
        x, y, z = position
        grid_x = int(x // self.position_tolerance)
        grid_y = int(y // self.position_tolerance)

        # Search nearby grid cells
        candidates: List[Tuple[float, DiscoveredLocation]] = []

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                gx, gy = grid_x + dx, grid_y + dy
                for loc_id in self._spatial_index.get((gx, gy), []):
                    loc = self._known_locations.get(loc_id)
                    if loc is None:
                        continue

                    # Check type match
                    if loc.location_type != location_type:
                        continue

                    # Check position
                    distance = self._distance(loc.position, position)
                    if distance <= self.position_tolerance:
                        candidates.append((distance, loc))

        if not candidates:
            return None

        # Sort by distance and return closest
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def add_location(self, location: DiscoveredLocation) -> None:
        """
        Add a discovered location to the index.

        Args:
            location: Location to add.
        """
        self._known_locations[location.location_id] = location

        # Update spatial index
        x, y, _ = location.position
        grid_x = int(x // self.position_tolerance)
        grid_y = int(y // self.position_tolerance)

        if (grid_x, grid_y) not in self._spatial_index:
            self._spatial_index[(grid_x, grid_y)] = []

        if location.location_id not in self._spatial_index[(grid_x, grid_y)]:
            self._spatial_index[(grid_x, grid_y)].append(location.location_id)

        logger.info(f"[DISCOVERY] Added location: {location.location_id} ({location.location_type.value})")

    def update_location(self, location_id: str, **updates) -> Optional[DiscoveredLocation]:
        """
        Update an existing location.

        Args:
            location_id: ID of location to update.
            **updates: Fields to update.

        Returns:
            Updated DiscoveredLocation or None.
        """
        loc = self._known_locations.get(location_id)
        if loc is None:
            return None

        for key, value in updates.items():
            if hasattr(loc, key):
                setattr(loc, key, value)

        # Special handling for confidence
        if "visits" in updates:
            # Increase confidence on each visit
            loc.confidence = min(1.0, loc.confidence + 0.05)

        # Confirm if confidence high enough
        if loc.confidence >= 0.8:
            loc.confirmed = True

        loc.last_visited = time.time()

        logger.debug(f"[DISCOVERY] Updated location: {location_id}")
        return loc

    def get_locations(
        self,
        location_type: Optional[LocationType] = None,
        confirmed_only: bool = False
    ) -> List[DiscoveredLocation]:
        """
        Get locations by type.

        Args:
            location_type: Filter by type.
            confirmed_only: Only return confirmed locations.

        Returns:
            List of matching locations.
        """
        locations = list(self._known_locations.values())

        if location_type:
            locations = [loc for loc in locations if loc.location_type == location_type]

        if confirmed_only:
            locations = [loc for loc in locations if loc.confirmed]

        # Sort by confidence (descending)
        locations.sort(key=lambda l: l.confidence, reverse=True)
        return locations

    @staticmethod
    def _distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


class AutoDiscovery:
    """
    Main auto-discovery system.
    Processes detections and automatically discovers new locations.
    """

    def __init__(self):
        """Initialize auto-discovery system."""
        self.matcher = LocationMatcher()
        self.patterns = LocationPatterns()
        self._discovered_sessions: Dict[str, float] = {}  # location_id -> last_seen

    def process_frame(
        self,
        detections: List[Dict[str, Any]],
        player_position: Optional[Dict[str, float]] = None
    ) -> List[DiscoveredLocation]:
        """
        Process a frame of detections and discover new locations.

        Args:
            detections: List of detection dictionaries.
            player_position: Current player position.

        Returns:
            List of newly discovered locations this frame.
        """
        if not detections:
            return []

        new_discoveries = []

        # 1. Group detections by type
        grouped = self._group_detections(detections)

        # 2. Process each group
        for detection_type, group_detections in grouped.items():
            # Skip enemies and NPCs for location discovery
            if detection_type in ["enemy", "creature"]:
                continue

            # Calculate average position for this group
            avg_position = self._calculate_average_position(group_detections, player_position)

            # Try to find type from labels
            labels = [d.get("label", "") for d in group_detections if d.get("label")]
            loc_type = self.patterns.identify_type(labels)

            if loc_type:
                # Check for existing match
                match = self.matcher.find_nearby_match(
                    avg_position,
                    loc_type,
                    labels[0] if labels else None
                )

                if match:
                    # Update existing location
                    self._update_with_detections(match, group_detections, avg_position)
                    self._discovered_sessions[match.location_id] = time.time()
                else:
                    # New location discovered
                    location = self._create_location(
                        loc_type,
                        labels[0] if labels else f"{loc_type.value}",
                        avg_position,
                        group_detections
                    )

                    self.matcher.add_location(location)
                    self._discovered_sessions[location.location_id] = time.time()
                    new_discoveries.append(location)
                    logger.info(f"[DISCOVERY] NEW {loc_type.value}: {location.label}")

        # 3. Check for compound locations
        compound = self._detect_compound_locations(new_discoveries)
        new_discoveries.extend(compound)

        return new_discoveries

    def _group_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group detections by type."""
        grouped = {}

        for det in detections:
            dtype = det.get("detection_type", "unknown")
            if dtype not in grouped:
                grouped[dtype] = []
            grouped[dtype].append(det)

        return grouped

    def _calculate_average_position(
        self,
        detections: List[Dict[str, Any]],
        player_position: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, float]:
        """Calculate average position from detections."""
        if not detections and player_position:
            return (
                player_position.get("x", 0),
                player_position.get("y", 0),
                player_position.get("z", 0)
            )

        # Try to get position from detections
        positions = []
        for det in detections:
            dist = det.get("distance")
            if dist:
                # Estimate position from distance and player position
                px = player_position.get("x", 0) if player_position else 0
                py = player_position.get("y", 0) if player_position else 0
                # Simple estimate: detection is at distance (this is simplified)
                # In real implementation, would use angle from detection center
                positions.append((px, py, 0))

        if positions:
            # Use detection positions (simplified - all same as player for now)
            return positions[0]
        elif player_position:
            return (
                player_position.get("x", 0),
                player_position.get("y", 0),
                player_position.get("z", 0)
            )
        else:
            return (0.0, 0.0, 0.0)

    def _create_location(
        self,
        location_type: LocationType,
        label: str,
        position: Tuple[float, float, float],
        detections: List[Dict[str, Any]]
    ) -> DiscoveredLocation:
        """Create a new discovered location."""
        # Generate unique ID
        hash_input = f"{location_type.value}_{position[0]}_{position[1]}_{label}".encode()
        location_id = f"{location_type.value}_{hashlib.md5(hash_input).hexdigest()[:8]}"

        # Calculate confidence
        confidence = self.patterns.get_confidence(location_type, detections)

        now = time.time()

        # Extract features
        features = {
            "detections": len(detections),
            "types": list(set(d.get("detection_type", "unknown") for d in detections)),
            "labels": [d.get("label", "") for d in detections if d.get("label")]
        }

        return DiscoveredLocation(
            location_id=location_id,
            location_type=location_type,
            label=label,
            position=position,
            discovered_at=now,
            confirmed=confidence >= 0.8,
            confidence=confidence,
            visits=1,
            last_visited=now,
            features=features
        )

    def _update_with_detections(
        self,
        location: DiscoveredLocation,
        detections: List[Dict[str, Any]],
        position: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Update location with new detection data."""
        location.visits += 1
        location.last_visited = time.time()

        # Update confidence
        location.confidence = min(1.0, location.confidence + 0.05)

        # Update features
        location.features["detections"] = location.features.get("detections", 0) + len(detections)

        # Confirm if confidence high enough
        if location.confidence >= 0.8:
            location.confirmed = True

    def _detect_compound_locations(
        self,
        new_discoveries: List[DiscoveredLocation]
    ) -> List[DiscoveredLocation]:
        """
        Detect compound locations (e.g., multiple vendors = market).

        Args:
            new_discoveries: Newly discovered locations this frame.

        Returns:
            List of compound locations detected.
        """
        compounds = []

        # Look for multiple vendors in small area
        vendors = self.matcher.get_locations(LocationType.VENDOR)

        if len(vendors) >= 3:
            # Check if vendors are clustered
            avg_x = sum(loc.position[0] for loc in vendors) / len(vendors)
            avg_y = sum(loc.position[1] for loc in vendors) / len(vendors)

            # Calculate average distance from center
            avg_dist = sum(
                math.hypot(loc.position[0] - avg_x, loc.position[1] - avg_y)
                for loc in vendors
            ) / len(vendors)

            if avg_dist < 100:  # Clustered within 100 units
                # Create market location
                market_id = f"market_cluster_{hashlib.md5(f'{avg_x}{avg_y}'.encode()).hexdigest()[:8]}"

                # Check if market already exists
                existing_match = self.matcher.find_nearby_match(
                    (avg_x, avg_y, 0),
                    LocationType.MARKET
                )

                if not existing_match:
                    compounds.append(DiscoveredLocation(
                        location_id=market_id,
                        location_type=LocationType.MARKET,
                        label=f"Market Area ({len(vendors)} vendors)",
                        position=(avg_x, avg_y, 0),
                        discovered_at=time.time(),
                        confirmed=True,
                        confidence=0.9,
                        visits=1,
                        last_visited=time.time(),
                        features={"vendor_count": len(vendors)}
                    ))

        return compounds

    def get_all_locations(self) -> List[DiscoveredLocation]:
        """Get all discovered locations."""
        return list(self.matcher._known_locations.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        locations = self.get_all_locations()

        by_type = {}
        for loc in locations:
            ltype = loc.location_type.value
            if ltype not in by_type:
                by_type[ltype] = {"total": 0, "confirmed": 0}
            by_type[ltype]["total"] += 1
            if loc.confirmed:
                by_type[ltype]["confirmed"] += 1

        return {
            "total_locations": len(locations),
            "confirmed_locations": sum(1 for loc in locations if loc.confirmed),
            "by_type": by_type
        }
