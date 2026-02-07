"""
Shared data models between Local PC (Spinal Cord) and VPS (Brain).
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
from datetime import datetime
import json


class SecurityTier(Enum):
    """Security assessment levels."""
    DANGER = 0          # Active combat or critical health - Highest Priority
    ATTACK_MAY_OCCUR = 1 # Hostiles nearby but not engaged
    SECURE = 2          # No threats - Safe for administrative tasks


class DetectionType(Enum):
    """Types of objects the vision system can detect."""
    ENEMY = "enemy"
    FRIENDLY = "friendly"
    PLAYER = "player"
    RESOURCE = "resource"
    ITEM = "item"
    UI_ELEMENT = "ui_element"
    MARKET = "market"
    OTHER = "other"


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        """Get center coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y2 - self.y1


@dataclass
class Detection:
    """A single object detection."""
    bbox: BoundingBox
    detection_type: DetectionType
    label: str
    confidence: float
    distance: Optional[float] = None  # Estimated distance for 3D games
    health_percent: Optional[float] = None  # For enemies/players

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": asdict(self.bbox),
            "detection_type": self.detection_type.value,
            "label": self.label,
            "confidence": self.confidence,
            "distance": self.distance,
            "health_percent": self.health_percent
        }


@dataclass
class PlayerState:
    """Current player state."""
    health_percent: float
    max_health: float
    health: float
    mana_percent: Optional[float] = None
    level: Optional[int] = None
    position: Optional[Dict[str, float]] = None  # {x, y, z} for 3D games
    is_in_combat: bool = False
    is_dead: bool = False
    inventory_full_percent: Optional[float] = None
    coordinates: Optional[Dict[str, float]] = None  # In-game map coordinates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LocalState:
    """Complete state from Local PC perception layer."""
    timestamp: float  # Unix timestamp
    player: PlayerState
    detections: List[Detection]
    security_tier: SecurityTier
    active_targets: List[str]  # IDs of engaged targets
    frame_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "player": self.player.to_dict(),
            "detections": [d.to_dict() for d in self.detections],
            "security_tier": self.security_tier.value,
            "active_targets": self.active_targets,
            "frame_id": self.frame_id
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ReflexCommand:
    """Immediate reflex command from Instinct layer."""
    command: Literal["POT", "FLEE", "ATTACK", "DEFEND", "NONE"]
    priority: int  # 0 = highest, 10 = lowest
    reasoning: str
    target_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SpatialMemoryEntry:
    """Entry for spatial memory in SQL database."""
    entry_id: Optional[int]
    object_type: str  # "market", "spawn_point", "resource_node", etc.
    label: str  # Specific identification
    position: Dict[str, float]  # {x, y, z} in game coordinates
    last_seen: float  # Unix timestamp
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result.pop("entry_id", None)
        return result


@dataclass
class StrategyPolicy:
    """High-level strategy policy from VPSBrain/LLM."""
    policy_id: Optional[int]
    priority: int
    action: str  # High-level action: "GO_TO_MARKET", "HUNT", "GATHER", "RETREAT"
    target: Optional[Dict[str, Any]]  # Target location/object details
    reasoning: str
    created_at: float
    estimated_duration: Optional[float] = None
    conditions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result.pop("policy_id", None)
        return result


@dataclass
class ActionCommand:
    """Final action command for Actuator layer."""
    reflex_command: Optional[ReflexCommand]
    strategy_policy: Optional[StrategyPolicy]
    authorized: bool
    execution_mode: Literal["REFLEX_ONLY", "STRATEGY_ONLY", "MERGED"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reflex_command": self.reflex_command.to_dict() if self.reflex_command else None,
            "strategy_policy": self.strategy_policy.to_dict() if self.strategy_policy else None,
            "authorized": self.authorized,
            "execution_mode": self.execution_mode
        }


# Message types for WebSocket communication
class MessageType(Enum):
    """WebSocket message types."""
    # Local -> VPS
    STATE_UPDATE = "state_update"
    HEARTBEAT = "heartbeat"
    ERROR_REPORT = "error_report"

    # VPS -> Local
    STRATEGY_UPDATE = "strategy_update"
    POLICY_CONFIRMATION = "policy_confirmation"
    DATABASE_QUERY = "database_query"
    DATABASE_RESPONSE = "database_response"

    # Bidirectional
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketMessage:
    """Base WebSocket message."""
    type: MessageType
    payload: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            payload=data["payload"],
            timestamp=data.get("timestamp", datetime.now().timestamp())
        )
