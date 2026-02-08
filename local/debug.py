"""
Debug Mode Module (Local)
Provides comprehensive debugging, logging, and inspection features for the reflex-strategy system.
Captures actions, state changes, detections, and other events for analysis and inspection.
"""
import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DebugEvent:
    """A debug event with timestamp and action details."""
    timestamp: float
    event_type: str
    source: str
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "time_str": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "tags": self.tags
        }


@dataclass
class ActionRecord:
    """Record of an action taken by the system."""
    timestamp: float
    action_type: str  # "reflex" or "strategy"
    action: str
    reasoning: str
    priority: int
    source: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "time_str": datetime.fromtimestamp(self.timestamp).isoformat(),
            "action_type": self.action_type,
            "action": self.action,
            "reasoning": self.reasoning,
            "priority": self.priority,
            "source": self.source,
            "context": self.context
        }


@dataclass
class DetectionRecord:
    """Record of detections for a frame."""
    timestamp: float
    frame_id: int
    detections: List[Dict[str, Any]]
    player_state: Dict[str, Any]
    security_tier: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "time_str": datetime.fromtimestamp(self.timestamp).isoformat(),
            "frame_id": self.frame_id,
            "detections": self.detections,
            "player_state": self.player_state,
            "security_tier": self.security_tier
        }


class DebugLogger:
    """
    Captures and logs debug events, actions, and state changes.
    Provides thread-safe storage and file output.
    """

    def __init__(
        self,
        max_events: int = 10000,
        max_actions: int = 1000,
        max_detections: int = 500,
        log_file: Optional[str] = None
    ):
        """
        Initialize debug logger.

        Args:
            max_events: Maximum number of events to keep in memory.
            max_actions: Maximum number of action records to keep.
            max_detections: Maximum number of detection records to keep.
            log_file: Path to log file (default: logs/debug_YYYYMMDD.jsonl).
        """
        self.max_events = max_events
        self.max_actions = max_actions
        self.max_detections = max_detections

        # Thread-safe storage
        self._events_lock = threading.Lock()
        self._actions_lock = threading.Lock()
        self._detections_lock = threading.Lock()

        # In-memory storage (deque for O(1) append/popleft)
        self._events: deque = deque(maxlen=max_events)
        self._actions: deque = deque(maxlen=max_actions)
        self._detections: deque = deque(maxlen=max_detections)

        # Statistics
        self._stats = defaultdict(int)
        self._stats_lock = threading.Lock()

        # Log file setup
        self.log_file = log_file or self._default_log_file()
        self._ensure_log_dir()

        # Track session
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _default_log_file(self) -> str:
        """Generate default log file path."""
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        return str(logs_dir / f"debug_{date_str}.jsonl")

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        tags: Optional[List[str]] = None
    ):
        """
        Log a debug event.

        Args:
            event_type: Type of event (e.g., "system_start", "error", "state_change").
            source: Source of the event (e.g., "coordinator", "vision", "actuator").
            data: Event data dictionary.
            tags: Optional tags for filtering.
        """
        event = DebugEvent(
            timestamp=time.time(),
            event_type=event_type,
            source=source,
            data=data,
            tags=tags or []
        )

        with self._events_lock:
            self._events.append(event)

        with self._stats_lock:
            self._stats[f"event_{event_type}"] += 1
            self._stats[f"source_{source}"] += 1

        # Write to file
        self._write_log_line(event.to_dict())

    def log_action(
        self,
        action_type: str,
        action: str,
        reasoning: str,
        priority: int,
        source: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log an action taken by the system.

        Args:
            action_type: "reflex" or "strategy".
            action: Action name.
            reasoning: Reasoning behind the action.
            priority: Action priority (1=highest).
            source: Source of the action.
            context: Additional context.
        """
        record = ActionRecord(
            timestamp=time.time(),
            action_type=action_type,
            action=action,
            reasoning=reasoning,
            priority=priority,
            source=source,
            context=context or {}
        )

        with self._actions_lock:
            self._actions.append(record)

        with self._stats_lock:
            self._stats[f"action_{action_type}"] += 1
            self._stats[f"action_{action}"] += 1

        # Write to file with action prefix
        log_entry = record.to_dict()
        log_entry["_type"] = "action"
        self._write_log_line(log_entry)

    def log_detections(
        self,
        frame_id: int,
        detections: List[Dict[str, Any]],
        player_state: Dict[str, Any],
        security_tier: str
    ):
        """
        Log detections for a frame.

        Args:
            frame_id: Frame identifier.
            detections: List of detection dictionaries.
            player_state: Player state at time of detection.
            security_tier: Current security tier.
        """
        record = DetectionRecord(
            timestamp=time.time(),
            frame_id=frame_id,
            detections=[self._sanitize_detection(d) for d in detections],
            player_state=player_state,
            security_tier=security_tier
        )

        with self._detections_lock:
            self._detections.append(record)

        with self._stats_lock:
            self._stats["detections_logged"] += len(detections)

        # Write to file
        log_entry = record.to_dict()
        log_entry["_type"] = "detection"
        self._write_log_line(log_entry)

    def _sanitize_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize detection data for logging (remove large arrays)."""
        result = detection.copy()
        # Simplify bbox to just center and size
        if "bbox" in result:
            bbox = result["bbox"]
            if hasattr(bbox, "center"):
                result["bbox"] = {"center": bbox.center, "size": bbox.size}
        return result

    def _write_log_line(self, entry: Dict[str, Any]):
        """Write a log line to file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[DEBUG] Failed to write log: {e}")

    def get_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get events, optionally filtered.

        Args:
            event_type: Filter by event type.
            source: Filter by source.
            tags: Filter by tags (any match).
            limit: Maximum number of events to return.

        Returns:
            List of event dictionaries.
        """
        with self._events_lock:
            events = list(self._events)

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        if tags:
            events = [e for e in events if any(t in e.tags for t in tags)]

        # Return most recent events
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in events[:limit]]

    def get_actions(
        self,
        action_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get action records, optionally filtered.

        Args:
            action_type: Filter by action type ("reflex" or "strategy").
            action: Filter by action name.
            limit: Maximum number of records.

        Returns:
            List of action dictionaries.
        """
        with self._actions_lock:
            actions = list(self._actions)

        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        if action:
            actions = [a for a in actions if a.action == action]

        actions.sort(key=lambda a: a.timestamp, reverse=True)
        return [a.to_dict() for a in actions[:limit]]

    def get_recent_detections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent detection records."""
        with self._detections_lock:
            detections = list(self._detections)

        detections.sort(key=lambda d: d.timestamp, reverse=True)
        return [d.to_dict() for d in detections[:limit]]

    def get_statistics(self) -> Dict[str, int]:
        """Get debug statistics."""
        with self._stats_lock:
            base_stats = dict(self._stats)

        return {
            **base_stats,
            "total_events": len(self._events),
            "total_actions": len(self._actions),
            "total_detections_records": len(self._detections),
            "session_duration_sec": time.time() - self.session_start,
            "session_id": self.session_id
        }

    def export_session(self, output_path: Optional[str] = None) -> str:
        """
        Export entire session to a JSON file.

        Args:
            output_path: Output file path (default: logs/session_SESSIONID.json).

        Returns:
            Path to exported file.
        """
        if output_path is None:
            logs_dir = Path(self.log_file).parent
            output_path = str(logs_dir / f"session_{self.session_id}.json")

        export_data = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "session_end": time.time(),
            "statistics": self.get_statistics(),
            "events": [e.to_dict() for e in self._events],
            "actions": [a.to_dict() for a in self._actions],
            "detections": [d.to_dict() for d in self._detections]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"[DEBUG] Session exported to {output_path}")
        return output_path

    def clear(self):
        """Clear all stored data."""
        with self._events_lock:
            self._events.clear()
        with self._actions_lock:
            self._actions.clear()
        with self._detections_lock:
            self._detections.clear()
        logger.info("[DEBUG] All data cleared")


class DebugUI:
    """
    Provides a live debug UI for visualization of system state.
    Uses terminal-based display with formatted output.
    """

    def __init__(self, debug_logger: DebugLogger, enabled: bool = True):
        """
        Initialize debug UI.

        Args:
            debug_logger: DebugLogger instance.
            enabled: Enable/disable UI display.
        """
        self.logger = debug_logger
        self.enabled = enabled
        self._last_display_time = 0
        self._min_display_interval = 0.5  # minimum seconds between displays

    def display_status(
        self,
        coordinator_state: Dict[str, Any],
        player_state: Optional[Dict[str, Any]] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
        last_action: Optional[Dict[str, Any]] = None
    ):
        """
        Display live system status.

        Args:
            coordinator_state: Coordinator state (fps, stats, etc.).
            player_state: Current player state.
            detections: Current detections.
            last_action: Last action taken.
        """
        if not self.enabled:
            return

        # Rate limit display
        now = time.time()
        if now - self._last_display_time < self._min_display_interval:
            return
        self._last_display_time = now

        # Build display
        lines = self._build_status_display(
            coordinator_state, player_state, detections, last_action
        )

        # Print to console with ANSI escape codes for terminal clearing
        self._print_display(lines)

    def _build_status_display(
        self,
        coordinator_state: Dict[str, Any],
        player_state: Optional[Dict[str, Any]],
        detections: Optional[List[Dict[str, Any]]],
        last_action: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Build status display lines."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("Reflex-Strategy System - Debug Mode")
        lines.append("=" * 80)

        # Session info
        stats = self.logger.get_statistics()
        uptime = stats["session_duration_sec"]
        lines.append(f"Session: {stats['session_id']} | Uptime: {uptime:.1f}s")

        # Coordinator Stats
        if coordinator_state:
            lines.append("\n[Coordinator]")
            fps = coordinator_state.get("fps", 0)
            lines.append(f"  FPS: {fps:.1f}")
            lines.append(f"  Frames: {coordinator_state.get('frames_processed', 0):,}")
            lines.append(f"  Reflex Commands: {coordinator_state.get('reflex_commands_executed', 0):,}")
            lines.append(f"  Strategy Commands: {coordinator_state.get('strategy_commands_executed', 0):,}")

        # Player State
        if player_state:
            lines.append("\n[Player]")
            lines.append(f"  Health: {player_state.get('health_percent', 0):.1f}%")
            lines.append(f"  Combat: {player_state.get('is_in_combat', False)}")
            lines.append(f"  Inventory: {player_state.get('inventory_full_percent', 0):.1f}%")

        # Detections
        if detections:
            lines.append(f"\n[Detections: {len(detections)}]")
            for det in detections[:5]:  # Show top 5
                label = det.get("label", "Unknown")
                dtype = det.get("detection_type", "unknown")
                conf = det.get("confidence", 0)
                dist = det.get("distance")
                dist_str = f"{dist:.1f}m" if dist else "?"
                lines.append(f"  - {label} ({dtype}) conf:{conf:.2f} dist:{dist_str}")

        # Last Action
        if last_action:
            atype = last_action.get("action_type", "unknown")
            action = last_action.get("action", "unknown")
            reasoning = last_action.get("reasoning", "")
            lines.append(f"\n[Last Action: {atype.upper()}]")
            lines.append(f"  Action: {action}")
            lines.append(f"  Reasoning: {reasoning[:60]}{'...' if len(reasoning) > 60 else ''}")

        # Debug Stats
        lines.append(f"\n[Debug Stats]")
        lines.append(f"  Events: {stats['total_events']:,}")
        lines.append(f"  Actions: {stats['total_actions']:,}")
        lines.append(f"  Detection Records: {stats['total_detections_records']:,}")

        # Footer
        lines.append("=" * 80)

        return lines

    def _print_display(self, lines: List[str]):
        """Print display with terminal clearing."""
        # ANSI escape codes for terminal operations
        clear = "\033[H\033[J"  # Clear screen and move cursor to top
        output = "\n".join(lines)
        print(clear + output, end="")

    def export_last_display(self, output_path: str):
        """
        Export the last display to a text file.

        Args:
            output_path: Path to output file.
        """
        lines = self._build_status_display({}, None, None, None)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def show_action_history(self, limit: int = 10):
        """Display recent action history."""
        actions = self.logger.get_actions(limit=limit)

        print("\n" + "=" * 80)
        print("Recent Action History")
        print("=" * 80)

        for action in actions:
            atype = action["action_type"].upper()
            name = action["action"]
            time_str = action["time_str"].split("T")[1][:8]
            print(f"[{time_str}] {atype:8s} {name:15s} - {action['reasoning'][:50]}")

        print("=" * 80)

    def show_event_log(self, limit: int = 20):
        """Display recent event log."""
        events = self.logger.get_events(limit=limit)

        print("\n" + "=" * 80)
        print("Recent Events")
        print("=" * 80)

        for event in events:
            etype = event["event_type"]
            source = event["source"]
            time_str = event["time_str"].split("T")[1][:8]
            print(f"[{time_str}] {source:15s} {etype}")

        print("=" * 80)


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> Optional[DebugLogger]:
    """Get the global debug logger instance."""
    return _debug_logger


def init_debug_mode(
    enabled: bool = True,
    max_events: int = 10000,
    log_file: Optional[str] = None
) -> Optional[DebugLogger]:
    """
    Initialize debug mode.

    Args:
        enabled: Enable debug mode.
        max_events: Maximum events to keep in memory.
        log_file: Path to log file.

    Returns:
        DebugLogger instance if enabled, None otherwise.
    """
    global _debug_logger

    if not enabled:
        _debug_logger = None
        return None

    _debug_logger = DebugLogger(
        max_events=max_events,
        log_file=log_file
    )

    logger.info(f"[DEBUG] Debug mode initialized. Log file: {_debug_logger.log_file}")

    return _debug_logger


def log_debug(*args, **kwargs):
    """
    Log a debug event if debug mode is enabled.
    Wrapper for convenience.

    Args:
        *args: Passed to DebugLogger.log_event.
        **kwargs: Passed to DebugLogger.log_event.
    """
    if _debug_logger:
        _debug_logger.log_event(*args, **kwargs)


def log_debug_action(*args, **kwargs):
    """
    Log a debug action if debug mode is enabled.
    Wrapper for convenience.

    Args:
        *args: Passed to DebugLogger.log_action.
        **kwargs: Passed to DebugLogger.log_action.
    """
    if _debug_logger:
        _debug_logger.log_action(*args, **kwargs)
