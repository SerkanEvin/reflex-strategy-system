"""
Local PC Configuration
Centralized configuration for the Spinal Cord (Local PC).
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import os


@dataclass
class CaptureConfig:
    """Screen capture configuration."""
    monitor_index: int = 1  # Primary monitor
    fps: int = 60
    buffer_size: int = 5
    grayscale: bool = False

    # Window capture (if you want to capture specific window)
    capture_window: bool = False
    window_title: Optional[str] = None  # Set to game window title


@dataclass
class VisionConfig:
    """Vision system configuration."""
    model_path: Optional[str] = None  # Path to YOLO model weights
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"  # "cpu" or "cuda"

    # Detection categories
    enabled_detections: List[str] = None  # None = all enabled

    # 3D game specific
    estimate_depth: bool = True
    min_detection_size: int = 20


@dataclass
class InstinctConfig:
    """Instinct layer configuration."""
    critical_health_threshold: float = 30.0
    low_health_threshold: float = 50.0
    combat_distance_threshold: float = 150.0
    safe_distance_threshold: float = 300.0

    reflex_cooldown: float = 1.0  # Seconds between reflex commands
    threat_decay_time: float = 10.0  # Seconds to forget threats


@dataclass
class ActuatorConfig:
    """Actuator configuration."""
    # Mouse movement settings
    mouse_speed_range: Tuple[float, float] = (100, 300)
    mouse_jitter: float = 2.0
    mouse_pause_chance: float = 0.1
    mouse_pause_range: Tuple[float, float] = (0.05, 0.2)

    # Keyboard settings
    press_duration_range: Tuple[float, float] = (0.05, 0.15)
    key_delay_range: Tuple[float, float] = (0.05, 0.1)

    # Key bindings (customize for your game)
    key_bindings: Dict[str, str] = None


@dataclass
class VPSConfig:
    """VPS connection configuration."""
    vps_url: str = "ws://167.86.105.39:8000/ws"
    reconnect_interval: float = 5.0
    heartbeat_interval: float = 30.0
    connection_timeout: float = 10.0


@dataclass
class LocalConfig:
    """Main configuration for Local PC."""
    capture: CaptureConfig
    vision: VisionConfig
    instinct: InstinctConfig
    actuator: ActuatorConfig
    vps: VPSConfig

    # Enable/disable features
    enable_capture: bool = True
    enable_vision: bool = True
    enable_instinct: bool = True
    enable_actuator: bool = True
    enable_vps_connection: bool = True

    # Debug settings
    debug_mode: bool = False
    log_detections: bool = False
    visualize: bool = False  # Show detection overlays

    @classmethod
    def from_env(cls) -> "LocalConfig":
        """Load configuration from environment variables."""
        return cls(
            capture=CaptureConfig(
                monitor_index=int(os.getenv("CAPTURE_MONITOR_INDEX", "1")),
                fps=int(os.getenv("CAPTURE_FPS", "60")),
                buffer_size=int(os.getenv("CAPTURE_BUFFER_SIZE", "5")),
                grayscale=os.getenv("CAPTURE_GRAYSCALE", "false").lower() == "true",
                capture_window=os.getenv("CAPTURE_WINDOW", "false").lower() == "true",
                window_title=os.getenv("CAPTURE_WINDOW_TITLE")
            ),
            vision=VisionConfig(
                model_path=os.getenv("VISION_MODEL_PATH"),
                confidence_threshold=float(os.getenv("VISION_CONFIDENCE", "0.5")),
                iou_threshold=float(os.getenv("VISION_IOU_THRESHOLD", "0.45")),
                device=os.getenv("VISION_DEVICE", "cpu"),
                estimate_depth=os.getenv("VISION_ESTIMATE_DEPTH", "true").lower() == "true",
                min_detection_size=int(os.getenv("VISION_MIN_SIZE", "20"))
            ),
            instinct=InstinctConfig(
                critical_health_threshold=float(os.getenv("INSTINCT_CRITICAL_HP", "30.0")),
                low_health_threshold=float(os.getenv("INSTINCT_LOW_HP", "50.0")),
                combat_distance_threshold=float(os.getenv("INSTINCT_COMBAT_DISTANCE", "150.0")),
                safe_distance_threshold=float(os.getenv("INSTINCT_SAFE_DISTANCE", "300.0")),
                reflex_cooldown=float(os.getenv("INSTINCT_REFLEX_COOLDOWN", "1.0")),
                threat_decay_time=float(os.getenv("INSTINCT_THREAT_DECAY", "10.0"))
            ),
            actuator=ActuatorConfig(
                key_bindings={
                    "MOVE_UP": os.getenv("KEY_MOVE_UP", "w"),
                    "MOVE_DOWN": os.getenv("KEY_MOVE_DOWN", "s"),
                    "MOVE_LEFT": os.getenv("KEY_MOVE_LEFT", "a"),
                    "MOVE_RIGHT": os.getenv("KEY_MOVE_RIGHT", "d"),
                    "ATTACK": os.getenv("KEY_ATTACK", "1"),
                    "SKILL_1": os.getenv("KEY_SKILL_1", "q"),
                    "SKILL_2": os.getenv("KEY_SKILL_2", "e"),
                    "USE_POTION": os.getenv("KEY_POTION", "space"),
                    "INTERACT": os.getenv("KEY_INTERACT", "f"),
                    "SPRINT": os.getenv("KEY_SPRINT", "shift"),
                }
            ),
            vps=VPSConfig(
                vps_url=os.getenv("VPS_URL", "ws://167.86.105.39:8000/ws"),
                reconnect_interval=float(os.getenv("VPS_RECONNECT_INTERVAL", "5.0")),
                heartbeat_interval=float(os.getenv("VPS_HEARTBEAT_INTERVAL", "30.0")),
                connection_timeout=float(os.getenv("VPS_CONNECTION_TIMEOUT", "10.0"))
            ),
            enable_capture=os.getenv("ENABLE_CAPTURE", "true").lower() == "true",
            enable_vision=os.getenv("ENABLE_VISION", "true").lower() == "true",
            enable_instinct=os.getenv("ENABLE_INSTINCT", "true").lower() == "true",
            enable_actuator=os.getenv("ENABLE_ACTUATOR", "true").lower() == "true",
            enable_vps_connection=os.getenv("ENABLE_VPS_CONNECTION", "true").lower() == "true",
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            log_detections=os.getenv("LOG_DETECTIONS", "false").lower() == "true",
            visualize=os.getenv("VISUALIZE", "false").lower() == "true"
        )


# Default configuration instance
default_config = LocalConfig(
    capture=CaptureConfig(),
    vision=VisionConfig(),
    instinct=InstinctConfig(),
    actuator=ActuatorConfig(),
    vps=VPSConfig()
)


def get_config() -> LocalConfig:
    """Get configuration (from env if available, else default)."""
    try:
        return LocalConfig.from_env()
    except:
        return default_config
