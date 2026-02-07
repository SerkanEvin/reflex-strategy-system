"""
Local Coordinator - Main Orchestration Module
Coordinates all local modules: Capture, Vision, Instinct, Actuator, and VPS Client.
Acts as the central hub for the Local PC (Spinal Cord).
"""
import asyncio
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import (
    LocalState, PlayerState, SecurityTier,
    ReflexCommand, StrategyPolicy, ActionCommand
)
from shared.constants import Thresholds

from capture import ScreenCapture
from vision import VisionSystem
from instinct import InstinctLayer
from actuator import Actuator
from client import VPSClient
from config import LocalConfig


@dataclass
class CoordinatorStats:
    """Statistics for coordinator."""
    frames_processed: int = 0
    detections_made: int = 0
    reflex_commands_executed: int = 0
    strategy_commands_executed: int = 0
    start_time: float = 0.0
    last_frame_time: float = 0.0
    fps: float = 0.0


class LocalCoordinator:
    """
    Main coordinator for Local PC operations.
    Orchestrates the reflex-strategy pipeline.
    """

    def __init__(self, config: Optional[LocalConfig] = None):
        """
        Initialize local coordinator.

        Args:
            config: Configuration object. If None, uses default.
        """
        self.config = config or LocalConfig(
            capture=type('CaptureConfig', (), {})(),
            vision=type('VisionConfig', (), {})(),
            instinct=type('InstinctConfig', (), {})(),
            actuator=type('ActuatorConfig', (), {})(),
            vps=type('VPSConfig', (), {})(),
        )

        # Core components
        self.capture = None
        self.vision = None
        self.instinct = None
        self.actuator = None
        self.vps_client = None

        # State
        self.running = False
        self.frame_id = 0
        self.latest_state: Optional[LocalState] = None
        self.current_strategy: Optional[StrategyPolicy] = None

        # Statistics
        self.stats = CoordinatorStats(start_time=time.time())

        # Synchronization
        self.state_update_interval = 0.5  # Seconds between state updates to VPS
        self.last_state_update = 0.0

    async def initialize(self):
        """Initialize all subsystems."""
        print("[COORDINATOR] Initializing Local PC (Spinal Cord)...")

        # Initialize capture
        if self.config.enable_capture:
            self.capture = ScreenCapture(
                fps=60,
                buffer_size=5,
                grayscale=False
            )
            self.capture.start()
            print("[COORDINATOR] Screen capture initialized")

        # Initialize vision
        if self.config.enable_vision:
            self.vision = VisionSystem(
                model_path=self.config.vision.model_path,
                confidence_threshold=self.config.vision.confidence_threshold,
                device=self.config.vision.device
            )
            print("[COORDINATOR] Vision system initialized")

        # Initialize instinct
        if self.config.enable_instinct:
            self.instinct = InstinctLayer()
            print("[COORDINATOR] Instinct layer initialized")

        # Initialize actuator
        if self.config.enable_actuator:
            self.actuator = Actuator()
            print("[COORDINATOR] Actuator initialized")

        # Initialize VPS client
        if self.config.enable_vps_connection:
            await self._initialize_vps_client()
            print("[COORDINATOR] VPS client initialized")

        print("[COORDINATOR] All systems initialized")

    async def _initialize_vps_client(self):
        """Initialize VPS WebSocket client."""
        self.vps_client = VPSClient(
            vps_url=self.config.vps.vps_url,
            reconnect_interval=self.config.vps.reconnect_interval
        )

        # Set up handlers
        self.vps_client.set_strategy_handler(self._on_strategy_update)
        self.vps_client.set_connect_handler(self._on_vps_connect)
        self.vps_client.set_disconnect_handler(self._on_vps_disconnect)

        # Start connection in background
        asyncio.create_task(self.vps_client.start_async())

    async def start(self):
        """Start the main processing loop."""
        self.running = True
        print("[COORDINATOR] Starting main loop...")

        # Start VPS connection
        if self.vps_client and self.config.enable_vps_connection:
            asyncio.create_task(self.vps_client.start_async())

        # Main processing loop
        while self.running:
            await self.process_frame()
            await asyncio.sleep(1.0 / 60.0)  # ~60 FPS

    async def process_frame(self):
        """Process a single frame through the pipeline."""
        start_time = time.time()

        # 1. Capture frame
        frame = None
        if self.capture:
            frame = self.capture.get_frame()

        if frame is None:
            await asyncio.sleep(0.01)
            return

        # 2. Vision - Detect objects
        detections = []
        if self.vision:
            detections = self.vision.detect(frame)

            if self.config.log_detections:
                print(f"[VISION] Found {len(detections)} detections")

        # 3. Estimate player state
        player_state = await self._estimate_player_state(frame, detections)

        # 4. Instinct - Security assessment
        security_tier = SecurityTier.SECURE
        reflex_command = None
        active_targets = []

        if self.instinct:
            security_tier = self.instinct.assess_security(player_state, detections)
            reflex_command = self.instinct.generate_reflex_command(
                security_tier, player_state, detections
            )
            active_targets = [
                d.label for d in detections
                if d.detection_type.value == "enemy"
            ]

        # 5. Build local state
        self.frame_id += 1
        self.latest_state = LocalState(
            timestamp=time.time(),
            player=player_state,
            detections=detections,
            security_tier=security_tier,
            active_targets=active_targets,
            frame_id=self.frame_id
        )

        # 6. Decision - Reflex vs Strategy
        await self._make_decision(reflex_command, security_tier)

        # 7. Send state update to VPS (throttled)
        current_time = time.time()
        if (self.vps_client and
            self.vps_client.is_connected() and
            current_time - self.last_state_update >= self.state_update_interval):
            await self.vps_client.send_state_update(self.latest_state)
            self.last_state_update = current_time

        # Update statistics
        self.stats.frames_processed += 1
        self.stats.detections_made += len(detections)
        self.stats.last_frame_time = time.time()
        if self.stats.frames_processed % 30 == 0:
            elapsed = time.time() - self.stats.start_time
            self.stats.fps = self.stats.frames_processed / elapsed

    async def _estimate_player_state(
        self,
        frame,
        detections
    ) -> PlayerState:
        """
        Estimate current player state from frame and detections.

        Args:
            frame: Captured frame.
            detections: List of detections.

        Returns:
            Estimated PlayerState.
        """
        # Default state
        player_state = PlayerState(
            health_percent=100.0,
            max_health=1000.0,
            health=1000.0,
            mana_percent=100.0,
            level=None,
            is_in_combat=False,
            is_dead=False,
            inventory_full_percent=None
        )

        # Extract from UI elements if vision available
        if self.vision and frame is not None:
            ui_elements = self.vision.detect_ui_elements(frame)
            estimated_state = self.vision.estimate_player_state(frame, ui_elements)

            # Update with estimated values
            player_state.health_percent = estimated_state.health_percent
            player_state.mana_percent = estimated_state.mana_percent
            player_state.level = estimated_state.level

            # Check for combat based on enemy detections
            enemies = [d for d in detections if d.detection_type.value == "enemy"]
            nearby_enemies = [
                e for e in enemies
                if e.distance and e.distance < Thresholds.COMBAT_DISTANCE.value
            ]
            player_state.is_in_combat = len(nearby_enemies) > 0

        return player_state

    async def _make_decision(
        self,
        reflex_command: Optional[ReflexCommand],
        security_tier: SecurityTier
    ):
        """
        Make action decision based on reflex and strategy.

        Args:
            reflex_command: Generated reflex command.
            security_tier: Current security tier.
        """
        if reflex_command is None:
            return

        # Build action command
        action = ActionCommand(
            reflex_command=reflex_command,
            strategy_policy=self.current_strategy,
            authorized=True,  # Local instincts always authorized
            execution_mode="MERGED"
        )

        # Check if reflex should override strategy
        if self.instinct and self.instinct.check_conflict(reflex_command, security_tier):
            action.execution_mode = "REFLEX_ONLY"
            print("[COORDINATOR] Reflex override - executing only reflex command")

        # Execute action
        if self.actuator:
            if action.execution_mode == "REFLEX_ONLY" or security_tier == SecurityTier.DANGER:
                self.actuator.execute_reflex(reflex_command)
                self.stats.reflex_commands_executed += 1

            elif self.current_strategy and action.execution_mode == "MERGED":
                # Check if strategy is compatible with current security
                if security_tier == SecurityTier.SECURE:
                    self.actuator.execute_strategy(self.current_strategy)
                    self.stats.strategy_commands_executed += 1

    async def _on_strategy_update(self, strategy: StrategyPolicy):
        """
        Handle strategy update from VPS.

        Args:
            strategy: New strategy policy.
        """
        print(f"[COORDINATOR] Strategy update: {strategy.action}")
        self.current_strategy = strategy

    def _on_vps_connect(self):
        """Handle VPS connection."""
        print("[COORDINATOR] Connected to VPS Brain")

    def _on_vps_disconnect(self):
        """Handle VPS disconnection."""
        print("[COORDINATOR] Disconnected from VPS Brain")

    async def stop(self):
        """Stop the coordinator and cleanup."""
        print("[COORDINATOR] Stopping...")
        self.running = False

        # Stop capture
        if self.capture:
            self.capture.stop()

        # Stop VPS client
        if self.vps_client:
            await self.vps_client.stop()

        # Stop actuator
        if self.actuator:
            self.actuator.stop_all_actions()

        print(f"[COORDINATOR] Stopped. Stats: {self.stats}")

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "frames_processed": self.stats.frames_processed,
            "detections_made": self.stats.detections_made,
            "reflex_executed": self.stats.reflex_commands_executed,
            "strategy_executed": self.stats.strategy_commands_executed,
            "fps": self.stats.fps,
            "uptime": time.time() - self.stats.start_time,
            "vps_connected": self.vps_client.is_connected() if self.vps_client else False
        }


def main():
    """Main entry point for Local PC."""
    import signal

    coordinator = LocalCoordinator()

    async def run():
        try:
            await coordinator.initialize()
            await coordinator.start()
        except KeyboardInterrupt:
            print("\n[COORDINATOR] Interrupted by user")
        finally:
            await coordinator.stop()

    # Run with signal handling
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[COORDINATOR] Shutting down...")


if __name__ == "__main__":
    main()
