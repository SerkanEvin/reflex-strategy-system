"""
Instinct Layer - The Spinal Cord
Handles immediate threats and reflex actions bypassing the VPS.
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import (
    Detection, DetectionType, LocalState, PlayerState,
    SecurityTier, ReflexCommand
)
from shared.constants import Thresholds, ReflexPriorities


class InstinctLayer:
    """
    Security assessment and reflex trigger system.
    Filters state and generates immediate reflex commands.
    """

    def __init__(self):
        """Initialize instinct layer."""
        # Thresholds
        self.critical_health_threshold = Thresholds.CRITICAL_HEALTH.value
        self.low_health_threshold = Thresholds.LOW_HEALTH.value
        self.combat_distance_threshold = Thresholds.COMBAT_DISTANCE.value

        # State tracking
        self.previous_security_tier = SecurityTier.SECURE
        self.in_combat_start_time = None
        self.last_reflex_command = None
        self.last_reflex_time = 0
        self.reflex_cooldown = 1.0  # Seconds between reflex commands

        # Threat tracking
        self.last_seen_enemies: Dict[str, float] = {}
        self.threat_decay_time = 10.0  # Seconds to forget a threat

    def assess_security(
        self,
        player_state: PlayerState,
        detections: List[Detection]
    ) -> SecurityTier:
        """
        Assess the security level of the current environment.

        Args:
            player_state: Current player state.
            detections: List of detected objects.

        Returns:
            SecurityTier classification.
        """
        current_time = time.time()

        # Check for DANGER conditions (Tier 0)
        if player_state.health_percent <= self.critical_health_threshold:
            return SecurityTier.DANGER

        if player_state.is_dead:
            return SecurityTier.DANGER

        if player_state.is_in_combat:
            self.in_combat_start_time = current_time
            return SecurityTier.DANGER

        # Check for active threats near player
        nearby_threats = self._count_nearby_threats(detections)
        if nearby_threats > 0:
            return SecurityTier.DANGER

        # Check for ATTACK_MAY_OCCUR conditions (Tier 1)
        if player_state.health_percent <= self.low_health_threshold:
            return SecurityTier.ATTACK_MAY_OCCUR

        if recent_threats := self._get_recent_threats(current_time):
            return SecurityTier.ATTACK_MAY_OCCUR

        # Check for enemies at medium range
        medium_range_threats = self._count_threats_at_range(
            detections,
            min_dist=Thresholds.COMBAT_DISTANCE.value,
            max_dist=Thresholds.SAFE_DISTANCE.value
        )
        if medium_range_threats > 0:
            return SecurityTier.ATTACK_MAY_OCCUR

        # Default to SECURE (Tier 2)
        return SecurityTier.SECURE

    def generate_reflex_command(
        self,
        security_tier: SecurityTier,
        player_state: PlayerState,
        detections: List[Detection]
    ) -> ReflexCommand:
        """
        Generate an immediate reflex command based on security assessment.

        Args:
            security_tier: Current security tier.
            player_state: Current player state.
            detections: List of detected objects.

        Returns:
            ReflexCommand to execute.
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_reflex_time < self.reflex_cooldown:
            return ReflexCommand(
                command="NONE",
                priority=ReflexPriorities.MEDIUM.value,
                reasoning="Reflex in cooldown",
                parameters=None
            )

        reflex_command = None

        # Generate command based on security tier
        if security_tier == SecurityTier.DANGER:
            reflex_command = self._handle_danger(player_state, detections)
        elif security_tier == SecurityTier.ATTACK_MAY_OCCUR:
            reflex_command = reflex_command or self._handle_potential_threat(player_state, detections)

        # Default: no action needed
        if reflex_command is None:
            reflex_command = ReflexCommand(
                command="NONE",
                priority=ReflexPriorities.LOW.value,
                reasoning="No immediate threat",
                parameters=None
            )

        self.last_reflex_command = reflex_command
        self.last_reflex_time = current_time

        return reflex_command

    def _handle_danger(
        self,
        player_state: PlayerState,
        detections: List[Detection]
    ) -> ReflexCommand:
        """
        Handle DANGER tier situations.

        Args:
            player_state: Current player state.
            detections: List of detected objects.

        Returns:
            ReflexCommand for emergency action.
        """
        # Critical health - POTION
        if player_state.health_percent <= self.critical_health_threshold:
            return ReflexCommand(
                command="POT",
                priority=ReflexPriorities.IMMEDIATE.value,
                reasoning=f"Critical health {player_state.health_percent:.1f}%",
                parameters={"health_threshold": self.critical_health_threshold}
            )

        # Player died
        if player_state.is_dead:
            return ReflexCommand(
                command="NONE",
                priority=ReflexPriorities.IMMEDIATE.value,
                reasoning="Player is dead - await resurrection",
                parameters=None
            )

        # Active combat - decide between attack or flee
        nearby_enemies = [
            d for d in detections
            if d.detection_type == DetectionType.ENEMY
            and d.bbox.center is not None
        ]

        if nearby_enemies:
            # Find closest enemy
            closest_enemy = min(nearby_enemies, key=lambda d: d.distance if d.distance else 1000)

            # Can we fight it?
            can_fight = (
                player_state.health_percent > 40.0 and  # Has some HP
                closest_enemy.health_percent and  # Enemy has known health
                closest_enemy.health_percent < 50.0  # Enemy is weak
            )

            if can_fight:
                return ReflexCommand(
                    command="ATTACK",
                    priority=ReflexPriorities.HIGH.value,
                    reasoning=f"Engaging weak enemy ({closest_enemy.health_percent:.1f}% HP)",
                    target_id=closest_enemy.label,
                    parameters={"target_health": closest_enemy.health_percent}
                )
            else:
                return ReflexCommand(
                    command="FLEE",
                    priority=ReflexPriorities.IMMEDIATE.value,
                    reasoning=f"Too dangerous - HP too low or enemy too strong",
                    parameters=None
                )

        return ReflexCommand(
            command="DEFEND",
            priority=ReflexPriorities.HIGH.value,
            reasoning="In combat but no target detected - prepare defense",
            parameters=None
        )

    def _handle_potential_threat(
        self,
        player_state: PlayerState,
        detections: List[Detection]
    ) -> Optional[ReflexCommand]:
        """
        Handle ATTACK_MAY_OCCUR tier situations.

        Args:
            player_state: Current player state.
            detections: List of detected objects.

        Returns:
            ReflexCommand for precautionary action, or None.
        """
        # Low health - pre-emptive potion
        if player_state.health_percent <= self.low_health_threshold:
            return ReflexCommand(
                command="POT",
                priority=ReflexPriorities.HIGH.value,
                reasoning=f"Low health precautionary heal {player_state.health_percent:.1f}%",
                parameters={"health_threshold": self.low_health_threshold}
            )

        return None

    def _count_nearby_threats(self, detections: List[Detection]) -> int:
        """
        Count threats within combat distance.

        Args:
            detections: List of detections.

        Returns:
            Number of nearby threats.
        """
        count = 0
        current_time = time.time()

        for detection in detections:
            # Skip non-enemies
            if detection.detection_type not in [DetectionType.ENEMY]:
                continue

            # Check distance
            if detection.distance and detection.distance <= self.combat_distance_threshold:
                count += 1
                # Track last seen
                self.last_seen_enemies[detection.label] = current_time

        return count

    def _count_threats_at_range(
        self,
        detections: List[Detection],
        min_dist: float,
        max_dist: float
    ) -> int:
        """
        Count threats within a specific distance range.

        Args:
            detections: List of detections.
            min_dist: Minimum distance.
            max_dist: Maximum distance.

        Returns:
            Number of threats in range.
        """
        count = 0
        current_time = time.time()

        for detection in detections:
            if detection.detection_type not in [DetectionType.ENEMY]:
                continue

            if detection.distance and min_dist <= detection.distance <= max_dist:
                count += 1
                self.last_seen_enemies[detection.label] = current_time

        return count

    def _get_recent_threats(self, current_time: float) -> List[str]:
        """
        Get list of recently seen threats.

        Args:
            current_time: Current timestamp.

        Returns:
            List of threat labels seen within decay time.
        """
        active_threats = []

        # Clean up old threats
        expired_threats = []
        for label, last_seen in self.last_seen_enemies.items():
            if current_time - last_seen > self.threat_decay_time:
                expired_threats.append(label)
            else:
                active_threats.append(label)

        # Remove expired threats
        for label in expired_threats:
            del self.last_seen_enemies[label]

        return active_threats

    def check_conflict(
        self,
        reflex_command: ReflexCommand,
        security_tier: SecurityTier
    ) -> bool:
        """
        Check if reflex command conflicts with security status.

        Args:
            reflex_command: The reflex command to check.
            security_tier: Current security tier.

        Returns:
            True if reflex should override strategy, False otherwise.
        """
        # DANGER tier always reflex-only
        if security_tier == SecurityTier.DANGER:
            if reflex_command.command in ["POT", "FLEE", "ATTACK", "DEFEND"]:
                return True

        # ATTACK_MAY_OCCUR tier can preempt some strategies
        if security_tier == SecurityTier.ATTACK_MAY_OCCUR:
            if reflex_command.command == "POT" and reflex_command.priority <= ReflexPriorities.HIGH.value:
                return True

        return False

    def update_threat_tracking(self, detections: List[Detection]):
        """
        Update internal threat tracking with new detections.

        Args:
            detections: List of new detections.
        """
        current_time = time.time()

        for detection in detections:
            if detection.detection_type == DetectionType.ENEMY:
                self.last_seen_enemies[detection.label] = current_time

        # Clean up old threats
        expired = [
            label for label, last_seen in self.last_seen_enemies.items()
            if current_time - last_seen > self.threat_decay_time
        ]
        for label in expired:
            del self.last_seen_enemies[label]

    def get_threat_summary(self) -> Dict[str, float]:
        """
        Get summary of currently tracked threats.

        Returns:
            Dictionary mapping threat labels to time since last seen.
        """
        current_time = time.time()
        return {
            label: current_time - last_seen
            for label, last_seen in self.last_seen_enemies.items()
        }
