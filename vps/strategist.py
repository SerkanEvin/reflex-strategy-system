"""
Strategist Module (VPS Brain)
LLM-based strategic reasoning for long-term decision making.
Analyzes state, recalls from spatial memory, and outputs high-level policies.
"""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from dataclasses import dataclass
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import (
    LocalState, StrategyPolicy, PlayerState, Detection, DetectionType, SecurityTier
)
from vps.database import SpatialDatabase
from vps.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """Context for strategic reasoning."""
    current_state: LocalState
    spatial_memory: List[Dict[str, Any]]
    recent_positions: List[Dict[str, Any]]
    session_history: List[Dict[str, Any]]
    current_strategy: Optional[StrategyPolicy]
    inventory_state: Dict[str, Any]


class Strategist:
    """
    LLM-powered strategic reasoning agent.
    Makes high-level decisions based on game state and spatial memory.
    """

    def __init__(
        self,
        database: SpatialDatabase,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        llm_api_url: Optional[str] = None,
        enable_llm: bool = True
    ):
        """
        Initialize strategist.

        Args:
            database: SpatialDatabase connection.
            llm_api_key: API key for LLM service (Synthetic API, OpenAI, etc.).
            llm_model: Model to use for reasoning.
            llm_api_url: Custom API URL (default uses Synthetic API).
            enable_llm: Enable LLM-based reasoning (fallback to rule-based if False).
        """
        self.database = database
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_api_url = llm_api_url or LLMClient.DEFAULT_API_URL
        self.enable_llm = enable_llm and llm_api_key is not None

        # LLM Client (lazy initialization)
        self._llm_client: Optional[LLMClient] = None

        # Policy tracking
        self.current_policy: Optional[StrategyPolicy] = None
        self.policy_history: List[StrategyPolicy] = []
        self.session_id: str = self._generate_session_id()

        # Rule-based fallbacks for critical scenarios
        self.critical_rules = self._initialize_critical_rules()

        # LLM statistics
        self.llm_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0
        }

    @property
    def llm_client(self) -> Optional[LLMClient]:
        """Get or create LLM client."""
        if not self.enable_llm:
            return None

        if self._llm_client is None:
            self._llm_client = LLMClient(
                api_key=self.llm_api_key,
                api_url=self.llm_api_url,
                model=self.llm_model
            )
            logger.info(f"[STRATEGIST] LLM client initialized with model: {self.llm_model}")

        return self._llm_client

    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        import uuid
        return str(uuid.uuid4())

    def _initialize_critical_rules(self) -> Dict[str, Any]:
        """Initialize critical rule-based decision trees."""
        return {
            "CRITICAL_HEALTH": {
                "condition": lambda state: state.player.health_percent < 20.0,
                "action": "RETREAT",
                "reasoning": "Critical health - immediate retreat required",
                "priority": 0
            },
            "FULL_INVENTORY": {
                "condition": lambda state: (
                    state.player.inventory_full_percent and
                    state.player.inventory_full_percent >= 90.0
                ),
                "action": "GO_TO_MARKET",
                "reasoning": "Inventory full - sell items",
                "priority": 2
            },
            "LOW_ON_POTIONS": {
                "condition": lambda state: self._check_low_potions(state),
                "action": "FIND_VENDOR",
                "reasoning": "Low on potions - restock",
                "priority": 3
            },
            "HIGH_VALUE_TARGET": {
                "condition": lambda state: self._has_high_value_target(state),
                "action": "HUNT",
                "reasoning": "High value target detected - engage",
                "priority": 1
            }
        }

    async def analyze_and_decide(self, state: LocalState) -> Optional[StrategyPolicy]:
        """
        Analyze current state and generate strategic policy.

        Args:
            state: Current local state from spinal cord.

        Returns:
            StrategyPolicy or None (no change needed).
        """
        # Update session history
        await self._log_state(state)

        # Gather context
        context = await self._gather_context(state)

        # Check critical rules first (bypass LLM for emergencies)
        if policy := self._apply_critical_rules(state, context):
            return await self._finalize_policy(state, policy)

        # Check if current policy still valid
        if self._is_current_policy_valid(state, context):
            return None

        # Use LLM for complex strategic decisions
        # TODO: Replace with actual LLM call
        policy = await self._llm_reason(state, context)

        return await self._finalize_policy(state, policy)

    async def _gather_context(self, state: LocalState) -> StrategyContext:
        """
        Gather all relevant context for strategic reasoning.

        Args:
            state: Current local state.

        Returns:
            StrategyContext with all relevant information.
        """
        # Get nearby markets, spawns, etc.
        # Use coordinates if available, otherwise use position (dict format)
        player_coords = (
            state.player.coordinates or
            state.player.position or
            {"x": 0, "y": 0, "z": 0}
        )
        if player_coords:
            nearby_entities = []
            for entity_type in ["market", "spawn_point", "vendor", "healer"]:
                results = await self.database.find_nearest(
                    player_coords,
                    entity_type,
                    limit=10
                )
                nearby_entities.extend([
                    {
                        "type": entity_type,
                        "label": entry.label,
                        "position": entry.position,
                        "distance": dist
                    }
                    for entry, dist in results
                ])

            spatial_memory = nearby_entities
        else:
            spatial_memory = []

        # Get recent player positions
        recent_positions = await self.database.get_recent_positions(
            self.session_id,
            limit=50
        )

        return StrategyContext(
            current_state=state,
            spatial_memory=spatial_memory,
            recent_positions=recent_positions,
            session_history=self.policy_history[-10:],
            current_strategy=self.current_policy,
            inventory_state=self._estimate_inventory_state(state)
        )

    def _apply_critical_rules(
        self,
        state: LocalState,
        context: StrategyContext
    ) -> Optional[Dict[str, Any]]:
        """
        Apply critical rule-based decisions.

        Args:
            state: Current state.
            context: Strategy context.

        Returns:
            Policy decision or None.
        """
        for rule_name, rule in self.critical_rules.items():
            if rule["condition"](state):
                return {
                    "action": rule["action"],
                    "reasoning": rule["reasoning"],
                    "priority": rule["priority"],
                    "source": "rule_" + rule_name
                }

        return None

    async def _llm_reason(
        self,
        state: LocalState,
        context: StrategyContext
    ) -> Dict[str, Any]:
        """
        Use LLM to reason about optimal strategy.

        Args:
            state: Current local state.
            context: Strategy context.

        Returns:
            Policy decision from LLM.
        """
        self.llm_stats["total_calls"] += 1

        # Check if LLM is enabled
        if not self.enable_llm or self.llm_client is None:
            logger.debug("[STRATEGIST] LLM disabled, using rule-based fallback")
            self.llm_stats["fallback_used"] += 1
            return self._rule_based_strategy(state, context)

        # Build reasoning prompt
        prompt = self._build_llm_prompt(state, context)

        try:
            # Call LLM API
            response: LLMResponse = await self._call_llm(prompt, state, context)

            # Convert to policy format
            policy = {
                "action": response.action,
                "reasoning": response.reasoning,
                "priority": response.priority,
                "source": "llm",
                "target": response.target
            }

            self.llm_stats["successful_calls"] += 1
            logger.info(f"[STRATEGIST] LLM decision: {policy['action']} - {policy['reasoning']}")

            return policy

        except Exception as e:
            logger.warning(f"[STRATEGIST] LLM call failed: {e}, using rule-based fallback")
            self.llm_stats["failed_calls"] += 1
            self.llm_stats["fallback_used"] += 1
            return self._rule_based_strategy(state, context)

    def _build_llm_prompt(
        self,
        state: LocalState,
        context: StrategyContext
    ) -> str:
        """
        Build LLM prompt for strategic reasoning.

        Args:
            state: Current state.
            context: Strategy context.

        Returns:
            Prompt string for LLM.
        """
        # Describe current situation
        situation = f"""
Current Situation:
- Security Level: {state.security_tier.name}
- Player Health: {state.player.health_percent}%
- In Combat: {state.player.is_in_combat}
- Inventory Full: {context.inventory_state.get('full_percent', 0)}%
- Active Targets: {len(state.active_targets)}

Detections:"""

        for detection in state.detections[:5]:  # Top 5 detections
            situation += f"\n- {detection.label} ({detection.detection_type.value}) - Confidence: {detection.confidence:.2f}"
            if detection.distance:
                situation += f" - Distance: {detection.distance:.1f}m"

        # Describe spatial memory
        memory_info = "\n\nKnown Locations:"
        for entity in context.spatial_memory[:10]:  # Top 10 known entities
            memory_info += f"\n- {entity['type']}: {entity['label']} at {entity['distance']:.1f}m away"

        # Build action description
        prompt = f"""{situation}{memory_info}

Available Actions:
- HUNT: Hunt for enemies/targets
- GATHER: Collect resources
- GO_TO_MARKET: Travel to market/vendor
- RETREAT: Fall back to safe location
- PATROL: Patrol known area
- EXPLORE: Explore new areas

Current Strategy: {context.current_strategy.action if context.current_strategy else "None"}

You are a strategic AI brain for a game automation system. Determine the best high-level action based on the situation.
Respond in this JSON format:
{{"action": "ACTION_NAME", "reasoning": "explanation", "priority": 1-10, "target": {{"type": "...", "position": {{...}}}}}}
"""
        return prompt

    async def _call_llm(
        self,
        prompt: str,
        state: LocalState,
        context: StrategyContext
    ) -> LLMResponse:
        """
        Call LLM API for reasoning.

        Args:
            prompt: Prompt to send to LLM.
            state: Current state (for context building).
            context: Strategy context.

        Returns:
            Parsed LLMResponse.

        Raises:
            Exception: If LLM call fails.
        """
        client = self.llm_client
        if client is None:
            raise Exception("LLM client not available")

        # Alternatively, use the client's built-in prompt builder
        state_dict = state.to_dict()
        context_dict = {
            "spatial_memory": context.spatial_memory,
            "inventory_state": context.inventory_state,
            "current_strategy": context.current_strategy.to_dict() if context.current_strategy else None
        }

        enhanced_prompt = client.get_strategy_prompt(state_dict, context_dict) + prompt

        response = await client.chat_completion(
            prompt=enhanced_prompt,
            max_tokens=300,
            temperature=0.7
        )

        return response

    def _rule_based_strategy(
        self,
        state: LocalState,
        context: StrategyContext
    ) -> Dict[str, Any]:
        """
        Rule-based strategy as fallback from LLM.

        Args:
            state: Current state.
            context: Strategy context.

        Returns:
            Policy decision.
        """
        # If in secure mode and no specific goals, patrol or explore
        if state.security_tier == SecurityTier.SECURE:
            # Check if we have gathered enough
            if context.inventory_state.get('full_percent', 0) >= 70.0:
                return {
                    "action": "GO_TO_MARKET",
                    "reasoning": "Inventory mostly full - selling items",
                    "priority": 3,
                    "target": self._find_nearest_market(context)
                }

            # Look for nearby resources or enemies
            enemies = [d for d in state.detections if d.detection_type == DetectionType.ENEMY]
            resources = [d for d in state.detections if d.detection_type == DetectionType.RESOURCE]

            if resources and len(resources) > len(enemies):
                return {
                    "action": "GATHER",
                    "reasoning": "Resources detected - gathering",
                    "priority": 4,
                    "target": {
                        "type": "resource",
                        "label": resources[0].label,
                        "position": self._bbox_to_position(resources[0].bbox)
                    }
                }

            if enemies:
                return {
                    "action": "HUNT",
                    "reasoning": f"Enemies detected - hunting ({len(enemies)} targets)",
                    "priority": 2,
                    "target": {
                        "type": "enemy",
                        "label": enemies[0].label,
                        "position": self._bbox_to_position(enemies[0].bbox)
                    }
                }

            # Default patrol
            return {
                "action": "PATROL",
                "reasoning": "Area secure - patrolling for targets",
                "priority": 6
            }

        # If potentially threatened, be cautious
        if state.security_tier == SecurityTier.ATTACK_MAY_OCCUR:
            return {
                "action": "PATROL",
                "reasoning": "Potential threats detected - cautious patrol",
                "priority": 4
            }

        # DANGER tier is handled by reflex (shouldn't reach here)
        return {
            "action": "RETREAT",
            "reasoning": "Danger detected - retreating",
            "priority": 1
        }

    def _find_nearest_market(self, context: StrategyContext) -> Dict[str, Any]:
        """Find nearest market from spatial memory."""
        markets = [e for e in context.spatial_memory if e["type"] == "market"]
        if markets:
            nearest = min(markets, key=lambda x: x["distance"])
            return {
                "type": "market",
                "label": nearest["label"],
                "position": nearest["position"],
                "distance": nearest["distance"]
            }
        return {}

    def _bbox_to_position(self, bbox) -> Dict[str, float]:
        """Convert bounding box to approximate position."""
        # This is game-specific - placeholder implementation
        center = bbox.center
        return {"x": center[0], "y": center[1], "z": 0.0}

    def _check_low_potions(self, state: LocalState) -> bool:
        """Check if potion count is low (placeholder)."""
        # TODO: Implement actual potion checking
        return False

    def _has_high_value_target(self, state: LocalState) -> bool:
        """Check if there's a high value target (placeholder)."""
        # TODO: Implement actual target value judgment
        for detection in state.detections:
            if detection.detection_type == DetectionType.ENEMY:
                if detection.health_percent and detection.health_percent < 40.0:
                    return True
        return False

    def _estimate_inventory_state(self, state: LocalState) -> Dict[str, Any]:
        """Estimate inventory state (placeholder)."""
        return {
            "full_percent": state.player.inventory_full_percent or 0.0,
            "items": []
        }

    def _is_current_policy_valid(
        self,
        state: LocalState,
        context: StrategyContext
    ) -> bool:
        """
        Check if current policy should continue.

        Args:
            state: Current state.
            context: Strategy context.

        Returns:
            True if current policy is still valid.
        """
        if self.current_policy is None:
            return False

        # Don't change strategy in danger unless reflex takes over
        if state.security_tier == SecurityTier.DANGER:
            return True

        # Don't change if we just started this strategy
        policy_age = datetime.now().timestamp() - self.current_policy.created_at
        if policy_age < 30:  # 30 seconds minimum per action
            return True

        return False

    async def _finalize_policy(
        self,
        state: LocalState,
        policy: Dict[str, Any]
    ) -> StrategyPolicy:
        """
        Finalize and store a new policy.

        Args:
            state: Current state.
            policy: Policy decision dictionary.

        Returns:
            StrategyPolicy object.
        """
        new_policy = StrategyPolicy(
            policy_id=None,
            priority=policy.get("priority", 5),
            action=policy["action"],
            target=policy.get("target"),
            reasoning=policy["reasoning"],
            estimated_duration=policy.get("estimated_duration"),
            conditions=policy.get("conditions"),
            created_at=datetime.now().timestamp()
        )

        # Update current policy
        self.current_policy = new_policy
        self.policy_history.append(new_policy)

        # Log to database
        await self.database.log_strategy(
            policy_id=None,
            action=new_policy.action,
            reasoning=new_policy.reasoning,
            priority=new_policy.priority,
            context={"state": state.to_dict(), "source": policy.get("source", "rule_based")}
        )

        print(f"[STRATEGIST] New policy: {new_policy.action} - {new_policy.reasoning}")

        return new_policy

    async def _log_state(self, state: LocalState):
        """
        Log state to database for tracking.

        Args:
            state: State to log.
        """
        # Use coordinates if available, otherwise use position
        player_coords = state.player.coordinates or state.player.position
        if player_coords:
            await self.database.log_player_position(
                session_id=self.session_id,
                position=player_coords,
                health_percent=state.player.health_percent,
                is_in_combat=state.player.is_in_combat,
                detected_objects=[d.to_dict() for d in state.detections]
            )

    def get_current_policy(self) -> Optional[StrategyPolicy]:
        """Get current strategy policy."""
        return self.current_policy

    def reset(self):
        """Reset strategist state."""
        self.current_policy = None
        self.policy_history = []
        self.session_id = self._generate_session_id()

    async def close(self):
        """Clean up resources."""
        if self._llm_client:
            await self._llm_client.close()
            self._llm_client = None

    def get_llm_stats(self) -> Dict[str, int]:
        """Get LLM usage statistics."""
        return self.llm_stats.copy()
