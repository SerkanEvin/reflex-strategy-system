"""
Shared constants for the Reflex-Strategy System.
"""
from enum import Enum


class Colors(Enum):
    """Color constants for game UI elements."""
    HP_BAR_GREEN = (0, 255, 0)
    HP_BAR_RED = (255, 0, 0)
    MP_BAR_BLUE = (0, 0, 255)
    ENEMY_HP = (255, 50, 50)
    FRIENDLY_HP = (100, 200, 100)


class Thresholds(Enum):
    """Safety thresholds."""
    CRITICAL_HEALTH = 30.0  # % - Triggers DANGER tier
    LOW_HEALTH = 50.0  # % - Triggers ATTACK_MAY_OCCUR tier
    INVENTORY_FULL_THRESHOLD = 90.0  # % - Triggers market visit
    COMBAT_DISTANCE = 150.0  # Screen pixels - Distance for danger assessment
    SAFE_DISTANCE = 300.0  # Screen pixels - Distance for SECURE tier


class GameActions(Enum):
    """Game actions for actuator."""
    MOVE_UP = "w"
    MOVE_DOWN = "s"
    MOVE_LEFT = "a"
    MOVE_RIGHT = "d"
    ATTACK = "1"
    SKILL_1 = "q"
    SKILL_2 = "e"
    SKILL_3 = "r"
    USE_POTION = "space"
    INTERACT = "f"
    ESCAPE = "esc"


class ReflexPriorities(Enum):
    """Priority levels for reflex commands."""
    IMMEDIATE = 0  # Execute now, override everything
    HIGH = 1
    MEDIUM = 5
    LOW = 10
