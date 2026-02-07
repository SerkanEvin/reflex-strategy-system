"""
Actuator Layer - The Muscles
Executes actions with human-like movement using Bezier curves.
"""
import time
import random
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import pyautogui
from pywinauto.keyboard import SendKeys
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import ActionCommand, ReflexCommand, StrategyPolicy
from shared.constants import GameActions


@dataclass
class BezierPoint:
    """Point for Bezier curve."""
    x: float
    y: float


class HumanMouse:
    """
    Human-like mouse movement using Bezier curves.
    Mimics natural mouse movement to avoid detection.
    """

    def __init__(
        self,
        speed_range: Tuple[float, float] = (100, 300),
        jitter: float = 2.0,
        pause_chance: float = 0.1,
        pause_range: Tuple[float, float] = (0.05, 0.2)
    ):
        """
        Initialize human-like mouse movement.

        Args:
            speed_range: Min and max pixels per second.
            jitter: Maximum random pixel offset.
            pause_chance: Probability of random pause during movement.
            pause_range: Pause duration range in seconds.
        """
        self.speed_range = speed_range
        self.jitter = jitter
        self.pause_chance = pause_chance
        self.pause_range = pause_range

        # Current position
        self.current_x, self.current_y = pyautogui.position()

    def move_to(self, target_x: float, target_y: float):
        """
        Move mouse to target with human-like bezier curve.

        Args:
            target_x: Target X coordinate.
            target_y: Target Y coordinate.
        """
        start_x, start_y = self.current_x, self.current_y

        # Generate control points for smooth curve
        control_points = self._generate_bezier_points(
            (start_x, start_y),
            (target_x, target_y),
            num_points=5
        )

        # Move through control points
        for i in range(len(control_points) - 1):
            p1 = control_points[i]
            p2 = control_points[i + 1]

            # Calculate duration based on distance and speed
            distance = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            duration = distance / random.uniform(*self.speed_range)

            # Execute movement with jitter
            self._move_with_jitter(p2.x, p2.y, duration)

            # Random pause
            if random.random() < self.pause_chance:
                time.sleep(random.uniform(*self.pause_range))

        # Update current position
        self.current_x, self.current_y = target_x, target_y

    def _generate_bezier_points(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        num_points: int = 5
    ) -> List[BezierPoint]:
        """
        Generate points along a Bezier curve.

        Args:
            start: Start coordinates.
            end: End coordinates.
            num_points: Number of intermediate points.

        Returns:
            List of BezierPoints.
        """
        start_x, start_y = start
        end_x, end_y = end

        points = [BezierPoint(start_x, start_y)]

        # Calculate control points with some randomness
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        offset_x = random.uniform(-50, 50)
        offset_y = random.uniform(-50, 50)

        # Generate intermediate points
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)

            # Quadratic Bezier: (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            x = ((1 - t)**2) * start_x + \
                2 * (1 - t) * t * (mid_x + offset_x) + \
                (t**2) * end_x

            y = ((1 - t)**2) * start_y + \
                2 * (1 - t) * t * (mid_y + offset_y) + \
                (t**2) * end_y

            # Add jitter
            x += random.uniform(-self.jitter, self.jitter)
            y += random.uniform(-self.jitter, self.jitter)

            points.append(BezierPoint(x, y))

        points.append(BezierPoint(end_x, end_y))

        return points

    def _move_with_jitter(self, x: float, y: float, duration: float):
        """
        Move to position with micro-jitter for natural movement.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            duration: Movement duration in seconds.
        """
        if duration <= 0:
            pyautogui.moveTo(x, y)
            return

        steps = max(5, int(duration * 60))  # 60 Hz
        for i in range(steps + 1):
            t = i / steps
            current_x = self.current_x + (x - self.current_x) * t
            current_y = self.current_y + (y - self.current_y) * t

            # Add micro-jitter
            jitter_x = random.uniform(-0.5, 0.5)
            jitter_y = random.uniform(-0.5, 0.5)

            pyautogui.moveTo(current_x + jitter_x, current_y + jitter_y)
            time.sleep(duration / steps)

    def click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
        clicks: int = 1,
        delay: float = 0.1
    ):
        """
        Click at position with human-like delay.

        Args:
            x: X coordinate. If None, click at current position.
            y: Y coordinate. If None, click at current position.
            button: Mouse button ("left", "right", "middle").
            clicks: Number of clicks.
            delay: Random delay before click.
        """
        if x is not None and y is not None:
            self.move_to(x, y)

        # Random delay before click
        human_delay = delay + random.uniform(-0.02, 0.05)
        time.sleep(max(0, human_delay))

        # Click with micro-variation
        for i in range(clicks):
            if i > 0:
                time.sleep(random.uniform(0.08, 0.15))

            pyautogui.click(button=button)

    def scroll(self, amount: int):
        """
        Scroll mouse wheel.

        Args:
            amount: Number of scroll clicks (positive for up, negative for down).
        """
        pyautogui.scroll(amount)


class KeyboardController:
    """
    Human-like keyboard input with timing variation.
    """

    def __init__(
        self,
        press_duration_range: Tuple[float, float] = (0.05, 0.15),
        key_delay_range: Tuple[float, float] = (0.05, 0.1)
    ):
        """
        Initialize keyboard controller.

        Args:
            press_duration_range: Min and max key press duration.
            key_delay_range: Min and max delay between key presses.
        """
        self.press_duration_range = press_duration_range
        self.key_delay_range = key_delay_range

    def press(self, key: str, hold: bool = False):
        """
        Press a key with human-like timing.

        Args:
            key: Key to press.
            hold: Whether to hold the key down.
        """
        duration = random.uniform(*self.press_duration_range)

        if hold:
            pyautogui.keyDown(key)
        else:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)

            # Small delay after press
            time.sleep(random.uniform(*self.key_delay_range))

    def release(self, key: str):
        """
        Release a held key.

        Args:
            key: Key to release.
        """
        pyautogui.keyUp(key)

    def type_text(self, text: str, speed: float = 0.05):
        """
        Type text with human-like typing speed variation.

        Args:
            text: Text to type.
            speed: Base typing speed per character.
        """
        for char in text:
            self.press(char)
            time.sleep(speed + random.uniform(-0.02, 0.05))

    def send_key_combo(self, keys: List[str]):
        """
        Send a combination of keys (e.g., Ctrl+Shift+R).

        Args:
            keys: List of keys to press simultaneously.
        """
        # Press all keys
        for key in keys:
            pyautogui.keyDown(key)
            time.sleep(random.uniform(0.01, 0.03))

        # Release all keys (in reverse order)
        for key in reversed(keys):
            time.sleep(random.uniform(0.01, 0.02))
            pyautogui.keyUp(key)


class Actuator:
    """
    Main actuator that executes commands with human-like behavior.
    """

    def __init__(self):
        """Initialize actuator."""
        self.mouse = HumanMouse()
        self.keyboard = KeyboardController()

        # Command queue for delayed execution
        self.command_queue: List[Tuple[str, Dict]] = []
        self.queue_thread = None
        self.running = False

    def execute_reflex(
        self,
        reflex: ReflexCommand,
        target_position: Optional[Tuple[float, float]] = None
    ):
        """
        Execute a reflex command.

        Args:
            reflex: Reflex command to execute.
            target_position: Optional screen position for targeting.
        """
        command = reflex.command.upper()

        if command == "POT":
            self._use_potion()

        elif command == "FLEE":
            self._flee()

        elif command == "ATTACK":
            self._attack(target_position)

        elif command == "DEFEND":
            self._defend()

        elif command == "NONE":
            pass

    def execute_strategy(
        self,
        strategy: StrategyPolicy,
        current_position: Optional[Tuple[float, float]] = None
    ):
        """
        Execute a high-level strategy by breaking it down into actions.

        Args:
            strategy: Strategy policy to execute.
            current_position: Optional current screen position.
        """
        action = strategy.action.upper()

        if action == "GO_TO_MARKET":
            self._navigate_to_strategy_target(strategy)

        elif action == "HUNT":
            self._hunt_targets(strategy)

        elif action == "GATHER":
            self._gather_resources(strategy)

        elif action == "RETREAT":
            self._flee()

    def _use_potion(self):
        """Use healing potion."""
        # Press potion key
        self.keyboard.press(GameActions.USE_POTION.value)
        print("[ACTUATOR] Used potion")

    def _flee(self):
        """Execute flee maneuver - move away from danger."""
        # Turn around and run in opposite direction
        # For 3D MMORPG, this might involve camera rotation + sprint

        # Sprint key held down
        self.keyboard.press("shift", hold=True)
        time.sleep(0.1)

        # Move forward
        self.keyboard.press("w", hold=True)
        time.sleep(random.uniform(1.5, 3.0))

        # Release
        self.keyboard.release("w")
        self.keyboard.release("shift")

        # Add some strafing for more human-like movement
        self.keyboard.press("a", hold=True)
        time.sleep(random.uniform(0.3, 0.8))
        self.keyboard.release("a")

        print("[ACTUATOR] Executed flee maneuver")

    def _attack(self, target_position: Optional[Tuple[float, float]] = None):
        """
        Execute attack on target.

        Args:
            target_position: Screen position of target.
        """
        if target_position:
            # Look at target
            self.mouse.move_to(*target_position)
            time.sleep(random.uniform(0.1, 0.2))
            self.mouse.click()

        # Attack rotation
        attack_seq = [
            GameActions.ATTACK.value,
            GameActions.SKILL_1.value,
            GameActions.ATTACK.value,
            GameActions.ATTACK.value,
        ]

        for key in attack_seq:
            self.keyboard.press(key)
            time.sleep(random.uniform(0.3, 0.6))

        print("[ACTUATOR] Executed attack sequence")

    def _defend(self):
        """Execute defensive stance."""
        # Block/dodge key press
        self.keyboard.press("s", hold=True)
        time.sleep(random.uniform(0.2, 0.5))
        self.keyboard.release("s")

        # Camera check
        self.mouse.move_to(
            self.mouse.current_x + random.uniform(-20, 20),
            self.mouse.current_y + random.uniform(-10, 10)
        )

        print("[ACTUATOR] Executed defensive stance")

    def _navigate_to_strategy_target(self, strategy: StrategyPolicy):
        """
        Navigate to a strategy target (e.g., market).

        Args:
            strategy: Strategy containing target information.
        """
        target = strategy.target

        if target and "position" in target:
            # Convert game coordinates to screen movement
            # This would be game-specific

            # Example: Look at target on minimap and navigate
            print(f"[ACTUATOR] Navigating to {strategy.action}: {strategy.reasoning}")

            # Hold forward to move
            self.keyboard.press("w", hold=True)

            # Adjust direction periodically
            for _ in range(random.randint(3, 7)):
                time.sleep(random.uniform(0.5, 1.5))

                # Slight direction adjustment
                self.keyboard.press("d", hold=True)
                time.sleep(random.uniform(0.1, 0.3))
                self.keyboard.release("d")

            # Stop
            self.keyboard.release("w")

    def _hunt_targets(self, strategy: StrategyPolicy):
        """
        Hunt for targets based on strategy.

        Args:
            strategy: Strategy containing hunt parameters.
        """
        print(f"[ACTUATOR] Hunting: {strategy.reasoning}")

        # Search pattern
        self.keyboard.press("w", hold=True)
        time.sleep(random.uniform(0.5, 1.0))

        # Rotate camera to search
        self.mouse.move_to(
            self.mouse.current_x, self.mouse.current_y + 30
        )
        time.sleep(0.2)
        self.mouse.move_to(
            self.mouse.current_x, self.mouse.current_y - 30
        )

    def _gather_resources(self, strategy: StrategyPolicy):
        """
        Gather resources based on strategy.

        Args:
            strategy: Strategy containing gather parameters.
        """
        print(f"[ACTUATOR] Gathering: {strategy.reasoning}")

        # Approach resource
        self.keyboard.press("w", hold=True)
        time.sleep(random.uniform(0.3, 0.8))

        # Interact
        self.keyboard.press(GameActions.INTERACT.value)

    def queue_action(self, action_type: str, params: Dict, delay: float = 0):
        """
        Queue an action for later execution.

        Args:
            action_type: Type of action ("reflex" or "strategy").
            params: Action parameters.
            delay: Delay before execution.
        """
        self.command_queue.append(time.time() + delay, (action_type, params))

    def process_queue(self):
        """Process queued actions whose delay has elapsed."""
        current_time = time.time()

        while self.command_queue and self.command_queue[0][0] <= current_time:
            _, (action_type, params) = self.command_queue.pop(0)

            if action_type == "reflex":
                self.execute_reflex(**params)
            elif action_type == "strategy":
                self.execute_strategy(**params)

    def stop_all_actions(self):
        """Stop all ongoing actions by releasing held keys."""
        # Release all movement keys
        for key in ["w", "a", "s", "d", "shift"]:
            try:
                self.keyboard.release(key)
            except:
                pass

        print("[ACTUATOR] All actions stopped")
