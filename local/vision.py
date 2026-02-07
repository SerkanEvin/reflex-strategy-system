"""
Vision System - YOLOv11 Inference
Detects enemies, stones, and UI elements from game frames.
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import shared models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import (
    Detection, DetectionType, BoundingBox, PlayerState
)


class VisionSystem:
    """YOLOv11 inference engine for object detection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize vision system.

        Args:
            model_path: Path to YOLOv11 model weights. If None, will use placeholder.
            confidence_threshold: Minimum confidence for detection.
            iou_threshold: IoU threshold for NMS.
            device: Device to run inference on ("cpu" or "cuda").
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model_path = model_path

        # Detection categories for 3D MMORPG
        self.categories = {
            0: DetectionType.ENEMY,
            1: DetectionType.FRIENDLY,
            2: DetectionType.PLAYER,
            3: DetectionType.RESOURCE,
            4: DetectionType.ITEM,
            5: DetectionType.UI_ELEMENT,
            6: DetectionType.MARKET,
        }

        # TODO: Load YOLOv11 model here
        # self.model = self._load_model(model_path)
        self.model = None
        self.model_loaded = False

        # UI element detection templates (fallback for HP bars, minimap, etc.)
        self.ui_templates = {}

    def _load_model(self, model_path: Optional[str]) -> Optional[object]:
        """
        Load YOLOv11 model.

        Args:
            model_path: Path to model weights.

        Returns:
            Loaded model or None.
        """
        if model_path is None:
            print("WARNING: No model path provided. Using placeholder mode.")
            return None

        try:
            # TODO: Load actual YOLOv11 model
            # from ultralytics import YOLO
            # model = YOLO(model_path)
            # model.to(self.device)
            # model.fuse()
            return None

        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def load_model(self, model_path: str):
        """
        Load or reload the YOLO model.

        Args:
            model_path: Path to model weights file.
        """
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.model_loaded = self.model is not None

    def detect(
        self,
        frame: np.ndarray,
        filter_types: Optional[List[DetectionType]] = None
    ) -> List[Detection]:
        """
        Run YOLO inference on a frame.

        Args:
            frame: Input frame (BGR image).
            filter_types: Optionally filter by detection types.

        Returns:
            List of Detection objects.
        """
        if frame is None or frame.size == 0:
            return []

        # Placeholder detection - replace with actual YOLO inference
        if self.model is None:
            return self._placeholder_detect(frame)

        # TODO: Run actual YOLO inference
        # results = self.model(
        #     frame,
        #     conf=self.confidence_threshold,
        #     iou=self.iou_threshold,
        #     verbose=False
        # )

        detections = []
        # Process results...

        return self._filter_detections(detections, filter_types)

    def _placeholder_detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Placeholder detection when no model is loaded.
        Uses basic color detection for UI elements.

        Returns:
            List of placeholder detections.
        """
        detections = []

        try:
            # Detect HP bars (red for enemies/UI)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red color range for HP bars
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])

            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Find contours for red regions (potential HP bars)
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum size threshold
                    x, y, w, h = cv2.boundingRect(contour)

                    # Create detection
                    bbox = BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=0.7,  # Placeholder confidence
                        label="hp_bar"
                    )

                    detection = Detection(
                        bbox=bbox,
                        detection_type=DetectionType.UI_ELEMENT,
                        label="hp_bar",
                        confidence=0.7
                    )
                    detections.append(detection)

        except Exception as e:
            print(f"Placeholder detection error: {e}")

        return detections

    def detect_ui_elements(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Detect UI elements like HP bars, minimap, etc.

        Args:
            frame: Input frame.

        Returns:
            Dictionary containing detected UI elements and their positions.
        """
        ui_elements = {
            "hp_bar": None,
            "mp_bar": None,
            "minimap": None,
            "inventory": None
        }

        try:
            # Detect HP bar (usually red, often in corner)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Look for HP bar in top-left corner
            corner_region = hsv[0:100, 0:200]

            red_mask1 = cv2.inRange(corner_region, np.array([0, 100, 100]), np.array([10, 255, 255]))
            if np.sum(red_mask1) > 1000:
                ui_elements["hp_bar"] = {"position": (0, 0), "visible": True}

            # Look for minimap (usually in corners)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=100,
                param1=50,
                param2=30,
                minRadius=50,
                maxRadius=150
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # First circle could be minimap
                if len(circles) > 0:
                    x, y, r = circles[0]
                    ui_elements["minimap"] = {
                        "position": (x, y),
                        "radius": r,
                        "visible": True
                    }

        except Exception as e:
            print(f"UI detection error: {e}")

        return ui_elements

    def estimate_player_state(
        self,
        frame: np.ndarray,
        ui_elements: Dict[str, any]
    ) -> PlayerState:
        """
        Estimate player state from UI elements.

        Args:
            frame: Input frame.
            ui_elements: Detected UI elements.

        Returns:
            PlayerState object.
        """
        # Default fallback values
        player_state = PlayerState(
            health_percent=100.0,
            max_health=1000.0,
            health=1000.0,
            mana_percent=100.0,
            level=1,
            is_in_combat=False,
            is_dead=False,
            inventory_full_percent=0.0
        )

        try:
            # Estimate HP from HP bar if detected
            if ui_elements.get("hp_bar", {}).get("visible"):
                # Extract HP percentage from bar
                hp_region = frame[
                    ui_elements["hp_bar"]["position"][1]:
                    ui_elements["hp_bar"]["position"][1] + 20,
                    0:200
                ]

                # Count red pixels
                red_mask = cv2.inRange(hp_region, (0, 0, 200), (50, 50, 255))
                total_pixels = hp_region.shape[0] * hp_region.shape[1]
                red_pixels = np.count_nonzero(red_mask)
                health_percent = (red_pixels / total_pixels) * 100

                player_state.health_percent = min(100.0, max(0.0, health_percent))
                player_state.health = (player_state.health_percent / 100.0) * player_state.max_health

        except Exception as e:
            print(f"Player state estimation error: {e}")

        return player_state

    def _filter_detections(
        self,
        detections: List[Detection],
        filter_types: Optional[List[DetectionType]] = None
    ) -> List[Detection]:
        """
        Filter detections by type.

        Args:
            detections: List of detections.
            filter_types: Types to keep. If None, keep all.

        Returns:
            Filtered list of detections.
        """
        if filter_types is None:
            return detections

        return [d for d in detections if d.detection_type in filter_types]

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw detection boxes on frame for visualization.

        Args:
            frame: Input frame.
            detections: List of detections.

        Returns:
            Frame with drawn detections.
        """
        vis_frame = frame.copy()

        colors = {
            DetectionType.ENEMY: (0, 0, 255),  # Red
            DetectionType.FRIENDLY: (0, 255, 0),  # Green
            DetectionType.PLAYER: (255, 255, 0),  # Yellow
            DetectionType.RESOURCE: (0, 255, 255),  # Cyan
            DetectionType.ITEM: (255, 0, 255),  # Magenta
            DetectionType.UI_ELEMENT: (128, 128, 128),  # Gray
            DetectionType.MARKET: (255, 165, 0),  # Orange
        }

        for det in detections:
            bbox = det.bbox
            color = colors.get(det.detection_type, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                color,
                2
            )

            # Draw label
            label = f"{det.label} {det.confidence:.2f}"
            if det.distance is not None:
                label += f" {det.distance:.1f}m"

            cv2.putText(
                vis_frame,
                label,
                (int(bbox.x1), int(bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return vis_frame
