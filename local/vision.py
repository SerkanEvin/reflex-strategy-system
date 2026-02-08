"""
Vision System - YOLOv11 Inference
Detects enemies, stones, and UI elements from game frames.
"""
import cv2
import numpy as np
import time
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

# Import shared models
import sys
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import (
    Detection, DetectionType, BoundingBox, PlayerState, GameProfile
)


class VisionSystem:
    """YOLOv11 inference engine for object detection."""

    # Default categories (fallback when no profile loaded)
    DEFAULT_CATEGORIES = {
        0: DetectionType.ENEMY,
        1: DetectionType.FRIENDLY,
        2: DetectionType.PLAYER,
        3: DetectionType.RESOURCE,
        4: DetectionType.ITEM,
        5: DetectionType.UI_ELEMENT,
        6: DetectionType.MARKET,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        game_profile: Optional[GameProfile] = None,
        download_dir: str = "models/"
    ):
        """
        Initialize vision system.

        Args:
            model_path: Path to YOLOv11 model weights. If None, will use placeholder.
            confidence_threshold: Minimum confidence for detection.
            iou_threshold: IoU threshold for NMS.
            device: Device to run inference on ("cpu" or "cuda").
            game_profile: Game profile with class mappings.
            download_dir: Directory for downloaded models.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model_path = model_path
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Game profile support
        self.game_profile = game_profile
        self.categories = {}

        # Initialize categories from profile or use default
        if game_profile:
            self._load_categories_from_profile(game_profile)
        else:
            self.categories = self.DEFAULT_CATEGORIES.copy()

        # TODO: Load YOLOv11 model here
        # self.model = self._load_model(model_path)
        self.model = None
        self.model_loaded = False

        # UI element detection templates (fallback for HP bars, minimap, etc.)
        self.ui_templates = {}

    def _load_categories_from_profile(self, game_profile: GameProfile):
        """
        Load category mapping from game profile.

        Args:
            game_profile: Game profile containing class mappings.
        """
        self.categories = {}
        for class_id, mapping in game_profile.class_mappings.items():
            self.categories[class_id] = mapping.detection_type

    def load_game_profile(self, profile_path: str):
        """
        Load game profile from YAML file.

        Args:
            profile_path: Path to game profile YAML file.
        """
        try:
            with open(profile_path, 'r') as f:
                profile_data = yaml.safe_load(f)

            self.game_profile = GameProfile.from_dict(profile_data)
            self._load_categories_from_profile(self.game_profile)

            print(f"[VISION] Loaded game profile: {self.game_profile.display_name}")
            print(f"[VISION] Classes: {self.game_profile.class_mappings}")

            # Attempt to download model if available and not found locally
            if self.game_profile.download_url and not self.model_path:
                model_name = f"{self.game_profile.name}_model.pt"
                local_model_path = self.download_dir / model_name
                if not local_model_path.exists():
                    print(f"[VISION] Model not found, attempting download from profile...")
                    if download_model(self.game_profile.download_url, str(local_model_path)):
                        self.model_path = str(local_model_path)
                        self._load_model(self.model_path)

        except Exception as e:
            print(f"[VISION] Failed to load game profile: {e}")
            # Fallback to default categories
            self.categories = self.DEFAULT_CATEGORIES.copy()

    def _load_model(self, model_path: Optional[str]) -> Optional[object]:
        """
        Load YOLOv11 model.

        Args:
            model_path: Path to model weights.

        Returns:
            Loaded model or None.
        """
        if model_path is None:
            # Check if profile has download URL
            if self.game_profile and self.game_profile.download_url:
                model_name = f"{self.game_profile.name}_model.pt"
                local_model_path = self.download_dir / model_name
                if local_model_path.exists():
                    model_path = str(local_model_path)
                else:
                    print(f"[VISION] Model not found locally, downloading from {self.game_profile.download_url}...")
                    if download_model(self.game_profile.download_url, str(local_model_path)):
                        model_path = str(local_model_path)
                    else:
                        print("WARNING: Failed to download model. Using placeholder mode.")
                        return None
            else:
                print("WARNING: No model path provided. Using placeholder mode.")
                return None

        # Check if model exists and is valid
        if not Path(model_path).exists():
            # Try to download if path looks like a URL
            if urlparse(model_path).scheme in ('http', 'https'):
                output_path = self.download_dir / Path(urlparse(model_path).path).name
                if download_model(model_path, str(output_path)):
                    model_path = str(output_path)
                else:
                    print(f"WARNING: Failed to download model from {model_path}")
                    return None
            else:
                print(f"WARNING: Model file not found: {model_path}")
                return None

        # Verify model file integrity
        if not verify_model(model_path):
            print(f"WARNING: Model file verification failed: {model_path}")
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
        # For YOLO results, map class IDs using:
        # if self.game_profile:
        #     detection_type = self.game_profile.get_detection_type(class_id)
        #     label = self.game_profile.get_label(class_id)
        # else:
        #     detection_type = self.categories.get(class_id, DetectionType.OTHER)
        #     label = f"class_{class_id}"

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


def download_model(url: str, output_path: str, timeout: int = 300) -> bool:
    """
    Download pretrained model from URL.

    Args:
        url: URL to download model from.
        output_path: Path where the model should be saved.
        timeout: Maximum time to wait for download in seconds.

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True, timeout=timeout)

        # Check if request was successful
        response.raise_for_status()

        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download file in chunks
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Optional: print progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Print every MB
                            print(f"\rDownloading: {progress:.1f}%", end='', flush=True)

        print(f"\nModel downloaded successfully to {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to download model: {e}")
        return False
    except Exception as e:
        print(f"Error during download: {e}")
        return False


def verify_model(model_path: str) -> bool:
    """
    Verify downloaded model file integrity.

    Args:
        model_path: Path to model file to verify.

    Returns:
        True if model appears valid, False otherwise.
    """
    try:
        path = Path(model_path)

        # Check if file exists
        if not path.exists():
            return False

        # Check file size (models should be at least 1MB)
        file_size = path.stat().st_size
        if file_size < 1024 * 1024:  # Less than 1MB
            print(f"Model file too small: {file_size} bytes")
            return False

        # Check file extension
        valid_extensions = ('.pt', '.onnx', '.engine', '.torchscript')
        if not path.suffix.lower() in valid_extensions:
            print(f"Invalid model file extension: {path.suffix}")
            return False

        # Attempt to read file header to verify it's not corrupted
        # For PyTorch models, check if it's a zip file (PyTorch uses zip format)
        if path.suffix.lower() == '.pt':
            try:
                import zipfile
                with zipfile.ZipFile(model_path, 'r') as _:
                    pass  # Just try to open it
            except zipfile.BadZipFile:
                # Old torch format, try loading it
                try:
                    import torch
                    torch.load(model_path, map_location='cpu')
                except Exception:
                    return False

        return True

    except Exception as e:
        print(f"Model verification failed: {e}")
        return False
