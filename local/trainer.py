"""
Model Trainer - YOLOv11 Training Module
Train YOLO models from datasets for game automation.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import GameProfile


class ModelTrainer:
    """Train YOLO models from datasets."""

    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "models/",
        config: Optional[GameProfile] = None
    ):
        """
        Initialize trainer with dataset path.

        Args:
            dataset_path: Path to YOLO dataset (Roboflow export format).
            output_dir: Directory to save trained models.
            config: Game profile with dataset configuration.
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.config = config

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model reference (will be loaded when training starts)
        self.model = None

    def prepare_dataset(self) -> bool:
        """
        Verify and prepare dataset for training.

        Returns:
            True if dataset is ready, False otherwise.
        """
        dataset_path = self.dataset_path

        # Check if dataset exists
        if not dataset_path.exists():
            print(f"[TRAIN] Dataset path does not exist: {dataset_path}")
            return False

        # Roboflow datasets typically have:
        # - data.yaml (dataset configuration)
        # - train/ (training images and labels)
        # - val/ (validation images and labels)
        # - test/ (optional test images and labels)

        data_yaml = dataset_path / "data.yaml"

        if not data_yaml.exists():
            print(f"[TRAIN] data.yaml not found in dataset: {data_yaml}")
            return False

        print(f"[TRAIN] Dataset found: {dataset_path}")
        print(f"[TRAIN] Configuration: {data_yaml}")

        return True

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = "cpu",
        pretrained: str = "yolo11n.pt"
    ) -> Optional[str]:
        """
        Train model and return path to trained model weights.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            image_size: Image size for training.
            device: Device to train on ("cpu" or "cuda").
            pretrained: Pretrained model to start from.

        Returns:
            Path to trained model weights, or None if training failed.
        """
        if not self.prepare_dataset():
            return None

        try:
            # Import ultralytics (required only for training)
            from ultralytics import YOLO

            print(f"[TRAIN] Starting training...")
            print(f"[TRAIN] Epochs: {epochs}, Batch size: {batch_size}, Image size: {image_size}")
            print(f"[TRAIN] Device: {device}")

            # Load pretrained model
            self.model = YOLO(pretrained)

            # Get data.yaml path
            data_yaml = self.dataset_path / "data.yaml"

            # Train model
            results = self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                project=str(self.output_dir),
                name="train",
                exist_ok=True,
                verbose=True
            )

            # Get path to best model weights
            model_path = Path(str(self.model.ckpt_path))
            print(f"[TRAIN] Training complete. Model saved to: {model_path}")

            return str(model_path)

        except ImportError:
            print("[TRAIN] ERROR: ultralytics package not installed.")
            print("[TRAIN] Install with: pip install ultralytics==8.1.0")
            return None
        except Exception as e:
            print(f"[TRAIN] Training failed: {e}")
            return None

    def resume_training(
        self,
        checkpoint_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = "cpu"
    ) -> Optional[str]:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pt).
            epochs: Total number of epochs (including completed).
            batch_size: Batch size for training.
            image_size: Image size for training.
            device: Device to train on ("cpu" or "cuda").

        Returns:
            Path to trained model weights, or None if training failed.
        """
        try:
            from ultralytics import YOLO

            print(f"[TRAIN] Resuming training from: {checkpoint_path}")

            self.model = YOLO(checkpoint_path)

            data_yaml = self.dataset_path / "data.yaml"

            results = self.model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                project=str(self.output_dir),
                name="train_resume",
                exist_ok=True,
                verbose=True,
                resume=True
            )

            model_path = Path(str(self.model.ckpt_path))
            print(f"[TRAIN] Training complete. Model saved to: {model_path}")

            return str(model_path)

        except Exception as e:
            print(f"[TRAIN] Failed to resume training: {e}")
            return None

    def export_model(
        self,
        model_path: str,
        format: str = "onnx"
    ) -> Optional[str]:
        """
        Export trained model to specified format.

        Args:
            model_path: Path to trained model weights.
            format: Export format ("onnx", "torchscript", "engine").

        Returns:
            Path to exported model, or None if export failed.
        """
        try:
            from ultralytics import YOLO

            print(f"[TRAIN] Exporting model to {format}...")

            model = YOLO(model_path)
            export_dir = self.output_dir / "exported"

            exported_path = model.export(format=format, project=str(export_dir))

            print(f"[TRAIN] Model exported to: {exported_path}")
            return str(exported_path)

        except Exception as e:
            print(f"[TRAIN] Failed to export model: {e}")
            return None

    def evaluate_model(
        self,
        model_path: str,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on validation set.

        Args:
            model_path: Path to trained model weights.
            metrics: List of metrics to compute.

        Returns:
            Dictionary of evaluation metrics.
        """
        try:
            from ultralytics import YOLO

            print(f"[TRAIN] Evaluating model: {model_path}")

            model = YOLO(model_path)
            data_yaml = self.dataset_path / "data.yaml"

            # Run validation
            results = model.val(data=str(data_yaml), verbose=True)

            return {
                "mAP50": float(results.box.map50),
                "mAP50_95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            }

        except Exception as e:
            print(f"[TRAIN] Failed to evaluate model: {e}")
            return {}


def create_minecraft_trainer(
    dataset_path: str = "minecraft.v10i.yolov11",
    output_dir: str = "models/"
) -> ModelTrainer:
    """
    Create a trainer configured for Minecraft dataset.

    Args:
        dataset_path: Path to Minecraft dataset.
        output_dir: Output directory for trained models.

    Returns:
        Configured ModelTrainer instance.
    """
    # Load Minecraft profile for configuration
    import yaml
    profile_path = Path(__file__).parent.parent / "profiles" / "minecraft.yaml"

    config = None
    if profile_path.exists():
        with open(profile_path, 'r') as f:
            profile_data = yaml.safe_load(f)
        from shared.models import GameProfile
        config = GameProfile.from_dict(profile_data)

    return ModelTrainer(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config
    )
