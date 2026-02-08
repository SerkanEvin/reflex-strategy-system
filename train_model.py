#!/usr/bin/env python3
"""
Training CLI Script
Train YOLO models for game automation from command line.

Usage:
    python train_model.py --profile profiles/minecraft.yaml --epochs 100 --device cpu
    python train_model.py --dataset path/to/dataset --output models/ --epochs 50
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent))

# Direct import to avoid dependency issues in __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location('trainer', 'local/trainer.py')
trainer_module = importlib.util.module_from_spec(spec)

# Mock imports that trainer.py needs but aren't needed for this script
sys.path.insert(0, str(Path(__file__).parent))
from shared.models import GameProfile

spec.loader.exec_module(trainer_module)
ModelTrainer = trainer_module.ModelTrainer
create_minecraft_trainer = trainer_module.create_minecraft_trainer


def load_profile(profile_path: str) -> GameProfile:
    """Load game profile from YAML file."""
    with open(profile_path, 'r') as f:
        profile_data = yaml.safe_load(f)
    return GameProfile.from_dict(profile_data)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO models for game automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train using Minecraft profile
  python train_model.py --profile profiles/minecraft.yaml --epochs 100

  # Train with custom dataset path
  python train_model.py --dataset minecraft.v10i.yolov11 --epochs 50 --batch 8

  # Train using GPU
  python train_model.py --profile profiles/minecraft.yaml --device cuda --epochs 200

  # Resume from checkpoint
  python train_model.py --resume models/train/weights/last.pt --epochs 150

  # Export model
  python train_model.py --export models/best.pt --format onnx
        """
    )

    # Input options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--profile", "-p",
        type=str,
        help="Path to game profile YAML file (e.g., profiles/minecraft.yaml)"
    )
    group.add_argument(
        "--dataset", "-d",
        type=str,
        help="Path to YOLO dataset directory"
    )

    # Training options
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        dest="batch_size",
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        dest="image_size",
        help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on (default: cpu)"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="yolov8n.pt",
        help="Pretrained model to start from (default: yolov8n.pt)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/",
        dest="output_dir",
        help="Directory to save trained models (default: models/)"
    )

    # Actions
    parser.add_argument(
        "--resume", "-r",
        type=str,
        metavar="CHECKPOINT",
        help="Resume training from checkpoint file"
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="MODEL_PATH",
        help="Export model to specified format (use with --format)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "engine"],
        help="Export format (default: onnx)"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        metavar="MODEL_PATH",
        help="Evaluate model on validation set"
    )

    # Other options
    parser.add_argument(
        "--verify-dataset",
        action="store_true",
        help="Only verify dataset, don't train"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Determine dataset path and config
    dataset_path = None
    config = None

    if args.profile:
        config = load_profile(args.profile)
        dataset_path = config.dataset_path
        if dataset_path is None:
            print(f"ERROR: Profile {args.profile} does not specify dataset_path")
            return 1

        # Convert relative dataset path to absolute
        if not Path(dataset_path).is_absolute():
            profile_dir = Path(args.profile).parent
            dataset_path = str(profile_dir / dataset_path)
    elif args.dataset:
        dataset_path = args.dataset
    else:
        # Default to Minecraft trainer
        trainer = create_minecraft_trainer()
        dataset_path = trainer.dataset_path

    # Handle export action
    if args.export:
        print(f"Exporting model: {args.export}")
        trainer = ModelTrainer(dataset_path, args.output_dir, config)
        result = trainer.export_model(args.export, args.format)
        if result:
            print(f"Exported to: {result}")
            return 0
        else:
            print("Export failed")
            return 1

    # Handle evaluate action
    if args.evaluate:
        print(f"Evaluating model: {args.evaluate}")
        trainer = ModelTrainer(dataset_path, args.output_dir, config)
        metrics = trainer.evaluate_model(args.evaluate)
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        return 0

    # Create trainer
    trainer = ModelTrainer(dataset_path, args.output_dir, config)

    # Verify dataset only
    if args.verify_dataset:
        print("Verifying dataset...")
        if trainer.prepare_dataset():
            print("Dataset is ready for training!")
            return 0
        else:
            print("Dataset verification failed!")
            return 1

    # Handle resume training
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        result = trainer.resume_training(
            checkpoint_path=args.resume,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device
        )
    else:
        # Regular training
        print("Starting training...")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {args.output_dir}")
        print(f"Device: {args.device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Image size: {args.image_size}")

        result = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
            pretrained=args.pretrained
        )

    if result:
        print(f"\nTraining complete! Model saved to: {result}")
        return 0
    else:
        print("\nTraining failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
