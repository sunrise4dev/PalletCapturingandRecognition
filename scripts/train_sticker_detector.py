#!/usr/bin/env python3
"""Train the sticker/barcode detection YOLOv8 model.

Usage:
    python scripts/train_sticker_detector.py --config configs/sticker_detection.yaml
    python scripts/train_sticker_detector.py --config configs/sticker_detection.yaml --resume
"""

import argparse
import logging
import shutil
from pathlib import Path

from src.detection.sticker_detector import StickerDetector
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train sticker detection model")
    parser.add_argument(
        "--config", default="configs/sticker_detection.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    data_yaml = config["training"]["data"]
    save_dir = Path(config["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting sticker detector training...")
    logger.info(f"  Data: {data_yaml}")
    logger.info(f"  Base model: {config['model']['base']}")
    logger.info(f"  Epochs: {config['training']['epochs']}")

    best_path = StickerDetector.train(data_yaml, config)

    # Copy best weights to output directory
    dest = save_dir / "best.pt"
    shutil.copy2(best_path, dest)
    logger.info(f"Best weights saved to {dest}")


if __name__ == "__main__":
    main()
