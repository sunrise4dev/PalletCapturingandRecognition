#!/usr/bin/env python3
"""Evaluate model performance on the test dataset.

Usage:
    python scripts/evaluate.py --model models/pallet_detection/best.pt --data data/pallet_detection.yaml
    python scripts/evaluate.py --model models/sticker_detection/best.pt --data data/sticker_detection.yaml --save-plots
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Path to trained .pt weights")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--save-plots", action="store_true", help="Save confusion matrix and PR curves")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Evaluating {model_path.name} on {args.split} split...")

    model = YOLO(str(model_path))
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        plots=args.save_plots,
    )

    # Print metrics
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.split} split)")
    print(f"{'='*50}")
    print(f"  mAP@50:      {results.box.map50:.4f}")
    print(f"  mAP@50-95:   {results.box.map:.4f}")
    print(f"  Precision:    {results.box.mp:.4f}")
    print(f"  Recall:       {results.box.mr:.4f}")

    # Per-class metrics
    if hasattr(results.box, "maps") and len(results.box.maps) > 1:
        print(f"\nPer-class mAP@50:")
        for i, map50 in enumerate(results.box.maps):
            print(f"  Class {i}: {map50:.4f}")

    if args.save_plots:
        print(f"\nPlots saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
