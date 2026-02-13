#!/usr/bin/env python3
"""Export trained models to ONNX, TFLite, or TensorFlow.js formats.

Usage:
    python scripts/export_model.py --model models/pallet_detection/best.pt --format onnx
    python scripts/export_model.py --model models/sticker_detection/best.pt --format tflite
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    "onnx": "ONNX",
    "tflite": "TensorFlow Lite",
    "tfjs": "TensorFlow.js",
    "torchscript": "TorchScript",
    "coreml": "CoreML",
}


def main():
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--model", required=True, help="Path to trained .pt weights")
    parser.add_argument(
        "--format", required=True, choices=SUPPORTED_FORMATS.keys(),
        help="Export format",
    )
    parser.add_argument("--output", default="models/exports", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_path.name} to {SUPPORTED_FORMATS[args.format]}...")

    model = YOLO(str(model_path))
    export_path = model.export(format=args.format, imgsz=args.imgsz)

    logger.info(f"Model exported to: {export_path}")

    # Print file size
    export_file = Path(export_path)
    if export_file.is_file():
        size_mb = export_file.stat().st_size / (1024 * 1024)
        logger.info(f"Export size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
