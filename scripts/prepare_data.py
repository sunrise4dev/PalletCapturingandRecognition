#!/usr/bin/env python3
"""Data preparation script for pallet recognition dataset.

Usage:
    python scripts/prepare_data.py convert --format voc --input data/raw/annotations --output data/raw/yolo_labels --classes pallet
    python scripts/prepare_data.py split --images data/raw/images --labels data/raw/yolo_labels --output data/processed/pallet_detection
    python scripts/prepare_data.py augment --images data/processed/pallet_detection/train/images --labels data/processed/pallet_detection/train/labels --output data/augmented
"""

import argparse
import logging

from src.data_preparation.augment import augment_dataset
from src.data_preparation.convert_annotations import convert_to_yolo_format
from src.data_preparation.split_dataset import split_dataset, verify_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for pallet recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert annotations to YOLO format")
    convert_parser.add_argument("--format", choices=["voc", "coco", "labelme"], required=True)
    convert_parser.add_argument("--input", required=True, help="Input annotations directory")
    convert_parser.add_argument("--output", required=True, help="Output YOLO labels directory")
    convert_parser.add_argument("--classes", nargs="+", required=True, help="Class names in order")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--images", required=True, help="Images directory")
    split_parser.add_argument("--labels", required=True, help="YOLO labels directory")
    split_parser.add_argument("--output", required=True, help="Output base directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.7)
    split_parser.add_argument("--val-ratio", type=float, default=0.2)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)
    split_parser.add_argument("--seed", type=int, default=42)

    # Augment command
    aug_parser = subparsers.add_parser("augment", help="Augment training data")
    aug_parser.add_argument("--images", required=True, help="Images directory")
    aug_parser.add_argument("--labels", required=True, help="Labels directory")
    aug_parser.add_argument("--output", required=True, help="Output directory")
    aug_parser.add_argument("--count", type=int, default=3, help="Augmentations per image")

    args = parser.parse_args()

    if args.command == "convert":
        class_map = {name: idx for idx, name in enumerate(args.classes)}
        convert_to_yolo_format(args.input, args.output, args.format, class_map)

    elif args.command == "split":
        counts = split_dataset(
            args.images, args.labels, args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(f"\nSplit complete: {counts}")
        verify_split(args.output)

    elif args.command == "augment":
        total = augment_dataset(
            args.images, args.labels, args.output,
            augmentations_per_image=args.count,
        )
        print(f"\nGenerated {total} augmented samples")


if __name__ == "__main__":
    main()
