#!/usr/bin/env python3
"""Run the full pallet recognition pipeline on images.

Usage:
    python scripts/run_inference.py --config configs/pipeline.yaml --image path/to/image.jpg
    python scripts/run_inference.py --config configs/pipeline.yaml --directory path/to/images/
    python scripts/run_inference.py --config configs/pipeline.yaml --image path/to/image.jpg --visualize
"""

import argparse
import json
import logging

from src.pipeline.full_pipeline import PalletRecognitionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run pallet recognition pipeline")
    parser.add_argument(
        "--config", default="configs/pipeline.yaml",
        help="Path to pipeline config YAML",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--directory", help="Path to a directory of images")
    parser.add_argument("--output", default="output/results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Save annotated images")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    pipeline = PalletRecognitionPipeline(args.config)

    if args.image:
        results = [pipeline.process_image_file(args.image)]
    else:
        results = pipeline.process_directory(args.directory)

    # Print summary
    for result in results:
        print(f"\n{'='*60}")
        print(f"Image: {result.image_path}")
        print(f"Processing time: {result.processing_time_ms:.1f} ms")
        print(f"Pallets found: {len(result.pallets)}")

        for i, pallet in enumerate(result.pallets):
            print(f"\n  Pallet {i+1} (conf: {pallet.confidence:.2f}, aligned: {pallet.aligned})")
            for j, sticker in enumerate(pallet.stickers):
                print(f"    Sticker {j+1} [{sticker.sticker_type}] (conf: {sticker.confidence:.2f})")
                if sticker.ocr_result:
                    print(f"      OCR text: {sticker.ocr_result.raw_text[:100]}")
                    if sticker.ocr_result.fields:
                        print(f"      Fields: {sticker.ocr_result.fields}")
                for bc in sticker.barcode_results:
                    print(f"      Barcode [{bc.barcode_type}]: {bc.data}")

        if args.verbose:
            print(f"\nFull JSON:")
            print(json.dumps(result.to_dict(), indent=2))

    # Save results
    pipeline.save_results(results, args.output, save_visualizations=args.visualize)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
