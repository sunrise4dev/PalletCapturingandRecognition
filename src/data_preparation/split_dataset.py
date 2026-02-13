"""Split annotated dataset into train/val/test sets in YOLO directory structure."""

import logging
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_base_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split images and labels into train/val/test sets.

    Creates the YOLO directory structure:
        output_base_dir/
            train/images/, train/labels/
            val/images/, val/labels/
            test/images/, test/labels/

    Args:
        images_dir: Directory containing image files.
        labels_dir: Directory containing YOLO format .txt label files.
        output_base_dir: Root output directory.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_base_dir = Path(output_base_dir)

    # Find all images
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Match images to labels
    paired = []
    unpaired = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            unpaired.append(img_path)

    if unpaired:
        logger.warning(f"{len(unpaired)} images have no matching label file")

    logger.info(f"Found {len(paired)} image-label pairs")

    # Split
    train_pairs, temp_pairs = train_test_split(
        paired, train_size=train_ratio, random_state=seed,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_pairs, test_pairs = train_test_split(
        temp_pairs, train_size=relative_val, random_state=seed,
    )

    # Copy files to output directories
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split_name, pairs in splits.items():
        img_out = output_base_dir / split_name / "images"
        lbl_out = output_base_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(label_path, lbl_out / label_path.name)

        logger.info(f"{split_name}: {len(pairs)} images")

    return {name: len(pairs) for name, pairs in splits.items()}


def verify_split(output_base_dir: str):
    """Print statistics about a dataset split."""
    output_base_dir = Path(output_base_dir)

    for split in ["train", "val", "test"]:
        img_dir = output_base_dir / split / "images"
        lbl_dir = output_base_dir / split / "labels"

        n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        # Count class distribution
        class_counts: dict[int, int] = {}
        if lbl_dir.exists():
            for label_file in lbl_dir.glob("*.txt"):
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        print(f"  {split}: {n_images} images, {n_labels} labels, classes: {class_counts}")
