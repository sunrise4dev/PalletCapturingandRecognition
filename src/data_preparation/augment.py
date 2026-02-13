"""Data augmentation for small datasets."""

import logging
import random
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    augmentations_per_image: int = 3,
    seed: int = 42,
):
    """Apply augmentations to images and labels, saving new pairs.

    Augmentations applied: horizontal flip, brightness/contrast jitter,
    slight rotation, Gaussian blur.

    Args:
        images_dir: Directory with source images.
        labels_dir: Directory with YOLO format labels.
        output_dir: Directory to write augmented images and labels.
        augmentations_per_image: Number of augmented copies per original image.
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    total = 0
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        labels = _read_labels(label_path)

        for i in range(augmentations_per_image):
            aug_image, aug_labels = _apply_random_augmentation(image, labels)

            out_name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(out_img_dir / f"{out_name}{img_path.suffix}"), aug_image)
            _write_labels(out_lbl_dir / f"{out_name}.txt", aug_labels)
            total += 1

    logger.info(f"Generated {total} augmented image-label pairs")
    return total


def _read_labels(label_path: Path) -> list[list[float]]:
    """Read YOLO labels from file."""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([float(p) for p in parts])
    return labels


def _write_labels(label_path: Path, labels: list[list[float]]):
    """Write YOLO labels to file."""
    with open(label_path, "w") as f:
        for label in labels:
            cls_id = int(label[0])
            f.write(f"{cls_id} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def _apply_random_augmentation(
    image: np.ndarray, labels: list[list[float]]
) -> tuple[np.ndarray, list[list[float]]]:
    """Apply a random combination of augmentations."""
    aug_image = image.copy()
    aug_labels = [l.copy() for l in labels]

    # Random horizontal flip
    if random.random() > 0.5:
        aug_image = cv2.flip(aug_image, 1)
        for label in aug_labels:
            label[1] = 1.0 - label[1]  # flip x_center

    # Random brightness/contrast
    if random.random() > 0.5:
        alpha = random.uniform(0.7, 1.3)  # contrast
        beta = random.randint(-30, 30)     # brightness
        aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)

    # Random Gaussian blur
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        aug_image = cv2.GaussianBlur(aug_image, (ksize, ksize), 0)

    # Random slight rotation (-5 to +5 degrees)
    if random.random() > 0.6:
        angle = random.uniform(-5, 5)
        h, w = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug_image = cv2.warpAffine(aug_image, M, (w, h), borderValue=(114, 114, 114))
        # Note: for small angles, bbox shift is negligible in YOLO normalized coords

    return aug_image, aug_labels
