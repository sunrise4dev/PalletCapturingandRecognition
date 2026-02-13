"""Common image I/O and manipulation utilities."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR numpy array."""
    path = str(path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(image: np.ndarray, path: str):
    """Save an image to disk, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resize_with_padding(
    image: np.ndarray,
    target_size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize image maintaining aspect ratio with letterbox padding."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def crop_region(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: float = 0.0,
) -> np.ndarray:
    """Crop a region from an image with optional padding.

    Args:
        image: Source image (BGR).
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        padding: Extra padding as fraction of bbox size.

    Returns:
        Cropped image region.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    if padding > 0:
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

    return image[y1:y2, x1:x2].copy()
