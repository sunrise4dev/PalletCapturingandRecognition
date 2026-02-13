"""Visualization utilities for detection results."""

import cv2
import numpy as np


# Color palette for different detection types
COLORS = {
    "pallet": (0, 255, 0),    # green
    "sticker": (255, 165, 0),  # orange
    "barcode": (0, 0, 255),    # red
    "text": (255, 255, 0),     # cyan
}


def draw_detections(
    image: np.ndarray,
    detections: list,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: BGR image.
        detections: List of objects with bbox, confidence, class_name attributes.
        color: Default BGR color for boxes.
        thickness: Line thickness.

    Returns:
        Annotated image copy.
    """
    result = image.copy()

    for det in detections:
        box_color = COLORS.get(det.class_name, color)
        x1, y1, x2, y2 = det.bbox

        cv2.rectangle(result, (x1, y1), (x2, y2), box_color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            result,
            (x1, y1 - label_size[1] - 6),
            (x1 + label_size[0], y1),
            box_color,
            -1,
        )
        cv2.putText(
            result, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return result


def draw_pipeline_result(image: np.ndarray, result) -> np.ndarray:
    """Draw full pipeline results with hierarchical color coding.

    Args:
        image: Original BGR image.
        result: PipelineResult instance.

    Returns:
        Annotated image copy.
    """
    output = image.copy()

    for pallet in result.pallets:
        # Draw pallet box
        x1, y1, x2, y2 = pallet.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), COLORS["pallet"], 3)
        cv2.putText(
            output, f"Pallet {pallet.confidence:.2f}",
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["pallet"], 2,
        )

        # Draw sticker/barcode boxes
        for sticker in pallet.stickers:
            sx1, sy1, sx2, sy2 = sticker.bbox
            # Offset by pallet position
            sx1 += x1
            sy1 += y1
            sx2 += x1
            sy2 += y1

            box_color = COLORS.get(sticker.sticker_type, COLORS["sticker"])
            cv2.rectangle(output, (sx1, sy1), (sx2, sy2), box_color, 2)

            # Show OCR text if available
            if sticker.ocr_result and sticker.ocr_result.raw_text.strip():
                text = sticker.ocr_result.raw_text.strip()[:40]
                cv2.putText(
                    output, text, (sx1, sy2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1,
                )

            # Show barcode data if available
            for bc in sticker.barcode_results:
                cv2.putText(
                    output, f"BC: {bc.data[:30]}",
                    (sx1, sy2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["barcode"], 1,
                )

    return output


def create_result_grid(
    images: list[np.ndarray], cols: int = 3
) -> np.ndarray:
    """Arrange multiple result images in a grid."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Resize all to same height
    target_h = 400
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(img, (int(w * scale), target_h)))

    # Pad widths to max
    max_w = max(img.shape[1] for img in resized)
    padded = []
    for img in resized:
        pad = np.zeros((target_h, max_w, 3), dtype=np.uint8)
        pad[:, : img.shape[1]] = img
        padded.append(pad)

    # Arrange in grid
    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i : i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)
