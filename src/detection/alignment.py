"""Perspective/affine correction for detected pallet regions."""

import cv2
import numpy as np


def align_pallet(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    method: str = "perspective",
    padding_ratio: float = 0.05,
) -> np.ndarray:
    """Align a detected pallet region to a fronto-parallel view.

    Steps:
    1. Crop the pallet region with padding.
    2. Detect edges and find the dominant quadrilateral contour.
    3. Apply perspective transform if a quadrilateral is found.
    4. Fall back to rotation correction via minimum area rectangle.

    Args:
        image: Full frame BGR image.
        bbox: Pallet bounding box (x1, y1, x2, y2).
        method: "perspective" or "affine".
        padding_ratio: Extra padding around bbox.

    Returns:
        Aligned pallet image.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Add padding
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = image[y1:y2, x1:x2].copy()

    # Try to find a quadrilateral contour
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    contour = _find_pallet_contour(gray)

    if contour is not None and method == "perspective":
        return _perspective_transform(cropped, contour)

    # Fallback: rotation correction
    return _rotation_correction(cropped, gray)


def _find_pallet_contour(gray: np.ndarray) -> np.ndarray | None:
    """Find the largest quadrilateral contour in the image.

    Returns:
        4-point contour array or None if not found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area, try largest contours first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(approx) > gray.shape[0] * gray.shape[1] * 0.1:
            return approx.reshape(4, 2).astype(np.float32)

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference

    return rect


def _perspective_transform(image: np.ndarray, src_points: np.ndarray) -> np.ndarray:
    """Apply perspective transform to make the pallet rectangular."""
    ordered = _order_points(src_points)
    tl, tr, br, bl = ordered

    # Compute output dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    if max_width <= 0 or max_height <= 0:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def _rotation_correction(image: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Correct rotation using minimum area rectangle angle."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[1]  # rotation angle

    # Normalize angle
    if angle < -45:
        angle += 90

    if abs(angle) < 1:
        return image  # no significant rotation

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))
