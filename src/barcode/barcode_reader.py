"""Barcode and QR code reading using pyzbar."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from pyzbar import pyzbar

logger = logging.getLogger(__name__)


@dataclass
class BarcodeResult:
    """A decoded barcode result."""
    data: str
    barcode_type: str
    bbox: tuple[int, int, int, int]  # x, y, width, height
    confidence: float  # 1.0 for pyzbar (binary decode)


class BarcodeReader:
    """Reads barcodes and QR codes from images."""

    def __init__(self, supported_types: list[str] | None = None):
        """
        Args:
            supported_types: List of pyzbar symbology names to accept (e.g., ["CODE128", "QRCODE"]).
                           None means accept all types.
        """
        self.supported_types = supported_types

    def read(self, image: np.ndarray) -> list[BarcodeResult]:
        """Decode all barcodes found in the image.

        Args:
            image: BGR numpy array.

        Returns:
            List of BarcodeResult instances.
        """
        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        decoded = pyzbar.decode(gray)
        return self._filter_results(decoded)

    def read_with_preprocessing(self, image: np.ndarray) -> list[BarcodeResult]:
        """Try multiple preprocessing strategies to maximize barcode detection.

        Strategies: original, sharpened, Otsu threshold, inverted, upscaled.
        Returns results from the first successful strategy.
        """
        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        strategies = [
            ("original", gray),
            ("sharpened", self._sharpen(gray)),
            ("otsu", self._otsu_threshold(gray)),
            ("inverted", cv2.bitwise_not(self._otsu_threshold(gray))),
            ("upscaled", self._upscale(gray)),
        ]

        for name, processed in strategies:
            decoded = pyzbar.decode(processed)
            results = self._filter_results(decoded)
            if results:
                logger.debug(f"Barcode decoded using '{name}' strategy")
                return results

        return []

    def _filter_results(self, decoded: list) -> list[BarcodeResult]:
        """Filter decoded barcodes by supported types."""
        results = []
        for barcode in decoded:
            barcode_type = barcode.type

            if self.supported_types and barcode_type not in self.supported_types:
                continue

            data = barcode.data.decode("utf-8", errors="replace")
            rect = barcode.rect

            results.append(BarcodeResult(
                data=data,
                barcode_type=barcode_type,
                bbox=(rect.left, rect.top, rect.width, rect.height),
                confidence=1.0,
            ))

        return results

    @staticmethod
    def _sharpen(gray: np.ndarray) -> np.ndarray:
        """Apply unsharp mask sharpening."""
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> np.ndarray:
        """Apply Otsu's threshold."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _upscale(gray: np.ndarray, factor: int = 2) -> np.ndarray:
        """Upscale image for better barcode detection."""
        h, w = gray.shape
        return cv2.resize(gray, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
