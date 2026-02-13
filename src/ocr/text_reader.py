"""OCR on sticker images using Tesseract."""

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR text extraction."""
    raw_text: str
    lines: list[str]
    confidence: float
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class TextBox:
    """A single detected text region with position."""
    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float


class TextReader:
    """Reads text from sticker images using Tesseract OCR."""

    def __init__(
        self,
        language: str = "eng",
        psm: int = 6,
        preprocessing: dict | None = None,
    ):
        """
        Args:
            language: Tesseract language code.
            psm: Page segmentation mode (6 = uniform block of text).
            preprocessing: Config dict with keys: grayscale, threshold,
                          threshold_method, denoise, scale_factor.
        """
        self.language = language
        self.psm = psm
        self.preprocessing = preprocessing or {
            "grayscale": True,
            "threshold": True,
            "threshold_method": "otsu",
            "denoise": True,
            "scale_factor": 2.0,
        }

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to improve OCR accuracy.

        Returns:
            Preprocessed grayscale image.
        """
        processed = image.copy()

        # Convert to grayscale
        if self.preprocessing.get("grayscale", True) and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Upscale
        scale = self.preprocessing.get("scale_factor", 1.0)
        if scale > 1.0:
            h, w = processed.shape[:2]
            processed = cv2.resize(
                processed,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Denoise
        if self.preprocessing.get("denoise", False):
            if len(processed.shape) == 2:
                processed = cv2.fastNlMeansDenoising(processed, h=10)
            else:
                processed = cv2.fastNlMeansDenoisingColored(processed, h=10)

        # Threshold
        if self.preprocessing.get("threshold", False) and len(processed.shape) == 2:
            method = self.preprocessing.get("threshold_method", "otsu")
            if method == "otsu":
                _, processed = cv2.threshold(
                    processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
            elif method == "adaptive":
                processed = cv2.adaptiveThreshold(
                    processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2,
                )

        return processed

    def read(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a single sticker image.

        Returns:
            OCRResult with extracted text and confidence.
        """
        processed = self.preprocess(image)
        config = f"--psm {self.psm}"

        # Get text
        raw_text = pytesseract.image_to_string(
            processed, lang=self.language, config=config,
        )

        # Get confidence from detailed output
        data = pytesseract.image_to_data(
            processed, lang=self.language, config=config,
            output_type=Output.DICT,
        )

        confidences = [
            int(c) for c, t in zip(data["conf"], data["text"])
            if int(c) > 0 and t.strip()
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        fields = self._parse_structured_text(lines)

        return OCRResult(
            raw_text=raw_text.strip(),
            lines=lines,
            confidence=avg_confidence,
            fields=fields,
        )

    def read_with_boxes(self, image: np.ndarray) -> list[TextBox]:
        """Run OCR and return individual text boxes with positions."""
        processed = self.preprocess(image)
        config = f"--psm {self.psm}"

        data = pytesseract.image_to_data(
            processed, lang=self.language, config=config,
            output_type=Output.DICT,
        )

        boxes = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if text and conf > 0:
                boxes.append(TextBox(
                    text=text,
                    bbox=(
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    ),
                    confidence=float(conf),
                ))

        return boxes

    @staticmethod
    def _parse_structured_text(lines: list[str]) -> dict[str, str]:
        """Attempt to parse key:value patterns from text lines.

        Looks for common pallet label patterns like:
        - "Weight: 500kg"
        - "PO# 12345"
        - "LOT: ABC-123"
        """
        fields = {}
        known_keys = [
            "weight", "po", "lot", "sku", "qty", "quantity",
            "date", "origin", "destination", "item", "batch",
            "serial", "ref", "order",
        ]

        for line in lines:
            # Try "key: value" or "key = value" pattern
            match = re.match(r"^([A-Za-z#\s]+?)\s*[:=]\s*(.+)$", line)
            if match:
                key = match.group(1).strip().lower().rstrip("#")
                value = match.group(2).strip()
                fields[key] = value
                continue

            # Try matching known keywords
            line_lower = line.lower()
            for key in known_keys:
                if key in line_lower:
                    # Extract the value part after the keyword
                    pattern = rf"{key}[#\s:=]*\s*(.+)"
                    m = re.search(pattern, line_lower)
                    if m:
                        fields[key] = m.group(1).strip()

        return fields
