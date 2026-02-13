"""Tests for OCR text reader preprocessing and parsing."""

import numpy as np

from src.ocr.text_reader import TextReader


def test_preprocessing_grayscale():
    """Test that preprocessing converts color image to grayscale."""
    reader = TextReader(preprocessing={"grayscale": True, "threshold": False, "denoise": False, "scale_factor": 1.0})
    color_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    processed = reader.preprocess(color_image)
    assert len(processed.shape) == 2  # grayscale has no channel dim


def test_preprocessing_upscale():
    """Test that preprocessing upscales image by configured factor."""
    reader = TextReader(preprocessing={"grayscale": True, "threshold": False, "denoise": False, "scale_factor": 2.0})
    image = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
    processed = reader.preprocess(image)
    assert processed.shape[0] == 100  # height doubled
    assert processed.shape[1] == 200  # width doubled


def test_parse_structured_text_key_value():
    """Test parsing key:value patterns from text lines."""
    lines = [
        "Weight: 500kg",
        "PO# 12345",
        "LOT: ABC-123",
    ]
    fields = TextReader._parse_structured_text(lines)
    assert "weight" in fields
    assert fields["weight"] == "500kg"


def test_parse_structured_text_empty():
    """Test parsing with no recognizable patterns."""
    fields = TextReader._parse_structured_text(["random text here"])
    # Should not crash, may or may not find fields
    assert isinstance(fields, dict)
