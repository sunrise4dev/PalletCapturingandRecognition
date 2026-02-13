"""Tests for barcode reader."""

import numpy as np

from src.barcode.barcode_reader import BarcodeReader


def test_barcode_reader_initialization():
    """Test BarcodeReader initializes with supported types."""
    reader = BarcodeReader(supported_types=["CODE128", "QRCODE"])
    assert reader.supported_types == ["CODE128", "QRCODE"]


def test_barcode_reader_no_filter():
    """Test BarcodeReader initializes without type filter."""
    reader = BarcodeReader()
    assert reader.supported_types is None


def test_read_blank_image():
    """Test that reading a blank image returns no barcodes."""
    reader = BarcodeReader()
    blank = np.zeros((100, 200), dtype=np.uint8)
    results = reader.read(blank)
    assert results == []


def test_read_with_preprocessing_blank():
    """Test multi-strategy preprocessing on a blank image."""
    reader = BarcodeReader()
    blank = np.zeros((100, 200, 3), dtype=np.uint8)
    results = reader.read_with_preprocessing(blank)
    assert results == []
