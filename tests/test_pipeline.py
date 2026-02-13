"""Tests for the pipeline result data structures and serialization."""

import numpy as np

from src.barcode.barcode_reader import BarcodeResult
from src.ocr.text_reader import OCRResult
from src.pipeline.full_pipeline import PalletResult, PipelineResult, StickerResult


def test_pipeline_result_to_dict():
    """Test PipelineResult serialization to dictionary."""
    result = PipelineResult(
        image_path="test.jpg",
        pallets=[
            PalletResult(
                bbox=(10, 20, 300, 400),
                confidence=0.95,
                aligned=True,
                stickers=[
                    StickerResult(
                        bbox=(5, 5, 100, 50),
                        sticker_type="sticker",
                        confidence=0.88,
                        ocr_result=OCRResult(
                            raw_text="Weight: 500kg",
                            lines=["Weight: 500kg"],
                            confidence=85.0,
                            fields={"weight": "500kg"},
                        ),
                        barcode_results=[],
                    ),
                    StickerResult(
                        bbox=(5, 60, 100, 100),
                        sticker_type="barcode",
                        confidence=0.92,
                        ocr_result=None,
                        barcode_results=[
                            BarcodeResult(
                                data="ABC123456",
                                barcode_type="CODE128",
                                bbox=(10, 10, 80, 30),
                                confidence=1.0,
                            ),
                        ],
                    ),
                ],
            ),
        ],
        processing_time_ms=150.5,
    )

    d = result.to_dict()
    assert d["image_path"] == "test.jpg"
    assert d["num_pallets"] == 1
    assert len(d["pallets"]) == 1
    assert len(d["pallets"][0]["stickers"]) == 2
    assert d["pallets"][0]["stickers"][0]["ocr_text"] == "Weight: 500kg"
    assert d["pallets"][0]["stickers"][1]["barcodes"][0]["data"] == "ABC123456"


def test_get_all_text():
    """Test extracting all OCR text from results."""
    result = PipelineResult(
        image_path=None,
        pallets=[
            PalletResult(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                stickers=[
                    StickerResult(
                        bbox=(0, 0, 50, 50),
                        sticker_type="sticker",
                        confidence=0.8,
                        ocr_result=OCRResult(
                            raw_text="Hello World",
                            lines=["Hello World"],
                            confidence=90.0,
                        ),
                    ),
                ],
            ),
        ],
        processing_time_ms=100.0,
    )

    texts = result.get_all_text()
    assert texts == ["Hello World"]


def test_get_all_barcodes():
    """Test extracting all barcode data from results."""
    result = PipelineResult(
        image_path=None,
        pallets=[
            PalletResult(
                bbox=(0, 0, 100, 100),
                confidence=0.9,
                stickers=[
                    StickerResult(
                        bbox=(0, 0, 50, 50),
                        sticker_type="barcode",
                        confidence=0.8,
                        barcode_results=[
                            BarcodeResult(data="CODE1", barcode_type="CODE128", bbox=(0, 0, 10, 10), confidence=1.0),
                            BarcodeResult(data="CODE2", barcode_type="QRCODE", bbox=(0, 0, 10, 10), confidence=1.0),
                        ],
                    ),
                ],
            ),
        ],
        processing_time_ms=100.0,
    )

    barcodes = result.get_all_barcodes()
    assert barcodes == ["CODE1", "CODE2"]


def test_empty_pipeline_result():
    """Test handling of empty results."""
    result = PipelineResult(
        image_path=None,
        pallets=[],
        processing_time_ms=10.0,
    )

    d = result.to_dict()
    assert d["num_pallets"] == 0
    assert result.get_all_text() == []
    assert result.get_all_barcodes() == []
