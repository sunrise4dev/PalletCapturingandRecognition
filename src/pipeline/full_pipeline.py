"""End-to-end pallet recognition pipeline."""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.barcode.barcode_reader import BarcodeReader, BarcodeResult
from src.detection.alignment import align_pallet
from src.detection.pallet_detector import PalletDetector
from src.detection.sticker_detector import StickerDetector
from src.ocr.text_reader import OCRResult, TextReader
from src.utils.config_loader import load_config
from src.utils.image_utils import load_image, save_image
from src.utils.visualization import draw_pipeline_result

logger = logging.getLogger(__name__)


@dataclass
class StickerResult:
    """Result for a single sticker/barcode region."""
    bbox: tuple[int, int, int, int]
    sticker_type: str  # "sticker" or "barcode"
    confidence: float
    ocr_result: OCRResult | None = None
    barcode_results: list[BarcodeResult] = field(default_factory=list)


@dataclass
class PalletResult:
    """Result for a single detected pallet."""
    bbox: tuple[int, int, int, int]
    confidence: float
    aligned: bool = False
    stickers: list[StickerResult] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete pipeline result for one image."""
    image_path: str | None
    pallets: list[PalletResult]
    processing_time_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "image_path": self.image_path,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "num_pallets": len(self.pallets),
            "pallets": [
                {
                    "bbox": list(p.bbox),
                    "confidence": round(p.confidence, 4),
                    "aligned": p.aligned,
                    "stickers": [
                        {
                            "bbox": list(s.bbox),
                            "type": s.sticker_type,
                            "confidence": round(s.confidence, 4),
                            "ocr_text": s.ocr_result.raw_text if s.ocr_result else None,
                            "ocr_fields": s.ocr_result.fields if s.ocr_result else {},
                            "ocr_confidence": round(s.ocr_result.confidence, 2) if s.ocr_result else None,
                            "barcodes": [
                                {"data": b.data, "type": b.barcode_type}
                                for b in s.barcode_results
                            ],
                        }
                        for s in p.stickers
                    ],
                }
                for p in self.pallets
            ],
            "metadata": self.metadata,
        }

    def get_all_text(self) -> list[str]:
        """Get all OCR text from all stickers across all pallets."""
        texts = []
        for pallet in self.pallets:
            for sticker in pallet.stickers:
                if sticker.ocr_result and sticker.ocr_result.raw_text.strip():
                    texts.append(sticker.ocr_result.raw_text.strip())
        return texts

    def get_all_barcodes(self) -> list[str]:
        """Get all barcode data strings."""
        barcodes = []
        for pallet in self.pallets:
            for sticker in pallet.stickers:
                for bc in sticker.barcode_results:
                    barcodes.append(bc.data)
        return barcodes


class PalletRecognitionPipeline:
    """End-to-end pipeline: Image -> Pallet -> Sticker -> OCR + Barcode -> Results."""

    def __init__(self, config_path: str):
        """Initialize all components from a pipeline.yaml config.

        Args:
            config_path: Path to pipeline configuration YAML file.
        """
        self.config = load_config(config_path)
        pipeline_cfg = self.config["pipeline"]

        # Pallet detector
        pallet_cfg = pipeline_cfg["pallet_detection"]
        self.pallet_detector = PalletDetector(
            model_path=pallet_cfg["model_path"],
            confidence=pallet_cfg["confidence_threshold"],
            iou_threshold=pallet_cfg["iou_threshold"],
        )

        # Alignment settings
        self.alignment_cfg = pipeline_cfg.get("alignment", {})
        self.align_enabled = self.alignment_cfg.get("enabled", True)

        # Sticker detector
        sticker_cfg = pipeline_cfg["sticker_detection"]
        self.sticker_detector = StickerDetector(
            model_path=sticker_cfg["model_path"],
            confidence=sticker_cfg["confidence_threshold"],
            iou_threshold=sticker_cfg["iou_threshold"],
        )

        # OCR reader
        ocr_cfg = pipeline_cfg.get("ocr", {})
        self.text_reader = TextReader(
            language=ocr_cfg.get("language", "eng"),
            psm=ocr_cfg.get("psm", 6),
            preprocessing=ocr_cfg.get("preprocessing"),
        )

        # Barcode reader
        barcode_cfg = pipeline_cfg.get("barcode", {})
        self.barcode_reader = BarcodeReader(
            supported_types=barcode_cfg.get("supported_types"),
        ) if barcode_cfg.get("enabled", True) else None

        # Output settings
        self.output_cfg = pipeline_cfg.get("output", {})

    def process_image(self, image: np.ndarray, image_path: str | None = None) -> PipelineResult:
        """Process a single image through the full pipeline.

        Args:
            image: BGR numpy array.
            image_path: Optional source file path for metadata.

        Returns:
            PipelineResult with all detections and extracted data.
        """
        start_time = time.time()
        pallets = []

        # Step 1: Detect pallets
        pallet_detections = self.pallet_detector.detect(image)

        for det in pallet_detections:
            pallet_image = det.cropped_image
            aligned = False

            # Step 2: Alignment
            if self.align_enabled and pallet_image is not None:
                try:
                    pallet_image = align_pallet(
                        image,
                        det.bbox,
                        method=self.alignment_cfg.get("method", "perspective"),
                        padding_ratio=self.alignment_cfg.get("padding_ratio", 0.05),
                    )
                    aligned = True
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}, using original crop")
                    pallet_image = det.cropped_image

            if pallet_image is None or pallet_image.size == 0:
                continue

            # Step 3: Detect stickers and barcodes
            sticker_results = []
            try:
                sticker_detections = self.sticker_detector.detect(pallet_image)
            except Exception as e:
                logger.warning(f"Sticker detection failed: {e}")
                sticker_detections = []

            for s_det in sticker_detections:
                sticker_result = StickerResult(
                    bbox=s_det.bbox,
                    sticker_type=s_det.class_name,
                    confidence=s_det.confidence,
                )

                if s_det.cropped_image is not None and s_det.cropped_image.size > 0:
                    # Step 4: OCR on sticker regions
                    if s_det.class_name == "sticker":
                        try:
                            sticker_result.ocr_result = self.text_reader.read(s_det.cropped_image)
                        except Exception as e:
                            logger.warning(f"OCR failed: {e}")

                    # Step 5: Barcode reading
                    if self.barcode_reader:
                        try:
                            sticker_result.barcode_results = (
                                self.barcode_reader.read_with_preprocessing(s_det.cropped_image)
                            )
                        except Exception as e:
                            logger.warning(f"Barcode reading failed: {e}")

                sticker_results.append(sticker_result)

            pallets.append(PalletResult(
                bbox=det.bbox,
                confidence=det.confidence,
                aligned=aligned,
                stickers=sticker_results,
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        return PipelineResult(
            image_path=image_path,
            pallets=pallets,
            processing_time_ms=elapsed_ms,
            metadata={
                "image_shape": list(image.shape),
                "num_pallets": len(pallets),
            },
        )

    def process_image_file(self, image_path: str) -> PipelineResult:
        """Load an image from disk and process it."""
        image = load_image(image_path)
        return self.process_image(image, image_path=image_path)

    def process_directory(
        self,
        directory: str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> list[PipelineResult]:
        """Process all images in a directory."""
        directory = Path(directory)
        image_files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in extensions
        )

        results = []
        for img_path in image_files:
            logger.info(f"Processing {img_path.name}")
            try:
                result = self.process_image_file(str(img_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")

        return results

    def save_results(
        self,
        results: list[PipelineResult],
        output_dir: str | None = None,
        save_visualizations: bool | None = None,
    ):
        """Save results as JSON files and optional visualizations."""
        output_dir = output_dir or self.output_cfg.get("results_dir", "output/results")
        save_vis = save_visualizations if save_visualizations is not None else \
            self.output_cfg.get("save_visualizations", False)
        vis_dir = self.output_cfg.get("visualization_dir", "output/visualizations")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for result in results:
            if result.image_path:
                name = Path(result.image_path).stem
            else:
                name = f"result_{id(result)}"

            # Save JSON
            json_path = output_path / f"{name}.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Save visualization
            if save_vis and result.image_path:
                try:
                    image = load_image(result.image_path)
                    vis_image = draw_pipeline_result(image, result)
                    vis_path = Path(vis_dir) / f"{name}_viz.jpg"
                    save_image(vis_image, str(vis_path))
                except Exception as e:
                    logger.warning(f"Failed to save visualization for {name}: {e}")

        logger.info(f"Saved {len(results)} results to {output_dir}")
