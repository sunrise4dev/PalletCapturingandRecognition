"""Sticker and barcode detection on cropped pallet regions using YOLOv8."""

import logging

import numpy as np
from ultralytics import YOLO

from .pallet_detector import Detection

logger = logging.getLogger(__name__)


class StickerDetector:
    """Detects stickers and barcodes on cropped pallet images using YOLOv8."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = "",
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, pallet_image: np.ndarray) -> list[Detection]:
        """Run sticker/barcode detection on a cropped pallet image.

        Args:
            pallet_image: BGR numpy array of a cropped pallet region.

        Returns:
            List of Detection instances (class_name: "sticker" or "barcode").
        """
        results = self.model.predict(
            pallet_image,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device or None,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = results[0].names[cls_id]

                h, w = pallet_image.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                cropped = pallet_image[cy1:cy2, cx1:cx2].copy()

                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_name=class_name,
                    cropped_image=cropped,
                ))

        logger.info(f"Detected {len(detections)} sticker/barcode region(s)")
        return detections

    @staticmethod
    def train(data_yaml: str, config: dict) -> str:
        """Train a new sticker detection model.

        Args:
            data_yaml: Path to YOLO dataset yaml.
            config: Training config dict (from configs/sticker_detection.yaml).

        Returns:
            Path to the best weights file.
        """
        model = YOLO(config["model"]["base"])
        train_cfg = config["training"]
        output_cfg = config["output"]

        results = model.train(
            data=data_yaml,
            epochs=train_cfg["epochs"],
            imgsz=train_cfg["image_size"],
            batch=train_cfg["batch_size"],
            patience=train_cfg["patience"],
            optimizer=train_cfg["optimizer"],
            lr0=train_cfg["lr0"],
            weight_decay=train_cfg["weight_decay"],
            augment=train_cfg["augment"],
            device=train_cfg["device"] or None,
            project=output_cfg["project"],
            name=output_cfg["name"],
        )

        best_path = str(results.save_dir / "weights" / "best.pt")
        logger.info(f"Training complete. Best weights: {best_path}")
        return best_path
