"""Pallet detection using YOLOv8."""

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection result."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    cropped_image: np.ndarray | None = None


class PalletDetector:
    """Detects pallets in full-frame images using YOLOv8."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "",
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run pallet detection on a single image.

        Args:
            image: BGR numpy array.

        Returns:
            List of Detection instances for detected pallets.
        """
        results = self.model.predict(
            image,
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

                # Crop the detected region
                h, w = image.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                cropped = image[cy1:cy2, cx1:cx2].copy()

                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_name=class_name,
                    cropped_image=cropped,
                ))

        logger.info(f"Detected {len(detections)} pallet(s)")
        return detections

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of images."""
        return [self.detect(img) for img in images]

    @staticmethod
    def train(data_yaml: str, config: dict) -> str:
        """Train a new pallet detection model.

        Args:
            data_yaml: Path to YOLO dataset yaml.
            config: Training config dict (from configs/pallet_detection.yaml).

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
