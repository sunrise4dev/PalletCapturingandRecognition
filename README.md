# Pallet Capturing and Recognition

A Python ML pipeline for detecting pallets in images, reading stickers and barcodes, and extracting structured data using computer vision and deep learning.

## Architecture

```
Image → Pallet Detection (YOLOv8) → Alignment → Sticker/Barcode Detection (YOLOv8) → OCR + Barcode → JSON Output
```

## Features

- **Pallet Detection** — YOLOv8-based object detection for locating pallets in full-frame images
- **Perspective Alignment** — Automatic correction of camera angle distortion
- **Sticker & Barcode Detection** — Second-stage detection for labels and barcodes on pallets
- **OCR** — Tesseract-based text extraction with adaptive preprocessing
- **Barcode Reading** — Multi-strategy barcode/QR decoding via pyzbar
- **End-to-End Pipeline** — Single command to process images and get structured JSON output

## Project Structure

```
├── configs/              # Training and pipeline configuration
├── data/                 # Dataset directories and YOLO configs
├── models/               # Trained model weights
├── scripts/              # CLI entry points
│   ├── prepare_data.py   # Data conversion, splitting, augmentation
│   ├── train_pallet_detector.py
│   ├── train_sticker_detector.py
│   ├── run_inference.py  # Full pipeline inference
│   ├── evaluate.py       # Model evaluation
│   └── export_model.py   # Export to ONNX/TFLite/TFJS
├── src/                  # Source modules
│   ├── data_preparation/ # Annotation conversion, dataset splitting
│   ├── detection/        # YOLOv8 wrappers and alignment
│   ├── ocr/              # Tesseract OCR with preprocessing
│   ├── barcode/          # pyzbar barcode reading
│   ├── pipeline/         # End-to-end pipeline
│   └── utils/            # Config loading, image I/O, visualization
└── tests/                # Unit tests
```

## Installation

### System Dependencies

```bash
# macOS
brew install tesseract zbar

# Ubuntu/Debian
sudo apt install tesseract-ocr libzbar0
```

### Python Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare Data

```bash
# Convert annotations (Pascal VOC example)
python scripts/prepare_data.py convert --format voc --input data/raw/annotations --output data/raw/yolo_labels --classes pallet

# Split into train/val/test
python scripts/prepare_data.py split --images data/raw/images --labels data/raw/yolo_labels --output data/processed/pallet_detection

# Augment training data (optional, recommended for small datasets)
python scripts/prepare_data.py augment --images data/processed/pallet_detection/train/images --labels data/processed/pallet_detection/train/labels --output data/augmented
```

### 2. Train Models

```bash
# Train pallet detector
python scripts/train_pallet_detector.py --config configs/pallet_detection.yaml

# Train sticker detector
python scripts/train_sticker_detector.py --config configs/sticker_detection.yaml
```

### 3. Run Inference

```bash
# Single image
python scripts/run_inference.py --config configs/pipeline.yaml --image path/to/image.jpg --visualize

# Directory of images
python scripts/run_inference.py --config configs/pipeline.yaml --directory path/to/images/
```

### 4. Evaluate

```bash
python scripts/evaluate.py --model models/pallet_detection/best.pt --data data/pallet_detection.yaml --save-plots
```

### 5. Export for Mobile

```bash
python scripts/export_model.py --model models/pallet_detection/best.pt --format onnx
python scripts/export_model.py --model models/pallet_detection/best.pt --format tflite
```

## Configuration

All parameters are controlled via YAML config files in `configs/`:

- `pallet_detection.yaml` — Pallet detector training hyperparameters
- `sticker_detection.yaml` — Sticker detector training hyperparameters
- `pipeline.yaml` — Full inference pipeline settings (thresholds, OCR params, etc.)

## Requirements

- Python 3.9+
- CUDA GPU recommended for training (CPU works for inference)
- ~100+ annotated pallet images for training

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Donate

If you find this project useful, consider supporting its development:

| Network | Address |
|---------|---------|
| **USDT / TRC20** | `TKkMqRFpoR64WveWaKQahQQ9EeAFz2AgMn` |
| **USDT / BEP20** | `0x1484f7e7e3a240c660d427a3ac253a8d29cbc8bf` |
| **USDT / SOL** | `FBPMCgkqTX8maefy2GyXBJh4qrtsXXpKZ3TVkxcjAL1A` |
| **USDT / ERC20** | `0x1484f7e7e3a240c660d427a3ac253a8d29cbc8bf` |
| **TAO / TAO** | `5FvNqfanTRk38TzMh56PBkcpkwWyXHbhijkXm9AVevQ8nmXo` |
