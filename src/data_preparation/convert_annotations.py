"""Convert annotation formats (Pascal VOC, COCO, LabelMe) to YOLO format."""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def convert_voc_to_yolo(xml_path: str, class_map: dict[str, int]) -> list[str]:
    """Convert a Pascal VOC XML annotation to YOLO format lines.

    Args:
        xml_path: Path to the VOC XML file.
        class_map: Mapping of class names to class indices.

    Returns:
        List of YOLO-format lines: "class_id x_center y_center width height"
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in class_map:
            logger.warning(f"Unknown class '{name}' in {xml_path}, skipping")
            continue

        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)

        # Convert to YOLO normalized format
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        if width <= 0 or height <= 0:
            logger.warning(f"Zero-area bbox in {xml_path}, skipping")
            continue

        class_id = class_map[name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    class_map: dict[str, int] | None = None,
):
    """Convert COCO JSON annotations to YOLO format txt files.

    Args:
        coco_json_path: Path to COCO format JSON file.
        output_dir: Directory to write per-image .txt files.
        class_map: Optional class name to index mapping. If None, uses COCO category IDs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image ID to filename mapping
    images = {img["id"]: img for img in coco["images"]}

    # Build category mapping
    if class_map is None:
        categories = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    else:
        cat_name_to_id = {cat["name"].lower(): cat["id"] for cat in coco["categories"]}
        categories = {}
        for name, idx in class_map.items():
            if name in cat_name_to_id:
                categories[cat_name_to_id[name]] = idx

    # Group annotations by image
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    for img_id, img_info in images.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        filename = Path(img_info["file_name"]).stem

        lines = []
        for ann in img_annotations.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in categories:
                continue

            x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            if norm_w <= 0 or norm_h <= 0:
                continue

            class_id = categories[cat_id]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = output_dir / f"{filename}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    logger.info(f"Converted {len(images)} images from COCO to YOLO format")


def convert_labelme_to_yolo(json_path: str, class_map: dict[str, int]) -> list[str]:
    """Convert a LabelMe JSON annotation to YOLO format lines.

    Args:
        json_path: Path to LabelMe JSON file.
        class_map: Mapping of class names to class indices.

    Returns:
        List of YOLO-format lines.
    """
    with open(json_path) as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    lines = []
    for shape in data["shapes"]:
        name = shape["label"].strip().lower()
        if name not in class_map:
            logger.warning(f"Unknown class '{name}' in {json_path}, skipping")
            continue

        if shape["shape_type"] != "rectangle":
            logger.warning(f"Non-rectangle shape in {json_path}, skipping")
            continue

        points = shape["points"]
        x1 = min(p[0] for p in points)
        y1 = min(p[1] for p in points)
        x2 = max(p[0] for p in points)
        y2 = max(p[1] for p in points)

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        if width <= 0 or height <= 0:
            continue

        class_id = class_map[name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def convert_to_yolo_format(
    input_dir: str,
    output_dir: str,
    format: str,
    class_map: dict[str, int],
    images_dir: str | None = None,
):
    """Convert annotations from a given format to YOLO format.

    Args:
        input_dir: Directory containing annotation files.
        output_dir: Directory to write YOLO .txt files.
        format: One of "voc", "coco", "labelme".
        class_map: Mapping of class names to class indices.
        images_dir: Optional path to images (used for VOC/LabelMe to verify dimensions).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "coco":
        # COCO is a single JSON file
        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {input_dir}")
        convert_coco_to_yolo(str(json_files[0]), str(output_dir), class_map)
        return

    converter = {"voc": _convert_voc_file, "labelme": _convert_labelme_file}
    if format not in converter:
        raise ValueError(f"Unsupported format: {format}. Use 'voc', 'coco', or 'labelme'.")

    extensions = {"voc": "*.xml", "labelme": "*.json"}
    files = list(input_dir.glob(extensions[format]))
    logger.info(f"Found {len(files)} annotation files in {format} format")

    converted = 0
    for ann_file in files:
        lines = converter[format](ann_file, class_map)
        if lines:
            label_path = output_dir / f"{ann_file.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(lines))
            converted += 1

    logger.info(f"Converted {converted}/{len(files)} annotation files to YOLO format")


def _convert_voc_file(xml_path: Path, class_map: dict[str, int]) -> list[str]:
    return convert_voc_to_yolo(str(xml_path), class_map)


def _convert_labelme_file(json_path: Path, class_map: dict[str, int]) -> list[str]:
    return convert_labelme_to_yolo(str(json_path), class_map)
