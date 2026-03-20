"""
src/inpainting/mask_generator.py
Generate binary masks from YOLO pole detections.

Each mask is a black image (same size as the source image)
with white pixels where poles were detected — ready for inpainting.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


POLE_CLASS_ID = 1  # 0: fachada, 1: poste


def generate_mask(image_path: Path, boxes, dilation_px: int = 10) -> np.ndarray:
    """Create a binary mask from bounding boxes of detected poles."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        cls = int(box.cls[0])
        if cls != POLE_CLASS_ID:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        mask[y1:y2, x1:x2] = 255

    # Optional: dilate the mask slightly to cover pole edges
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Generate inpainting masks from detections")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Folder with images")
    parser.add_argument("--output", type=str, default="results/masks/")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--dilation", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(args.source).glob("*.jpg")) + \
                  list(Path(args.source).glob("*.png")) + \
                  list(Path(args.source).glob("*.jpeg"))

    for img_path in image_paths:
        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        boxes = results[0].boxes
        mask = generate_mask(img_path, boxes, dilation_px=args.dilation)
        mask_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"Saved mask: {mask_path}")

    print(f"\nAll masks saved to: {args.output}")


if __name__ == "__main__":
    main()
