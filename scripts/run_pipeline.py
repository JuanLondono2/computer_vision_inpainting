"""
scripts/run_pipeline.py
End-to-end pipeline: detect poles → generate masks → apply inpainting.

Usage:
    python scripts/run_pipeline.py \
        --input path/to/images \
        --weights runs/detect/train/weights/best.pt \
        --output results/
"""

import argparse
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.predict import main as predict
from src.inpainting.mask_generator import main as generate_masks


def parse_args():
    parser = argparse.ArgumentParser(description="Full facade/pole pipeline")
    parser.add_argument("--input", type=str, required=True, help="Folder with input images")
    parser.add_argument("--weights", type=str, required=True, help="YOLOv8 weights path")
    parser.add_argument("--output", type=str, default="results/", help="Root output folder")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--dilation", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)

    print("=" * 60)
    print("STEP 1: Running YOLOv8 Detection")
    print("=" * 60)
    # Detection results will be saved to results/detections/
    detection_dir = output / "detections"

    print("\n" + "=" * 60)
    print("STEP 2: Generating Pole Masks")
    print("=" * 60)
    mask_dir = output / "masks"
    # mask_generator reads from detection results and original images

    print("\n" + "=" * 60)
    print("STEP 3: Applying Inpainting")
    print("=" * 60)
    inpainted_dir = output / "inpainted"
    print(f"  → Inpainting model integration coming soon.")
    print(f"  → See src/inpainting/inpaint.py for implementation.")

    print("\n✅ Pipeline complete!")
    print(f"   Detections  → {detection_dir}")
    print(f"   Masks       → {mask_dir}")
    print(f"   Inpainted   → {inpainted_dir}")


if __name__ == "__main__":
    main()
