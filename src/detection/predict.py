"""
src/detection/predict.py
Run inference with the trained YOLOv8 model and save annotated results.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Image folder or file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save-results", type=str, default="results/detections/")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=True,
        project=str(Path(args.save_results).parent),
        name=str(Path(args.save_results).name),
    )

    print(f"\nPredictions saved to: {args.save_results}")
    return results


if __name__ == "__main__":
    main()
