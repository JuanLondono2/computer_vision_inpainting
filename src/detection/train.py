"""
src/detection/train.py
Fine-tuning YOLOv8 on the facade & pole dataset.
"""

import argparse
from ultralytics import YOLO
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on facade/pole dataset")
    parser.add_argument("--data", type=str, default="configs/data.yaml") #Apunta a los datos usando el archivo data.yaml
    parser.add_argument("--model", type=str, default="yolov8n.pt") #De los modelos precargados usa nano Yolov8
    parser.add_argument("--epochs", type=int, default=100) #Número de épocas del entrenamiento 
    parser.add_argument("--imgsz", type=int, default=640) # Se ajusta con base en lo que se exporta de Roboflow
    parser.add_argument("--batch", type=int, default=16) # Número de imágenes que ve al mismo tiempo
    parser.add_argument("--device", type=str, default=None,  # None = auto-detecta si se usa la cpu o si hay disponible gpu
                        help="cuda device (e.g. '0') or 'cpu'. Auto-detected if not set.")
    parser.add_argument("--project", type=str, default="runs/detect") # Donde se guarda
    parser.add_argument("--name", type=str, default="train")
    return parser.parse_args()

def get_device(device_arg):
    """Auto-detect the best available device if none is specified."""
    if device_arg is not None:
        return device_arg  # user explicitly chose, respect it
    if torch.cuda.is_available():
        print("GPU detected — training on CUDA.")
        return "0"
    else:
        print("No GPU detected — training on CPU.")
        return "cpu"

def main():
    args = parse_args()
    device = get_device(args.device)
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device= device,
        project=args.project,
        name=args.name,
        patience=20,          # Early stopping
        save=True,
        plots=True,
    )

    print(f"\nTraining complete. Best weights saved to: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
