# 🏙️ Facade & Pole Detection with Inpainting

> **Computer Vision pipeline** for detecting building facades and utility poles in urban imagery, followed by automated pole removal using deep learning inpainting models.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project implements a two-stage computer vision pipeline:

1. **Detection** — Fine-tuned YOLOv8 model trained to detect `fachada` (facade) and `poste` (pole) classes in urban images.
2. **Inpainting** — Detected poles are masked and removed from images using a deep learning inpainting model, reconstructing the background behind them.

The dataset was built and labeled using [Roboflow](https://roboflow.com), containing **170 images** (with augmentation applied) exported in YOLOv8 format.

---

## 🗂️ Project Structure

```
facade-pole-detection/
│
├── data/                          # Dataset (not tracked by Git)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── src/
│   ├── detection/
│   │   ├── train.py               # YOLOv8 fine-tuning script
│   │   ├── evaluate.py            # Evaluation: mAP, precision, recall
│   │   └── predict.py             # Run inference on new images
│   │
│   ├── inpainting/
│   │   ├── mask_generator.py      # Generate masks from YOLO detections
│   │   └── inpaint.py             # Apply inpainting model to masked images
│   │
│   └── utils/
│       ├── visualize.py           # Draw bounding boxes and masks
│       └── io_utils.py            # File and image I/O helpers
│
├── scripts/
│   ├── run_pipeline.py            # End-to-end pipeline script
│   └── batch_process.py           # Batch inference on a folder of images
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_training.ipynb          # Training walkthrough
│   └── 03_results_analysis.ipynb  # Results and metric visualization
│
├── configs/
│   └── data.yaml                  # Dataset config for YOLOv8
│
├── results/
│   ├── detections/                # Images with bounding boxes
│   ├── masks/                     # Binary masks of detected poles
│   └── inpainted/                 # Final images with poles removed
│
├── docs/
│   └── pipeline_diagram.png       # Visual overview of the pipeline
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/facade-pole-detection.git
cd facade-pole-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the dataset

Download your Roboflow dataset and place the `train/`, `valid/`, and `test/` folders inside a `data/` directory at the root of the project. Your `configs/data.yaml` should already point to these paths.

```yaml
# configs/data.yaml
path: ../data
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['fachada', 'poste']
```

---

## 🧠 Pipeline

### Step 1 — Train the Detection Model

Fine-tune YOLOv8 on our labeled dataset:

```bash
python src/detection/train.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640
```

### Step 2 — Evaluate the Model

```bash
python src/detection/evaluate.py \
  --weights runs/detect/train/weights/best.pt \
  --data configs/data.yaml
```

Metrics reported: **mAP@0.5**, **Precision**, **Recall**

### Step 3 — Run Inference

```bash
python src/detection/predict.py \
  --weights runs/detect/train/weights/best.pt \
  --source data/test/images/ \
  --save-results results/detections/
```

### Step 4 — Generate Pole Masks

```bash
python src/inpainting/mask_generator.py \
  --predictions results/detections/ \
  --output results/masks/
```

### Step 5 — Remove Poles via Inpainting

```bash
python src/inpainting/inpaint.py \
  --images data/test/images/ \
  --masks results/masks/ \
  --output results/inpainted/
```

### Full Pipeline (Single Command)

```bash
python scripts/run_pipeline.py \
  --input <path_to_images> \
  --weights runs/detect/train/weights/best.pt \
  --output results/
```

---

## 📊 Dataset

| Split      | Images |
|------------|--------|
| Train      | ~136   |
| Validation | ~17    |
| Test       | ~17    |
| **Total**  | **170** |

**Preprocessing applied by Roboflow:**
- Auto-orientation (EXIF stripping)
- Resize to 640×640 (stretch)

**Augmentations applied (2× per image):**
- 50% probability horizontal flip
- Random brightness: ±15%
- Random exposure: ±10%

**Classes:** `fachada` (facade), `poste` (utility pole)

**Annotation format:** YOLOv8 bounding boxes

---

## 🎨 Inpainting Models

This project supports the following inpainting backends:

| Model | Repository | Notes |
|-------|-----------|-------|
| **LaMa** | [advimman/lama](https://github.com/advimman/lama) | Best quality, recommended |
| **Inpaint-Anything** | [geekyutao/Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) | SAM-based, flexible |

---

## 📦 Requirements

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 🎬 Demo Video

📺 [Watch on YouTube](<youtube-link-here>)

---

## 👥 Team

Developed as part of a Master's program in Machine Learning & Deep Learning.

---

## 📄 License

This project is licensed under the MIT License.
