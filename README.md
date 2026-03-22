# 🏙️ Detección de fachadas y postes con Inpainting

> **Computer Vision pipeline** para la detección de fachadas de edificios y postes de servicios públicos en imágenes urbanas, seguida de la eliminación automatizada de postes mediante modelos de relleno de aprendizaje profundo.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Descripción general

Este proyecto implementa un proceso de visión artificial de dos etapas:

1. **Detección** — Modelo YOLOv8 ajustado para detectar las clases `fachada` (fachadas) y `poste` (postes) en imágenes urbanas
2. **Inpainting** — Los postes detectados se enmascaran y se eliminan de las imágenes mediante un modelo de relleno de aprendizaje profundo, reconstruyendo el fondo que hay detrás de ellos.



El conjunto de datos fue construido y etiquetado utilizando [Roboflow](https://roboflow.com), que contiene **170 images** (con aumento aplicado) Exportado en formato YOLOv8.

---

## 🗂️ Project Structure

```
facade-pole-detection/
│
├── data/                          # Conjunto de datos (no cargado en Git)
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
│   │   └── predict.py             # Ejecutar inferencia en imágenes nuevas
│   │
│   ├── inpainting/
│   │   ├── mask_generator.py      # Generar máscaras a partir de detecciones de YOLO
│   │   └── inpaint.py             # Aplicar el modelo de relleno a imágenes enmascaradas
│   │
│   └── utils/
│       ├── visualize.py           # Dibujar cuadros delimitadores y máscaras
│       └── io_utils.py            # Ayudantes de entrada/salida de archivos e imágenes
│
├── scripts/
│   ├── run_pipeline.py            # End-to-end pipeline script
│   └── batch_process.py           # Inferencia por lotes en una carpeta de imágenes
│
├── notebooks/
│   ├── 01_eda.ipynb               # Análisis de datos exploratorios.
│   ├── 02_training.ipynb          # Training walkthrough
│   └── 03_results_analysis.ipynb  # Visualización de resultados y métricas
│
├── configs/
│   └── data.yaml                  # Configuración del conjunto de datos para YOLOv8
│
├── results/
│   ├── detections/                # Imágenes con cuadros delimitadores
│   ├── masks/                     # Máscaras binarias de polos detectados
│   └── inpainted/                 # Imágenes finales con los postes retirados.
│
├── docs/
│   └── pipeline_diagram.png       # Descripción general visual del proceso
│
├── requirements.txt
└── README.md
```

---

## 🚀 Empezando

### 1. Clone el repositorio

```bash
git clone https://github.com/<your-username>/facade-pole-detection.git
cd facade-pole-detection
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar el conjunto de datos

Descargar el conjunto de datos de Roboflow y coloca las carpetas `train/`, `valid/` y `test/` dentro de un directorio `data/` en la raíz del proyecto. Tu archivo `configs/data.yaml` ya debería apuntar a estas rutas.


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

### Paso 1: Entrenar el modelo de detección.

Ajuste fino de YOLOv8 en nuestro conjunto de datos etiquetados:

```bash
python src/detection/train.py \
  --data configs/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640
```

### Paso 2 — Evaluar el modelo

```bash
python src/detection/evaluate.py \
  --weights runs/detect/train/weights/best.pt \
  --data configs/data.yaml
```

Métricas reportadas: **mAP@0.5**, **Precision**, **Recall**

### Paso 3: Ejecutar inferencia

```bash
python src/detection/predict.py \
  --weights runs/detect/train/weights/best.pt \
  --source data/test/images/ \
  --save-results results/detections/
```

### Paso 4 — Generar máscaras de postes

```bash
python src/inpainting/mask_generator.py \
  --predictions results/detections/ \
  --output results/masks/
```

### Paso 5: Eliminar postes mediante relleno.

```bash
python src/inpainting/inpaint.py \
  --images data/test/images/ \
  --masks results/masks/ \
  --output results/inpainted/
```

### Pipeline completo (comando único)

```bash
python scripts/run_pipeline.py \
  --input <path_to_images> \
  --weights runs/detect/train/weights/best.pt \
  --output results/
```

---

## 📊 Conjunto de datos

| Split      | Images |
|------------|--------|
| Train      | ~136   |
| Validation | ~17    |
| Test       | ~17    |
| **Total**  | **170** |

**Preprocesamiento aplicado por Roboflow:**
- Autoorientación (eliminación de datos EXIF)
- Cambiar tamaño a 640×640 (estirar)

**Aumentos aplicados (2 veces por imagen):**
- 50% de probabilidad de giro horizontal
- Brillo aleatorio: ±15%
- RExposición aleatoria: ±10%

**Clases:** `fachada` (Fachadas), `poste` (postes)

**Formato de anotación:** Cuadros delimitadores YOLOv8

---

## 🎨Repintado de modelos

Este proyecto admite los siguientes backends de relleno de imágenes:

| Modelo | Repositorio | Notas |
|-------|-----------|-------|
| **LaMa** | [advimman/lama](https://github.com/advimman/lama) | Best quality, recommended |
| **Inpaint-Anything** | [geekyutao/Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) | SAM-based, flexible |

---

## 📦 Requisitos

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

## 🎬 Vídeo de demostración

📺 [Ver en youtube](<youtube-link-here>)

---

## 👥 Team

Desarrollado como parte de un programa de maestría en Aprendizaje Automático y Aprendizaje Profundo. 