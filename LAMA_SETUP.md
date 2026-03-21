# 🖌️ LaMa Inpainting — Setup Guide

LaMa is a separate research repository and cannot be installed as a regular pip package.
It must be cloned and set up independently from the main project.

---

## ⚠️ Requirements

- Python **3.10.11** specifically (3.11+ and 3.13+ will fail during dependency installation)
- Git
- ~500 MB free disk space for the model weights

---

## Step 1 — Install Python 3.10.11

Download from: https://www.python.org/downloads/release/python-31011/

During installation, make sure to check **"Add Python to PATH"**.

Verify it installed correctly:
```bash
py -0
# Should show -V:3.10 in the list
```

---

## Step 2 — Clone LaMa outside the project

LaMa must be cloned **outside** this project folder to avoid nested git repository issues.

```bash
# Navigate one level up from this project, or to any neutral folder
cd ..

# Clone LaMa
git clone https://github.com/advimman/lama.git

# Your folder structure should look like:
# 📁 some_folder/
# ├── 📁 computer_vision_inpainting/   ← this project
# └── 📁 lama/                         ← lama lives here
```

---

## Step 3 — Download the pretrained weights

1. Go to the LaMa releases page:
   https://disk.yandex.ru/d/ouP6l8VJ0HpMZg (official LaMa weights mirror)
   
   Or find `big-lama.zip` from the LaMa GitHub releases.

2. Download **`big-lama.zip`** (≈364 MB) — ignore the other files.

3. Unzip it and place the `big-lama/` folder inside your lama directory:

```
lama/
├── big-lama/
│   └── models/
│       └── best.ckpt      ← this file is what matters
├── bin/
├── saicinpainting/
└── requirements.txt
```

---

## Step 4 — Create a virtual environment with Python 3.10

Navigate into the lama folder and create the venv:

```bash
cd lama
py -3.10 -m venv venv_lama
venv_lama\Scripts\activate     # Windows
# source venv_lama/bin/activate  # Mac/Linux
```

You should see `(venv_lama)` at the start of your terminal prompt.

---

## Step 5 — Install dependencies

LaMa's `requirements.txt` pins very old package versions that don't build on modern systems.
Install dependencies manually in this exact order:

```bash
# Core scientific stack
pip install numpy scipy scikit-image opencv-python

# PyTorch (CPU version)
pip install torch torchvision

# Fix pkg_resources issue before continuing
pip install setuptools

# LaMa specific dependencies
pip install pyyaml tqdm easydict
pip install scikit-image scikit-learn
pip install kornia==0.5.0
pip install hydra-core==1.1.0
pip install pytorch-lightning==1.2.9
pip install pyyaml tqdm easydict kornia==0.5.0 hydra-core==1.1.0 pytorch-lightning==1.2.9 pandas albumentations joblib h5py matplotlib tensorboard
```

> **Do NOT run `pip install -r requirements.txt` directly** — the pinned versions
> in that file are incompatible with Python 3.10 and will fail to build.

---

## Step 6 — Note your LaMa path

You will need the absolute path to your lama folder when running the inpainting script.
Update `src/inpainting/inpaint.py` with your local path:

```python
# Windows example
LAMA_PATH = r"C:\Users\yourname\lama"

# Mac/Linux example
LAMA_PATH = "/home/yourname/lama"
```

---

## ✅ Verify the setup

With `(venv_lama)` active, run:

```bash
python -c "import torch; import kornia; import yaml; print('LaMa dependencies OK')"
```

You should see `LaMa dependencies OK` with no errors.

---

## 🚀 Running inpainting

Once set up, go back to the main project and run:

```bash
# Make sure you are in the project root, NOT inside lama/
cd ../computer_vision_inpainting

python src/inpainting/inpaint.py \
  --images data/test/images/ \
  --masks results/masks/ \
  --output results/inpainted/
```

The script will use LaMa's venv Python automatically — you do not need to activate
the LaMa venv manually when running from the main project.

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'pkg_resources'` | `pip install setuptools` |
| `Failed to build scikit-image` | Don't use `requirements.txt`, follow Step 5 above |
| `Python 3.10 not found` | Install from python.org, verify with `py -0` |
| `best.ckpt not found` | Check that `big-lama/models/best.ckpt` exists inside your lama folder |
| `venv_lama\Scripts\activate` not recognized | Make sure you are inside the `lama/` folder when running this |
