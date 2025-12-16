# Efficient Gesture Recognition (Assignment 3)

Leiden University — 4032COVIX: Computer Vision (Autumn 2025)

Authors: Yotam Lev, Piotr Perski

This repository contains the code and report for Assignment 3: Gesture Recognition on the Jester dataset. We implement two models:

- `baseline`: a 2D ResNet18 that processes a single frame.
- `tsm`: a ResNet18 augmented with Temporal Shift Modules (TSM) for efficient temporal modeling.

Results (reported):
- Training accuracy (best): 92%
- Validation accuracy (best): 85%

---

## Repo layout (relevant files)

- `train.py`            — training script
- `evaluate.py`         — evaluation script (load checkpoint + run on validation set)
- `test_loader.py`      — small dataset sanity-check script
- `models/`             — `baseline.py`, `tsm.py`, `ops.py`
- `data/dataset.py`     — `JesterDataset` implementation
- `report.tex`          — LaTeX report (CVPR-style)

---

## Requirements

Recommended: use a virtual environment (venv) or conda.

Python (3.8+)
PyTorch and torchvision matching your CUDA / CPU setup.

Example using `venv` (macOS / Linux):

```bash
cd "/Users/piotrek/Desktop/development/computer vision 3/ComputerVision_Assignment3"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you run into `externally-managed-environment` on macOS, create and use a virtualenv or conda env as shown above.

Conda example (useful for lab GPUs):

```bash
conda create --name pytorch python=3.9 -y
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## How to compile the report (`report.tex`) locally

Option A — latexmk (recommended if you have TeX Live / MacTeX installed):

```bash
# from repo root
cd "/Users/piotrek/Desktop/development/computer vision 3/ComputerVision_Assignment3"
latexmk -pdf report.tex
# output: report.pdf
```

Option B — pdflatex + bibtex (manual sequence):

```bash
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

If you don't have a local TeX installation, use Overleaf: create a new project, upload `report.tex` and any figures, then compile there.

Note: If you use the CVPR style package provided in the repo or a local TeX installation, ensure all style files are available (MacTeX / TeX Live include them). On macOS, `brew install --cask mactex` or download MacTeX from tug.org.

---

## How to run evaluation on a checkpoint

From the repo root, example commands:

```bash
# Baseline
python3 evaluate.py --model baseline --checkpoint ./baseline_best.pth --batch_size 8

# TSM (v2)
python3 evaluate.py --model tsm --checkpoint ./tsm_v2_best.pth --batch_size 4

# TSM (original)
python3 evaluate.py --model tsm --checkpoint ./tsm_best.pth --batch_size 4
```

Optional flags:
- `--csv_val` : path to validation CSV (default `./datasets/jester-v1-validation.csv`)
- `--image_root` : root folder with frames (default `./small-20bn-jester-v1`)
- `--labels` : labels CSV (default `./datasets/jester-v1-labels.csv`)

If the dataset path is different on your machine, pass `--image_root /path/to/frames`.

---

## Dataset

Download the full Jester dataset here:
https://www.qualcomm.com/developer/software/jester-dataset/downloads

For quick experiments the repo expects the small 20% subset structure (default `./small-20bn-jester-v1`). Ensure `datasets/*.csv` files are present.

---


