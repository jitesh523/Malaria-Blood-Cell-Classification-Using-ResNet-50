# 🩸 Malaria Blood Cell Classification (ResNet-50)

Deep learning pipeline that classifies single-cell blood smear images as **Parasitized** or **Uninfected**, using transfer learning on a **ResNet-50** backbone (TensorFlow/Keras). Includes a full training/evaluation CLI, a Jupyter notebook for one-click experimentation, and a **Streamlit app with Grad-CAM explainability** for interactive predictions.

**Latest results (Azure CPU):** Validation Accuracy **~92.6%** · Validation Loss **~0.20**

---

## ✨ Features

- **Transfer learning** on ResNet-50 pretrained on ImageNet, with a custom classification head and staged fine-tuning
- **tf.data** input pipeline with augmentation and ResNet-specific preprocessing
- **Reproducible splits** — stratified train/val/test split generation from raw image folders
- **Evaluation reports** — confusion matrix and metrics saved to disk
- **Grad-CAM visualizations** to explain *why* the model flagged a cell as parasitized
- **Streamlit demo app** for uploading an image and getting an instant prediction + heatmap
- **Notebook mode** — `malaria.ipynb` downloads the NIH dataset and trains end-to-end, no manual setup required

## 🖼️ Demo

Upload a cell image in the Streamlit app to get a label, prediction probability, and a Grad-CAM overlay highlighting the regions that drove the decision.

```
python -m streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
.
├── app/
│   └── streamlit_app.py       # Interactive demo: upload → predict → Grad-CAM overlay
├── notebooks/
│   ├── malaria.ipynb          # End-to-end training (auto-downloads dataset)
│   └── Malaria_ResNet50.ipynb # Exploratory / alternate training notebook
├── src/
│   ├── data_prep.py           # Stratified train/val/test split creation
│   ├── data_prep_nosklearn.py # Split creation without scikit-learn dependency
│   ├── create_manifests.py    # Build dataset manifests
│   ├── datasets_tf.py         # tf.data pipeline: loading, augmentation, preprocessing
│   ├── model_tf.py            # ResNet-50 base + custom classification head
│   ├── train_tf.py            # Training loop: early stopping, LR scheduling, fine-tuning
│   ├── eval_tf.py             # Test-set evaluation: metrics + confusion matrix
│   └── gradcam.py             # Grad-CAM heatmap generation
├── models/                    # Saved model checkpoints (.keras / .h5)
├── reports/                   # Metrics and figures from evaluation
└── requirements.txt
```

## 🚀 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 📦 Dataset

This project uses the [NIH Malaria Cell Images dataset](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html).

**Option A — Notebook (recommended):** Run `notebooks/malaria.ipynb`, which downloads the dataset automatically.

**Option B — CLI:** Download the dataset manually and arrange it as:

```
/path/to/dataset/cell_images/
├── Parasitized/
└── Uninfected/
```

## 🏋️ Training (CLI)

**1. Create train/val/test splits**

```bash
python -m src.data_prep \
  --raw_dir /path/to/dataset/cell_images \
  --out_dir data/splits \
  --val_size 0.15 --test_size 0.15 --seed 42
```

**2. Train the model**

```bash
python -m src.train_tf \
  --data_dir data/splits \
  --out models/best_resnet50.keras \
  --img_size 224 --batch_size 32 \
  --epochs_head 10 --epochs_ft 10 \
  --lr_head 1e-4 --lr_ft 1e-5
```

**3. Evaluate**

```bash
python -m src.eval_tf \
  --model models/best_resnet50.keras \
  --data_dir data/splits \
  --out_dir reports
```

Outputs:
- `reports/metrics.txt`
- `reports/figures/confusion_matrix.png`

> Prefer notebooks? Open and run `notebooks/malaria.ipynb` for the equivalent end-to-end flow.

## 🔍 Explainability (Grad-CAM)

`src/gradcam.py` generates class activation heatmaps over the last convolutional block (`conv5_block3_out`), overlaid on the input image so you can visually inspect what the model is attending to. This is wired directly into the Streamlit app.

## 🧰 Tech Stack

TensorFlow / Keras · ResNet-50 · scikit-learn · OpenCV · Streamlit · Matplotlib / Seaborn


