# Malaria Blood Cell Classification (ResNet-50, TensorFlow/Keras)

End-to-end pipeline to classify segmented blood cell images as Parasitized vs Uninfected using transfer learning with ResNet-50.

**Recent Results (Azure CPU):** Validation Accuracy ~92.6% | Validation Loss ~0.20

## Project structure
- `malaria.ipynb`: **(New)** Standalone Jupyter Notebook for end-to-end training (downloads data automatically).
- src/
  - data_prep.py: Create stratified train/val/test splits from raw folders.
  - datasets_tf.py: tf.data pipeline with augmentation and ResNet preprocessing.
  - model_tf.py: ResNet-50 base + custom head.
  - train_tf.py: Training with early stopping, LR scheduling, staged fine-tuning.
  - eval_tf.py: Evaluation on test set with metrics and confusion matrix plot.
- models/: saved models
- reports/: metrics and figures
- requirements.txt

## Setup

python3 -m venv .venv source .venv/bin/activate pip install -r requirements.txt


## Dataset
**Option A (Notebook):** Run `malaria.ipynb` to automatically download the dataset from NIH.

**Option B (CLI):** Place the NIH Malaria dataset manually with the following structure:

/path/to/dataset/cell_images/ Parasitized/ Uninfected/


## 1) Create splits (CLI only)

python -m src.data_prep --raw_dir /path/to/dataset/cell_images

--out_dir data/splits --val_size 0.15 --test_size 0.15 --seed 42


## 2) Train
**Notebook:** Open and run `malaria.ipynb`.

**CLI:**

python -m src.train_tf --data_dir data/splits --out models/best_resnet50.keras

--img_size 224 --batch_size 32 --epochs_head 10 --epochs_ft 10 --lr_head 1e-4 --lr_ft 1e-5

## 3) Evaluate

python -m src.eval_tf --model models/best_resnet50.keras --data_dir data/splits --out_dir reports

Outputs:

- reports/figures/confusion_matrix.png
- reports/metrics.txt
