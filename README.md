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

## 1) Create splits (CLI only)

## 2) Train
**Notebook:** Open and run `malaria.ipynb`.

**CLI:**
