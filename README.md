# Malaria Blood Cell Classification (ResNet-50, TensorFlow/Keras)

End-to-end pipeline to classify segmented blood cell images as Parasitized vs Uninfected using transfer learning with ResNet-50.

## Project structure
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
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
Place or point to the NIH Malaria dataset with the following structure:
```
/Users/jitesh/Downloads/cell_images/
  Parasitized/
  Uninfected/
```

## 1) Create splits
```
python -m src.data_prep --raw_dir /Users/jitesh/Downloads/cell_images \
  --out_dir data/splits --val_size 0.15 --test_size 0.15 --seed 42
```

## 2) Train
```
python -m src.train_tf --data_dir data/splits --out models/best_resnet50.h5 \
  --img_size 224 --batch_size 32 --epochs_head 10 --epochs_ft 10 --lr_head 1e-4 --lr_ft 1e-5
```

## 3) Evaluate
```
python -m src.eval_tf --model models/best_resnet50.h5 --data_dir data/splits --out_dir reports
```

Outputs:
- reports/figures/confusion_matrix.png
- reports/metrics.txt
