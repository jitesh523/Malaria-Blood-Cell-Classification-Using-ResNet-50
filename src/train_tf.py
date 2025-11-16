import os
import argparse
from pathlib import Path
from datetime import datetime
import csv

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.datasets_tf import get_datasets, get_datasets_from_manifests
from src.model_tf import build_resnet50, build_efficientnetb0


def _build_model(backbone: str, img_size: int):
    input_shape = (img_size, img_size, 3)
    if backbone == "resnet50":
        return build_resnet50(input_shape=input_shape)
    elif backbone == "efficientnetb0":
        return build_efficientnetb0(input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def _log_experiment(row: dict, experiments_path: Path):
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = experiments_path.exists()
    fieldnames = [
        "timestamp",
        "backbone",
        "img_size",
        "batch_size",
        "epochs_head",
        "epochs_ft",
        "lr_head",
        "lr_ft",
        "seed",
        "val_accuracy",
        "val_auc",
        "test_accuracy",
        "test_auc",
        "notes",
    ]

    with open(experiments_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(data_dir: str, out_model: str, img_size: int = 224, batch_size: int = 32,
          epochs_head: int = 10, epochs_ft: int = 10, lr_head: float = 1e-4, lr_ft: float = 1e-5,
          patience: int = 4, manifests: bool = False, backbone: str = "resnet50",
          seed: int = 42, notes: str = ""):
    if manifests:
        train_ds, val_ds, test_ds, class_names = get_datasets_from_manifests(
            data_dir, img_size=img_size, batch_size=batch_size, seed=seed
        )
    else:
        train_ds, val_ds, test_ds, class_names = get_datasets(
            data_dir, img_size=img_size, batch_size=batch_size, seed=seed
        )

    model, base = _build_model(backbone, img_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_head),
                  loss='binary_crossentropy',
                  metrics=[
                      'accuracy',
                      tf.keras.metrics.AUC(name='auc')
                  ])

    out_model_path = Path(out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=max(1, patience-1), min_lr=1e-6),
        ModelCheckpoint(filepath=str(out_model_path), monitor='val_auc', mode='max', save_best_only=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, callbacks=callbacks)

    # Fine-tune: unfreeze upper blocks depending on backbone
    base.trainable = True
    trainable = False
    if backbone == "resnet50":
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) and 'conv5' in layer.name:
                trainable = True
            layer.trainable = trainable
    elif backbone == "efficientnetb0":
        # Unfreeze from block6 onwards for EfficientNetB0
        for layer in base.layers:
            name = layer.name
            if "block6" in name:
                trainable = True
            layer.trainable = trainable

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_ft),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft, callbacks=callbacks)

    # Ensure final best is saved
    model.save(out_model_path)

    # Evaluate on validation quickly
    val_metrics = model.evaluate(val_ds, return_dict=True)
    print('Validation metrics:', val_metrics)

    # Evaluate on test set
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print('Test metrics:', test_metrics)

    # Log experiment
    experiments_path = Path("reports") / "experiments.csv"
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "backbone": backbone,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs_head": epochs_head,
        "epochs_ft": epochs_ft,
        "lr_head": lr_head,
        "lr_ft": lr_ft,
        "seed": seed,
        "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
        "val_auc": float(val_metrics.get("auc", float("nan"))),
        "test_accuracy": float(test_metrics.get("accuracy", float("nan"))),
        "test_auc": float(test_metrics.get("auc", float("nan"))),
        "notes": notes,
    }
    _log_experiment(row, experiments_path)


def main():
    parser = argparse.ArgumentParser(description='Train CNN backbones on Malaria cell images (TF/Keras)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with train/val/test OR manifests folder')
    parser.add_argument('--manifests', action='store_true', help='Treat data_dir as a manifests directory (train.csv, val.csv, test.csv)')
    parser.add_argument('--out', type=str, default='models/best_resnet50.h5', help='Path to save best model')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_head', type=int, default=10)
    parser.add_argument('--epochs_ft', type=int, default=10)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_ft', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnetb0'],
                        help='Backbone architecture to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling/splits')
    parser.add_argument('--notes', type=str, default='', help='Free-form notes to store with the experiment log')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        out_model=args.out,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs_head=args.epochs_head,
        epochs_ft=args.epochs_ft,
        lr_head=args.lr_head,
        lr_ft=args.lr_ft,
        patience=args.patience,
        manifests=args.manifests,
        backbone=args.backbone,
        seed=args.seed,
        notes=args.notes,
    )


if __name__ == '__main__':
    main()
