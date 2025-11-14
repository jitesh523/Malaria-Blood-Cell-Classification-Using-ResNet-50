import os
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.datasets_tf import get_datasets
from src.model_tf import build_resnet50


def train(data_dir: str, out_model: str, img_size: int = 224, batch_size: int = 32,
          epochs_head: int = 10, epochs_ft: int = 10, lr_head: float = 1e-4, lr_ft: float = 1e-5,
          patience: int = 4):
    train_ds, val_ds, test_ds, class_names = get_datasets(data_dir, img_size=img_size, batch_size=batch_size)

    model, base = build_resnet50(input_shape=(img_size, img_size, 3))

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

    # Fine-tune: unfreeze last convolutional block
    base.trainable = True
    trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and 'conv5' in layer.name:
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


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on Malaria cell images (TF/Keras)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with train/val/test subfolders')
    parser.add_argument('--out', type=str, default='models/best_resnet50.h5', help='Path to save best model')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_head', type=int, default=10)
    parser.add_argument('--epochs_ft', type=int, default=10)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_ft', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=4)
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
    )


if __name__ == '__main__':
    main()
