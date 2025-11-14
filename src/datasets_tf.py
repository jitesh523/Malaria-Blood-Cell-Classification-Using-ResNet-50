from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
import csv
import numpy as np


def _augment():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


def _prep_layer():
    return layers.Lambda(preprocess_input)


def _build_pipeline(ds, training):
    autotune = tf.data.AUTOTUNE
    if training:
        ds = ds.shuffle(1000)
        ds = ds.map(lambda x, y: (_augment()(x), y), num_parallel_calls=autotune)
    ds = ds.map(lambda x, y: (_prep_layer()(x), y), num_parallel_calls=autotune)
    ds = ds.prefetch(autotune)
    return ds


def get_datasets(data_dir: str, img_size: int = 224, batch_size: int = 32, seed: int = 42):
    data_root = Path(data_dir)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    image_size = (img_size, img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode="binary", image_size=image_size, batch_size=batch_size, seed=seed
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, label_mode="binary", image_size=image_size, batch_size=batch_size, seed=seed
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, label_mode="binary", image_size=image_size, batch_size=batch_size, shuffle=False
    )

    class_names = train_ds.class_names

    train_ds = _build_pipeline(train_ds, training=True)
    val_ds = _build_pipeline(val_ds, training=False)
    test_ds = _build_pipeline(test_ds, training=False)

    return train_ds, val_ds, test_ds, class_names


def _read_manifest(csv_path: Path):
    paths = []
    labels = []
    with open(csv_path, 'r') as f:
        r = csv.reader(f)
        header = next(f)
        f.seek(0)
        r = csv.DictReader(f)
        for row in r:
            paths.append(row['path'])
            labels.append(int(row['label']))
    return np.array(paths), np.array(labels, dtype=np.int32)


def _make_ds_from_manifest(paths, labels, img_size, batch_size, shuffle, seed=None):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img, tf.cast(tf.reshape(label, [1]), tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds


def get_datasets_from_manifests(manifest_dir: str, img_size: int = 224, batch_size: int = 32, seed: int = 42):
    mroot = Path(manifest_dir)
    train_csv = mroot / 'train.csv'
    val_csv = mroot / 'val.csv'
    test_csv = mroot / 'test.csv'

    train_paths, train_labels = _read_manifest(train_csv)
    val_paths, val_labels = _read_manifest(val_csv)
    test_paths, test_labels = _read_manifest(test_csv)

    train_ds = _make_ds_from_manifest(train_paths, train_labels, img_size, batch_size, shuffle=True, seed=seed)
    val_ds = _make_ds_from_manifest(val_paths, val_labels, img_size, batch_size, shuffle=False)
    test_ds = _make_ds_from_manifest(test_paths, test_labels, img_size, batch_size, shuffle=False)

    class_names = ["Parasitized", "Uninfected"]

    train_ds = _build_pipeline(train_ds, training=True)
    val_ds = _build_pipeline(val_ds, training=False)
    test_ds = _build_pipeline(test_ds, training=False)

    return train_ds, val_ds, test_ds, class_names
