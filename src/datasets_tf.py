from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input


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
