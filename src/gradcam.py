import numpy as np
import tensorflow as tf
from tensorflow.keras import models


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute a Grad-CAM heatmap for a single preprocessed image.

    img_array: np.ndarray or tf.Tensor of shape (1, H, W, 3)
    model: Keras model with a convolutional backbone and final sigmoid output.
    last_conv_layer_name: name of the last conv layer in the backbone (e.g. 'conv5_block3_out').
    pred_index: index of the class to visualize (0 for positive class in binary); if None, uses model prediction.
    """
    if isinstance(img_array, np.ndarray):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the final predictions.
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = models.Model([
        model.inputs
    ], [
        last_conv_layer.output,
        model.output,
    ])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of the target class w.r.t. conv feature map
    grads = tape.gradient(class_channel, conv_outputs)
    # Global-average-pool the gradients over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4, cmap="jet"):
    """Overlay a Grad-CAM heatmap on top of an RGB image.

    heatmap: 2D numpy array in [0, 1]
    image: uint8 RGB image array (H, W, 3)
    Returns: uint8 RGB image with heatmap overlay.
    """
    import matplotlib.cm as cm

    # Rescale heatmap to 0-255 and apply colormap
    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap(cmap)
    colored = colormap(heatmap)
    colored = np.uint8(colored[:, :, :3] * 255)

    # Resize to match image
    colored = tf.image.resize(colored, (image.shape[0], image.shape[1])).numpy().astype("uint8")

    # Overlay
    overlay = np.uint8(alpha * colored + (1 - alpha) * image)
    return overlay
