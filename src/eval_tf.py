import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.datasets_tf import get_datasets


def evaluate(model_path: str, data_dir: str, img_size: int = 224, batch_size: int = 32, out_dir: str = "reports"):
    _, _, test_ds, class_names = get_datasets(data_dir, img_size=img_size, batch_size=batch_size)

    model = tf.keras.models.load_model(model_path)

    # Gather predictions and labels
    y_true = []
    y_prob = []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().ravel().tolist())

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    print("Test results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    out_root = Path(out_dir)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)

    # Confusion matrix plot
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = out_root / 'figures' / 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()

    # Save metrics
    metrics_txt = out_root / 'metrics.txt'
    with open(metrics_txt, 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")

    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved metrics to: {metrics_txt}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained TF/Keras model on the malaria test set')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model .h5')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with train/val/test subfolders')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', type=str, default='reports')
    args = parser.parse_args()

    evaluate(args.model, args.data_dir, args.img_size, args.batch_size, args.out_dir)


if __name__ == '__main__':
    main()
