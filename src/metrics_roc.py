"""Multiclass ROC/AUC helpers (one-vs-rest) for the 7 JAFFE classes.

roc_auc_score needs both positives and negatives present for each one-vs-rest
problem; on a 34-image validation fold a class can be missing, so every helper
here skips absent classes and macro-averages only over the ones present.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def _onehot(labels, num_classes):
    return label_binarize(labels, classes=list(range(num_classes)))


def macro_auc(labels, probs, num_classes):
    """Macro one-vs-rest AUC over classes that have >=1 positive AND negative."""
    labels = np.asarray(labels)
    onehot = _onehot(labels, num_classes)
    aucs = []
    for c in range(num_classes):
        column = onehot[:, c]
        if column.sum() == 0 or column.sum() == len(column):
            continue
        aucs.append(roc_auc_score(column, probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_auc(labels, probs, num_classes):
    labels = np.asarray(labels)
    onehot = _onehot(labels, num_classes)
    out = []
    for c in range(num_classes):
        column = onehot[:, c]
        if column.sum() == 0 or column.sum() == len(column):
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(column, probs[:, c])))
    return out


def plot_roc(labels, probs, class_names, out_path, title):
    """Per-class ROC curves + macro-average, saved as a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    num_classes = len(class_names)
    labels = np.asarray(labels)
    onehot = _onehot(labels, num_classes)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    used = 0
    for c in range(num_classes):
        column = onehot[:, c]
        if column.sum() == 0 or column.sum() == len(column):
            continue
        fpr, tpr, _ = roc_curve(column, probs[:, c])
        auc = roc_auc_score(column, probs[:, c])
        ax.plot(fpr, tpr, alpha=0.6, lw=1.2,
                label=f"{class_names[c]} (AUC={auc:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        used += 1
    if used:
        mean_tpr /= used
        macro = macro_auc(labels, probs, num_classes)
        ax.plot(all_fpr, mean_tpr, color="black", lw=2.4, linestyle="--",
                label=f"makro (AUC={macro:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
