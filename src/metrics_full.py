"""Six metrics for the JAFFE experiments, all macro one-vs-rest where relevant:
Accuracy, Precision, Recall, F1, Specificity, AUC.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             multilabel_confusion_matrix, precision_score,
                             recall_score)

from metrics_roc import macro_auc, per_class_auc


def all_metrics(labels, probs, num_classes):
    labels = np.asarray(labels)
    preds = probs.argmax(1)
    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec = float(recall_score(labels, preds, average="macro", zero_division=0))
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    # Specificity = TN / (TN + FP), per class, macro-averaged.
    mcm = multilabel_confusion_matrix(labels, preds, labels=list(range(num_classes)))
    spec = []
    for tn_fp_fn_tp in mcm:
        tn, fp = tn_fp_fn_tp[0, 0], tn_fp_fn_tp[0, 1]
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity = float(np.mean(spec))
    auc = macro_auc(labels, probs, num_classes)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": specificity, "auc": auc,
        "per_class_auc": per_class_auc(labels, probs, num_classes),
    }


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "specificity", "auc"]
