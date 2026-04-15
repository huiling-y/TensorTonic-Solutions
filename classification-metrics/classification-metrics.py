import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    accuracy = np.mean(y_true == y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "binary":
        classes = np.array([pos_label])

    tp = np.array([np.sum((y_pred == c) & (y_true == c)) for c in classes])
    fp = np.array([np.sum((y_pred == c) & (y_true != c)) for c in classes])
    fn = np.array([np.sum((y_true == c) & (y_pred != c)) for c in classes])

    if average == 'micro':
        tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average in ('macro', "binary"):
        per_precision = np.where(tp+fp > 0, tp / (tp+fp), 0.0)
        per_recall = np.where(tp+fn > 0, tp / (tp+fn), 0.0)
        per_f1 = np.where(per_precision + per_recall > 0,
                                 2 * per_precision * per_recall / (per_precision + per_recall), 0.0)
        precision = per_precision.mean()
        recall = per_recall.mean()
        f1 = per_f1.mean()

    elif average == 'weighted':
        weights = np.array([np.sum(y_true == c) for c in classes])
        per_precision = np.where(tp+fp > 0, tp / (tp+fp), 0.0)
        per_recall = np.where(tp+fn > 0, tp / (tp+fn), 0.0)
        per_f1 = np.where(per_precision + per_recall > 0,
                                 2 * per_precision * per_recall / (per_precision + per_recall), 0.0)
        precision = np.average(per_precision, weights=weights)
        recall = np.average(per_recall, weights=weights)
        f1 = np.average(per_f1, weights=weights)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }