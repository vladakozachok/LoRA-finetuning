import numpy as np


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def compute_binary_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    true_positives = np.logical_and(predictions == 1, labels == 1).sum()
    false_positives = np.logical_and(predictions == 1, labels == 0).sum()
    false_negatives = np.logical_and(predictions == 0, labels == 1).sum()

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": compute_accuracy(predictions, labels),
        "f1": compute_binary_f1(predictions, labels),
    }
