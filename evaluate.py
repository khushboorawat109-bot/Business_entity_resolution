"""Macro-averaged F_0.5 scorer, matching the competition definition exactly."""


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision + recall)
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred_ids, true_ids):
    """Per-Source-1-entity score. Empty/empty => 1.0 (correct singleton)."""
    pred = set(pred_ids)
    true = set(true_ids)
    if not true and not pred:
        return 1.0
    if not pred:
        precision = 0.0
        recall = 0.0
    else:
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
    return f_beta(precision, recall, beta=0.5)


def macro_f05(predictions: dict, ground_truth: dict):
    """
    predictions, ground_truth: dict[source1_id] -> list[matched_ids]
    Scored over the union of keys present in ground_truth (evaluation set).
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, [])
        scores.append(score_entity(pred_ids, true_ids))
    return sum(scores) / len(scores) if scores else 0.0
