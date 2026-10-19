"""metrics.py —— 分类性能指标（纯 numpy 实现，不强制依赖 scikit-learn）。

指标定义（都是多分类、100 类、类别不严格均衡）
--------------------------------------------
    Accuracy        : 总体正确率
    Top-5 Accuracy  : 正确答案落在概率前 5 名里的比例（100 类时更能反映"接近对"）
    Macro Precision / Recall / F1 : 先对每一类算，再对 100 类取平均
        —— 因为稀有鸟种与常见鸟种同样重要，macro 不会被大类淹没
    Weighted F1     : 按各类样本数加权，接近 Accuracy 的"平滑版"
    Per-class Acc   : 每一类的召回率（错误分析里"哪几种鸟最难分"）
"""
from __future__ import annotations

import numpy as np


def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0


def per_class_accuracy(y_true, y_pred, num_classes: int) -> np.ndarray:
    """每一类的召回率 = 该类被正确识别的比例。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            acc[c] = np.mean(y_pred[mask] == c)
    return acc


def macro_scores(y_true, y_pred, num_classes: int) -> tuple:
    """返回 (macro_precision, macro_recall, macro_f1)。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def weighted_f1(y_true, y_pred, num_classes: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = len(y_true)
    if total == 0:
        return 0.0
    score = 0.0
    for c in range(num_classes):
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
        score += f * float(np.sum(y_true == c)) / total
    return float(score)


def topk_accuracy(scores, labels, k: int = 5) -> float:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 2 or len(labels) == 0:
        return 0.0
    k = int(min(k, scores.shape[1]))
    topk = np.argsort(-scores, axis=1)[:, :k]
    hit = [labels[i] in topk[i] for i in range(len(labels))]
    return float(np.mean(hit))


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def compute_metrics(y_true, y_pred, probs=None, num_classes: int = 100,
                    classes: list = None) -> dict:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    macro_p, macro_r, macro_f1 = macro_scores(y_true, y_pred, num_classes)
    cls_acc = per_class_accuracy(y_true, y_pred, num_classes)

    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": accuracy,
        "top5_accuracy": topk_accuracy(probs, y_true, 5) if probs is not None else None,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1(y_true, y_pred, num_classes),
        "per_class_accuracy": [float(v) for v in cls_acc],
        "per_class_support": [int(np.sum(y_true == c)) for c in range(num_classes)],
    }
    if classes is not None:
        metrics["classes"] = list(classes)
    return metrics


def print_metrics(metrics: dict, title: str = "", logger=None) -> None:
    def out(text=""):
        if logger is not None:
            logger.info(text)
        else:
            print(text)

    if title:
        out(f"---------- {title} ----------")
    out(f"  samples      : {metrics['num_samples']}")
    out(f"  accuracy     : {metrics['accuracy']:.4f}")
    if metrics.get("top5_accuracy") is not None:
        out(f"  top-5 acc    : {metrics['top5_accuracy']:.4f}")
    out(f"  macro P/R/F1 : {metrics['macro_precision']:.4f} / "
        f"{metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    out(f"  weighted F1  : {metrics['weighted_f1']:.4f}")
    out("-" * 34)


def hardest_classes(metrics: dict, topk: int = 10) -> list:
    """返回最难分的类别：[(class_name, acc, support), ...]（只统计有样本的类）。"""
    classes = metrics.get("classes")
    if not classes:
        return []
    rows = []
    for idx, (acc, support) in enumerate(zip(metrics["per_class_accuracy"],
                                             metrics["per_class_support"])):
        if support > 0:
            rows.append((classes[idx], acc, support))
    rows.sort(key=lambda r: (r[1], -r[2]))
    return rows[:topk]


def top_confusion_pairs(cm: np.ndarray, classes: list, topk: int = 20) -> list:
    """最容易混淆的类别对：把混淆矩阵里非对角的最大元素排出来。"""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": classes[i] if classes else str(i),
                    "pred": classes[j] if classes else str(j),
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda r: -r["count"])
    return pairs[:topk]
