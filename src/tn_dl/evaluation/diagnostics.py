from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def write_classification_diagnostics(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int,
    output_dir: Path,
) -> None:
    """
    Evalua el modelo sobre un dataloader y guarda diagnosticos por clase.

    Archivos generados:
        - summary_metrics.csv
        - confusion_matrix.csv
        - confusion_matrix.png
        - per_class_metrics.csv
        - prediction_distribution.csv
        - top_confusions.csv
        - precision_recall_curve.csv
        - precision_recall_curve.png
        - f1_confidence_curve.csv
        - f1_confidence_curve.png

    Nota:
        classification_mAP es mAP de clasificacion sobre scores por clase. No es
        mAP@50 de deteccion, porque este modelo no predice bounding boxes ni IoU.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    labels, predictions, confidences, probabilities = _collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    confusion = _build_confusion_matrix(
        labels=labels,
        predictions=predictions,
        n_classes=n_classes,
    )

    per_class_rows = _build_per_class_metrics(
        confusion=confusion,
        labels=labels,
        probabilities=probabilities,
    )
    pr_rows = _build_precision_recall_curve_rows(
        labels=labels,
        probabilities=probabilities,
        n_classes=n_classes,
    )
    confidence_rows = _build_confidence_curve_rows(
        labels=labels,
        predictions=predictions,
        confidences=confidences,
    )

    _write_summary_metrics(
        confusion=confusion,
        per_class_rows=per_class_rows,
        confidence_rows=confidence_rows,
        labels=labels,
        probabilities=probabilities,
        path=output_dir / "summary_metrics.csv",
    )
    _write_confusion_matrix_csv(confusion, output_dir / "confusion_matrix.csv")
    _write_confusion_matrix_plot(confusion, output_dir / "confusion_matrix.png")
    _write_rows(per_class_rows, output_dir / "per_class_metrics.csv")
    _write_prediction_distribution(
        labels=labels,
        predictions=predictions,
        confidences=confidences,
        n_classes=n_classes,
        path=output_dir / "prediction_distribution.csv",
    )
    _write_top_confusions(confusion, output_dir / "top_confusions.csv")
    _write_rows(pr_rows, output_dir / "precision_recall_curve.csv")
    _write_precision_recall_plot(
        pr_rows=pr_rows,
        per_class_rows=per_class_rows,
        path=output_dir / "precision_recall_curve.png",
    )
    _write_rows(confidence_rows, output_dir / "f1_confidence_curve.csv")
    _write_confidence_curve_plot(
        rows=confidence_rows,
        path=output_dir / "f1_confidence_curve.png",
    )


def _collect_predictions(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    all_labels = []
    all_predictions = []
    all_confidences = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())
            all_probabilities.append(probabilities.cpu())
    return (
        torch.cat(all_labels),
        torch.cat(all_predictions),
        torch.cat(all_confidences),
        torch.cat(all_probabilities),
    )


def _build_confusion_matrix(
    *,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """
    Filas: clase real.
    Columnas: clase predicha.
    """
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)

    for label, prediction in zip(labels, predictions, strict=True):
        confusion[int(label), int(prediction)] += 1

    return confusion


def _build_per_class_metrics(
    *,
    confusion: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for class_id in range(confusion.shape[0]):
        true_positive = int(confusion[class_id, class_id].item())
        total_real = int(confusion[class_id, :].sum().item())
        total_predicted = int(confusion[:, class_id].sum().item())

        recall = true_positive / max(total_real, 1)
        precision = true_positive / max(total_predicted, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        average_precision = _compute_average_precision(
            labels=labels,
            scores=probabilities[:, class_id],
            class_id=class_id,
        )

        rows.append(
            {
                "class_id": class_id,
                "num_real": total_real,
                "num_predicted": total_predicted,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "average_precision": average_precision,
            }
        )

    return rows


def _compute_average_precision(
    *,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_id: int,
) -> float:
    targets = (labels == class_id).float()
    num_positive = int(targets.sum().item())
    if num_positive == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order]

    true_positive = torch.cumsum(sorted_targets, dim=0)
    false_positive = torch.cumsum(1.0 - sorted_targets, dim=0)

    precision = true_positive / (true_positive + false_positive).clamp_min(1e-12)
    recall = true_positive / max(num_positive, 1)

    recall_values = [0.0, *recall.tolist(), 1.0]
    precision_values = [1.0, *precision.tolist(), 0.0]

    for index in range(len(precision_values) - 2, -1, -1):
        precision_values[index] = max(precision_values[index], precision_values[index + 1])

    average_precision = 0.0
    for index in range(1, len(recall_values)):
        delta_recall = recall_values[index] - recall_values[index - 1]
        if delta_recall > 0.0:
            average_precision += delta_recall * precision_values[index]

    return float(average_precision)


def _build_precision_recall_curve_rows(
    *,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    n_classes: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for class_id in range(n_classes):
        targets = (labels == class_id).float()
        num_positive = int(targets.sum().item())
        if num_positive == 0:
            continue

        scores = probabilities[:, class_id]
        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        sorted_targets = targets[order]

        true_positive = torch.cumsum(sorted_targets, dim=0)
        false_positive = torch.cumsum(1.0 - sorted_targets, dim=0)

        precision = true_positive / (true_positive + false_positive).clamp_min(1e-12)
        recall = true_positive / max(num_positive, 1)

        previous_score: float | None = None
        for score, precision_value, recall_value in zip(
            sorted_scores.tolist(),
            precision.tolist(),
            recall.tolist(),
            strict=True,
        ):
            if previous_score is not None and abs(score - previous_score) < 1e-12:
                continue
            rows.append(
                {
                    "class_id": class_id,
                    "confidence_threshold": score,
                    "precision": precision_value,
                    "recall": recall_value,
                }
            )
            previous_score = score

    return rows


def _build_confidence_curve_rows(
    *,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    correct = predictions == labels
    total_items = labels.numel()

    for step in range(101):
        threshold = step / 100
        accepted = confidences >= threshold
        true_positive = int((accepted & correct).sum().item())
        false_positive = int((accepted & ~correct).sum().item())
        false_negative = total_items - true_positive
        accepted_count = int(accepted.sum().item())

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        coverage = accepted_count / max(total_items, 1)
        accepted_accuracy = true_positive / max(accepted_count, 1)

        rows.append(
            {
                "confidence_threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )

    return rows


def _write_summary_metrics(
    *,
    confusion: torch.Tensor,
    per_class_rows: list[dict[str, float | int]],
    confidence_rows: list[dict[str, float | int]],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    path: Path,
) -> None:
    total_items = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    weights = [int(row["num_real"]) for row in per_class_rows]
    total_weight = max(sum(weights), 1)

    macro_precision = _mean([float(row["precision"]) for row in per_class_rows])
    macro_recall = _mean([float(row["recall"]) for row in per_class_rows])
    macro_f1 = _mean([float(row["f1"]) for row in per_class_rows])
    classification_map = _mean([float(row["average_precision"]) for row in per_class_rows])

    weighted_precision = (
        sum(float(row["precision"]) * int(row["num_real"]) for row in per_class_rows) / total_weight
    )
    weighted_recall = (
        sum(float(row["recall"]) * int(row["num_real"]) for row in per_class_rows) / total_weight
    )
    weighted_f1 = (
        sum(float(row["f1"]) * int(row["num_real"]) for row in per_class_rows) / total_weight
    )

    top5_accuracy = _compute_topk_accuracy(
        labels=labels,
        probabilities=probabilities,
        k=5,
    )
    best_confidence_row = max(confidence_rows, key=lambda row: float(row["f1"]))

    rows = [
        {"metric": "num_samples", "value": total_items},
        {"metric": "accuracy", "value": correct / max(total_items, 1)},
        {"metric": "top5_accuracy", "value": top5_accuracy},
        {"metric": "macro_precision", "value": macro_precision},
        {"metric": "macro_recall", "value": macro_recall},
        {"metric": "macro_f1", "value": macro_f1},
        {"metric": "weighted_precision", "value": weighted_precision},
        {"metric": "weighted_recall", "value": weighted_recall},
        {"metric": "weighted_f1", "value": weighted_f1},
        {"metric": "micro_precision", "value": correct / max(total_items, 1)},
        {"metric": "micro_recall", "value": correct / max(total_items, 1)},
        {"metric": "micro_f1", "value": correct / max(total_items, 1)},
        {"metric": "classification_mAP", "value": classification_map},
        {
            "metric": "best_f1_confidence_threshold",
            "value": best_confidence_row["confidence_threshold"],
        },
        {"metric": "best_f1_confidence", "value": best_confidence_row["f1"]},
        {
            "metric": "precision_at_best_f1_confidence",
            "value": best_confidence_row["precision"],
        },
        {
            "metric": "recall_at_best_f1_confidence",
            "value": best_confidence_row["recall"],
        },
        {
            "metric": "coverage_at_best_f1_confidence",
            "value": best_confidence_row["coverage"],
        },
    ]

    _write_rows(rows, path)


def _compute_topk_accuracy(
    *,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    k: int,
) -> float:
    k = min(k, probabilities.shape[1])
    topk = probabilities.topk(k=k, dim=1).indices
    matches = topk.eq(labels.unsqueeze(1)).any(dim=1)
    return float(matches.float().mean().item())


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _write_rows(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix_csv(confusion: torch.Tensor, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class/predicted_class", *range(confusion.shape[1])])

        for class_id, row in enumerate(confusion.tolist()):
            writer.writerow([class_id, *row])


def _write_top_confusions(confusion: torch.Tensor, path: Path, top_k: int = 30) -> None:
    rows = []

    for true_class in range(confusion.shape[0]):
        for predicted_class in range(confusion.shape[1]):
            if true_class == predicted_class:
                continue

            count = int(confusion[true_class, predicted_class].item())
            if count > 0:
                rows.append(
                    {
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": count,
                    }
                )

    rows.sort(key=lambda row: row["count"], reverse=True)
    rows = rows[:top_k]

    _write_rows(rows, path)


def _write_prediction_distribution(
    *,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    n_classes: int,
    path: Path,
) -> None:
    rows = []

    for class_id in range(n_classes):
        real_mask = labels == class_id
        predicted_mask = predictions == class_id

        num_real = int(real_mask.sum().item())
        num_predicted = int(predicted_mask.sum().item())

        if predicted_mask.any():
            mean_confidence = float(confidences[predicted_mask].mean().item())
        else:
            mean_confidence = 0.0

        rows.append(
            {
                "class_id": class_id,
                "num_real": num_real,
                "num_predicted": num_predicted,
                "mean_confidence_when_predicted": mean_confidence,
            }
        )

    _write_rows(rows, path)


def _write_confusion_matrix_plot(confusion: torch.Tensor, path: Path) -> None:
    normalized = confusion.float()
    normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1.0)

    plt.figure(figsize=(10, 10))
    plt.imshow(normalized.numpy(), cmap="Blues", interpolation="nearest")
    plt.colorbar(label="fraction of real class")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _write_precision_recall_plot(
    *,
    pr_rows: list[dict[str, float | int]],
    per_class_rows: list[dict[str, float | int]],
    path: Path,
) -> None:
    by_class: dict[int, list[dict[str, float | int]]] = {}
    for row in pr_rows:
        by_class.setdefault(int(row["class_id"]), []).append(row)

    plt.figure(figsize=(8, 6))
    for rows in by_class.values():
        recalls = [float(row["recall"]) for row in rows]
        precisions = [float(row["precision"]) for row in rows]
        plt.plot(recalls, precisions, color="0.75", linewidth=0.8, alpha=0.7)

    recall_grid = [index / 100 for index in range(101)]
    macro_precision = []
    for recall_level in recall_grid:
        values = []
        for rows in by_class.values():
            candidates = [
                float(row["precision"]) for row in rows if float(row["recall"]) >= recall_level
            ]
            values.append(max(candidates) if candidates else 0.0)
        macro_precision.append(_mean(values))

    classification_map = _mean([float(row["average_precision"]) for row in per_class_rows])
    plt.plot(
        recall_grid,
        macro_precision,
        color="tab:blue",
        linewidth=2.5,
        label=f"Macro PR (classification mAP={classification_map:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _write_confidence_curve_plot(
    *,
    rows: list[dict[str, float | int]],
    path: Path,
) -> None:
    confidence = [float(row["confidence_threshold"]) for row in rows]
    precision = [float(row["precision"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]
    f1 = [float(row["f1"]) for row in rows]
    coverage = [float(row["coverage"]) for row in rows]

    best = max(rows, key=lambda row: float(row["f1"]))

    plt.figure(figsize=(8, 6))
    plt.plot(confidence, f1, label="F1")
    plt.plot(confidence, precision, label="Precision")
    plt.plot(confidence, recall, label="Recall")
    plt.plot(confidence, coverage, label="Coverage", linestyle="--")
    plt.axvline(
        float(best["confidence_threshold"]),
        color="0.3",
        linestyle=":",
        label=f"Best F1 @ {float(best['confidence_threshold']):.2f}",
    )
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric")
    plt.title("F1-Confidence curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
