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
        class_names: list[str] | None = None,
        split_name: str = "valid",
) -> None:
    """
    Evalua el modelo sobre un dataloader y guarda diagnósticos por clase.

    Archivos generados:
        - confusion_matrix.csv
        - confusion_matrix.png
        - per_class_metrics.csv
        - prediction_distribution.csv
        - top_confusions.csv
        - summary_metrics.csv (accuracy global, macro/weighted precision/recall/F1)

    Si se pasan class_names, se usan como etiquetas legibles en la matriz de
    confusión y en per_class_metrics, lo que facilita la lectura para el TFG.
    """

    if class_names is not None and len(class_names) != n_classes:
        class_names = None

    output_dir.mkdir(parents = True, exist_ok = True)

    labels, predictions, confidences = _collect_predictions(
        model = model,
        dataloader = dataloader,
        device = device,
    )

    confusion = _build_confusion_matrix(
        labels = labels,
        predictions = predictions,
        n_classes = n_classes,
    )

    _write_confusion_matrix_csv(confusion, output_dir / "confusion_matrix.csv")
    _write_confusion_matrix_plot(
        confusion, output_dir / "confusion_matrix.png", class_names = class_names
    )
    _write_per_class_metrics(
        confusion, output_dir / "per_class_metrics.csv", class_names = class_names
    )
    _write_prediction_distribution(
        labels = labels,
        predictions = predictions,
        confidences = confidences,
        n_classes = n_classes,
        path = output_dir / "prediction_distribution.csv"
    )
    _write_top_confusions(confusion, output_dir / "top_confusions.csv")
    _write_summary_metrics(
        confusion, output_dir / "summary_metrics.csv", split_name = split_name
    )

def _collect_predictions(
        *,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    all_labels = []
    all_predictions = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device = device, dtype = torch.long)

            logits = model(images)
            probabilities = torch.softmax(logits, dim = 1)
            confidences, predictions = probabilities.max(dim = 1)

            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())
    return (
        torch.cat(all_labels),
        torch.cat(all_predictions),
        torch.cat(all_confidences)
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

    confusion[real, predicha] cuenta cuantas imágenes de la clase real acabaron clasificadas como
    predicha.
    """
    confusion = torch.zeros((n_classes, n_classes), dtype = torch.long)

    for label, prediction in zip(labels, predictions, strict = True):
        confusion[int(label), int(prediction)] += 1

    return confusion

def _write_confusion_matrix_csv(confusion: torch.Tensor, path: Path) -> None:
    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class/predicted_class", *range(confusion.shape[1])])

        for class_id, row in enumerate(confusion.tolist()):
            writer.writerow([class_id, *row])

def _write_per_class_metrics(
        confusion: torch.Tensor,
        path: Path,
        class_names: list[str] | None = None,
) -> None:
    rows = []

    for class_id in range(confusion.shape[0]):
        true_positive = int(confusion[class_id, class_id].item())
        total_real = int(confusion[class_id, :].sum().item())
        total_predicted = int(confusion[:, class_id].sum().item())

        recall = true_positive / max(total_real, 1)
        precision = true_positive / max(total_predicted, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_names else str(class_id),
                "num_real": total_real,
                "num_predicted": total_predicted,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_metrics(
        confusion: torch.Tensor,
        path: Path,
        split_name: str = "valid",
) -> None:
    """
    Resume la clasificación en una sola fila a partir de la matriz de confusión:
    accuracy global y medias macro/weighted de precision, recall y F1.

    El promedio macro trata todas las clases por igual; el weighted pondera por el
    número de muestras reales de cada clase. Son las métricas estándar para comparar
    modelos multiclase (TN vs ResNet50).
    """
    n_classes = confusion.shape[0]
    total = int(confusion.sum().item())
    correct = int(confusion.diagonal().sum().item())
    accuracy = correct / max(total, 1)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    for class_id in range(n_classes):
        true_positive = int(confusion[class_id, class_id].item())
        total_real = int(confusion[class_id, :].sum().item())
        total_predicted = int(confusion[:, class_id].sum().item())

        recall = true_positive / max(total_real, 1)
        precision = true_positive / max(total_predicted, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(total_real)

    total_support = max(sum(supports), 1)

    def _macro(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    def _weighted(values: list[float]) -> float:
        return sum(v * s for v, s in zip(values, supports, strict=True)) / total_support

    row = {
        "split": split_name,
        "num_samples": total,
        "num_classes": n_classes,
        "accuracy": accuracy,
        "macro_precision": _macro(precisions),
        "macro_recall": _macro(recalls),
        "macro_f1": _macro(f1s),
        "weighted_precision": _weighted(precisions),
        "weighted_recall": _weighted(recalls),
        "weighted_f1": _weighted(f1s),
    }

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

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
    
    rows.sort(key = lambda row: row["count"], reverse = True)
    rows = rows[:top_k]

    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["true_class", "predicted_class", "count"])
        writer.writeheader()
        writer.writerows(rows)

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
    
    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _write_confusion_matrix_plot(
        confusion: torch.Tensor,
        path: Path,
        class_names: list[str] | None = None,
) -> None:
    normalized = confusion.float()
    normalized = normalized / normalized.sum(dim = 1, keepdim = True).clamp_min(1.0)

    plt.figure(figsize=(10, 10))
    plt.imshow(normalized.numpy(), cmap = "Blues", interpolation = "nearest")
    plt.colorbar(label = "fraction of real class")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized confusion matrix")

    if class_names is not None:
        ticks = range(len(class_names))
        plt.xticks(ticks, class_names, rotation = 90, fontsize = 6)
        plt.yticks(ticks, class_names, fontsize = 6)

    plt.tight_layout()
    plt.savefig(path, dpi = 300)
    plt.close()
