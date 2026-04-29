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
    Evalua el modelo sobre un dataloader y guarda diagnósticos por clase.

    Archivos generados:
        - confusion_matrix.csv
        - confusion_matrix.png
        - per_class_metrics.csv
        - prediction_distribution.csv
        - top_confusions.csv
    """

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
    _write_confusion_matrix_plot(confusion, output_dir / "confusion_matrix.png")
    _write_per_class_metrics(confusion, output_dir / "per_class_metrics.csv")
    _write_prediction_distribution(
        labels = labels,
        predictions = predictions,
        confidences = confidences,
        n_classes = n_classes,
        path = output_dir / "prediction_distribution.csv"
    )
    _write_top_confusions(confusion, output_dir / "top_confusions.csv")

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

def _write_per_class_metrics(confusion: torch.Tensor, path: Path) -> None:
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

def _write_confusion_matrix_plot(confusion: torch.Tensor, path: Path) -> None:
    normalized = confusion.float()
    normalized = normalized / normalized.sum(dim = 1, keepdim = True).clamp_min(1.0)

    plt.figure(figsize=(10, 10))
    plt.imshow(normalized.numpy(), cmap = "Blues", interpolation = "nearest")
    plt.colorbar(label = "fraction of real class")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi = 300)
    plt.close()
