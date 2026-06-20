from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.utils import make_grid

from tn_dl.data.lab_apparatus import (
    LabApparatusCropDataset,
    build_lab_apparatus_transform,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def run_lab_apparatus_audit(*, data_config, output_dir: Path) -> None:
    """
    No se vuelve a parsear YOLO aquÃ­; esta auditorÃ­a usa LabApparatusCropDataset.samples porque eso
    audita los crops que luego va a entrenar el modelo
    """
    output_dir.mkdir(parents = True, exist_ok = True)

    datasets = {
        "train": LabApparatusCropDataset(data_config.data_dir, split = "train", transform = None),
        "valid": LabApparatusCropDataset(data_config.data_dir, split = "valid", transform = None),
        "test": LabApparatusCropDataset(data_config.data_dir, split = "test", transform = None),
    }

    _write_class_counts_by_split(datasets, output_dir / "class_counts_by_split.csv")
    _write_object_counts_by_split(datasets, output_dir / "object_counts_by_split.csv")
    _write_image_object_count_distribution(
        datasets,
        output_dir / "image_object_count_distribution.csv",
    )
    _write_bbox_size_distribution(datasets, output_dir / "bbox_size_distribution.csv")
    _write_crop_samples(data_config, datasets, output_dir / "crop_samples.png")
    _write_bbox_overlay_samples(datasets, output_dir / "bbox_overlay_samples.png")

    logger.info("Auditoria Lab Apparatus completada en %s", output_dir)

def _write_class_counts_by_split(datasets, path: Path) -> None:
    """
    Cuenta objetos por clase, no cuenta imÃ¡genes.
    """
    class_names = datasets["train"].class_names
    counts = {
        split: Counter(sample.class_id for sample in dataset.samples)
        for split, dataset in datasets.items()
    }

    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "train", "valid", "test", "total"])

        for class_id, class_name in enumerate(class_names):
            train = counts["train"][class_id]
            valid = counts["valid"][class_id]
            test = counts["test"][class_id]
            writer.writerow([class_id, class_name, train, valid, test, train + valid + test])

def _write_object_counts_by_split(datasets, path: Path) -> None:
    """
    Crea un resumen por split.
    """
    rows = []

    for split, dataset in datasets.items():
        object_counts = Counter(sample.image_path for sample in dataset.samples)
        values = list(object_counts.values())

        rows.append({
            "split": split,
            "num_images_with_objects": len(object_counts),
            "num_objects": len(dataset.samples),
            "mean_objects_per_image": sum(values) / max(len(values), 1),
            "max_objects_per_image": max(values) if values else 0,
        })

    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _write_image_object_count_distribution(datasets, path: Path) -> None:
    """
    Esta parte responde a la pregunta:
    cuantas imagenes tienen 1 objeto, 2 objetos, 3 objetos...
    """
    rows = []

    for split, dataset in datasets.items():
        per_image = Counter(sample.image_path for sample in dataset.samples)
        distribution = Counter(per_image.values())

        for num_objects, num_images in sorted(distribution.items()):
            rows.append({
                "split": split,
                "num_objects": num_objects,
                "num_images": num_images,
            })

    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = ["split", "num_objects", "num_images"])
        writer.writeheader()
        writer.writerows(rows)

def _write_bbox_size_distribution(datasets, path: Path) -> None:
    """
    Sirve para detectar cajas minÃºsculas, objetos muy alargados...etc.
    """
    rows = []

    for split, dataset in datasets.items():
        for sample in dataset.samples:
            left, top, right, bottom = sample.bbox_pixels
            width_px = right - left
            height_px = bottom - top
            _, _, width_norm, height_norm = sample.bbox_normalized

            rows.append({
                "split": split,
                "class_id": sample.class_id,
                "class_name": sample.class_name,
                "width_norm": width_norm,
                "height_norm": height_norm,
                "area_norm": width_norm * height_norm,
                "aspect_ratio": width_norm / max(height_norm, 1e-12),
                "width_px": width_px,
                "height_px": height_px,
                "area_px": width_px * height_px,
                "image_path": str(sample.image_path),
            })

    with path.open("w", newline = "", encoding = "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _write_crop_samples(data_config, datasets, path: Path, n_per_split: int = 8) -> None:
    """
    Genera una grid de crops tal y como los verÃ¡ el modelo. Es importante para ver si
    `Resize(224, 224)` deforma demasiado la imÃ¡gen o si las augmentations son agresivas.
    """
    images: list[torch.Tensor] = []

    for split in datasets:
        transform = build_lab_apparatus_transform(train=(split == "train"))
        dataset = LabApparatusCropDataset(
            data_config.data_dir,
            split = split,
            transform = transform,
            max_samples = n_per_split,
        )

        for index in range(min(n_per_split, len(dataset))):
            image, _ = dataset[index]
            if not isinstance(image, torch.Tensor):
                raise TypeError(f"Expected transformed crop to be a Tensor, got {type(image)}.")
            images.append(_denormalize(image))

    grid = make_grid(images, nrow = n_per_split, padding = 4, pad_value = 1.0)

    plt.figure(figsize = (16, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Crop samples: train / valid / test")
    plt.tight_layout()
    plt.savefig(path, dpi = 150)
    plt.close()

def _write_bbox_overlay_samples(datasets, path: Path, n_images: int = 12) -> None:
    rendered = []

    for dataset in datasets.values():
        by_image = defaultdict(list)
        for sample in dataset.samples:
            by_image[sample.image_path].append(sample)

        for image_path, samples in list(by_image.items())[:n_images]:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            for sample in samples:
                draw.rectangle(sample.bbox_pixels, outline = "red", width = 4)
                draw.text(
                    (sample.bbox_pixels[0], sample.bbox_pixels[1]),
                    str(sample.class_id),
                    fill = "red",
                )

            image.thumbnail((320, 320))
            rendered.append(torchvision.transforms.ToTensor()(image))

    grid = make_grid(rendered, nrow = 4, padding = 4, pad_value = 1.0)

    plt.figure(figsize = (12, 9))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Bounding box overlay samples")
    plt.tight_layout()
    plt.savefig(path, dpi = 150)
    plt.close()

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
