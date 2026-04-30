from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
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

def _write_object_counts_by_split(datatsets, path: Path) -> None:
    rows = []

    for split, dataset in datatsets.items():
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