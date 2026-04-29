from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from tn_dl.data.flowers import build_flowers_dataset

logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def run_data_audit(*, data_config, output_dir: Path, seed: int = 7) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = build_flowers_dataset(data_config, train=False)
    n_total = len(base_dataset)

    train_size = int(0.7 * n_total)
    eval_size = int(0.15 * n_total)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator).tolist()
    train_indices = permutation[:train_size]
    eval_indices = permutation[train_size:train_size + eval_size]
    test_indices = permutation[train_size + eval_size:]

    all_labels = base_dataset.labels

    splits = {
        "train": train_indices,
        "eval": eval_indices,
        "test": test_indices,
    }

    _write_class_counts(all_labels, splits, output_dir / "class_counts.csv")
    _write_class_counts_plot(all_labels, splits, output_dir / "class_counts.png")
    _write_split_summary(splits, output_dir / "split_summary.txt")
    _check_label_alignment(data_config, train_indices, output_dir / "alignment_check.txt")
    _write_train_batch_variety(data_config, train_indices, output_dir / "batch_variety.txt")
    _write_augmentation_grid(
        data_config, train_indices, output_dir / "augmentation_samples.png"
    )

    logger.info("Auditoría de datos completada en %s", output_dir)


def _write_class_counts(
    all_labels, splits: dict[str, list[int]], path: Path
) -> None:
    class_ids = sorted(set(int(lbl) for lbl in all_labels))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "total", "train", "eval", "test"])

        for cid in class_ids:
            total = sum(1 for lbl in all_labels if int(lbl) == cid)
            row = [cid, total]
            for split_name in ("train", "eval", "test"):
                indices = splits[split_name]
                count = sum(1 for i in indices if int(all_labels[i]) == cid)
                row.append(count)
            writer.writerow(row)


def _write_class_counts_plot(
    all_labels, splits: dict[str, list[int]], path: Path
) -> None:
    class_ids = sorted(set(int(lbl) for lbl in all_labels))

    train_counts = []
    eval_counts = []
    test_counts = []

    for cid in class_ids:
        train_counts.append(
            sum(1 for i in splits["train"] if int(all_labels[i]) == cid)
        )
        eval_counts.append(
            sum(1 for i in splits["eval"] if int(all_labels[i]) == cid)
        )
        test_counts.append(
            sum(1 for i in splits["test"] if int(all_labels[i]) == cid)
        )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, counts, name in zip(
        axes, [train_counts, eval_counts, test_counts], ["train", "eval", "test"],
        strict = True
    ):
        ax.bar(class_ids, counts, width=1.0)
        ax.set_ylabel("count")
        ax.set_title(f"{name} — images per class (min={min(counts)}, max={max(counts)})")

    axes[-1].set_xlabel("class_id")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_split_summary(splits: dict[str, list[int]], path: Path) -> None:
    lines = []
    total = sum(len(v) for v in splits.values())
    for name, indices in splits.items():
        pct = 100 * len(indices) / max(total, 1)
        lines.append(f"{name:6s}: {len(indices):5d} images ({pct:.1f}%)")
    lines.append(f"{'total':6s}: {total:5d} images")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_label_alignment(
    data_config, train_indices: list[int], path: Path, n_checks: int = 20
) -> None:
    dataset_no_aug = build_flowers_dataset(data_config, train=False, indices=train_indices)

    lines = ["Alignment check: index -> real_index -> filename -> label"]
    lines.append("-" * 60)

    check_positions = list(range(min(n_checks, len(dataset_no_aug))))

    all_ok = True
    for pos in check_positions:
        real_index = dataset_no_aug.indices[pos]
        expected_label = int(dataset_no_aug.labels[real_index])
        _, returned_label = dataset_no_aug[pos]

        match = expected_label == returned_label
        if not match:
            all_ok = False

        filename = f"image_{real_index + 1:05d}.jpg"
        status = "OK" if match else "MISMATCH"
        lines.append(
            f"pos={pos:4d}  real_idx={real_index:5d}  file={filename}  "
            f"expected={expected_label:3d}  got={returned_label:3d}  [{status}]"
        )

    lines.append("-" * 60)
    if all_ok:
        lines.append(f"RESULTADO: {len(check_positions)} muestras verificadas, todas OK.")
    else:
        lines.append("RESULTADO: HAY DESALINEACION. Revisar OxfordFlowersDataset.__getitem__.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_train_batch_variety(
    data_config, train_indices: list[int], path: Path
) -> None:
    dataset = build_flowers_dataset(data_config, train=True, indices=train_indices)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=0,
    )

    lines = ["Batch variety: classes present in the first 5 batches"]
    lines.append("-" * 60)

    for i, (_, labels) in enumerate(loader):
        if i >= 5:
            break
        classes = sorted(set(int(lbl) for lbl in labels))
        lines.append(
            f"Batch {i}: {len(classes)} distinct classes out of {len(labels)} samples: "
            f"{classes}"
        )

    lines.append("-" * 60)
    lines.append(
        "Si cada batch tiene muchas clases distintas, el shuffle funciona bien."
    )
    lines.append(
        "Si aparecen pocas clases repetidas, puede haber un problema con el sampler."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.view(3, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _write_augmentation_grid(
    data_config, train_indices: list[int], path: Path
) -> None:
    raw_dataset = build_flowers_dataset(data_config, train=False, indices=train_indices)
    aug_dataset = build_flowers_dataset(data_config, train=True, indices=train_indices)

    n_samples = 4
    n_augmentations = 4

    images = []
    for sample_idx in range(n_samples):
        original, _ = raw_dataset[sample_idx]
        images.append(original)
        for _ in range(n_augmentations):
            augmented, _ = aug_dataset[sample_idx]
            images.append(augmented)

    images = [_denormalize(img) for img in images]
    grid = make_grid(images, nrow=n_augmentations + 1, padding=4, pad_value=1.0)

    fig, ax = plt.subplots(1, 1, figsize=(3 * (n_augmentations + 1), 3 * n_samples))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title("Original (col 1) + augmentations (cols 2-5)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()