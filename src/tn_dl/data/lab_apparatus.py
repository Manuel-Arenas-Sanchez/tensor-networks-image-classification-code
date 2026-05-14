from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml
from PIL import Image

from tn_dl.data import DataLoaders

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

@dataclass(frozen=True)
class LabApparatusSample:
    image_path: Path
    label_path: Path
    split: str
    object_index: int
    class_id: int
    class_name: str
    bbox_normalized: tuple[float, float, float, float]
    bbox_pixels: tuple[int, int, int, int]

class LabApparatusCropDataset(torch.utils.data.Dataset):
    """
    Dataset de clasificación por instancia para ChemEq25.

    Cada linea YOLO se convierte en una muestra independiente:

        imagen completa + bbox YOLO -> crop del objeto -> etiqueta de clase
    
    Así mantenemos el problema como clasificación, que encaja con AIM + HybridCustomTTN, sin
    convertir la red tensorial en un detector.
    """

    def __init__(
            self,
            root_dir,
            split: str,
            transform = None,
            crop_margin: float = 0.05,
            max_samples: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = _normalize_split(split)
        self.transform = transform
        self.crop_margin = crop_margin

        self.class_names = _load_class_names(self.root_dir)
        self.samples = _build_samples(
            root_dir = self.root_dir,
            split = self.split,
            class_names = self.class_names,
            crop_margin = crop_margin,
        )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.labels = [sample.class_id for sample in self.samples]

        if len(self.samples) == 0:
            raise ValueError(
                f"No se encontraron crops para split = '{self.split}' en {self.root_dir}."
            )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        crop = image.crop(sample.bbox_pixels)

        if self.transform:
            crop = self.transform(crop)
        
        return crop, int(sample.class_id)
    
    def get_metadata(self, index: int) -> LabApparatusSample:
        return self.samples[index]

class ResizeWithPadding:
    """
    Redimensiona manteniendo aspect ratio y rellena hasta un cuadrado.

    El padding usa aproximadamente la media ImageNet en RGB, de modo que tras Normalize queda cerca
    de cero y no introduce bordes negros dominantes.
    """

    def __init__(
            self,
            size: int,
            fill: tuple[int, int, int] = (124, 116, 104),
    ) -> None:
        self.size = size
        self.fill = fill
    
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        scale = min(self.size / width, self.size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        resized = image.resize(
            (new_width, new_height),
            resample = Image.Resampling.BICUBIC,
        )

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))

        return canvas

def build_lab_apparatus_transform(
        train: bool = False,
        *,
        image_size: int = 224,
        preserve_aspect_ratio: bool = False,
        augmentation_name: str = "basic",
        rotation_degrees: float = 0.0,
        random_erasing_p: float = 0.0,
    ):

    normalize = torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    if preserve_aspect_ratio:
        resize_transform = ResizeWithPadding(image_size)
    else:
        resize_transform = torchvision.transforms.Resize(
            (image_size, image_size),
            interpolation = torchvision.transforms.InterpolationMode.BICUBIC,
        )

    if train:
        if preserve_aspect_ratio:
            transform_steps: list[Callable[[Any], Any]] = [resize_transform]
        else:
            transform_steps = [
                torchvision.transforms.RandomResizedCrop(
                    image_size,
                    scale = (0.85, 1.0),
                    ratio = (0.8, 1.25),
                    interpolation = torchvision.transforms.InterpolationMode.BICUBIC,
                )
            ]

        if augmentation_name in ("basic", "light"):
            if augmentation_name == "light" and rotation_degrees > 0.0:
                transform_steps.append(
                    torchvision.transforms.RandomRotation(
                        degrees = rotation_degrees,
                        interpolation = torchvision.transforms.InterpolationMode.BICUBIC,
                        fill = 124,
                    )
                )

            transform_steps.extend([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ColorJitter(
                    brightness = 0.2,
                    contrast = 0.2,
                    saturation = 0.15,
                    hue = 0.03,
                ),
            ])

            if augmentation_name == "light":
                transform_steps.append(
                    torchvision.transforms.RandomAffine(
                        degrees = 0,
                        translate = (0.03, 0.03),
                        scale = (0.95, 1.05),
                        interpolation = torchvision.transforms.InterpolationMode.BICUBIC,
                        fill = 124,
                    )
                )

        elif augmentation_name != "none":
            raise ValueError("Expected augmentation_name to be 'none', 'basic' or 'light'.")

        transform_steps.extend([
            torchvision.transforms.ToTensor(),
            normalize,
        ])

        if augmentation_name == "light" and random_erasing_p > 0.0:
            transform_steps.append(
                torchvision.transforms.RandomErasing(
                    p=random_erasing_p,
                    scale = (0.01, 0.04),
                    ratio = (0.3, 3.3),
                    value = 0,
                )
            )

        return torchvision.transforms.Compose(transform_steps)

    return torchvision.transforms.Compose([
       resize_transform,
       torchvision.transforms.ToTensor(),
       normalize,
    ])


def build_lab_apparatus_dataset(data_config, split: str, train: bool = False):
    if getattr(data_config, "download", False):
        logger.warning(
            "download = True se ignora para Lab Apparatus: se espera el dataset local."
        )
    
    transform = build_lab_apparatus_transform(
        train=train,
        image_size=getattr(data_config, "image_size", 224),
        preserve_aspect_ratio=getattr(data_config, "preserve_aspect_ratio", False),
        augmentation_name=getattr(data_config, "augmentation_name", "basic"),
        rotation_degrees=getattr(data_config, "rotation_degrees", 0.0),
        random_erasing_p=getattr(data_config, "random_erasing_p", 0.0),
    )

    dataset = LabApparatusCropDataset(
        root_dir = data_config.data_dir,
        split = split,
        transform = transform,
        crop_margin = getattr(data_config, "crop_margin", 0.05),
        max_samples = getattr(data_config, "smoke_samples", None),
    )

    logger.info(
        "Dataset Lab Apparatus cargado: split = %s, crops = %s, clases = %s",
        dataset.split,
        len(dataset),
        len(dataset.class_names),
    )
    return dataset

def build_lab_apparatus_dataloaders(data_config):
    train_dataset = build_lab_apparatus_dataset(data_config, split = "train", train = True)
    eval_dataset = build_lab_apparatus_dataset(data_config, split = "valid", train = False)
    test_dataset = build_lab_apparatus_dataset(data_config, split = "test", train = False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = data_config.batch_size,
        shuffle = True,
        num_workers = data_config.num_workers,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=data_config.batch_size,
        shuffle = False,
        num_workers = data_config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = data_config.batch_size,
        shuffle = False,
        num_workers = data_config.num_workers,
    )

    logger.info(
        "Split Lab Apparatus creado: train = %s, eval = %s, test = %s, batch_size = %s",
        len(train_dataset),
        len(eval_dataset),
        len(test_dataset),
        data_config.batch_size,
    )

    return DataLoaders(
        train_loader = train_loader,
        eval_loader = eval_loader,
        test_loader = test_loader,
    )

def _normalize_split(split: str) -> str:
    normalized = split.lower()
    if normalized == "eval":
        return "valid"
    if normalized not in ("train", "valid", "test"):
        raise ValueError("Expected split to be 'train', 'valid', 'eval' or 'test'.")
    return normalized

def _load_class_names(root_dir: Path) -> list[str]:
    yaml_path = root_dir / "data.yaml"
    with yaml_path.open("r", encoding = "utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    names = payload.get("names")
    nc = payload.get("nc")

    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Expected a list of class names in {yaml_path}.")
    
    if nc is not None and int(nc) != len(names):
        raise ValueError(f"data.yaml says nc = {nc}, but names has {len(names)} entries.")
    
    return names

def _build_samples(
        root_dir: Path,
        split: str,
        class_names: list[str],
        crop_margin: float,
) -> list[LabApparatusSample]:
    images_dir = root_dir / split / "images"
    labels_dir = root_dir/ split / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels diretory: {labels_dir}")
    
    samples = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _find_image_path(images_dir, label_path.stem)

        with Image.open(image_path) as image:
            image_width, image_height = image.size
        
        object_index = 0
        for line_number, line in enumerate(
            label_path.read_text(encoding = "utf-8").splitlines(),
            start = 1,
        ):
            stripped = line.strip()
            if not stripped:
                continue

            class_id, bbox_normalized = _parse_yolo_line(
                stripped,
                label_path = label_path,
                line_number = line_number,
            )

            if not 0 <= class_id < len(class_names):
                raise ValueError(
                    f"Class id {class_id} out of range in {label_path}:{line_number}."
                )
            
            bbox_pixels = _yolo_to_pixel_bbox(
                bbox_normalized = bbox_normalized,
                image_width = image_width,
                image_height = image_height,
                margin = crop_margin,
            )

            samples.append(
                LabApparatusSample(
                    image_path = image_path,
                    label_path = label_path,
                    split = split,
                    object_index=object_index,
                    class_id = class_id,
                    class_name = class_names[class_id],
                    bbox_normalized = bbox_normalized,
                    bbox_pixels = bbox_pixels,
                )
            )
            object_index += 1
        
    return samples

def _find_image_path(images_dir: Path, stem: str) -> Path:
    for extension in _IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{extension}"
        if image_path.is_file():
            return image_path
    
    matches = [
        path for path in images_dir.glob(f"{stem}.*")
        if path.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if matches:
        return sorted(matches)[0]
    
    raise FileNotFoundError(f"No image found for label stem '{stem}' in {images_dir}.")

def _parse_yolo_line(
        line: str,
        *,
        label_path: Path,
        line_number: int,
) -> tuple[int, tuple[float, float, float, float]]:
    parts = line.split()

    if len(parts) < 5:
        raise ValueError(
            f"Expected at least 5 YOLO fields in {label_path}:{line_number}, got {len(parts)}."
        )
    
    class_id = int(parts[0])

    if len(parts) == 5:
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    else:
        coordinates = [float(value) for value in parts[1:]]
        if len(coordinates) % 2 != 0:
            raise ValueError(
                f"Expected polygon coordinates in x/y pairs in {label_path}:{line_number}."
            )

        xs = coordinates[0::2]
        ys = coordinates[1::2]

        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)

        width = right - left
        height = bottom - top
        x_center = left + width / 2
        y_center = top + height / 2

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox size in {label_path}:{line_number}.")
    
    return class_id, (x_center, y_center, width, height)

def _yolo_to_pixel_bbox(
        *,
        bbox_normalized: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
        margin: float,
) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = bbox_normalized

    box_width = width * image_width
    box_height = height * image_height

    left = x_center * image_width - box_width / 2
    top = y_center * image_height - box_height / 2
    right = x_center * image_width + box_width / 2
    bottom = y_center * image_height + box_height / 2

    margin_x = box_width * margin
    margin_y = box_height * margin

    left = max(0, math.floor(left - margin_x))
    top = max(0, math.floor(top - margin_y))
    right = min(image_width, math.ceil(right + margin_x))
    bottom = min(image_height, math.ceil(bottom + margin_y))

    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)

    return left, top, right, bottom
