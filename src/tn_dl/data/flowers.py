from __future__ import annotations

import logging
import os
import tarfile
import urllib.request
from pathlib import Path

import scipy
import torch
import torch.utils.data
import torchvision
from PIL import Image

from tn_dl.data import DataLoaders

logger = logging.getLogger(__name__)

class OxfordFlowersDataset(torch.utils.data.Dataset):
    """
    Necesito responder tres preguntas con estos tres `def`.

    1. ¿Donde encuentro los datos?
        def __init__(self, root_dir):
        
    2. ¿Cuantas muestras en total?
        def __len__(self):
        
    3. ¿Como puedo coger las imágenes y su etiqueta correspondiente?
        def __getitem__(self, idx)
    """
    def __init__(self, root_dir, transform=None, indices = None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, '102flowers', 'jpg')

        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))

        self.labels = labels_mat['labels'][0] - 1
        self.transform = transform
        self.indices = list(range(len(self.labels))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        real_index = self.indices[index]
        img_name = f'image_{real_index+1:05d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[real_index])

        if self.transform:
            image = self.transform(image)
        return image, label

_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

def _flowers_is_present(root_dir) -> bool:
    root = Path(root_dir)
    jpg_dir = root/"102flowers"/"jpg"
    return(
        (root/"imagelabels.mat").is_file()
        and jpg_dir.is_dir()
        and any(jpg_dir.glob("image_*.jpg"))
    )

def download_flowers(root_dir) -> None:
    root = Path(root_dir)
    root.mkdir(parents = True, exist_ok = True)

    labels_path = root/"imagelabels.mat"
    if not labels_path.is_file():
        print(f"[flowers] descargando labels -> {labels_path}")
        urllib.request.urlretrieve(_LABELS_URL, labels_path)

    jpg_dir = root/"102flowers"/"jpg"
    if not jpg_dir.is_dir() or not any(jpg_dir.glob("image_*.jpg")):
        tgz_path = root/"102flowers.tgz"
        if not tgz_path.is_file():
            print(f"[flowers] descargando imágenes -> {tgz_path}")
            urllib.request.urlretrieve(_IMAGES_URL, tgz_path)
        print(f"[flowers] extrayendo {tgz_path.name} en {root}")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(root)
        
        extracted = root/"jpg"
        if extracted.is_dir() and not jpg_dir.is_dir():
            jpg_dir.parent.mkdir(parents = True, exist_ok = True)
            extracted.rename(jpg_dir)
            
        tgz_path.unlink(missing_ok = True)

def build_flowers_transform(train: bool = False):
    normalize = torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )

    if train:
        return torchvision.transforms.Compose([
            # Cambia el encuadre sin destruir mucho la flor principal.
            torchvision.transforms.RandomResizedCrop(
                224,
                scale = (0.70, 1.0),
                ratio = (0.85, 1.15),
                interpolation = torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p = 0.5),

            # Variación suave de iluminación/color
            torchvision.transforms.ColorJitter(
                brightness = 0.20,
                contrast = 0.20,
                saturation = 0.15,
                hue = 0.03,
            ),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

def build_flowers_dataset(data_config, train: bool = False, indices = None):
    if getattr(data_config, "download", False) and not _flowers_is_present(data_config.data_dir):
        download_flowers(data_config.data_dir)
    
    transform = build_flowers_transform(train = train)
    dataset = OxfordFlowersDataset(
        root_dir=data_config.data_dir,
        transform=transform,
        indices = indices,
    )
    logger.info("Dataset Flowers cargado correctamente: %s imágenes", len(dataset))
    return dataset

def build_flowers_dataloaders(data_config):
    base_dataset = build_flowers_dataset(data_config, train = False)

    train_size = int(0.7*len(base_dataset))
    eval_size = int(0.15*len(base_dataset))
    test_size = len(base_dataset) - train_size - eval_size

    permutation = torch.randperm(len(base_dataset)).tolist()
    train_indices = permutation[:train_size]
    eval_indices = permutation[train_size:train_size + eval_size]
    test_indices = permutation[train_size + eval_size:]

    train_dataset = build_flowers_dataset(data_config, train = True, indices = train_indices)
    eval_dataset = build_flowers_dataset(data_config, train = False, indices = eval_indices)
    test_dataset = build_flowers_dataset(data_config, train = False, indices = test_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    logger.info(
        "Split Flowers creado: train = %s, eval = %s, test = %s, batch_size = %s",
        train_size,
        eval_size,
        test_size,
        data_config.batch_size,
    )
    return DataLoaders(train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader)
