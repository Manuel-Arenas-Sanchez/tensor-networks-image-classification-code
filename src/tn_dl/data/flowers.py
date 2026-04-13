from __future__ import annotations

import os

import scipy
import torch.utils.data
import torchvision
from PIL import Image

from tn_dl.data import DataLoaders


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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, '102flowers', 'jpg')

        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))

        self.labels = labels_mat['labels'][0] - 1
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_name = f'image_{index+1:05d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        image=Image.open(img_path).convert("RGB")
        label=self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, label

def build_flowers_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256), #Recorta el eje mas corto a 256 pixeles
        torchvision.transforms.CenterCrop(224), #Recorta de manera igual en ambos ejes para que las imagenes queden como 224x224 pixeles
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
    ])

def build_flowers_dataset(data_config):
    transform = build_flowers_transform()
    dataset = OxfordFlowersDataset(
        root_dir=data_config.data_dir,
        transform=transform,
    )
    return dataset

def build_flowers_dataloaders(data_config):
    dataset = build_flowers_dataset(data_config)

    train_size = int(0.7*len(dataset))
    eval_size = int(0.15*len(dataset))
    test_size = len(dataset) - train_size - eval_size

    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=data_config.num_workers)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.num_workers)

    return DataLoaders(train_loader=train_loader, eval_loader=eval_loader, test_loader=test_loader)
