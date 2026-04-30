from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class DataLoaders:
    train_loader: DataLoader
    eval_loader: DataLoader
    test_loader: DataLoader | None=None

def build_dataloaders(data_config):
    dataset_name = data_config.dataset.lower()

    if dataset_name == "flowers":
        from tn_dl.data import flowers
        return flowers.build_flowers_dataloaders(data_config)

    if dataset_name == "lab_apparatus_crop":
        from tn_dl.data import lab_apparatus
        return lab_apparatus.build_lab_apparatus_dataloaders(data_config)
    
    raise ValueError(f"Unsupported dataset '{data_config.dataset}'.")
