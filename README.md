# Tensor Network-Based Machine Learning - Image Classification Benchmarks

Bachelor's Thesis (TFG) project exploring tensor-network-based machine learning
for image classification.

The project started with Oxford Flowers 102 as a preliminary benchmark. The
current main dataset is Chemistry Lab Apparatus / ChemEq25, introduced by
Hossain et al. (2025), converted from an object-detection dataset into an
instance-classification problem by cropping each annotated laboratory object.

## Current Status

The repository contains an end-to-end training pipeline with:

- a reference Matrix Product State (MPS) baseline based on TensorKrowch;
- a custom hierarchical Tree Tensor Network (TTN);
- a hybrid TTN implementation with PyTorch transformations between tensor
  network levels;
- an AIM-based spatial front-end inspired by the DTTN paper;
- YAML-driven experiments, checkpointing, metrics CSVs and loss-curve plots;
- dataset audits for both Flowers 102 and ChemEq25 crops.

Flowers 102 remains as a useful preliminary benchmark. ChemEq25 is now the main
TFG dataset: the model classifies object crops generated from YOLO annotations,
using AIM + HybridCustomTTN as the primary architecture.

## Goals

- Build a reproducible image-classification pipeline driven by YAML configs.
- Compare tensor-network models under the same data and training conditions.
- Use a stable MPS baseline as a reference.
- Develop and analyze a custom TTN architecture.
- Improve the TTN with spatial inductive bias before flattening images into
  tensor-network sequences.
- Evaluate tensor-network architectures on real laboratory apparatus images.
- Produce training curves and experimental evidence suitable for the TFG memory.

## Main Ideas

### Baseline MPS

The baseline route is:

```text
image -> embedding -> sequence [B, L, d] -> MPS -> logits
```

This model is used as a stable reference. It should remain comparable with the
custom TTN experiments.

### Hybrid Custom TTN

The original custom TTN was a monolithic TensorKrowch network. It worked
technically, but early training curves showed weak learning.

The current version is hybrid:

```text
sequence [B, L, local_dim]
-> TTNLevel
-> LayerNorm + Linear + ReLU
-> TTNLevel
-> ...
-> root vector [B, bond_dim]
-> Linear(bond_dim, n_classes)
```

`TTNLevel` performs one level of tensor-network contraction with TensorKrowch.
`HybridCustomTTN` orchestrates several independent levels and applies PyTorch
modules between them. This avoids modifying TensorKrowch resultant nodes and
keeps the model trainable.

### AIM Spatial Front-End

The project now includes an AIM front-end inspired by Nie et al. (2025):

> Deep Tree Tensor Networks for Image Recognition

The AIM route is:

```text
image [B, 3, 224, 224]
-> CNN stem
-> AIM2D blocks
-> Conv2d 1x1 to local_dim
-> flatten + zigzag
-> sequence [B, 196, local_dim]
-> HybridCustomTTN
-> logits
```

This keeps spatial structure before the image is flattened into a sequence.
AIM2D combines two linear branches in opposite order and multiplies them with a
Hadamard product. This produces multiplicative interactions that are natural
from a tensor-network point of view while also adding convolutional parameter
sharing.

## Current Progress

- [x] Oxford Flowers 102 data pipeline.
- [x] ChemEq25 / Lab Apparatus crop dataset for instance classification.
- [x] YAML configuration system.
- [x] Reference MPS baseline through TensorKrowch.
- [x] `TrigPixelEmbedding`.
- [x] `PatchEmbedding`.
- [x] Zigzag linearization for image-to-sequence conversion.
- [x] Custom TTN based on 3-node MPS blocks.
- [x] Structural padding with non-trainable neutral nodes.
- [x] Hybrid TTN with `LayerNorm -> Linear -> ReLU` between levels.
- [x] Synthetic forward/backward validation for `L = 9`, `L = 27` and
  `L = 196`.
- [x] End-to-end Flowers 102 training with `layer_name: custom_ttn`.
- [x] Logging during training.
- [x] `LossTracker` writing `metrics.csv` and `loss_curve.png`.
- [x] AIM2D layer.
- [x] AIMPatchEmbedding.
- [x] Integration for `embedding_name: aim_patch`.
- [x] Learning-rate scheduler with `ReduceLROnPlateau`.
- [x] Data augmentation and richer metrics for Flowers 102.
- [x] Classification diagnostics and data audits.
- [x] ChemEq25-specific audit with crop samples and bbox overlays.
- [x] Smoke checks for ChemEq25 dataloaders and AIM + TTN forward/backward.
- [ ] Full AIM + TTN training run on ChemEq25 crops.
- [ ] Comparison against the MPS baseline.
- [ ] ChemEq25 MPS and TTN baseline comparisons.

## Project Structure

```text
src/tn_dl/
|-- cli/            # train, eval and SLURM entry points
|-- config/         # dataclass schema and YAML loader
|-- data/           # Flowers and Lab Apparatus datasets/dataloaders
|-- embeddings/     # TrigPixelEmbedding, PatchEmbedding, AIMPatchEmbedding
|-- layers/         # MPS baseline, AIM2D and custom TTN layers
|-- models/         # ImageClassifier and build_model factory
|-- training/       # training loop, evaluation, checkpoints, loss tracking
|-- evaluation/     # diagnostics, data audits and Lab Apparatus audit
`-- utils/          # device, I/O and seeding helpers

configs/
|-- experiments/    # experiment YAML files
`-- runtime/        # local and SLURM runtime YAML files
```

## Pipeline

1. Data loading: images or object crops are resized to 224 x 224, converted to
   tensors and normalized.
2. Embedding: each image becomes a sequence `[B, sequence_length, local_dim]`.
3. Tensor network layer: the embedded sequence is processed by either the MPS
   baseline or the custom TTN.
4. Classification: the final representation is mapped to the dataset classes.
5. Training: AdamW, cross-entropy loss, checkpointing, CSV metrics and loss
   plots.

## Embeddings

| Embedding | Config name | Output sequence length |
|-----------|-------------|------------------------|
| TrigPixelEmbedding | `trig` | `C * H * W` |
| PatchEmbedding | `patch` | `(H / patch_size) * (W / patch_size)` |
| AIMPatchEmbedding | `aim_patch` | `(H / patch_size) * (W / patch_size)` |

All tensor-network layers consume the same contract:

```text
embedding(image) -> [B, sequence_length, local_dim]
```

This keeps the tensor-network code independent from the image representation.

## Linearization Order

The model supports:

- `row`: standard row-major flattening;
- `zigzag`: alternating rows are reversed before flattening.

Zigzag keeps consecutive sequence positions closer to spatial neighbors at row
boundaries. The tensor network itself remains agnostic to the order; it only
receives the final sequence.

## AIM2D

`AIM2D` lives in:

```text
src/tn_dl/layers/aim.py
```

It expects:

```text
[B, C, H, W]
```

and returns the same shape.

Conceptually:

```text
f1 = Linear(C -> C*R) + depthwise Conv2d
f2 = grouped Conv2d + Linear(C*R -> C*R)
interaction = f1(x) * f2(x)
out = x + scale * Conv2d_1x1(interaction)
```

The product `f1(x) * f2(x)` is the key point: if both branches are linear in
the input, their Hadamard product is bilinear. This gives AIM a natural
connection with polynomial and tensor-network representations.

## AIMPatchEmbedding

`AIMPatchEmbedding` lives in:

```text
src/tn_dl/embeddings/aim_patch.py
```

For Flowers 102 with 224 x 224 images and `patch_size: 16`:

```text
[B, 3, 224, 224]
-> [B, aim_channels, 14, 14]
-> [B, local_dim, 14, 14]
-> [B, 196, local_dim]
```

Unlike the original `PatchEmbedding`, it does not immediately flatten each
patch. It first processes the image as a spatial feature map, then flattens the
final grid.

## Learning-Rate Scheduler

The planned scheduler is:

```yaml
scheduler_name: reduce_on_plateau
```

This maps to PyTorch's `ReduceLROnPlateau`.

It monitors `eval_loss`. If validation loss stops improving for a fixed number
of epochs, the learning rate is reduced. For example:

```yaml
learning_rate: 0.0005
scheduler_factor: 0.5
scheduler_patience: 5
scheduler_min_lr: 0.000001
```

This means:

```text
0.0005 -> 0.00025 -> 0.000125 -> ...
```

whenever `eval_loss` reaches a plateau, never going below `scheduler_min_lr`.
This is useful here because the main problem is generalization, so the scheduler
reacts to validation behavior instead of training loss alone.

## Example Config: AIM + TTN

Flowers 102:

```yaml
name: flowers_aim_ttn
seed: 7
device: auto

data:
  dataset: flowers
  data_dir: flower_data
  batch_size: 16
  num_workers: 0
  smoke_samples:
  download: true

model:
  embedding_name: aim_patch
  layer_name: custom_ttn
  input_shape: [3, 224, 224]
  local_dim: 8
  bond_dim: 8
  n_classes: 102
  patch_size: 16
  linearization_order: zigzag
  aim_channels: 32
  num_aim_blocks: 2
  aim_expansion_ratio: 3
  aim_kernel_size: 3
  aim_use_ln: true

training:
  epochs: 50
  learning_rate: 0.0005
  weight_decay: 0.001
  log_every_n_steps: 50
  scheduler_name: reduce_on_plateau
  scheduler_factor: 0.5
  scheduler_patience: 5
  scheduler_min_lr: 0.000001

io:
  root_dir: runs
  checkpoint_name: best.pt
  metrics_name: metrics.csv
```

ChemEq25 crop classification:

```yaml
name: ChemEq
seed: 7
device: auto

data:
  dataset: lab_apparatus_crop
  data_dir: ChemistryLabApparatus-25
  batch_size: 16
  num_workers: 0
  smoke_samples:
  download: false

model:
  embedding_name: aim_patch
  layer_name: custom_ttn
  input_shape: [3, 224, 224]
  local_dim: 8
  bond_dim: 8
  n_classes: 25
  patch_size: 16
  linearization_order: zigzag
  aim_channels: 32
  num_aim_blocks: 3
  aim_expansion_ratio: 3
  aim_kernel_size: 3
  aim_use_ln: true

training:
  epochs: 100
  learning_rate: 0.0005
  weight_decay: 0.002
  log_every_n_steps: 100
  scheduler_name: reduce_on_plateau
  scheduler_factor: 0.5
  scheduler_patience: 4
  scheduler_min_lr: 0.000001

io:
  root_dir: runs
  checkpoint_name: best.pt
  metrics_name: metrics.csv
```

## Running

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Train the original custom TTN:

```bash
python -m tn_dl.cli.train --config configs/experiments/flowers_custom_ttn.yaml
```

Train the AIM + TTN experiment:

```bash
python -m tn_dl.cli.train --config configs/experiments/flowers_aim_ttn.yaml
```

Train the ChemEq25 AIM + TTN crop-classification experiment:

```bash
python -m tn_dl.cli.train --config configs/experiments/ChemEq.yaml
```

Evaluate a checkpoint:

```bash
python -m tn_dl.cli.eval \
  --config configs/experiments/flowers_mps.yaml \
  --checkpoint runs/<timestamp>-flowers_mps/best.pt
```

Each run directory contains:

- `best.pt`;
- `metrics.csv`;
- `loss_curve.png`;
- resolved `config.yaml`.

## Data Audits

The audit CLI dispatches automatically from `data.dataset`.

Flowers 102:

```bash
python -m tn_dl.cli.audit \
  --config configs/experiments/flowers_aim_ttn.yaml \
  --output-dir audit_flowers
```

ChemEq25 crops:

```bash
python -m tn_dl.cli.audit \
  --config configs/experiments/ChemEq.yaml \
  --output-dir audit_lab
```

The ChemEq25 audit writes:

- `class_counts_by_split.csv`;
- `object_counts_by_split.csv`;
- `image_object_count_distribution.csv`;
- `bbox_size_distribution.csv`;
- `crop_samples.png`;
- `bbox_overlay_samples.png`.

## Useful Synthetic Checks

HybridCustomTTN:

```bash
python -c "import torch; from tn_dl.layers.mps_custom import MPSClassifier_1; m=MPSClassifier_1(sequence_length=196, local_dim=2, n_classes=102, bond_dim=4); x=torch.randn(2,196,2); y=m(x); print(y.shape); y.sum().backward(); print([n for n,p in m.named_parameters() if p.requires_grad and p.grad is None])"
```

AIMPatchEmbedding:

```bash
python -c "import torch; from tn_dl.embeddings import AIMPatchEmbedding; e=AIMPatchEmbedding(patch_size=16, local_dim=8, in_channels=3, order='zigzag', grid_h=14, grid_w=14); x=torch.randn(2,3,224,224); y=e(x); print(y.shape); y.sum().backward(); print('OK')"
```

Full AIM + TTN forward:

```bash
python -c "import torch; from pathlib import Path; from tn_dl.config.loader import load_experiment_config; from tn_dl.models.factory import build_model; cfg=load_experiment_config(Path('configs/experiments/flowers_aim_ttn.yaml')); m=build_model(cfg.model); x=torch.randn(2,3,224,224); y=m(x); print(y.shape); y.sum().backward(); print([n for n,p in m.named_parameters() if p.requires_grad and p.grad is None])"
```

## Dataset

### Oxford Flowers 102

Oxford Flowers 102 is downloaded automatically when `data.download: true` is
set in the YAML config.

Expected manual layout:

```text
flower_data/
|-- 102flowers/jpg/image_00001.jpg ... image_08189.jpg
`-- imagelabels.mat
```

Dataset source:

<https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>

### Chemistry Lab Apparatus / ChemEq25

ChemEq25 is expected as a local dataset and is not downloaded automatically.
The project uses the official train/valid/test folders and creates one
classification sample per annotated object.

The dataset comes from Hossain et al. (2025), "Real-world chemistry lab image
dataset for equipment recognition across 25 apparatus categories".

Expected manual layout:

```text
ChemistryLabApparatus-25/
|-- data.yaml
|-- Metadata.csv
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
`-- test/
    |-- images/
    `-- labels/
```

YOLO bbox labels and YOLO polygon labels are both supported. Polygon labels are
converted to their enclosing bounding box before cropping.

Validated crop counts:

```text
train = 4863
valid = 1393
test  = 704
```

## Tech Stack

- Python 3.11 / 3.12
- PyTorch
- TorchVision
- TensorKrowch
- PyYAML
- SciPy
- Matplotlib
- pytest and ruff for development

Windows/Linux use the versions defined by `pyproject.toml`.

On Intel macOS, PyTorch must be pinned manually to the last compatible stack:

```bash
torch==2.2.2
torchvision==0.17.2
numpy<2
```

Then install the project with:

```bash
pip install -e . --no-deps
```

## Author

Manuel Arenas Sanchez  
Email: manuelarenas2002@hotmail.es  
GitHub: [Manuel-Arenas-Sanchez](https://github.com/Manuel-Arenas-Sanchez)

Bachelor's Thesis (TFG), BSc Physics.  
Universidad Europea de Madrid. Internship at ITCL.

## References

This project uses ChemEq25 / Chemistry Lab Apparatus as the main laboratory
image dataset:

```bibtex
@article{hossain2025realworld,
  title        = {Real-world chemistry lab image dataset for equipment recognition across 25 apparatus categories},
  author       = {Hossain, Md. Sakhawat and Haque, Md. Sadman and Rahman, Md. Mostafizur and Mousum, Md. Mosaddik Mashrafi and Razzaque, Zobaer Ibn and Robin, Robiul Awoul and Rahman, Raiyan and Noor, Jannatun},
  journal      = {Scientific Data},
  volume       = {12},
  pages        = {1747},
  year         = {2025},
  publisher    = {Nature Portfolio},
  doi          = {10.1038/s41597-025-05952-3}
}
```

The AIM-inspired image embedding used before the tensor-network classifier is
based on the DTTN paper:

```bibtex
@misc{nie2025deeptreetensornetworks,
  title         = {Deep Tree Tensor Networks for Image Recognition},
  author        = {Chang Nie and Junfang Chen and Yajie Chen},
  year          = {2025},
  eprint        = {2502.09928},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2502.09928}
}
```

## Notes

The current experimental question is no longer "does the custom TTN run?" but
"how can the custom TTN generalize better?". The AIM front-end, learning-rate
scheduler, data augmentation and controlled comparisons against MPS are the next
steps in answering that question.
