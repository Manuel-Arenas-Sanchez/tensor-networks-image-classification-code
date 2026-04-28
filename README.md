# Tensor Network-Based Machine Learning - Flowers Benchmark

Bachelor's Thesis (TFG) project exploring tensor-network-based machine learning
for image classification.

The current benchmark is Oxford Flowers 102. The long-term goal is to use this
benchmark to validate the pipeline, architectures and methodology before moving
to domain-specific laboratory imaging data.

## Current Status

The repository contains an end-to-end training pipeline for Flowers 102 with:

- a reference Matrix Product State (MPS) baseline based on TensorKrowch;
- a custom hierarchical Tree Tensor Network (TTN);
- a hybrid TTN implementation with PyTorch transformations between tensor
  network levels;
- an AIM-based spatial front-end inspired by the DTTN paper;
- YAML-driven experiments, checkpointing, metrics CSVs and loss-curve plots.

The most important current result is that the custom TTN is no longer merely a
synthetic prototype: it trains end-to-end on Flowers 102. A longer run showed
that it learns the training set but overfits, so the current research focus is
improving generalization.

## Goals

- Build a reproducible image-classification pipeline driven by YAML configs.
- Compare tensor-network models under the same data and training conditions.
- Use a stable MPS baseline as a reference.
- Develop and analyze a custom TTN architecture.
- Improve the TTN with spatial inductive bias before flattening images into
  tensor-network sequences.
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

The project now includes an AIM front-end inspired by:

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
- [x] Planned integration for `embedding_name: aim_patch`.
- [x] Learning-rate scheduler design with `ReduceLROnPlateau`.
- [ ] Smoke run for AIM + TTN + scheduler.
- [ ] Full AIM + TTN training run.
- [ ] Data augmentation for Flowers 102.
- [ ] Comparison against the MPS baseline.
- [ ] Migration to laboratory imaging datasets.

## Project Structure

```text
src/tn_dl/
|-- cli/            # train, eval and SLURM entry points
|-- config/         # dataclass schema and YAML loader
|-- data/           # Oxford Flowers dataset and dataloaders
|-- embeddings/     # TrigPixelEmbedding, PatchEmbedding, AIMPatchEmbedding
|-- layers/         # MPS baseline, AIM2D and custom TTN layers
|-- models/         # ImageClassifier and build_model factory
|-- training/       # training loop, evaluation, checkpoints, loss tracking
`-- utils/          # device, I/O and seeding helpers

configs/
|-- experiments/    # experiment YAML files
`-- runtime/        # local and SLURM runtime YAML files
```

## Pipeline

1. Data loading: Oxford Flowers 102 images are resized/cropped to 224 x 224,
   converted to tensors and normalized.
2. Embedding: each image becomes a sequence `[B, sequence_length, local_dim]`.
3. Tensor network layer: the embedded sequence is processed by either the MPS
   baseline or the custom TTN.
4. Classification: the final representation is mapped to 102 flower classes.
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

## Notes

The current experimental question is no longer "does the custom TTN run?" but
"how can the custom TTN generalize better?". The AIM front-end, learning-rate
scheduler, data augmentation and controlled comparisons against MPS are the next
steps in answering that question.
