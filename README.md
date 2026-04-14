# Tensor Network-Based Machine Learning — Flowers Benchmark

Bachelor's Thesis (TFG) project exploring **Tensor Networks** as an alternative to classical deep learning models for image classification.

The **Oxford Flowers 102** dataset is used as a preliminary benchmark before migrating to domain-specific laboratory imaging data.

> **Status:** active development. This repository evolves from a PyTorch/TensorKrowch template into a custom tensor-network classifier built by the author.

---

## Goals

- Build an end-to-end image classification pipeline driven by YAML configs.
- Experiment with different **image embeddings** feeding into a tensor-network model.
- Replace the reference MPS layer with a **custom tensor network architecture** designed from scratch.
- Compare tensor-network approaches against classical deep-learning baselines.

---

## Current Progress

- [x] Data pipeline for Oxford Flowers 102 (`OxfordFlowersDataset`, transforms, train/eval/test split).
- [x] Config system (YAML → typed dataclasses).
- [x] Reference MPS layer wired through `tensorkrowch` (provisional, will be replaced).
- [x] `TrigPixelEmbedding` (per-pixel cos/sin embedding).
- [x] `PatchEmbedding` implemented and integrated into the model factory.
- [ ] Custom tensor-network layer (in design).
- [ ] Full training & evaluation sweep on Flowers 102.
- [ ] Migration to laboratory imaging datasets.

---

## Project Structure

```
src/tn_dl/
├── cli/            # Entry points: train, eval, submit_slurm
├── config/         # Dataclass schema + YAML loader
├── data/           # OxfordFlowersDataset + DataLoader factory
├── embeddings/     # TrigPixelEmbedding, PatchEmbedding
├── layers/         # MPSClassifier (provisional), tensor-network base class
├── models/         # ImageClassifier + build_model factory
├── training/       # Training loop, evaluation, checkpointing
└── utils/          # Device, I/O, seeding helpers

configs/
├── experiments/    # Per-experiment YAML (flowers_mps.yaml, ...)
└── runtime/        # Runtime YAML (local.yaml, slurm.yaml)

tests/              # Pytest suite
scripts/            # Local + SLURM launch scripts
```

---

## Pipeline Overview

1. **Data loading** — `OxfordFlowersDataset` reads `imagelabels.mat` and the raw JPG files. Images are resized to 256, center-cropped to 224 × 224, converted to tensors and normalized with ImageNet statistics.
2. **Split** — dataset randomly divided into 70 % train, 15 % eval, 15 % test.
3. **Embedding** — each image is mapped into a sequence of local feature vectors with shape `[B, sequence_length, local_dim]`. Two embeddings are currently available:
   - `TrigPixelEmbedding`: one site per pixel-channel, using `(cos θ, sin θ)` with `θ = π · pixel / 2`.
   - `PatchEmbedding`: splits the image into non-overlapping patches of size `patch_size × patch_size` and projects each flattened patch to `local_dim` with a linear layer.
4. **Tensor network layer** — consumes the embedded sequence and produces class logits. Currently a Matrix Product State (`tensorkrowch.models.MPSLayer`), soon to be replaced by a custom architecture.
5. **Training** — AdamW optimizer, cross-entropy loss, best-checkpoint saving by validation accuracy. Per-epoch metrics are written to CSV.

---

## Embedding ↔ Tensor Network Contract

The tensor-network layer is agnostic to the embedding choice. The contract is:

```
embedding(image)  →  tensor  [B, sequence_length, local_dim]
```

`sequence_length` is the number of sites of the tensor network and depends on the embedding:

| Embedding             | `sequence_length`                  |
|-----------------------|------------------------------------|
| `TrigPixelEmbedding`  | `C · H · W`                        |
| `PatchEmbedding`      | `(H / patch_size) · (W / patch_size)` |

This keeps the tensor-network layer decoupled from the image representation and allows plugging new embeddings or new TN architectures without touching the training engine.

---

## Running an Experiment

Install the package (Python 3.11 or 3.12):

```bash
pip install -e .
```

Train with a YAML experiment config:

```bash
python -m tn_dl.cli.train --config configs/experiments/flowers_mps.yaml
```

Evaluate a checkpoint:

```bash
python -m tn_dl.cli.eval \
  --config configs/experiments/flowers_mps.yaml \
  --checkpoint runs/<timestamp>-flowers_mps/best.pt
```

Runtime overrides (device, output directory, runtime YAML, resume checkpoint) are available as CLI flags.

---

## Configuration Example

`configs/experiments/flowers_mps.yaml` (patch embedding):

```yaml
name: flowers_mps
seed: 7
device: auto

data:
  dataset: flowers
  data_dir: ./flower_data
  batch_size: 32
  num_workers: 0
  download: false

model:
  embedding_name: patch
  layer_name: mps
  input_shape: [3, 224, 224]
  local_dim: 2
  bond_dim: 8
  n_classes: 102
  patch_size: 16

training:
  epochs: 10
  learning_rate: 0.0005
  weight_decay: 0.0
  log_every_n_steps: 10

io:
  root_dir: runs
  checkpoint_name: best.pt
  metrics_name: metrics.csv
```

With `patch_size=16` and `input_shape=[3,224,224]`, the resulting sequence has `14 · 14 = 196` sites.

---

## Dataset

The Oxford Flowers 102 dataset must be placed under `data_dir` with the following layout:

```
flower_data/
├── 102flowers/jpg/image_00001.jpg ... image_08189.jpg
└── imagelabels.mat
```

Available at <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>.

---

## Tech Stack

- Python 3.11 / 3.12
- PyTorch ≥ 2.8, TorchVision ≥ 0.26
- [TensorKrowch](https://github.com/joserapa98/tensorkrowch) ≥ 1.1.6 (reference MPS)
- PyYAML for configuration
- pytest + ruff (dev dependencies)

---

## Author

**Manuel Arenas Sánchez**
BSc Physics — Final Year Project (TFG)
Universidad Europea de Madrid · Internship at ITCL

---

## Notes

This repository is continuously updated as the project evolves. The current focus is finishing the embedding stage before designing and implementing the custom tensor-network classifier.
