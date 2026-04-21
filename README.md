# Tensor Network-Based Machine Learning — Flowers Benchmark

Bachelor's Thesis (TFG) project exploring **Tensor Networks** as an alternative to classical deep learning models for image classification.

The **Oxford Flowers 102** dataset is used as a preliminary benchmark before migrating to domain-specific laboratory imaging data.

> **Status:** active development. The reference MPS baseline and the custom hierarchical TTN (`CustomTTN`) both run end-to-end on Flowers 102 via the same training engine. Current focus: longer training runs on `CustomTTN` and a like-for-like benchmark against the MPS baseline.

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
- [x] `CustomTTN`: hierarchical 3-to-1 TTN built from 3-node MPS blocks, implemented on top of `tensorkrowch.TensorNetwork` in `src/tn_dl/layers/mps_custom.py`.
- [x] Structural padding (non-trainable `[1, 0, ..., 0]` nodes) for sequence lengths that are not a power of three.
- [x] Zigzag (boustrophedon) linearization available in both `TrigPixelEmbedding` and `PatchEmbedding` via an `order` flag, surfaced through `ModelConfig.linearization_order`.
- [x] `build_model(...)` routes `layer_name: custom_ttn` to the custom TTN wrapper (`MPSClassifier_1`).
- [x] CustomTTN forward + backward validated on synthetic batches for `L ∈ {9, 27, 196}`, with every parameter receiving gradients.
- [x] End-to-end training run with `layer_name: custom_ttn` on Flowers 102 (PatchEmbedding, `patch_size=16`, `L=196`) validated on a 2-epoch smoke test.
- [x] Automatic download of Oxford Flowers 102 with two-layer idempotency (skips entirely if the dataset is already present, only fetches missing pieces otherwise).
- [x] Loss tracking utility that writes `metrics.csv` and `loss_curve.png` for train/eval loss monitoring across epochs.
- [ ] Long training run on `CustomTTN` (`epochs=20`, `bond_dim=8`) for a real convergence curve.
- [ ] Benchmark `CustomTTN` vs. reference MPS baseline on Flowers 102.
- [ ] Scale to the raw `TrigPixelEmbedding` regime (`L = C·H·W = 150 528`): evaluate migrating the contraction loop to the `op.stack` / `op.unbind` pattern used by `tensorkrowch.models.Tree`.
- [ ] Migration to laboratory imaging datasets.

---

## Project Structure

```
src/tn_dl/
├── cli/            # Entry points: train, eval, submit_slurm
├── config/         # Dataclass schema + YAML loader
├── data/           # OxfordFlowersDataset + DataLoader factory
├── embeddings/     # TrigPixelEmbedding, PatchEmbedding
├── layers/         # Reference MPS layer + custom TTN prototype + TN base class
├── models/         # ImageClassifier + build_model factory
├── training/       # Training loop, evaluation, checkpointing, loss tracking
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
4. **Tensor network layer** — consumes the embedded sequence and produces class logits. Two backends are wired into the factory:
   - `layer_name: mps` → reference Matrix Product State (`tensorkrowch.models.MPSLayer`), the default baseline.
   - `layer_name: custom_ttn` → `CustomTTN`, a hierarchical 3-to-1 tree tensor network built from 3-node MPS blocks.
5. **Training** — AdamW optimizer, cross-entropy loss, best-checkpoint saving by validation accuracy. Per-epoch metrics are written to CSV and train/eval loss curves are exported as `loss_curve.png`.

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

## Linearization Order (`row` vs. `zigzag`)

Flattening a 2D image to a 1D sequence breaks spatial proximity between adjacent rows. To partially recover it, both embeddings accept an `order` flag, surfaced at the config level as `model.linearization_order`:

- `row` (default for new MPS-style experiments): standard row-major flattening.
- `zigzag` (boustrophedon): alternating rows are reversed so that consecutive sites in the sequence also correspond to spatially adjacent pixels at row boundaries. This is the common baseline for MPS-based image classifiers.

Implementation:

- `PatchEmbedding` applies zigzag over the `(H/patch, W/patch)` patch grid.
- `TrigPixelEmbedding` applies zigzag per-channel (variant *a*): each channel is zigzagged spatially and the three channels are concatenated in their natural order — `perm[c·H·W + i] = c·H·W + zigzag_hw[i]`.

The tensor network itself remains agnostic to the order; it only sees the permuted `[B, sequence_length, local_dim]` tensor.

---

## Custom TTN (`CustomTTN`)

The repository implements a custom hierarchical tree tensor network in `src/tn_dl/layers/mps_custom.py`. The file exposes two classes:

- `CustomTTN(tk.TensorNetwork)` — the tensor network itself. Holds the parametric nodes of every level, the structural padding nodes, and implements `set_data_nodes` / `contract`.
- `MPSClassifier_1(BaseTensorNetworkLayer)` — thin PyTorch wrapper. Runs the `CustomTTN` on the input batch and applies a final `nn.Linear(bond_dim, n_classes)` outside of the tensor network.

### Architecture

- Base block: a 3-node MPS (`left`, `center`, `right`) registered as `tk.ParamNode` instances so their tensors are materialised at construction and show up as `nn.Parameter`s of the parent network.
  - `left : (bond, input)`, axes `(right, input)`
  - `center : (bond, bond, input, output)`, axes `(left, right, input, output)`
  - `right : (bond, input)`, axes `(left, input)`
  - Internal bonds: `left[right] ^ center[left]` and `center[right] ^ right[left]`. The free `output` edge lives on the `center` node and is what feeds the next level (or becomes the root).
  - Initialisation: `init_method="randn"`, `std=0.02`, matching the reference MPS layer.
- Levels are built bottom-up until a single root block remains:
  - Level 0: `input_dim = local_dim`.
  - Levels `k > 0`: `input_dim = bond_dim` (they consume the `output` edges of the previous level).
- Final classification happens outside the TN, in the `MPSClassifier_1` wrapper, to keep the contraction logic simple while the prototype stabilises.

### Structural padding (option B)

When the number of sites at a given level is not a multiple of three, the last block of the next level needs fewer than 3 real inputs. Instead of padding data or supporting variable-arity blocks, `CustomTTN` inserts non-trainable `tk.Node`s with a fixed `[1, 0, ..., 0]` tensor on the surplus input edges. These nodes are registered per level in `self.padding_nodes_by_level` and do not appear in `model.parameters()`, so they contribute no gradients.

Example for `sequence_length = 196` (PatchEmbedding with `patch_size=16`):

| Level | Sites | Blocks | Capacity | Padding nodes |
|-------|-------|--------|----------|---------------|
| 0     | 196   | 66     | 198      | 2             |
| 1     | 66    | 22     | 66       | 0             |
| 2     | 22    | 8      | 24       | 2             |
| 3     | 8     | 3      | 9        | 1             |
| 4     | 3     | 1      | 3        | 0 (root)      |

### Contraction

`contract()` runs bottom-up. At level 0 the partners of each block's three `input` edges are discovered through `neighbours("input")` (they are the data nodes created by `set_data_nodes` plus the padding nodes connected in `_connect_levels`). At levels `k > 0`, `contract()` passes the partners — the composites produced by level `k − 1` plus the padding nodes of level `k` — **explicitly** to `_contract_block`, because `@` creates a new composite node in TensorKrowch without rerouting the `node1`/`node2` pointers of the shared edges to it. The composites do inherit the shared edge reference, so `@` still detects and contracts it correctly.

### Validation status

Synthetic batch driver in `debug_ttn.py` confirms, for `bond_dim=4`, `local_dim=2`, `n_classes=3`, `batch=2`:

- Forward + backward green for `L ∈ {9, 27, 196}`.
- Every trainable parameter receives gradients (`14` params for `L=9`, `41` for `L=27`, `302` for `L=196`).
- Padding kicks in at multiple levels for `L = 196`, as tabulated above.

---

## Running an Experiment

Install the package (Python 3.11 or 3.12):

```bash
pip install -e .
```

Train with a YAML experiment config. Two experiments are provided out of the box:

```bash
# Reference MPS baseline
python -m tn_dl.cli.train --config configs/experiments/flowers_mps.yaml

# Custom hierarchical TTN
python -m tn_dl.cli.train --config configs/experiments/flowers_custom_ttn.yaml
```

Evaluate a checkpoint:

```bash
python -m tn_dl.cli.eval \
  --config configs/experiments/flowers_mps.yaml \
  --checkpoint runs/<timestamp>-flowers_mps/best.pt
```

Runtime overrides (device, output directory, runtime YAML, resume checkpoint) are available as CLI flags.

Each run directory contains:

- `best.pt`: best checkpoint by validation accuracy.
- `metrics.csv`: epoch-level train loss, eval loss and eval accuracy.
- `loss_curve.png`: train/eval loss plot generated after each epoch.
- `config.yaml`: resolved experiment configuration.

---

## Configuration Example

`configs/experiments/flowers_mps.yaml` (patch embedding, reference MPS). An analogous `configs/experiments/flowers_custom_ttn.yaml` ships with `layer_name: custom_ttn` for the hierarchical TTN backend.

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
  layer_name: mps            # or "custom_ttn" to use CustomTTN
  input_shape: [3, 224, 224]
  local_dim: 2
  bond_dim: 8
  n_classes: 102
  patch_size: 16
  linearization_order: zigzag  # "row" or "zigzag"

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

When `data.download: true` is set in the YAML (the default for `flowers_custom_ttn.yaml`), the Oxford Flowers 102 dataset is downloaded automatically on first run into `data_dir`. The download is idempotent: if the dataset is already present it is skipped entirely, and if only part is missing (e.g. `imagelabels.mat` exists but images are gone) only the missing pieces are fetched.

Alternatively, the dataset can be placed manually under `data_dir` with the following layout:

```
flower_data/
├── 102flowers/jpg/image_00001.jpg ... image_08189.jpg
└── imagelabels.mat
```

Source: <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>.

---

## Tech Stack

- Python 3.11 / 3.12
- PyTorch ≥ 2.8, TorchVision ≥ 0.26 (Windows / Linux)
- [TensorKrowch](https://github.com/joserapa98/tensorkrowch) ≥ 1.1.6 (reference MPS + `CustomTTN` backbone)
- PyYAML for configuration
- Matplotlib for loss-curve visualization
- pytest + ruff (dev dependencies)

> **macOS Intel note:** PyTorch stopped publishing x86_64 wheels after 2.2.2. On Intel Macs, `pyproject.toml` cannot be satisfied directly; install the pinned stack manually (`torch==2.2.2`, `torchvision==0.17.2`, `tensorkrowch==1.1.6`, `numpy<2`, `scipy`, `PyYAML`, `pytest`, `ruff`) and then `pip install -e . --no-deps`. `scipy` is required at runtime by `OxfordFlowersDataset` to read `imagelabels.mat` and is easy to miss when using `--no-deps`. The project code is kept compatible with both torch 2.2.2 and torch 2.8.

---

## Author

**Manuel Arenas Sánchez**
BSc Physics — Final Year Project (TFG)
Universidad Europea de Madrid · Internship at ITCL

---

## Notes

This repository is continuously updated as the project evolves. The first end-to-end training run with `layer_name: custom_ttn` on Flowers 102 is now green, and each training run now exports both tabular metrics and a visual train/eval loss curve for easier TFG analysis. The current focus is launching a longer `CustomTTN` run (`epochs=20`, `bond_dim=8`) for a real convergence curve and benchmarking it against the reference MPS baseline, followed by scaling the contraction loop to handle the raw `TrigPixelEmbedding` regime (`L = 150 528`).
