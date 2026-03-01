# ToxicityGNN

A Graph Neural Network (GNN) framework for predicting **Sites of Metabolism (SOM)** — the specific atoms or bonds within a drug molecule most likely to undergo metabolic transformation. This project represents molecules as molecular graphs and trains GNNs to perform atom-level toxicity and metabolism predictions, with support for integrating molecular property descriptors from QikProp.

---

## Overview

Drug metabolism prediction is a critical task in computational pharmacology. This project applies Graph Neural Networks to model the metabolic behavior of small molecules, specifically targeting:

- **Node-level SOM prediction**: Identifying which atoms in a molecule are most likely to be metabolized
- **CYP enzyme filtering**: Optionally restricting predictions to CYP (cytochrome P450) or non-CYP substrates
- **Multi-featurization**: Combining element-level features, KCF (KEGG Chemical Function) atom types, bond structure, and optional QikProp ADMET descriptors
- **Hyperparameter search**: Automated grid search over GNN depth, width, convolution type, and pooling strategy

---

## Repository Structure

```
toxicitygnn/
├── gnn-node-final.py       # Extended GNN training script with pooling, edge attributes, and QikProp support
├── gnn-node.py             # Core GNN training script (baseline node-level SOM prediction)
├── run-gnns.py             # Parallel GPU job scheduler for grid search over GNN configurations
├── analyze.py              # Evaluation script: computes AUROC, R-Precision, Precision@K, and Top-2 metrics
├── som.models.py           # GNN model definitions for node-level SOM prediction (ChebConv-based)
├── __init__.py             # (empty init)
├── featurizers/
│   ├── sort_pkl.py         # Molecule featurization: element encoding, KCF types, edge construction, dataset splits
│   ├── add_qikprop_features.py  # Pipeline for integrating QikProp ADMET descriptors into dataset
│   ├── original/           # Original featurization utilities
│   └── clintox/            # ClinTox dataset-specific featurization
├── som/
│   └── common.py           # Task runner class (subprocess management) and KEGG rpair parser
└── data/                   # Pickled datasets, prediction outputs, and model checkpoints (not tracked)
```

---

## How It Works

### 1. Molecular Featurization (`featurizers/sort_pkl.py`)

Each molecule (provided as a SMILES string) is converted into a graph representation:

- **Node features (per atom)**:
  - One-hot encoding of element type (22 elements + `other`)
  - One-hot encoding of KCF atom type (72 KEGG Chemical Function types)
  - Atom degree (0–8), formal charge, radical electrons, aromaticity, ring membership, explicit valence, mass, chiral tag, total H count
- **Edge features**: Bidirectional bond connectivity (undirected bonds stored as directed pairs)
- **Labels**: Binary SOM label per atom (1 = site of metabolism, 0 = not)

Molecules are split into train/validation/test sets (80/20 by default) with 10 pre-generated random splits for cross-validation.

### 2. Optional QikProp ADMET Features (`featurizers/add_qikprop_features.py`)

QikProp (Schrödinger) computes ~51 ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) molecular descriptors. This pipeline:

1. Maps molecule tracking IDs between the dataset and QikProp CSV output
2. Normalizes descriptors using precomputed means/standard deviations
3. Fills missing values with feature averages
4. Appends descriptors to the per-molecule feature tensors

### 3. GNN Architecture (`gnn-node-final.py`, `som.models.py`)

The GNN is built using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and supports a wide variety of graph convolution operators:

| Convolution | Description |
|---|---|
| `cheb`, `cheb1k`–`cheb15k` | Chebyshev Spectral Conv (K=1–15) |
| `gcn` | Graph Convolutional Network |
| `gat` | Graph Attention Network v2 (GATv2) |
| `gin`, `gin2`, `gin3` | Graph Isomorphism Network variants |
| `arma` | ARMA Graph Convolution |
| `resgat` | Residual Gated Graph Conv |
| `transform` | Transformer Conv |
| `gine` | GINE Conv (edge-aware GIN) |
| `crystal` | Crystal Graph Conv (CGConv) |
| `gen` | GEN Conv |
| `path` | PDN Conv |
| `general` | General Conv |

The architecture stacks `depth` convolutional layers of hidden dimension `width`, with ReLU activations and 0.5 dropout between layers. The final layer outputs a scalar per node (atom), trained with **Binary Cross-Entropy with logits** and class-imbalance weighting (`pos_weight = negatives / positives`).

Pooling strategies (`mean`, `max`, `add`) are available for graph-level tasks in `gnn-node-final.py`.

### 4. Training

- **Optimizer**: Adam (`lr=0.001`, `weight_decay=5e-4`)
- **Loss**: Weighted BCE with logits (handles SOM class imbalance)
- **Metrics reported every 10 epochs**: Train AUROC, Validation AUROC, Balanced Accuracy
- **Reproducibility**: Fixed random seeds (`torch.manual_seed(123)`, `random.seed(123)`)
- GPU support via CUDA with configurable device index

### 5. Evaluation (`analyze.py`)

After training, predictions are saved to pickle files and evaluated with:

| Metric | Description |
|---|---|
| **Molecular AUROC** | Mean per-molecule ROC AUC score |
| **Molecular R-Precision** | Fraction of true SOMs in top-R predictions, averaged per molecule |
| **Precision@1, @2, @3** | Precision at top-K predicted atoms |
| **Top-2 Correctness Rate** | Fraction of molecules where a true SOM appears in the top 2 predictions |
| **Atomic AUROC** | AUROC computed across all atoms pooled |
| **Atomic R-Precision** | R-Precision computed across all atoms pooled |

All metrics are compared against random-chance baselines.

Optionally generates score distribution plots (true SOMs vs. non-SOMs) using matplotlib.

### 6. Grid Search Runner (`run-gnns.py`)

`run-gnns.py` orchestrates parallel GPU training runs across all combinations of:
- Depths: `[1, 2, 3, 4]`
- Widths: `[32, 64, 128, 256, 512]`
- Convolutions: `[gat, resgat, transform, gine, crystal, gen, path, general]`
- Pooling: `[mean]`
- Splits: configurable (default 3)

The scheduler (`som/common.py`) manages a pool of GPUs, spawning one training process per GPU and dynamically re-queuing tasks if GPUs are added or removed during a run.

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit](https://www.rdkit.org/)
- [DeepChem](https://deepchem.io/)
- [kcfconvoy](https://github.com/JungeAlexander/kcfconvoy) (for KCF atom type featurization)
- scikit-learn
- numpy
- matplotlib (for score distribution plots)

Install core dependencies:
```bash
pip install torch torch-geometric rdkit deepchem scikit-learn numpy matplotlib
```

---

## Usage

### Featurize a Dataset

```bash
cd featurizers
python sort_pkl.py
```

### Train a GNN

```bash
python gnn-node-final.py \
  --data=dataset2_addprop_encode \
  --conv=gat \
  --pool=mean \
  --width=256 \
  --depth=3 \
  --epochs=200 \
  --split=0 \
  --gpu=0
```

**Key arguments:**
| Argument | Default | Description |
|---|---|---|
| `--data` | required | Dataset pickle filename (without `.pkl`) |
| `--conv` | `cheb` | Graph convolution operator |
| `--pool` | required | Pooling operator (`mean`, `max`, `add`) |
| `--width` | `512` | Hidden layer width |
| `--depth` | `3` | Number of GNN layers |
| `--epochs` | `400` | Training epochs |
| `--split` | `0` | Dataset split index |
| `--gpu` | `0` | GPU device index |
| `--cyp` | — | Filter to CYP substrates only |
| `--noncyp` | — | Filter to non-CYP substrates only |
| `--save` | — | Save trained model weights |

### Run a Grid Search

```bash
python run-gnns.py dataset2_addprop_encode output/ \
  --depth=3 \
  --width=256 \
  --epochs=200 \
  --splits=3
```

### Evaluate Predictions

```bash
python analyze.py data/pred-SomGnnNode-gat-mean256x3-200epochs-0split.pkl
python analyze.py data/pred-....pkl --test     # evaluate on test set
python analyze.py data/pred-....pkl --dist     # plot score distributions
```

---

## Data Format

Datasets are stored as Python pickle files with the following structure:

```python
{
  'mols': [
    {
      'tracking': 'molecule_id',
      'node': {
        'x': torch.Tensor,    # [num_atoms, num_features]
        'y': list,             # [num_atoms] binary SOM labels
        'edges': list,         # list of (atom_i, atom_j) tuples
        'edge_attr': torch.Tensor,  # [num_edges, edge_features] (optional)
        'elements': list,      # atom element symbols
        'types': list,         # KCF atom type strings
      },
      'isCyp': bool,           # CYP substrate flag
      'isNonCyp': bool,
    },
    ...
  ],
  'splits': {
    0: {'train': [...], 'valid': [...], 'test': [...]},
    ...
  },
  'features': {
    'element': [...],   # element list used for encoding
    'kcfType': [...],   # KCF type list used for encoding
    'degree': [...],
  }
}
```

