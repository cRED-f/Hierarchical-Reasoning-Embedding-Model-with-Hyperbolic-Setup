# Hierarchical Reasoning Embedding Model with Hyperbolic Setup

A hierarchy-aware embedding model that maps text into hyperbolic (Poincaré ball) space for semantic retrieval. The model produces multiple embedding "levels" per text (coarse to fine), with increasing radius in the Poincaré ball representing different levels of semantic granularity.

---

## Overview

This project implements a Hierarchical Reasoning Model (HRM) that:

1. Uses a **frozen Transformer backbone** (e.g., BGE, BERT) to encode text as token hidden states
2. Applies a **learnable token attention pooler** to aggregate tokens into a single vector
3. Runs **HRM-style refinement dynamics** for multiple segments, producing hierarchical tangents
4. Maps tangents to the **Poincaré ball** using exponential mapping with a radial schedule
5. Optimizes **coarse-to-fine hyperbolic NCE** with explicit negatives across all segments

The result is a model that captures semantic hierarchy naturally within hyperbolic space, where distances to the origin encode granularity levels.

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────┐
│  Frozen Backbone (e.g., BGE-small-en-v1.5)   │
│  → Token hidden states H ∈ ℝ(L×d_base)       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  TokenAttentionPooler                        │
│  → Learned token weighting + aggregation    │
│  → x ∈ ℝ(d_base)                            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Linear Projection                           │
│  → u₀ = W·x ∈ ℝ(d_hrm)                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  HRM Refinement (M segments)                │
│  for m in 1..M:                              │
│    z_H, z_L, h^(m) = run_segment(u₀, z_H, z_L) │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Hierarchy by Construction                   │
│  for each segment m:                        │
│    u^(m) = s_m · (h^(m)/||h^(m)||)         │
│    y^(m) = exp₀^c(u^(m))  ∈ ℙⁿ              │
└─────────────────────────────────────────────┘
    │
    ▼
  Multiple embeddings y^(1), y^(2), ..., y^(M)
  (coarse to fine, increasing radius)
```

### Core Components

#### 1. TokenAttentionPooler (`models/hrm_encoder.py`)
A learnable attention mechanism that weights tokens before aggregation:

```
scores = ScoreNet(LayerNorm(H))  → [B, L, heads]
alpha   = softmax(scores, dim=1)  → attention weights
pooled  = Σ(alpha_i · H_i)        → aggregated representation
```

The `ScoreNet` is a small MLP with GELU activations that learns which tokens are most important for the task.

#### 2. HRMRefinementEncoder (`models/hrm_encoder.py`)
Implements hierarchical refinement with two types of updates:

- **Low update**: `z_L' = z_L + MLP([z_L, z_H, x])` — refines local context
- **High update**: `z_H' = z_H + MLP([z_H, z_L])` — captures higher-level semantics

The schedule follows `n_cycles × t_low` total steps, with high-level updates every `t_low` steps.

#### 3. Poincaré Utilities (`models/poincare_utils.py`)
Implements hyperbolic geometry operations:

- `expmap0_poincare(u, c)` — Exponential map at origin
- `poincare_distance_batch(q, cand, c)` — Batch pairwise distances
- `poincare_distance_matrix(a, b, c)` — Full distance matrix

---

## Mathematical Foundation

### Poincaré Ball Model

The Poincaré ball is a model for n-dimensional hyperbolic space:

$$\mathbb{B}^n_c = \{x \in \mathbb{R}^n : c\|x\|^2 < 1\}$$

with curvature `-c` (higher `c` = more negatively curved = smaller effective radius).

### Exponential Map at Origin

Maps a tangent vector `u` at the origin to a point in the ball:

$$\exp_0^c(u) = \tanh(\sqrt{c}\|u\|) \cdot \frac{u}{\sqrt{c}\|u\|}$$

This creates a mapping where small tangents near the origin stay close to the origin, while larger tangents get pushed toward the ball's boundary.

### Poincaré Distance

The hyperbolic distance between two points `x, y ∈ ℙⁿ` is:

$$d_\mathbb{B}(x, y) = \frac{1}{\sqrt{c}} {arcosh}\left(1 + 2c\frac{\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\right)$$

### Hierarchy by Construction

The model creates hierarchy through radial scaling, not through explicit hierarchy losses:

1. Each segment produces a tangent vector `h^(m)`
2. Normalize to unit direction: `ĥ^(m) = h^(m) / ||h^(m)||`
3. Scale by segment-dependent scalar: `u^(m) = s_m · ĥ^(m)`
4. Map to Poincaré ball: `y^(m) = exp_0^c(u^(m))`

Increasing `s_m` for later segments means later embeddings naturally have larger radii, representing finer semantic distinctions.

---

## Repository Structure

```
Hierarchical-Reasoning-Embedding-Model-with-Hyperbolic-Setup/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── constants.py          # Task names, dataset mappings, instructions
├── models/
│   ├── __init__.py
│   ├── poincare_utils.py     # Hyperbolic geometry functions
│   └── hrm_encoder.py        # HRM encoder and pooler
├── data/
│   ├── __init__.py
│   ├── loader.py             # MTEB retrieval dataset loading
│   └── kalm_loader.py        # KaLM finetuning data loader
├── encoding/
│   ├── __init__.py
│   └── encoder.py            # Text embedding functions
├── evaluation/
│   ├── __init__.py
│   └── retriever.py          # Retrieval metrics computation
├── training/
│   ├── __init__.py
│   ├── utils.py              # Training utilities (seed, device, lr scheduler)
│   └── losses.py             # Hyperbolic NCE loss
├── scripts/
│   ├── train.py              # Training entry point
│   └── evaluate.py           # Evaluation entry point
├── README.md
└── LICENSE
```

---

## Installation

```bash
pip install torch transformers datasets huggingface_hub tqdm
```

Requirements:
- Python 3.9+
- PyTorch (CUDA recommended)
- Hugging Face Transformers
- Datasets library
- huggingface_hub
- tqdm

---

## Quick Start

### Training

```bash
python scripts/train.py \
    --backbone_name BAAI/bge-small-en-v1.5 \
    --output_dir hrm_hyp_hier_runs \
    --epochs 1 \
    --batch_size 128 \
    --num_segments 4 \
    --num_negs 4 \
    --temperature 0.05 \
    --hyp_c 1.0
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint hrm_hyp_hier_runs/checkpoint_best.pt \
    --task fever \
    --max_corpus 200000
```

---

## Supported Tasks

The evaluation script supports these MTEB retrieval tasks:

| Task | Description | Dataset |
|------|-------------|---------|
| `fever` | Fact verification retrieval | mteb/FEVER |
| `scifact` | Scientific claim verification | mteb/SciFact |
| `nfcorpus` | Medical question answering | mteb/NFCorpus |
| `dbpedia` | Entity retrieval | mteb/DBPedia |
| `hotpotqa` | Multi-hop question answering | mteb/HotpotQA |
| `nq` | Wikipedia QA | mteb/NQ |
| `scidocs` | Scientific paper citation | mteb/SCIDOCS |
| `fiqa` | Financial QA | mteb/FiQA2018 |
| `cqadupstack` | Community QA duplicate detection | mteb/CQADupstack |
| `climatefever` | Climate change claim verification | mteb/ClimateFEVER |
| `arguana` | Argument retrieval | mteb/Arguana |

### Evaluation Metrics

The model computes:
- **Hits@K** — Percentage of queries with at least one relevant document in top K
- **Recall@K** — Fraction of relevant documents retrieved in top K
- **NDCG@K** — Normalized Discounted Cumulative Gain
- **MRR@K** — Mean Reciprocal Rank
- **MAP@K** — Mean Average Precision

---

## Key Hyperparameters

### Hierarchy Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_segments M` | Number of hierarchical segments | 4 |
| `--s_scales` | Radial schedule (comma-separated) | 1,2,3,4 |
| `--w_segments` | Segment weights (normalized sum to 1) | linear increasing |
| `--alpha_segments` | Coarse-to-fine mixing (0=coarse, 1=fine) | linear 0→1 |

### Model Architecture

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--d_hrm` | HRM embedding dimension | 512 |
| `--n_cycles` | HRM cycles per segment | 2 |
| `--t_low` | Low-level update frequency | 2 |
| `--hrm_hidden_mult` | MLP hidden dimension multiplier | 4 |
| `--pool_heads` | Attention pooler heads | 1 |
| `--pool_hidden_mult` | Pooler MLP hidden multiplier | 2 |
| `--pool_dropout` | Pooler dropout rate | 0.0 |

### Hyperbolic Geometry

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hyp_c` | Poincaré ball curvature (c > 0) | 1.0 |
| `--temperature` | NCE temperature | 0.05 |

### Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Learning rate | 1e-4 |
| `--weight_decay` | AdamW weight decay | 0.01 |
| `--warmup_ratio` | Warmup proportion | 0.1 |
| `--hrm_grad_window` | Backprop through last N steps (0=all) | 0 |
| `--amp` | Enable mixed precision training | false |

### Example with Custom Hierarchy

```bash
python scripts/train.py \
    --num_segments 4 \
    --s_scales "1,2,4,7" \
    --w_segments "0.1,0.2,0.3,0.4" \
    --alpha_segments "0.0,0.33,0.66,1.0" \
    --hyp_c 1.0 \
    --temperature 0.05
```

---

## Advanced Features

### Gradient Windowing

The `--hrm_grad_window` parameter controls how many HRM steps are backpropped through:

- `0` (default): Full unroll, gradient flows through all steps
- Positive integer: Only the last N steps receive gradients (saves memory)

This is useful when training with many segments to reduce memory usage.

### Mixed Precision Training

Enable AMP for faster training on CUDA devices:

```bash
python scripts/train.py --amp --batch_size 256
```

### Checkpoint Resume

Resume training from a saved checkpoint:

```bash
python scripts/train.py --resume_from hrm_hyp_hier_runs/checkpoint_last.pt
```

### Backbone-Only Mode (Cosine)

For comparison, evaluate with backbone embeddings (no HRM, cosine similarity):

```bash
python scripts/evaluate.py \
    --checkpoint checkpoint.pt \
    --task fever \
    --no_hrm \
    --backbone_name BAAI/bge-small-en-v1.5
```

---

## Dataset Format

The default training dataset is `KaLM-Embedding/KaLM-embedding-finetuning-data`.

Expected JSON format per example:

```json
{
  "query": "What is the capital of France?",
  "pos": ["Paris is the capital of France.", "France's capital city is Paris."],
  "neg": ["London is the capital of UK.", "Berlin is Germany's capital."]
}
```

The collator creates:
- `q` — query (as-is)
- `pf` — fine positive (randomly sampled from `pos`)
- `pc` — coarse positive (shorter version of positive, or shorter of two positives)
- `K` negatives — sampled from `neg`

---

## Inference / Scoring

For retrieval, use the final segment (most fine-grained):

```python
# Embed query and candidate
y_query = encode_text(query)      # y^(M)(q) in Poincaré ball
y_candidate = encode_text(candidate)  # y^(M)(p)

# Score = negative hyperbolic distance (higher is better)
score = -poincare_distance(y_query, y_candidate, c=hyp_c)
```

For multi-stage retrieval, use earlier segments for coarse filtering and later segments for reranking.

---

## Checkpoints

Training saves checkpoints to `--output_dir`:

- `checkpoint_last.pt` — Latest checkpoint after each epoch
- `checkpoint_best.pt` — Best validation loss checkpoint
- `checkpoint_final.pt` — Final epoch checkpoint

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training args
- epoch, global_step, best_val metrics

---

## References

- **MTEB**: "MTEB: Massive Text Embedding Benchmark" — Evaluation benchmark for text embeddings
- **Poincaré Embeddings**: "Poincaré Embeddings for Learning Hierarchical Representations" — Hyperbolic embeddings
- **HRM**: Hierarchical Reasoning Model concepts for multi-granularity representation

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
