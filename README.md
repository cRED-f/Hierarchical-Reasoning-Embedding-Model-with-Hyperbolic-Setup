# Hierarchical Reasoning Embedding Model with Hyperbolic Setup

Train a hierarchy-aware embedding model by mapping text into hyperbolic (Poincaré ball) space using:

- a frozen Transformer backbone (e.g., BGE)
- a trainable token attention pooler
- HRM-style refinement run for multiple segments
- a hierarchy-by-construction radial schedule (no separate hierarchy loss)
- a single unified objective: coarse-to-fine hyperbolic NCE with explicit negatives

The result is a model that produces multiple embedding “levels” per text (coarse to fine), with increasing radius in the Poincaré ball.

---

## What this project does

Given training samples of the form:

- `query`: string
- `pos`: list of positive strings
- `neg`: list of negative strings

We train a model so that:

- the query is close to a coarse positive early in the hierarchy (segment 1)
- the query is close to a fine positive later in the hierarchy (segment M)
- negatives are pushed away at every segment
- hierarchy is enforced via a radial schedule (scales `s_m`), not an explicit hierarchy loss

Ranking during evaluation uses hyperbolic distance:

\[
ext{score}(q,p) = - d\_{\mathbb{B}}(y^{(M)}(q),\ y^{(M)}(p))
\]

---

## Core ideas

### 1) Frozen backbone, learned pooling

- Backbone encodes token states `H` (no gradients).
- A trainable TokenAttentionPooler learns how to aggregate tokens into a single vector `x`.

### 2) HRM refinement with segments

- `x` is projected to HRM space, then refined by HRM dynamics.
- Refinement is run repeatedly for `M` segments producing tangents `h^(1)...h^(M)`.

### 3) Hyperbolic embedding and hierarchy by construction

Each tangent is normalized to a direction and scaled by a segment-dependent scalar `s_m`:

\[
u^{(m)} = s_m \cdot \frac{h^{(m)}}{\|h^{(m)}\|}
\]

Then mapped to Poincaré ball via the exponential map at the origin:

\[
y^{(m)} = \exp_0^c(u^{(m)})
\]

Increasing `s_m` makes later segments lie at larger radii, representing finer detail.

### 4) Single unified objective (coarse-to-fine hyperbolic NCE)

For each segment `m`, we blend coarse and fine positives using `alpha_m`:

\[
L_m = (1-\alpha_m)\,\text{NCE}(q, pc, \text{negs}) + \alpha_m\,\text{NCE}(q, pf, \text{negs})
\]

Total loss is a weighted sum over segments:

\[
L = \sum_m w_m L_m
\]

---

## Repository structure (suggested)

```
.
├── train_hrm_backbone.py          # training script
├── README.md
└── hrm_hyp_hier_runs/             # default output directory
		├── checkpoint_last.pt
		├── checkpoint_best.pt
		└── checkpoint_final.pt
```

---

## Requirements

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Datasets
- huggingface_hub
- tqdm

Install:

```bash
pip install torch transformers datasets huggingface_hub tqdm
```

---

## Dataset

Default dataset is:

- `KaLM-Embedding/KaLM-embedding-finetuning-data`

Expected fields per example:

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```json
{
  "query": "string",
  "pos": ["positive text 1", "positive text 2", "..."],
  "neg": ["negative text 1", "negative text 2", "..."]
}
```

Per training example, the script constructs:

- `pf` (fine positive): sampled from `pos`
- `pc` (coarse positive):
  - if `len(pos) >= 2`, choose two positives, coarse = shorter, fine = longer
  - else `pc = coarsen(pf)` (heuristic truncation)

- `K` negatives: sampled from `neg` (with replacement if needed)

---

## Quick start (training)

Basic run:

```bash
python Train.py \
	--backbone_name BAAI/bge-small-en-v1.5 \
	--output_dir hrm_hyp_hier_runs \
	--epochs 1 \
	--batch_size 128 \
	--num_segments 4 \
	--num_negs 4 \
	--temperature 0.05 \
	--hyp_c 1.0
```

````

---

## Key hyperparameters

### Hierarchy-related

- `--num_segments M`
  Number of hierarchical segments (levels).
- `--s_scales` (comma list length M)
  Radial schedule. Default is `1..M`.
- `--alpha_segments` (comma list length M)
  Coarse-to-fine mixing, default linear `0..1`.
- `--w_segments` (comma list length M)
  Segment weights (normalized to sum to 1). Default increases with depth.

Example:

```bash
python Train.py \
	--num_segments 4 \
	--s_scales "1,2,4,7" \
	--w_segments "0.1,0.2,0.3,0.4" \
	--alpha_segments "0.0,0.33,0.66,1.0"
````

### Hyperbolic geometry

- `--hyp_c`
  Curvature parameter `c > 0` (ball curvature is `-c`).
- `--temperature`
  NCE temperature.

### HRM compute/grad trade-offs

- `--n_cycles`, `--t_low`
  Control HRM update schedule.
- `--hrm_grad_window`
  Backprop only through the last N HRM steps per segment (saves memory). `0` means full unroll.

---

## Checkpoints and outputs

Saved to `--output_dir`:

- `checkpoint_last.pt` after every epoch
- `checkpoint_best.pt` when validation improves
- `checkpoint_final.pt` after training completes

Resume:

```bash
python train_hrm_backbone.py --resume_from hrm_hyp_hier_runs/checkpoint_last.pt
```

---

## Evaluation and ranking

This project’s intended retrieval scoring uses hyperbolic distance on the final segment:

- Embed query: `y^(M)(q)`
- Embed candidate: `y^(M)(p)`
- Rank by `- d_B(y^(M)(q), y^(M)(p))`

You can also use earlier segments for coarse pruning and rerank with deeper segments.

---

## Practical notes

- Padding token: the script sets `tokenizer.pad_token` if missing (uses EOS/CLS when available).
- Stability: Poincaré distance uses clamping and fp32 math internally.
- Speed: `batch_size` and `num_negs` increase compute because negatives are encoded too.
