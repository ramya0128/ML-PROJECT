# **runwAI**

A deep learning system that recommends compatible fashion items using FashionCLIP embeddings and a trained Siamese network. Given a garment image, a text mood/occasion, or both, the system surfaces matching tops, bottoms, or complete outfits from a Fashion200K catalog.

---

## How It Works

1. **FashionCLIP** (`marqo-fashionSigLIP`) encodes both garment images and text queries into a shared 768-dimensional embedding space.
2. A **Siamese Network** trained on the Polyvore dataset learns to score compatibility between top–bottom pairs.
3. **FAISS** indexes enable fast nearest-neighbor search across the catalog.
4. Style rules (pattern-on-pattern avoidance, color clash detection) and a user feedback loop refine the final recommendations.

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| Polyvore | `Marqo/polyvore` (HuggingFace) | Training data for the Siamese model |
| Fashion200K | `Marqo/fashion200k` (HuggingFace) | Recommendation catalog |

---

## Model Architecture

### Siamese Network
- **Input:** Two 768-dim FashionCLIP image vectors (one top, one bottom)
- **Branch:** `Linear(768→256) → BN → ReLU → Dropout → Linear(256→128) → BN → ReLU → Dropout`
- **Fusion:** `|emb_A − emb_B|` concatenated with `emb_A * emb_B` → 256-dim
- **Classifier:** `Linear(256→64) → BN → ReLU → Dropout → Linear(64→1)`
- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=0.0005, weight_decay=1e-4) with ReduceLROnPlateau scheduler
- **Training:** Up to 50 epochs with early stopping (patience=8)

### Training Data (Polyvore)
Positive pairs are tops and bottoms from the **same outfit**; negative pairs are tops and bottoms from **different outfits**. The dataset is balanced 1:1.

---

## Recommendation Modes

### Text only
Describe a mood or occasion (e.g., `"casual beach day"`, `"formal office wear"`).

- Intent is detected from keywords (dress / top / bottom / full outfit).
- **Full outfit** mode returns 3 dress options + 3 complete top–bottom pairs.
- **Single category** mode returns the top matching items from that category.

### Image + Text
Upload a garment image and describe your mood.

- The query vector is a weighted blend of image (60%) and text (40%) features.
- The Siamese network scores compatibility between the uploaded item and catalog candidates.
- User feedback adjusts scores in future sessions.

---

## Style Filters

Two rule-based filters are applied before returning results:

**Pattern clash** — avoids recommending a patterned item to pair with another patterned item. Pattern detection uses keyword matching on item text (floral, striped, plaid, etc.) plus CLIP zero-shot classification on uploaded images.

**Color clash** — avoids known clashing color pairs (e.g., red + orange, blue + green) by matching color keywords extracted from item text.

Filters fall back gracefully: if too few results pass, restrictions are relaxed in stages.

---

## User Feedback

A lightweight feedback loop stores liked/disliked item IDs in `user_feedback.json`. On subsequent requests, candidate scores are adjusted:
- Items visually similar to liked items receive a **+0.3 × cosine similarity** boost.
- Items visually similar to disliked items receive a **−0.5 × cosine similarity** penalty.

---

## Setup

### Requirements

```
torch
open_clip_torch
faiss-cpu          # or faiss-gpu
datasets
pandas
numpy
Pillow
gradio
scikit-learn
matplotlib
```

Install with:

```bash
pip install torch open_clip_torch faiss-cpu datasets pandas numpy Pillow gradio scikit-learn matplotlib
```

### File Paths

The notebook uses local Windows paths — update these before running:

| Variable | Default path | What it is |
|---|---|---|
| `feature_dict_polyvore.npy` | `D:/feature_dict_polyvore.npy` | Cached Polyvore CLIP features |
| `feature_dict_f200k_clip.npy` | `C:/Users/.../feature_dict_f200k_clip.npy` | Cached Fashion200K CLIP features |
| `fashion200k_subset.csv` | `C:/Users/.../fashion200k_subset.csv` | Subset of Fashion200K metadata |
| `best_model_polyvore.pth` | `C:/Users/.../best_model_polyvore.pth` | Trained Siamese model weights |
| `f200k_id_to_index.npy` | `C:/Users/.../f200k_id_to_index.npy` | Item ID → dataset index lookup |
| `user_feedback.json` | `./user_feedback.json` | Persisted user feedback |

---

## Running the Notebook

Execute cells in order:

1. Load and filter the **Polyvore** dataset; build training pairs.
2. Extract FashionCLIP features for all Polyvore items (saves to `.npy`).
3. Train the Siamese network; the best checkpoint is saved automatically.
4. Load **Fashion200K**, extract features, and build FAISS indexes.
5. Run the recommendation functions or launch the Gradio UI.

### Gradio UI

```python
demo.launch(share=True)
```

The UI exposes:
- An image upload field (optional)
- A text mood/occasion field
- A garment type selector (`top` / `bottom` / `dress`)

Results are displayed in labelled image grids for compatible items, dresses, and complete outfit pairs.

---

## Project Structure

```
fashion_latest.ipynb       # Main notebook (data prep → training → inference → UI)
user_feedback.json         # Auto-generated; stores liked/disliked item IDs
feature_dict_polyvore.npy  # Cached CLIP features for Polyvore (generated)
feature_dict_f200k_clip.npy# Cached CLIP features for Fashion200K (generated)
best_model_polyvore.pth    # Best Siamese model checkpoint (generated)
f200k_id_to_index.npy      # ID → index lookup for Fashion200K (generated)
fashion200k_subset.csv     # Fashion200K metadata subset (user-provided)
```
