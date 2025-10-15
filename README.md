# kr8cht_review_anonymous

**Investigating study choice in Flemish schools with multi-target regression (MTR) on Dutch text.**  
This repository accompanies the submission of the article and contains all code, notebooks, configs, and precomputed artifacts needed to **reproduce the results for review** without access to private data.

> **TL;DR findings (from the paper):**
> - Frozen **sentence-level** transformers + MTR (esp. **E5 + chain_ERCcv_lr**) consistently outperform frozen token-level and static embeddings.
> - **Local / chain-based** MTR > **global** MTR on this small dataset.
> - **Fine-tuning** (LoRA) on the tiny dataset does **not** beat frozen+MTR.
> - **Taxonomy-guided augmentation** (Gemma → teacher pseudo-labels) improves **local_lasso** with statistical significance.

---

## Repository map

- **notebooks/** – Re-runnable analysis notebooks for each stage (static, frozen, fine-tuned, augmentation, final figures). Each has a docstring listing inputs/outputs and idempotent behavior.
- **scripts/** – Script equivalents of the notebooks plus orchestrator (`run_all.sh`) to re-run everything in order.
- **outputs/** – All result artifacts (plots/tables/arrays) produced by the pipeline and used in the paper (organized per step).
- **docs/** – Background documentation and assets for **taxonomy-guided prompting** (Appendix A), Mermaid source trees + SVG/PNG exports, and a small helper render script.
- **envs/** – Conda environments (`environment.yml` and optional `environment.cuda.yml`) to reproduce the software stack.
- **models/** – Placeholder for local checkpoints if needed (no private data included).
- **MODEL_CARDS.md** – Short cards describing model families and embeddings used.
- **REPRODUCIBILITY.md** – Exact steps to rebuild the environment and re-run notebooks/scripts from saved artifacts.
- **CITATION.cff** – Placeholder for citation (anonymous review; will be updated post-acceptance).
- **LICENSE** – Code license (Apache-2.0). *Data are not distributed.*

> **Note:** There is **no `src/` package**; code lives in notebooks and `scripts/` by design for clarity during review.

### Artifacts at a glance

- Baseline static arrays: `outputs/a_static/results/`
- Frozen transformer arrays: `outputs/b_frozen/results/`
- Fine-tuned arrays: `outputs/d_fine_tuned/results/`
- Final report figures/tables: `outputs/f_final_report/`  
  Subfolders:  
  `f_1_wilcoxon_heatmap/`, `f_2_pooling_families_cd/`, `f_3_embedding_cd/`, `f_4_frozen_vs_finetuned/`, `f_5_average_diversity/`, `f_6_target_analysis/`

---

## Quick start

**Environment (CPU-only is fine for review):**
```bash
conda create -n kr8cht_review_anonymous python=3.11 -y
conda activate kr8cht_review_anonymous
conda env update -n kr8cht_review_anonymous -f envs/environment.yml
```

**Re-run:**

- **Interactive (recommended):** open JupyterLab and run each notebook with the `kr8cht_review_anonymous` kernel.
- **Batch (shell):**
```bash
bash scripts/run_all.sh          # continues on errors; logs to ./logs
bash scripts/run_all.sh --strict # aborts on first error
```

Full details and troubleshooting: **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

### Mode selection

```
Most notebooks implement a toggle at the top:
REVIEW_MODE = True    # artifact-only, no raw data or training
REVIEW_MODE = False   # full compute, requires data/models

If a notebook has no explicit toggle, it is review-only and consumes committed artifacts.
For **f_6_target_analysis** Part B, use:
export FORCE_REBUILD_PART_B=1   # recompute diagnostics from ./data; omit for review
```

---

## Execution modes and idempotency

**Definition of modes**  
• **Compute mode**: end-to-end reproduction that trains or generates artifacts. Requires access to private data and, where applicable, model weights or external LLMs.  
• **Review mode**: artifact-only reproduction that re-renders tables, statistics, and figures from committed outputs. No access to raw data is required. All review notebooks are idempotent.

**Per-notebook mode matrix**

| Notebook | Mode support | Review inputs (minimum) | Notes on idempotency and outputs |
|---|---|---|---|
| a_static.ipynb | Compute and review (`REVIEW_MODE`) | `outputs/a_static/results/baseline_{embedding}_loocv_rrmse.npy` | Recomputes all statistics and figures from artifacts only; writes the same filenames as compute mode. |
| b_frozen.ipynb | Compute and review (`REVIEW_MODE`) | `outputs/b_frozen/results/{embedding}_loocv_rrmse.npy` | Re-creates per-embedding and cross-embedding stats and CD plots; prints overall ranking. |
| c_interim_report.ipynb | Review only (artifact-only) | Baseline and frozen `{embedding}_loocv_rrmse.npy` from A and B | Consolidates and analyzes without reading raw data; fully idempotent. |
| d_fine_tuned.ipynb | Compute and review (`REVIEW_MODE`) | `outputs/d_fine_tuned/results/partial_ft_{emb}_loocv_rrmse.npy` | In review, rebuilds all CSVs/plots from fine-tuned artifacts; no training. |
| e_1_synth_augmentation.ipynb | Compute only (resumable) | — | Fully resumable; clean no-op if all target finals exist. Suitable to skip during review. |
| e_2_teacher_labeling.ipynb | Compute and review (`REVIEW_MODE`) | Existing `g2f_labels_fold{ii}_*.csv` under `outputs/e_2_teacher_labeling/` | Review scans artifacts to rebuild `run_summary.csv` and a light `run_config.json`. |
| e_3_student_scoring.ipynb | Compute with summarization-only path | Per-fold CSVs under `outputs/e_3_student_scoring/results/` | Idempotent per configuration; can run summarization/plots only from existing CSVs. |
| f_1_wilcoxon_heatmap.ipynb | Review only (artifact-only) | A and B fold arrays | Builds Wilcoxon heatmaps and significance CSVs for overall comparisons; idempotent. |
| f_2_pooling_families_cd.ipynb | Review only (artifact-only) | A and B fold arrays | Produces **family/paradigm** (local/chain/global) CD diagrams and stats from artifacts. |
| f_3_embedding_cd.ipynb | Review only (artifact-only) | A and B fold arrays | Produces **embedding-group** CD diagrams and statistics; artifact-only. |
| f_4_frozen_vs_finetuned.ipynb | Review only (artifact-only) | Frozen arrays from B and fine-tuned arrays from D | Generates frozen vs fine-tuned heatmaps/CD diagrams without training. |
| f_5_average_diversity.ipynb | Review only (artifact-only) | Augmentation artifacts from E (saved embeddings/metrics) | Aggregates augmentation **diversity** (1−cos) across sizes; outputs CSV + LaTeX table with absolute averages and relative Δ%. |
| f_6_target_analysis.ipynb | Review by default; optional compute for Part B | A and B fold arrays; Part B precomputed tables for review | Part B recomputes only if `FORCE_REBUILD_PART_B=1`; otherwise uses committed outputs. |

**Common environment flags**  
• `REVIEW_MODE=True|False` where available (top of notebook).  
• `FORCE_REBUILD_PART_B=1` to force recomputation of `f_6` Part B diagnostics from `./data`; omit or set to `0` to keep review behavior.

**Scope of data-free review**  
The complete figure and table set reported in the paper can be regenerated in review mode from the committed artifacts under `outputs/`, without access to raw data or private model assets. Steps `e_1`, `e_2`, and full compute in `e_3` are not required for review.

---

## Key results & paths

- **Overall model × embedding comparisons**
  - Wilcoxon heatmaps: `outputs/f_final_report/f_1_wilcoxon_heatmap/`
  - Family/paradigm CD (local vs chain vs global): `outputs/f_final_report/f_2_pooling_families_cd/`
  - Embedding-group CD: `outputs/f_final_report/f_3_embedding_cd/`

- **Frozen vs. fine-tuned**
  - Heatmaps/CD and stats: `outputs/f_final_report/f_4_frozen_vs_finetuned/`

- **Augmentation diversity**
  - Average diversity (1−cos) across augmentation sizes, including relative Δ%: `outputs/f_final_report/f_5_average_diversity/`

- **Per-target analyses & diagnostics**
  - Tables + plots: `outputs/f_final_report/f_6_target_analysis/`

Each corresponding notebook in `notebooks/` regenerates the figures/tables from these saved arrays idempotently.

---

## Background & docs

- **Taxonomy-guided prompting:** `docs/taxonomy-guided-prompting.md`  
  Visuals and Mermaid sources in `docs/trees/`, render helper in `docs/scripts/render_trees.sh`.

**Preprint:** link to be added upon acceptance (kept anonymous during review).

---

## Models & embeddings (high level)

- **MTR families:** local (Linear/Lasso/RF), chain-based (ERCcv with base LR/Lasso/RF), global (RF/Bagging); PCA grid for dimensionality.
- **Embeddings:** static (Word2Vec, fastText), frozen token-level (BERTje, RobBERT v2/2023, DeBERTa-v3, XLM-R large, ReMBERT with mean/max/CLS pooling), frozen sentence-level (SBERT-BERT/RoBERTa, E5 base/large, LaBSE, SimCSE-XLM-R).
- **Fine-tuning:** LoRA on three sentence encoders (SBERT, SimCSE-XLM-R, E5-large) – optional, heavier, does not outperform frozen+MTR on this dataset.

Details in **[MODEL_CARDS.md](MODEL_CARDS.md)** and the notebook/script docstrings.

---

## Reproducibility conventions

- All plots and tables are regenerated deterministically from NumPy arrays and CSVs.
- When both legacy and current filename conventions exist (for static embeddings), loaders try both to ensure portability.

---

## Data availability

The **raw dataset cannot be shared** and is **not included**.  
All review computations are derived from **precomputed artifacts** in `outputs/`, and notebooks/scripts are designed to **load these idempotently**.

---

### What’s included in this review package

This repository includes all **code**, **environment specs**, and the **artifacts required to re-render every figure
and table** in review mode, without access to private data. In particular, we keep:
- A/B/D LOOCV arrays used across the F-steps,
- augmentation finals and caches needed for downstream steps (e.g., `facet_pools/*.txt`, `g_raw_gemma.jsonl`,
  `e_2_teacher_labeling/cache/synth_embeds/*`, and `e_3_student_scoring/cache/X_seed_e5_base.npy`),
- and all finalized paper assets under `outputs/f_final_report/**`.

For a precise, path-level list of what is **kept** vs **omitted** (e.g., trainer detritus, teacher pickles, raw data),
see **ARTIFACTS_MANIFEST.md** in the repo root.

---

## License & citation

- **Code license:** Apache License 2.0 — see **[LICENSE](LICENSE)**.  
- **Citation:** see **[CITATION.cff](CITATION.cff)** (placeholder; final metadata added post-acceptance).

---

## Contact

Omitted for double‑blind review. Please use the conference/journal review channel.
