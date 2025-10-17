# kr8cht_review_anonymous

**Investigating study choice in Flemish schools with multi-target regression (MTR) on Dutch text.**  
This repository evaluates text representation and multi-target regression strategies for predicting fourteen continuous targets from Dutch activity descriptions in the Kr8cht educational guidance game. The targets comprise eight curricular domains and six RIASEC personality traits. The dataset contains ninety-six expert-annotated activities with target scores in the range −6 to +6.

## Table of Contents
- [1. Scope and problem definition](#1-scope-and-problem-definition)
- [2. Data](#2-data)
- [3. Methods & representations](#3-methods--representations)
- [4. Overall leaderboard (Top-20)](#4-overall-leaderboard-top-20)
- [5. Repository map](#5-repository-map)
- [6. What’s included in this review package](#6-whats-included-in-this-review-package)
- [7. Artifacts at a glance](#7-artifacts-at-a-glance)
- [8. Quick start](#8-quick-start)
- [9. Modes & idempotency](#9-modes--idempotency)
- [10. Key results & paths](#10-key-results--paths)
- [11. Reproducibility conventions](#11-reproducibility-conventions)
- [12. Background & docs](#12-background--docs)
- [13. License & citation](#13-license--citation)
- [14. Contact](#14-contact)

---

## 1. Scope and problem definition

**Objective.** Learn a predictive model that maps a short Dutch activity description to **fourteen real-valued targets**:
- eight curricular domains of Flemish secondary education, and
- six RIASEC personality traits (Realistic, Investigative, Artistic, Social, Enterprising, Conventional).

**Task type.** Multi-target regression evaluated primarily via **RRMSE** (RMSE relative to a per-target mean baseline).  
An **RRMSE < 1** indicates improvement over the mean baseline.

**Statistical comparisons.** We assess approaches with aligned/original **Friedman + Nemenyi**, paired **Wilcoxon** (Holm), and **Cliff’s Δ** effect sizes.

---

## 2. Data

**Source.** 96 activities with expert scores for 14 targets. No personal data are included.

**Targets.** Eight curricular domains (Taal en Cultuur; STEM; Kunst en Creatie; Land- en Tuinbouw; Economie en Organisatie; Sport; Maatschappij en Welzijn; Voeding en Horeca) and six RIASEC traits.

**Static embeddings (external).** Place required files under `embeddings/`:
- `word2vec_costoo.bin` (Coosto Dutch Word2Vec)
- `cc.nl.300.bin.gz` (FastText Common Crawl Dutch)

**Data availability.** The **raw dataset cannot be shared** and is **not included**. All review computations are derived from **precomputed artifacts** in `outputs/`, and notebooks/scripts are designed to **load these idempotently**.

---

## 3. Methods & representations

**Representations.**
- **Static embeddings.** Word2Vec and FastText token vectors (mean pooled).
- **Frozen transformers.** Dutch & multilingual encoders with token-level pooling (mean, max, CLS) and sentence encoders (e.g., SBERT, E5, LaBSE, SimCSE). Encoders remain frozen; regression is classical.
- **Parameter-efficient fine-tuning (optional).** LoRA adapters on selected sentence encoders (SBERT, SimCSE-XLM-R, E5-large).

**Taxonomy-guided augmentation (E-steps).**
- **E1** synthetic generation: facet-driven prompts produce nested-prefix datasets of sizes **N ∈ {96, 192, 384, 768, 1536, 3072}** per method, with strict validation + diversity selection.
- **E2** teacher labeling: per-fold, leak-free teachers (selected MTRs) label synthetics for chosen embeddings.
- **E3** student scoring: students reuse **per-fold hyperparameters** from teachers (no tuning) and are trained on **%K** subsets (**10, 20, 50, 100, 200, 400**%) of labeled synthetics.  
  Multiple *label sources (teachers)* are supported:
  - default: `teacher_e5_base_chainERCcv_lr`
  - additional: `teacher_e5_base_local_lasso`  
  The resulting student models are distinguished by **(student, teacher, augmentation)**; files may include an optional `__labels_*` tag to encode the teacher (e.g., `__labels_local_lasso`).
  *Students trained on augmented data*
  - On labels of `e5_base_chainERCcv_lr`: `e5_base_chainERCcv_lr`, `e5_local_lasso`, `e5_local_rf`, `e5_global_rf`, `e5_chain_ERCcv_rf`
  - On labels of `e5_local_lasso`: `e5_base_chainERCcv_lr`, `e5_local_lasso`

**Multi-target regressors (MTR).**
- **Local:** `local_lr`, `local_lasso`, `local_rf`
- **Chain-based (ERCcv):** `chain_ERCcv_lr`, `chain_ERCcv_lasso`, `chain_ERCcv_rf`
- **Global:** `global_rf`, `global_bag`

**Embedding families (examples).**
- **Static:** Word2Vec, FastText
- **Frozen token-level:** BERTje, RobBERT v2/2023, DeBERTa-v3, XLM-R large, RemBERT (pool: mean/max/CLS)
- **Frozen sentence-level:** SBERT-BERT, SBERT-RoBERTa, E5-base/large, LaBSE, SimCSE-XLM-R

*Family labels used in the leaderboard:* `static + MTR`, `frozen token mean + MTR`, `frozen token max + MTR`, `frozen token cls + MTR`, `frozen sentence + MTR`, `fine-tuned LoRA`, and `augmented + …` variants.

---

## 4. Overall leaderboard (Top-20)

Ranked by **global median RRMSE** (lower is better) across folds × targets. Columns:  
1) **Rank**, 2) **Family**, 3) **Embedding**, 4) **MTR model**, 5) **Teacher** (label source), 6) **Augm.** (`NA` or `A{10|20|50|100|200|400}`), 7) **RRMSE**.

```text
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ rank ┃ representation_family             ┃ embedding ┃ mtr_model      ┃ teacher                       ┃ augmentation ┃ rrmse  ┃
┣━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━┫
│ 1    │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_chainERCcv_lr │ A20          │ 0.6676 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 2    │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_chainERCcv_lr │ A50          │ 0.6737 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 3    │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_chainERCcv_lr │ A10          │ 0.6738 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 4    │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_local_lasso   │ A100         │ 0.6756 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 5    │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_local_lasso   │ A50          │ 0.6756 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 6    │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_chainERCcv_lr │ A50          │ 0.6778 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 7    │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_local_lasso   │ A20          │ 0.6780 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 8    │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_chainERCcv_lr │ A100         │ 0.6788 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 9    │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_local_lasso   │ A50          │ 0.6790 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 10   │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_local_lasso   │ A100         │ 0.6813 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 11   │ frozen sentence + MTR             │ e5_base   │ chain_ERCcv_lr │ NA                            │ NA           │ 0.6816 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 12   │ frozen sentence + MTR             │ e5_large  │ local_lasso    │ NA                            │ NA           │ 0.6820 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 13   │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_local_lasso   │ A200         │ 0.6841 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 14   │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_local_lasso   │ A10          │ 0.6842 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 15   │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_chainERCcv_lr │ A200         │ 0.6847 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 16   │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_chainERCcv_lr │ A400         │ 0.6847 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 17   │ augmented + frozen sentence + MTR │ e5_base   │ chain_ERCcv_lr │ teacher_e5_base_chainERCcv_lr │ A100         │ 0.6847 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 18   │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_chainERCcv_lr │ A200         │ 0.6852 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 19   │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_chainERCcv_lr │ A20          │ 0.6882 │
├──────┼───────────────────────────────────┼───────────┼────────────────┼───────────────────────────────┼──────────────┼────────┤
│ 20   │ augmented + frozen sentence + MTR │ e5_base   │ local_lasso    │ teacher_e5_base_local_lasso   │ A10          │ 0.6903 │
┗━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━┛

```

**Full report (all configurations):** `reports/leaderboard/leaderboard.md`  
**Tables/plots powering this section:** `outputs/r_1_overall/results/*.csv`, `outputs/r_1_overall/plots/*.png`  
<sub>_Tie-breaks:_ rrmse → family label → embedding (A–Z) → model (A–Z) → **teacher (A–Z)** → augmentation (NA before A10…A400).</sub>

---

## 5. Repository map

- **notebooks/** – Re-runnable analysis notebooks for each stage (static, frozen, fine-tuned, augmentation, final figures). Each has a docstring listing inputs/outputs and idempotent behavior.
- **scripts/** – Script equivalents of the notebooks plus orchestrator (`run_all.sh`) to re-run everything in order.
- **outputs/** – All result artifacts (plots/tables/arrays) produced by the pipeline and used in the paper (organized per step).
- **docs/** – Background documentation and assets for **taxonomy-guided prompting**, Mermaid source trees + SVG/PNG exports, and a small helper render script.
- **envs/** – Conda environments (`environment.yml` and optional `environment.cuda.yml`) to reproduce the software stack.
- **models/** – Placeholder for local checkpoints if needed (no private data included).
- **MODEL_CARDS.md** – Short cards describing model families and embeddings used.
- **REPRODUCIBILITY.md** – Exact steps to rebuild the environment and re-run notebooks/scripts from saved artifacts.
- **CITATION.cff** – Placeholder for citation (anonymous review; will be updated post-acceptance).
- **LICENSE** – Code license (Apache-2.0). *Data are not distributed.*
- **reports/leaderboard/** – Markdown report for the overall ranking (Top-20 + full) with small summary figures.
- **outputs/r_1_overall/** – CSVs and plots for the leaderboard (`results/top20.csv`, `results/full_leaderboard.csv`).
- **scripts/r_1_overall_leaderboard.py** – Aggregates artifacts and writes the leaderboard tables/plots/report.

> **Note:** There is **no `src/` package**; code lives in notebooks and `scripts/` by design for clarity during review.

---

## 6. What’s included in this review package

This repository includes all **code**, **environment specs**, and the **artifacts required to re-render every figure and table** in review mode, without access to private data. In particular, we keep:
- A/B/D LOOCV arrays used across the F-steps,
- augmentation finals and caches needed for downstream steps (e.g., `facet_pools/*.txt`, `g_raw_gemma.jsonl`,
  `e_2_teacher_labeling/cache/synth_embeds/*`, and `e_3_student_scoring/cache/X_seed_e5_base.npy`),
- all finalized paper assets under `outputs/f_final_report/**`,
- `outputs/r_1_overall/**` and `reports/leaderboard/leaderboard.md` (leaderboard CSVs/plots and markdown report).

For a precise, path-level list of what is **kept** vs **omitted** (e.g., trainer detritus, teacher pickles, raw data),
see **ARTIFACTS_MANIFEST.md** in the repo root.

---

## 7. Artifacts at a glance

- Baseline static arrays: `outputs/a_static/results/`
- Frozen transformer arrays: `outputs/b_frozen/results/`
- Fine-tuned arrays: `outputs/d_fine_tuned/results/`
- Final report figures/tables: `outputs/f_final_report/`  
  Subfolders:  
  `f_1_wilcoxon_heatmap/`, `f_2_pooling_families_cd/`, `f_3_embedding_cd/`, `f_4_frozen_vs_finetuned/`, `f_5_average_diversity/`, `f_6_target_analysis/`
- Leaderboard tables/plots: `outputs/r_1_overall/`
- Leaderboard report (markdown): `reports/leaderboard/leaderboard.md`

---

## 8. Quick start

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

**Optional: regenerate the leaderboard (artifact-only)**
```bash
python scripts/r_1_overall_leaderboard.py
```
This reads A/B/D/E3 artifacts under `outputs/**`, writes `outputs/r_1_overall/**`, and renders `reports/leaderboard/leaderboard.md`. No raw data or training is required.

---

## 9. Modes & idempotency

```
Most notebooks implement a toggle at the top:
REVIEW_MODE = True    # artifact-only, no raw data or training
REVIEW_MODE = False   # full compute, requires data/models

If a notebook has no explicit toggle, it is review-only and consumes committed artifacts.
For **f_6_target_analysis** Part B, use:
export FORCE_REBUILD_PART_B=1   # recompute diagnostics from ./data; omit for review
```

**Definition of modes**  
• **Compute mode**: end-to-end reproduction that trains or generates artifacts. Requires access to private data and, where applicable, model weights or external LLMs.  
• **Review mode**: artifact-only reproduction that re-renders tables, statistics, and figures from committed outputs. No access to raw data is required. All review notebooks are idempotent.  
• **Leaderboard aggregator (`scripts/r_1_overall_leaderboard.py`)**: review-only, artifact-based summarization. Deterministic and idempotent; never touches raw data or trains models.

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

## 10. Key results & paths

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

- **Top-20 leaderboard (quick scan):** see [4. Overall leaderboard (Top-20)](#4-overall-leaderboard-top-20) and `reports/leaderboard/leaderboard.md`.
  - The *teacher* column reflects the label source used to train augmented students (e.g., `teacher_e5_base_chainERCcv_lr` or `teacher_e5_base_local_lasso`). Files with `__labels_*` are mapped to canonical teacher names.


Each corresponding notebook in `notebooks/` regenerates the figures/tables from these saved arrays idempotently.

---

## 11. Reproducibility conventions

- All plots and tables are regenerated deterministically from NumPy arrays and CSVs.
- When both legacy and current filename conventions exist (for static embeddings), loaders try both to ensure portability.
- The leaderboard (`outputs/r_1_overall/**`) is computed deterministically from committed artifacts; re-running the aggregator reproduces identical CSVs/plots given identical inputs.

---

## 12. Background & docs

- **Taxonomy-guided prompting:** `docs/taxonomy-guided-prompting.md`  
  Visuals and Mermaid sources in `docs/trees/`, render helper in `docs/scripts/render_trees.sh`.

**Preprint:** link to be added upon acceptance (kept anonymous during review).

---

## 13. License & citation

- **Code license:** Apache License 2.0 — see **[LICENSE](LICENSE)**.  
- **Citation:** see **[CITATION.cff](CITATION.cff)** (placeholder; final metadata added post-acceptance).

---

## 14. Contact

Omitted for double-blind review. Please use the conference/journal review channel.
