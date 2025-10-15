# ARTIFACTS_MANIFEST (Review Package)

This manifest lists **what is included** in the public review repository and **what is intentionally omitted**, so
reviewers can understand exactly which artifacts the notebooks/scripts consume in review mode.

- Repository: `kr8cht_review_anonymous`
- Purpose: enable full **review-mode** re-runs (no private data), regenerating **all figures/tables** from committed artifacts.
- See also: `README.md` (project overview) and `REPRODUCIBILITY.md` (how to run).

---

## What’s included (kept in the repo)

### Code & configuration
- `notebooks/*.ipynb` — analysis notebooks (A–F).
- `scripts/*.py` — script equivalents and orchestration.
- `envs/*.yml` — Conda environment specs.
- `config/prompt_variable_blocks/*.json` — prompt variable blocks used in augmentation.
- `docs/**` — documentation, Mermaid sources, exported SVG/PNG assets, and `docs/data/**` taxonomy files.

### Outputs — Step A (static embeddings)
- `outputs/a_static/results/*_loocv_rrmse.npy` — **required** for review.
- `outputs/a_static/{plots,results}/*` — figures and stats used downstream.

### Outputs — Step B (frozen transformers)
- `outputs/b_frozen/results/*_loocv_rrmse.npy` — **required** for review.
- `outputs/b_frozen/{plots,results}/*` — figures and stats used downstream.

### Outputs — Step C (interim report)
- `outputs/c_interim_report/**/*` — consolidated stats and figures (consumed by F-steps).

### Outputs — Step D (fine-tuned, LoRA)
- `outputs/d_fine_tuned/results/partial_ft_*_loocv_rrmse.npy` — **required** for F4 comparisons.
- `outputs/d_fine_tuned/{plots,results}/*` — summary CSVs and figures.
- *(Trainer temp runs are omitted; see exclusions.)*

### Outputs — Step E1 (synthetic augmentation)
- `outputs/e_1_synth_augmentation/facet_pools/*.txt` — kept for transparency/reproducibility.
- `outputs/e_1_synth_augmentation/g_raw_gemma.jsonl` — raw candidates (kept).
- `outputs/e_1_synth_augmentation/g_intrinsic_metrics.csv` — intrinsic metrics.
- `outputs/e_1_synth_augmentation/g_final_n{N}_gemma.csv` for N ∈ {96,192,384,768,1536,3072}.
- `outputs/e_1_synth_augmentation/run_summary.csv` and `run.log` (light).

### Outputs — Step E2 (teacher labeling)
- `outputs/e_2_teacher_labeling/g2f_labels_fold{ii}_n{M}_{method}__{embedding}__{model}.csv` — **required** downstream.
- `outputs/e_2_teacher_labeling/cache/synth_embeds/*__{embedding}.npy` and `*__index.csv` — **required** cache.
- `outputs/e_2_teacher_labeling/run_summary.csv`, `run_config.json` — manifests/config.

### Outputs — Step E3 (student scoring)
- `outputs/e_3_student_scoring/results/*.csv` — **required** for F5 tables and E3 summaries.
- `outputs/e_3_student_scoring/tables/*.csv`, `outputs/e_3_student_scoring/plots/*.png` — summaries/figures.
- `outputs/e_3_student_scoring/cache/X_seed_e5_base.npy` — kept as **fallback** seed vectors for F5.
- `outputs/e_3_student_scoring/run_config.json` (and light logs).

### Final paper assets — Step F
- `outputs/f_final_report/**` — all subfolders kept:
  - `f_1_wilcoxon_heatmap/`
  - `f_2_pooling_families_cd/`
  - `f_3_embedding_cd/`
  - `f_4_frozen_vs_finetuned/`
  - `f_5_average_diversity/`
  - `f_6_target_analysis/`

---

## What’s omitted (pruned for clarity/weight)

- **Raw/private data** (repo-root): `/data/**` (not included).
- **Teacher model pickles**: `models/teacher/*.pkl` (not required for review paths).
- **Trainer detritus** (fine-tuning): `outputs/d_fine_tuned/temp/**`.
- **Verbose raw label logs**: `outputs/e_2_teacher_labeling/**/*.jsonl` (CSV labels are kept).
- **Per-fold scratch notebooks** (labeling): `outputs/e_2_teacher_labeling/run_fold*.ipynb`, `pm_run_*.ipynb`.
- **Student-scoring caches**: `outputs/e_3_student_scoring/cache/**` **except** `X_seed_e5_base.npy`.
- **Logs**: `logs/**`, `*.log` (light run logs may persist where helpful).
- **Local caches**: any accidental `huggingface/`, `**/hf_cache/`, `**/.huggingface/`, `**/datasets-cache/`.
- **Secrets**: `.env*`, `*.key`, `*.pem`, etc.

These omissions do **not** affect review-mode runs; all notebooks/scripts read the committed artifacts listed above.

---

## Conditional recomputation

- **F6 Part B** (diagnostics on raw data) is **skipped** in review by default and uses committed outputs under
  `outputs/f_final_report/f_6_target_analysis/`. To recompute Part B (requires raw data under `./data`), set:
  ```bash
  export FORCE_REBUILD_PART_B=1
  ```
- **Fine-tuned comparisons (F4)** rely on `outputs/d_fine_tuned/results/partial_ft_*_loocv_rrmse.npy`. No training is required in review.

---

## Quick verification checklist

After cloning, verify that the following exist (non-exhaustive):
- `outputs/a_static/results/baseline_fasttext_loocv_rrmse.npy`
- `outputs/b_frozen/results/e5_base_loocv_rrmse.npy`
- `outputs/d_fine_tuned/results/partial_ft_e5_large_loocv_rrmse.npy`
- `outputs/e_1_synth_augmentation/g_final_n3072_gemma.csv`
- `outputs/e_2_teacher_labeling/cache/synth_embeds/g_final_n3072_gemma__e5_base.npy`
- `outputs/e_3_student_scoring/cache/X_seed_e5_base.npy`
- `outputs/f_final_report/f_5_average_diversity/tables/diversity_values.csv`

If any item is missing, see `REPRODUCIBILITY.md` and the notebook docstrings for recovery behavior.
