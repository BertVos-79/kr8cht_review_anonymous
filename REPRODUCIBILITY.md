# REPRODUCIBILITY

This document explains how to recreate the environment and re-execute the notebooks/scripts in **`kr8cht_review_anonymous`**.

> **Scope.** Two environment files are provided:
> - `envs/environment.yml` (CPU, default)
> - `envs/environment.cuda.yml` (GPU/CUDA, optional)
>
> All figures/tables can be regenerated **idempotently** from precomputed artifacts in `outputs/`, so **GPU is not required** for review.

---

## 1) System Requirements

- **OS:** macOS (Apple Silicon) or Linux  
- **Python:** 3.11 (managed by Conda)  
- **Conda:** Miniconda/Anaconda (mamba optional)  
- **Jupyter:** JupyterLab

---

## 2) Repository Layout (top level)

```
kr8cht_review_anonymous/
├── CITATION.cff
├── config/
├── docs/
├── envs/
│   ├── environment.yml
│   └── environment.cuda.yml
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── LICENSE
├── README.md
└── REPRODUCIBILITY.md
```

> **Included artifacts manifest.**  
> The exact list of committed artifacts used by the notebooks in **review mode**
> is documented in **ARTIFACTS_MANIFEST.md** (repo root). If an expected file is
> missing locally, consult that manifest first to confirm it should be present.

---

## 3) Obtain the Sources

The anonymized repository is available at:
<https://anonymous.4open.science/r/kr8cht_review_anonymous/>

1) Click **Download Repository** to get a ZIP.  
2) Unzip and enter the folder:

```bash
cd ~/Downloads
unzip kr8cht_review_anonymous*.zip
cd kr8cht_review_anonymous*/
```

---

## 4) Create the Conda Environment

### Option A (recommended): one command from the spec

```bash
conda env create -n kr8cht_review_anonymous -f envs/environment.yml
conda activate kr8cht_review_anonymous
python -m ipykernel install --user   --name=kr8cht_review_anonymous   --display-name "kr8cht_review_anonymous"
```

### Option B: GPU/CUDA (optional)

```bash
conda env create -n kr8cht_review_anonymous -f envs/environment.cuda.yml
conda activate kr8cht_review_anonymous
python -m ipykernel install --user   --name=kr8cht_review_anonymous   --display-name "kr8cht_review_anonymous"
```

> If the solver proposes updates/downgrades, accept them—the spec pins the study versions.

---

## 5) Launch JupyterLab

```bash
jupyter lab
```

In each notebook, select the **kr8cht_review_anonymous** kernel.

---

## 6) How to Re-Run the Analysis

You can either (A) run notebooks interactively, (B) run them in batch, or (C) use the convenience runner script (added in `scripts/run_all.sh`).

### A) Interactive (recommended for inspection)

1. Open `notebooks/` in JupyterLab.  
2. For each notebook (`a_static.ipynb` … `f_6_target_analysis.ipynb`):
   - Ensure the **kr8cht_review_anonymous** kernel is selected.
   - `Run → Run All Cells`.

Figures/tables will be written under `outputs/` (final paper assets under `outputs/f_final_report/`).

### B) Batch-execute notebooks (optional)

```bash
# From repo root, environment active
find notebooks -type f -name "*.ipynb"   -not -path "*/.ipynb_checkpoints/*"   -print0 | while IFS= read -r -d '' nb; do
    echo "Executing: $nb"
    jupyter nbconvert --to notebook --inplace --execute       --ExecutePreprocessor.timeout=0 "$nb"
done
```

### C) One-shot scripts runner (provided)

Using scripts/run_all.sh (see README), you can re-render all figures/tables from existing artifacts:

```bash
bash scripts/run_all.sh
```

This script only calls plotting/statistics steps that read from `outputs/` and **do not** require raw/private data.

---

## 7) Determinism & Idempotency

- Evaluation uses LOOCV with **precomputed artifacts** committed in `outputs/`.  
- Reruns are deterministic for review; any residual plotting randomness does not affect reported metrics.  
- Scripts are **idempotent**: if raw data are absent, they load existing result files and skip rebuilding from raw.

---

## 8) Troubleshooting

- **Kernel not visible:**  
  ```bash
  conda activate kr8cht_review_anonymous
  python -m ipykernel install --user     --name=kr8cht_review_anonymous     --display-name "kr8cht_review_anonymous"
  ```

- **Solver conflicts:**  
  ```bash
  conda update -n base -c defaults conda -y
  conda env update -n kr8cht_review_anonymous -f envs/environment.yml
  ```

- **GPU/MPS/CUDA warnings:** Safe to ignore for CPU review.

Please include OS, `conda --version`, and `conda list` (within the env) if reporting issues.

---

## 9) Transparency

All figures/tables supporting the paper are reproducible from this repo without external credentials or proprietary data/services. No additional downloads are required for verification.
