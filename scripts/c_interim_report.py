"""
c_interim_report.ipynb
───────────────────────────────────────────────────────────────────────────────
Interim report synthesizing results from script a_static (comparison multi-target regression
models on static embeddings) and b_frozen (comparison frozen transformer embeddings).

This script:
1. Consolidates LOOCV results:
   - Loads RRMSE arrays for baseline (Word2Vec, FastText) and transformer embeddings.
   - Organizes data for 26 embeddings (24 frozen, 2 static) and the appropriate model 
     sets (10 for baseline, 6 for frozen).
2. Builds analysis matrices:
   - Per-target (targets × models; embeddings collapsed).
   - Per-embedding (embeddings × models; targets collapsed).
   - Per-model (models × embeddings; targets collapsed).
   - Per-target-embedding (targets × embeddings; models collapsed).
3. Performs hierarchical statistical testing:
   - Aligned and original Friedman tests with Iman–Davenport F.
   - Post-hoc Nemenyi (nblocks ≥ 10) or Conover–Iman (Holm-adjusted) when blocks < 10.
   - Pairwise Wilcoxon tests with Holm–Bonferroni correction.
   - Computes Cliff’s Δ effect sizes.
4. Conducts group comparisons:
   - Model families: Local vs Global vs Chain-based.
   - Pooling strategies: mean vs max vs CLS.
   - Embedding types: word-level vs sentence-level.
   - Architecture classes: static vs frozen transformers.
5. Generates reports and figures:
   - Exports median performance CSVs and p-value tables.
   - Saves critical-difference diagrams and grouped bar charts.

Inputs:
- `outputs/a_static/results/baseline_{word2vec|fasttext}_loocv_rrmse.npy`
- `outputs/b_frozen/results/{embedding}_loocv_rrmse.npy`

Outputs:
- `outputs/c_interim_report/{analysis}_median.csv`
- `outputs/c_interim_report/{analysis}_nemenyi_p.csv`
- `outputs/c_interim_report/{analysis}_conover_p.csv`
- `outputs/c_interim_report/{analysis}_wilcoxon_raw_p.csv`
- `outputs/c_interim_report/{analysis}_wilcoxon_holm_p.csv`
- `outputs/c_interim_report/{analysis}_cliffs_delta.csv`
- `outputs/c_interim_report/fig/{analysis}_cd.png`
- `outputs/c_interim_report/fig/baseline_models_2view_cd.png`
- `outputs/c_interim_report/fig/transformer_models_2view_cd.png`
- `outputs/c_interim_report/fig/transformer_embeddings_2view_cd.png`
- Group charts:
  - `outputs/c_interim_report/fig/group_a_bar.png`
  - `outputs/c_interim_report/fig/group_b_bar.png`
  - `outputs/c_interim_report/fig/group_c_bar.png`
  - `outputs/c_interim_report/fig/group_d_bar.png`
  - `outputs/c_interim_report/fig/sent_static_token_bar.png`

"""

# ────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────

from itertools import combinations
import json
import os
from pathlib import Path
import re
import textwrap
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

# ────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────

# suppress the FutureWarning coming from scikit_posthocs’ plotting
warnings.filterwarnings(
    "ignore",
    message="Series.__getitem__ treating keys as positions is deprecated",
    category=FutureWarning,
    module="scikit_posthocs._plotting"
)

# Helper to find project root
def get_project_root(marker: str = "LICENSE") -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(f"Could not find '{marker}' in {cwd} or any parent directories.")

ROOT         = get_project_root()
DATA_DIR     = ROOT / "data"
OUTPUT_DIR   = ROOT / "outputs"
A_STATIC_DIR = OUTPUT_DIR / "a_static" / "results"
B_FROZEN_DIR = OUTPUT_DIR / "b_frozen" / "results"
REPORT_DIR   = OUTPUT_DIR / "c_interim_report"
FIG_DIR      = REPORT_DIR / "fig"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True,     exist_ok=True)

BASELINE_EMB      = ["word2vec", "fasttext"]
STATIC_EMB        = set(BASELINE_EMB)

baseline_models = [
    "local_lr", "local_lasso", "local_rf", "local_mean",
    "global_bag", "global_rf",
    "chain_ERCcv_lr", "chain_ERCcv_lasso",
    "chain_ERCcv_rf", "chain_ERCcv_mean",
]

transformer_models = [
    "local_mean", "local_lasso", "local_rf",
    "global_rf", "chain_ERCcv_lr", "chain_ERCcv_rf",
]

models = transformer_models


# ──────────────────────────────────────────────────────────────
# 0. Load arrays  →  data[embedding][model] = np.ndarray(folds, targets)
# ──────────────────────────────────────────────────────────────
def _discover_loocv_files(base_dir: Path, prefix: str = ""):
    pat = re.compile(rf"{prefix}(.+)_loocv_rrmse\.npy$")
    for f in base_dir.glob(f"{prefix}*_loocv_rrmse.npy"):
        m = pat.match(f.name)
        if m:
            yield m.group(1), f

# baseline LOOCV arrays (10-model runs on word2vec + fasttext)
baseline_data = {}
for emb, file in _discover_loocv_files(A_STATIC_DIR, prefix="baseline_"):
    baseline_data[emb] = {
        m: np.asarray(a)
        for m, a in np.load(file, allow_pickle=True).item().items()
    }

# transformer LOOCV arrays (6-model runs on all embeddings)
data = {}
for emb, file in _discover_loocv_files(B_FROZEN_DIR, prefix=""):
    data[emb] = {
        m: np.asarray(a)
        for m, a in np.load(file, allow_pickle=True).item().items()
    }

embeddings      = sorted(data.keys())
n_targets       = next(iter(next(iter(data.values())).values())).shape[1]
print(f"Loaded {len(embeddings)} embeddings   ×   {len(models)} models   ×   {n_targets} targets")


def get_rrmse(embedding: str, model: str) -> np.ndarray:
    """
    Return the (folds x targets) RRMSE array for (embedding, model),
    looking first in transformer results (`data`), then in static baseline
    (`baseline_data`). Raises if missing in both.
    """
    if embedding in data and model in data[embedding]:
        return data[embedding][model]
    if embedding in baseline_data and model in baseline_data[embedding]:
        return baseline_data[embedding][model]
    raise KeyError(f"Missing RRMSE for embedding={embedding}, model={model}")

# ──────────────────────────────────────────────────────────────
# 1.  Statistical helpers
# ──────────────────────────────────────────────────────────────
def section(title):
    """Print section header"""
    bar = "═" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")

def _save_and_show(fig, path: str):
    """Save and display figure"""
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Plot saved → {path}")

def aligned_ranks(mat):
    """Hodges–Lehmann alignment + ranking along rows (lower is better)"""
    aligned = mat - np.median(mat, axis=1, keepdims=True)
    return np.apply_along_axis(lambda r: np.argsort(np.argsort(r)) + 1, 1, aligned)

def friedman_aligned(mat):
    """Aligned-Friedman χ² and Iman–Davenport F-statistic (expects ranks or aligned data)"""
    k = mat.shape[1]
    from scipy.stats import friedmanchisquare
    chi2, _ = friedmanchisquare(*[mat[:, i] for i in range(k)])
    Ff = ((mat.shape[0] - 1) * chi2) / (mat.shape[0] * (k - 1) - chi2)
    return chi2, Ff

def wilcoxon_matrix(mat, labels):
    """Pairwise two-sided Wilcoxon (zero-method = zsplit)"""
    df = pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)
    for i, j in combinations(range(len(labels)), 2):
        diff = mat[:, i] - mat[:, j]
        p    = 1.0 if np.allclose(diff, 0) else wilcoxon(diff, zero_method="zsplit")[1]
        df.iat[i, j] = df.iat[j, i] = p
    return df.round(4)

def holm_correct_and_effects(raw_p, data, labels):
    """Holm–Bonferroni correction and Cliff's Δ effect sizes"""
    idx = list(combinations(range(len(labels)), 2))
    pvals = [raw_p.iat[i, j] for i, j in idx]
    _, p_adj, _, _ = multipletests(pvals, method="holm")

    adj_df = raw_p.copy()
    for (i, j), p in zip(idx, p_adj):
        adj_df.iat[i, j] = adj_df.iat[j, i] = p
    adj_df[np.eye(len(labels), dtype=bool)] = 1.0

    def cliffs_delta(x, y):
        diffs = np.subtract.outer(x, y)
        n = len(x) * len(y)
        return (np.sum(diffs > 0) - np.sum(diffs < 0)) / n

    delta_df = pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)
    for (i, j) in idx:
        d_ij = cliffs_delta(data[:, i], data[:, j])
        delta_df.iat[i, j] = d_ij
        delta_df.iat[j, i] = -d_ij

    return adj_df.round(4), delta_df.round(3)

def conover_posthoc(ranks, labels, fname_tag):
    """Conover–Iman test with Holm correction"""
    p_df = sp.posthoc_conover_friedman(ranks, p_adjust="holm")
    p_df.index = p_df.columns = labels
    out = REPORT_DIR / f"{fname_tag}_conover_p.csv"
    p_df.to_csv(out)
    print("\nConover–Iman post-hoc p-values (Holm-adjusted):")
    print(p_df.round(4).to_string())
    print("  ↳ saved →", out)
    return p_df

def run_friedman(mat, block_name, col_labels, fname_tag):
    """Generic routine for Friedman analysis with post-hoc tests (FULL verbose printout)"""
    k       = len(col_labels)
    nblocks = mat.shape[0]

    # Save & print medians (PRINT SORTED low→high; CSV keeps original order)
    col_meds = pd.Series(np.median(mat, axis=0), index=col_labels)
    med_path = REPORT_DIR / f"{fname_tag}_median.csv"
    col_meds.to_csv(med_path, header=["median_rrmse"])
    print(f"\nMedian RRMSE per {block_name[:-1] if block_name.endswith('s') else block_name} (sorted low→high):")
    print(col_meds.sort_values().round(3).to_string())
    print("  ↳ saved →", med_path)

    # Only two blocks → Wilcoxon only
    if nblocks == 2:
        print(f"\nOnly two {block_name} → skipping Friedman/post-hoc.")
        wilc = wilcoxon_matrix(mat, col_labels)
        print("\nWilcoxon pairwise p-values:")
        print(wilc.round(4).to_string())
        wilc_path = REPORT_DIR / f"{fname_tag}_wilcoxon_raw_p.csv"
        wilc.to_csv(wilc_path)
        print("  ↳ saved →", wilc_path)

        adj, delta = holm_correct_and_effects(wilc, mat, col_labels)
        print("\nHolm–Bonferroni adjusted p-values:")
        print(adj.round(4).to_string())
        adj_path = REPORT_DIR / f"{fname_tag}_wilcoxon_holm_p.csv"
        adj.to_csv(adj_path)
        print("  ↳ saved →", adj_path)

        print("\nCliff's Δ effect sizes:")
        print(delta.round(3).to_string())
        delta_path = REPORT_DIR / f"{fname_tag}_cliffs_delta.csv"
        delta.to_csv(delta_path)
        print("  ↳ saved →", delta_path)
        return

    # Only two methods → paired Wilcoxon only
    if k == 2:
        p = wilcoxon(mat[:, 0], mat[:, 1], zero_method="zsplit")[1]
        print(f"\nPaired Wilcoxon ({col_labels[0]} vs {col_labels[1]}): p = {p:.5g}")
        return

    # Friedman statistics (aligned + original)
    ranks = aligned_ranks(mat)
    chi2_a, Ff_a = friedman_aligned(ranks)
    chi2_o, p_o  = friedmanchisquare(*[mat[:, i] for i in range(k)])
    Ff_o = ((nblocks - 1) * chi2_o) / (nblocks * (k - 1) - chi2_o)

    print(f"\n*Aligned-Friedman* (blocks = {block_name})")
    print(f"  χ²_F = {chi2_a:.3f}    F_F = {Ff_a:.3f}")
    print(f"\n*Original-Friedman* (blocks = {block_name})")
    print(f"  χ²_F = {chi2_o:.3f}    p = {p_o:.3g}    F_F = {Ff_o:.3f}")

    # Post-hoc: Conover (few blocks) or Nemenyi (many blocks)
    if nblocks < 10:
        conover_posthoc(ranks, col_labels, fname_tag)
    else:
        pvals_nem = sp.posthoc_nemenyi_friedman(ranks)
        pvals_nem.index = pvals_nem.columns = col_labels
        nem_path = REPORT_DIR / f"{fname_tag}_nemenyi_p.csv"
        pvals_nem.to_csv(nem_path)
        print("\nNemenyi p-values (aligned post-hoc):")
        print(pvals_nem.round(4).to_string())
        print("  ↳ saved →", nem_path)

    # Wilcoxon raw + Holm + Cliff’s Δ
    wilc = wilcoxon_matrix(mat, col_labels)
    print("\nWilcoxon pairwise p-values:")
    print(wilc.round(4).to_string())
    wilc_path = REPORT_DIR / f"{fname_tag}_wilcoxon_raw_p.csv"
    wilc.to_csv(wilc_path)
    print("  ↳ saved →", wilc_path)

    adj, delta = holm_correct_and_effects(wilc, mat, col_labels)
    print("\nHolm–Bonferroni adjusted p-values:")
    print(adj.round(4).to_string())
    adj_path = REPORT_DIR / f"{fname_tag}_wilcoxon_holm_p.csv"
    adj.to_csv(adj_path)
    print("  ↳ saved →", adj_path)

    print("\nCliff's Δ effect sizes:")
    print(delta.round(3).to_string())
    delta_path = REPORT_DIR / f"{fname_tag}_cliffs_delta.csv"
    delta.to_csv(delta_path)
    print("  ↳ saved →", delta_path)

def cd_plot(matrix, labels, title, fname):
    """Critical-distance diagram with robust p-value alignment to labels."""
    if matrix.shape[1] < 2:
        print(f"⚠  Skipping CD-plot '{title}' (need ≥2 methods, got {matrix.shape[1]})")
        return

    ranks = aligned_ranks(matrix)

    # Compute post-hoc p-values and FORCE index/columns to match `labels`
    pvals_raw = sp.posthoc_nemenyi_friedman(ranks)
    if not isinstance(pvals_raw, pd.DataFrame):
        pvals = pd.DataFrame(pvals_raw, index=range(len(labels)), columns=range(len(labels)))
    else:
        pvals = pvals_raw.copy()

    # Defensive shape fix (trim/pad unlikely; trim covers rare inconsistencies)
    if pvals.shape != (len(labels), len(labels)):
        pvals = pvals.iloc[:len(labels), :len(labels)]
        if pvals.shape != (len(labels), len(labels)):
            # Last resort: identity p-values (no significant lines)
            pvals = pd.DataFrame(np.ones((len(labels), len(labels))), index=range(len(labels)), columns=range(len(labels)))

    # Align names to your model labels, sanitize & symmetrize
    pvals.index = labels
    pvals.columns = labels
    pvals = pvals.astype(float).fillna(1.0)
    pvals = pd.DataFrame(np.minimum(pvals.values, pvals.values.T), index=labels, columns=labels)
    np.fill_diagonal(pvals.values, 1.0)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
    sp.critical_difference_diagram(pd.Series(ranks.mean(0), index=labels), pvals, ax=ax)
    ax.set_title(title, pad=10)
    _save_and_show(fig, FIG_DIR / fname)


def cd_plot_dual(matrix1, labels1, matrix2, labels2, title1, title2, fname):
    """Two CD-diagrams side-by-side with robust p-value alignment."""
    if matrix1.shape[1] < 2 or matrix2.shape[1] < 2:
        print("⚠  Skipping dual CD-plot (need ≥2 methods for both)")
        return

    def _aligned_pvals(M, lbls):
        r = aligned_ranks(M)
        raw = sp.posthoc_nemenyi_friedman(r)
        if not isinstance(raw, pd.DataFrame):
            P = pd.DataFrame(raw, index=range(len(lbls)), columns=range(len(lbls)))
        else:
            P = raw.copy()
        if P.shape != (len(lbls), len(lbls)):
            P = P.iloc[:len(lbls), :len(lbls)]
            if P.shape != (len(lbls), len(lbls)):
                P = pd.DataFrame(np.ones((len(lbls), len(lbls))), index=range(len(lbls)), columns=range(len(lbls)))
        P.index = lbls
        P.columns = lbls
        P = P.astype(float).fillna(1.0)
        P = pd.DataFrame(np.minimum(P.values, P.values.T), index=lbls, columns=lbls)
        np.fill_diagonal(P.values, 1.0)
        return r, P

    ranks1, pvals1 = _aligned_pvals(matrix1, labels1)
    ranks2, pvals2 = _aligned_pvals(matrix2, labels2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3), dpi=120)
    sp.critical_difference_diagram(pd.Series(ranks1.mean(0), index=labels1), pvals1, ax=ax1)
    ax1.set_title(title1, pad=10)
    sp.critical_difference_diagram(pd.Series(ranks2.mean(0), index=labels2), pvals2, ax=ax2)
    ax2.set_title(title2, pad=10)
    plt.tight_layout()
    _save_and_show(fig, FIG_DIR / fname)

    
# ──────────────────────────────────────────────────────────────
# 2.  Build matrices for analyses
# ──────────────────────────────────────────────────────────────

def _resolve(arg, fallback):
    """Utility: if arg is None, fall back to module-level default."""
    return fallback if arg is None else arg

def matrix_per_target_compare_models(
        emb_subset,
        data_dict=None,
        model_list=None):
    """
    rows   = targets
    cols   = models
    folds  and embeddings are aggregated by median.

    Parameters
    ----------
    emb_subset : list[str]
        Embedding names to include.
    data_dict  : dict, optional
        Mapping  embedding → {model → ndarray(folds, targets)}.
        Defaults to the module-level `data`.
    model_list : list[str], optional
        Ordered list of models.  Defaults to the
        module-level `models`.
    """
    D = _resolve(data_dict, data)
    M = _resolve(model_list, models)

    return np.column_stack([
        np.median(
            np.concatenate([D[e][m] for e in emb_subset], axis=0),
            axis=0
        )
        for m in M
    ])

def matrix_per_embedding_compare_models(
        emb_subset,
        data_dict=None,
        model_list=None):
    """
    rows = embeddings
    cols = models
    folds & targets aggregated by median.
    """
    D = _resolve(data_dict, data)
    M = _resolve(model_list, models)

    return np.vstack([
        [np.median(D[e][m]) for m in M] for e in emb_subset
    ])

def matrix_per_target_compare_embeddings(
        emb_subset,
        data_dict=None,
        model_list=None):
    """
    rows = targets
    cols = embeddings
    folds & models aggregated by median.
    """
    D = _resolve(data_dict, data)
    M = _resolve(model_list, models)

    return np.column_stack([
        np.median(
            np.stack([D[e][m] for m in M], axis=0),
            axis=(0, 1)
        )
        for e in emb_subset
    ])

def matrix_per_model_compare_embeddings(
        emb_subset=None,
        data_dict=None,
        model_list=None):
    """
    rows = models
    cols = embeddings
    folds & targets aggregated by median.

    If `emb_subset` is None → use *all* embeddings present in
    the supplied `data_dict`.
    """
    D = _resolve(data_dict, data)
    M = _resolve(model_list, models)
    E = list(D.keys()) if emb_subset is None else emb_subset

    return np.array([
        [np.median(D[e][m]) for e in E] for m in M
    ])

# ──────────────────────────────────────────────────────────────
# 3.  Reporting functions
# ──────────────────────────────────────────────────────────────
def section(title):
    bar = "═" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")

def run_friedman(mat, block_name, col_labels, fname_tag):
    """
    Generic routine for all 2-D analyses.

    Decision tree
    ─────────────
      • nblocks == 2              → pairwise Wilcoxon only
      • k == 2                    → paired Wilcoxon only
      • 3 ≤ nblocks < 10          → Aligned-Friedman + Conover–Iman
      • nblocks ≥ 10 and k > 2    → Aligned-Friedman + Nemenyi

    Always at the end: pairwise Wilcoxon matrix + Holm–Bonferroni + Cliff’s Δ
    (except when nblocks == 2, which early-exits here).
    """
    k       = len(col_labels) # nr. of columns = nr of models
    nblocks = mat.shape[0] # nr. of blocks

    # 0) Save & print medians
    col_meds = pd.Series(np.median(mat, axis=0), index=col_labels)
    med_path = REPORT_DIR / f"{fname_tag}_median.csv"
    col_meds.to_csv(med_path, header=["median_rrmse"])
    print(f"\nMedian RRMSE per {block_name[:-1] if block_name.endswith('s') else block_name}:")
    print(col_meds.round(3).to_string())
    print("  ↳ saved →", med_path)

    # ──────────────── Case A: only two blocks ────────────────
    if nblocks == 2:
        print(f"\nOnly two {block_name} → skipping Friedman/post-hoc.")
        wilc = wilcoxon_matrix(mat, col_labels)
        wilc_path = REPORT_DIR / f"{fname_tag}_wilcoxon_raw_p.csv"
        wilc.to_csv(wilc_path)
        print("\nRaw Wilcoxon p-values saved →", wilc_path)
        print(wilc.to_string())

        adj, delta = holm_correct_and_effects(wilc, mat, col_labels)
        adj_path   = REPORT_DIR / f"{fname_tag}_wilcoxon_holm_p.csv"
        delta_path = REPORT_DIR / f"{fname_tag}_cliffs_delta.csv"
        adj.to_csv(adj_path)
        delta.to_csv(delta_path)
        print("\nHolm–Bonferroni adjusted p-values saved →", adj_path)
        print(adj.to_string())
        print("\nCliff’s Δ effect sizes saved →", delta_path)
        print(delta.to_string())
        return

    # ──────────────── Case B: only two methods ────────────────
    if k == 2:
        p = wilcoxon(mat[:,0], mat[:,1], zero_method="zsplit")[1]
        print(f"\nPaired Wilcoxon ({col_labels[0]} vs {col_labels[1]}):  p = {p:.5g}")
        return

    # ────────── Aligned-Friedman statistics ──────────
    ranks   = aligned_ranks(mat)
    chi2_a, Ff_a = friedman_aligned(ranks)
    chi2_o, p_o  = friedmanchisquare(*[mat[:,i] for i in range(k)])
    Ff_o = ((nblocks-1)*chi2_o)/(nblocks*(k-1)-chi2_o)

    print(f"\n*Aligned-Friedman*  (blocks = {block_name})")
    print(f"  χ²_F = {chi2_a:.3f}    F_F = {Ff_a:.3f}")
    print(f"\n*Original-Friedman*  (blocks = {block_name})")
    print(f"  χ²_F = {chi2_o:.3f}    p = {p_o:.3g}    F_F = {Ff_o:.3f}")

    # ───────── Case C: few blocks → Conover–Iman ─────────
    if nblocks < 10:
        conover_posthoc(ranks, col_labels, fname_tag)

    # ───────── Case D: many blocks → Nemenyi ─────────
    else:
        pvals_nem = sp.posthoc_nemenyi_friedman(ranks)
        pvals_nem.index = pvals_nem.columns = col_labels
        nem_path = REPORT_DIR / f"{fname_tag}_nemenyi_p.csv"
        pvals_nem.to_csv(nem_path)
        print("\nNemenyi p-values (aligned post-hoc):")
        print(pvals_nem.round(4).to_string())
        print("  ↳ saved →", nem_path)

    # ───────────── Final: Wilcoxon + Holm + Cliff’s Δ ─────────────
    wilc = wilcoxon_matrix(mat, col_labels)
    wilc_path = REPORT_DIR / f"{fname_tag}_wilcoxon_raw_p.csv"
    wilc.to_csv(wilc_path)
    print("\nRaw Wilcoxon p-values saved →", wilc_path)
    print(wilc.to_string())

    adj, delta = holm_correct_and_effects(wilc, mat, col_labels)
    adj_path   = REPORT_DIR / f"{fname_tag}_wilcoxon_holm_p.csv"
    delta_path = REPORT_DIR / f"{fname_tag}_cliffs_delta.csv"
    adj.to_csv(adj_path)
    delta.to_csv(delta_path)
    print("\nHolm–Bonferroni adjusted p-values saved →", adj_path)
    print(adj.to_string())
    print("\nCliff’s Δ effect sizes saved →", delta_path)
    print(delta.to_string())


# ──────────────────────────────────────────────────────────────
# 4.  Main function
# ──────────────────────────────────────────────────────────────

def main():
    # A.  BASELINE   (two static embeddings) –  MODEL COMPARISON
    # ──────────────────────────────────────────────────────────────

    section("A  BASELINE  –  Select most performant models")

    # build matrices using baseline_data and baseline_models
    mat_bt = np.column_stack([  
        np.median(np.concatenate([baseline_data[e][m] for e in BASELINE_EMB], axis=0), axis=0)
        for m in baseline_models
    ])
    mat_be = np.vstack([
        [np.median(baseline_data[e][m]) for m in baseline_models]
        for e in BASELINE_EMB
    ])

    # Two-view CD‐diagrams side-by-side
    ranks_bt = aligned_ranks(mat_bt);  pvals_bt = sp.posthoc_nemenyi_friedman(ranks_bt)
    ranks_be = aligned_ranks(mat_be);  pvals_be = sp.posthoc_nemenyi_friedman(ranks_be)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharey=False)
    sp.critical_difference_diagram(
        pd.Series(ranks_bt.mean(0), index=baseline_models), pvals_bt, ax=ax1
    )
    ax1.set_title("Baseline: per-target", pad=12)

    sp.critical_difference_diagram(
        pd.Series(ranks_be.mean(0), index=baseline_models), pvals_be, ax=ax2
    )
    ax2.set_title("Baseline: per-embedding", pad=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR/"baseline_models_2view_cd.png", bbox_inches="tight")
    plt.show()

    # Per-target analysis (blocks = targets)
    print("\n➤ Baseline: per-target analysis (embeddings + folds collapsed)\n")
    run_friedman(mat_bt, "targets", baseline_models, "baseline_per_target_models")

    # Per-embedding analysis (blocks = embeddings)
    print("\n➤ Baseline: per-embedding analysis (targets + folds collapsed)\n")
    run_friedman(mat_be, "embeddings", baseline_models, "baseline_per_embedding_models")


    # B.  TRANSFORMER  (all embeddings)  –  MODEL + EMBEDDING COMPARISON
    # ──────────────────────────────────────────────────────────────
    section("B1  TRANSFORMER MODELS  –  Two-view MODEL comparison")

    mat_tt = matrix_per_target_compare_models(embeddings)
    mat_te = matrix_per_embedding_compare_models(embeddings)

    # Two-view CD for transformer MODELS
    ranks_tt = aligned_ranks(mat_tt);  pvals_tt = sp.posthoc_nemenyi_friedman(ranks_tt)
    ranks_te = aligned_ranks(mat_te);  pvals_te = sp.posthoc_nemenyi_friedman(ranks_te)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharey=False)
    sp.critical_difference_diagram(
        pd.Series(ranks_tt.mean(0), index=models), pvals_tt, ax=ax1
    )
    ax1.set_title("Transformer: per-target", pad=12)

    sp.critical_difference_diagram(
        pd.Series(ranks_te.mean(0), index=models), pvals_te, ax=ax2
    )
    ax2.set_title("Transformer: per-embedding", pad=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR/"transformer_models_2view_cd.png", bbox_inches="tight")
    plt.show()


    # Per-target analysis (blocks = targets)
    print("\n➤ Transformer: per-target analysis (embeddings + folds collapsed)\n")
    run_friedman(mat_tt, "targets", models, "transf_per_target_models")

    # Per-embedding analysis (blocks = embeddings)
    print("\n➤ Transformer: per-embedding analysis (targets + folds collapsed)\n")
    run_friedman(mat_te, "embeddings", models, "transf_per_embedding_models")


    section("B2  TRANSFORMER EMBEDDINGS  –  Two-view EMBEDDING comparison")

    mat_em = matrix_per_model_compare_embeddings()
    mat_et = matrix_per_target_compare_embeddings(embeddings)

    # Two-view CD for transformer EMBEDDINGS
    ranks_em = aligned_ranks(mat_em);  pvals_em = sp.posthoc_nemenyi_friedman(ranks_em)
    ranks_et = aligned_ranks(mat_et);  pvals_et = sp.posthoc_nemenyi_friedman(ranks_et)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharey=False)
    sp.critical_difference_diagram(
        pd.Series(ranks_em.mean(0), index=embeddings), pvals_em, ax=ax1
    )
    ax1.set_title("Embeddings: per-model", pad=12)

    sp.critical_difference_diagram(
        pd.Series(ranks_et.mean(0), index=embeddings), pvals_et, ax=ax2
    )
    ax2.set_title("Embeddings: per-target", pad=12)

    plt.tight_layout()
    fig.savefig(FIG_DIR/"transformer_embeddings_2view_cd.png", bbox_inches="tight")
    plt.show()

    # Per-model embedding analysis (blocks = models)
    print("\n➤ Embeddings: per-model analysis (blocks = models)\n")
    run_friedman(mat_em, "models", embeddings, "transf_per_model_embeddings")

    # Per-target embedding analysis (blocks = targets)
    print("\n➤ Embeddings: per-target analysis (blocks = targets)\n")
    run_friedman(mat_et, "targets", embeddings, "transf_per_target_embeddings")


    # ────────────────────────────────────────────────────────────────────────────
    # C.  GROUP COMPARISONS
    # ────────────────────────────────────────────────────────────────────────────
    section("C  GROUP COMPARISONS")

    # helper to format significance sentences
    def say_sig(name1, name2, p, med1, med2):
        verdict = "SIGNIFICANT" if p < .05 else "not significant"
        return (f"Wilcoxon p = {p:.3g}  →  {name1} median = {med1:.3f}, "
                f"{name2} median = {med2:.3f}  →  difference is {verdict}")

    # ── C1) Model families (Local / Global / Chain)
    print("\na)  Model families (Local / Global / Chain)")
    families = {
        "Local":  [m for m in models if m.startswith("local_")],
        "Global": [m for m in models if m.startswith("global_")],
        "Chain":  [m for m in models if m.startswith("chain_")],
    }
    # prepare performance arrays
    family_perf = {
        name: np.array([
            np.median([np.median(data[e][m]) for m in model_list])
            for e in embeddings
        ])
        for name, model_list in families.items()
    }
    # bar-plot for families
    fam_med = {name: np.median(arr) for name, arr in family_perf.items()}
    pd.Series(fam_med).sort_values().plot.bar(
        figsize=(4,3), title="Group a: Family medians (Local/Global/Chain)"
    )
    plt.ylabel("Median RRMSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"group_a_bar.png", bbox_inches="tight")
    plt.show()

    # Wilcoxon for families
    for (k1, v1), (k2, v2) in combinations(family_perf.items(), 2):
        p = wilcoxon(v1, v2, zero_method="zsplit")[1]
        print(" •", say_sig(k1, k2, p, np.median(v1), np.median(v2)))

    # ── C2) Pooling strategies (mean / max / CLS)
    print("\nb)  Token-level pooling strategies (mean / max / CLS)")
    pool_groups = {
        "mean": [e for e in embeddings if "_mean" in e],
        "max":  [e for e in embeddings if "_max"  in e],
        "cls":  [e for e in embeddings if any(pat in e for pat in ("_cls","_cls_s"))],
    }
    pool_perf = {
        name: np.array([
            np.median([np.median(data[e][m]) for e in emb_list])
            for m in models
        ])
        for name, emb_list in pool_groups.items() if emb_list
    }
    # bar-plot for pooling
    pool_med = {name: np.median(arr) for name, arr in pool_perf.items()}
    pd.Series(pool_med).sort_values().plot.bar(
        figsize=(4,3), title="Group b: Pooling medians (mean/max/cls)"
    )
    plt.ylabel("Median RRMSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"group_b_bar.png", bbox_inches="tight")
    plt.show()

    # Wilcoxon for pooling
    for (k1, v1), (k2, v2) in combinations(pool_perf.items(), 2):
        p = wilcoxon(v1, v2, zero_method="zsplit")[1]
        print(" •", say_sig(k1, k2, p, np.median(v1), np.median(v2)))

    # ── C3) Word- vs sentence-level embeddings
    print("\nc)  Transformer *word* vs *sentence* embeddings")
    word_embs = [e for e in embeddings if any(pat in e for pat in ("_mean","_max","_cls"))]
    sent_embs = [e for e in embeddings if e not in word_embs and e not in STATIC_EMB]
    word_vec  = np.array([np.median([np.median(data[e][m]) for e in word_embs])
                          for m in models])
    sent_vec  = np.array([np.median([np.median(data[e][m]) for e in sent_embs])
                          for m in models])
    # bar-plot for word vs sent
    pd.Series({"word-level": np.median(word_vec),
               "sentence-level": np.median(sent_vec)}
    ).plot.bar(
        figsize=(4,3), title="Group c: Word vs Sentence embeddings"
    )
    plt.ylabel("Median RRMSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"group_c_bar.png", bbox_inches="tight")
    plt.show()

    # Wilcoxon for word vs sent
    p = wilcoxon(word_vec, sent_vec, zero_method="zsplit")[1]
    print(" •", say_sig("word-level", "sentence-level", p,
                       np.median(word_vec), np.median(sent_vec)))

    # ── C4) Static vs Transformer embeddings
    print("\nd)  Static (Word2Vec + FastText) vs Transformer embeddings")

    # static embeddings come from a_static (baseline_data), transformers from b_frozen (data)
    static_vec  = np.array([
        np.median([np.median(get_rrmse(e, m)) for e in STATIC_EMB])
        for m in models
    ])

    transf_embs = [e for e in embeddings if e not in STATIC_EMB]  # embeddings = transformer-only list
    transf_vec  = np.array([
        np.median([np.median(get_rrmse(e, m)) for e in transf_embs])
        for m in models
    ])

    # bar-plot for static vs transformer
    pd.Series({"static": np.median(static_vec),
           "transformer": np.median(transf_vec)}
    ).plot.bar(
        figsize=(4,3), title="Group d: Static vs Transformer embeddings"
    )
    plt.ylabel("Median RRMSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR/"group_d_bar.png", bbox_inches="tight")
    plt.show()

    # Wilcoxon for static vs transformer
    p = wilcoxon(static_vec, transf_vec, zero_method="zsplit")[1]
    print(" •", f"Wilcoxon p = {p:.3g}  →  static median = {np.median(static_vec):.3f}, "
                  f"transformer median = {np.median(transf_vec):.3f}  →  "
                  f"difference is {'SIGNIFICANT' if p < .05 else 'not significant'}")


    print(f"\nAll group-comparison tables & plots are in {REPORT_DIR}")


    # ── C5) Sentence vs Static vs Token-level (frozen) embeddings
    # Compares median RRMSE aggregated across folds/targets and grouped as:
    #   • "static"      → {word2vec, fasttext}
    #   • "token_mean"  → *_mean
    #   • "token_max"   → *_max
    #   • "token_cls"   → *_cls / *_cls_s
    #   • "token_avg"   → median over (mean, max, cls)
    #   • "sentence"    → transformer sentence embeddings (all others, non-static)
    # Blocks for stats = models available in all groups (intersection).
    # Saves: CSV + Wilcoxon raw/adjusted p + Cliff’s Δ, and a bar chart.
    # ─────────────────────────────────────────────────────────────────────────────
    section("Δ  Sentence vs Static vs Token-level (frozen)")

    def _models_available_for_static():
        have_static_in_data = all(e in data for e in STATIC_EMB)
        if have_static_in_data:
            inter = set.intersection(*[set(data[e].keys()) for e in STATIC_EMB])
        else:
            inter = set.intersection(*[set(baseline_data[e].keys()) for e in STATIC_EMB])
        return sorted(inter)

    static_model_candidates = _models_available_for_static()
    model_list = sorted(set(models).intersection(static_model_candidates))
    if not model_list:
        raise RuntimeError("No overlapping models between transformer `models` and the available static results.")
    print(f"Models used as blocks (n={len(model_list)}): {', '.join(model_list)}")

    token_mean = [e for e in embeddings if "_mean" in e]
    token_max  = [e for e in embeddings if "_max"  in e]
    token_cls  = [e for e in embeddings if any(pat in e for pat in ("_cls","_cls_s"))]
    token_all  = sorted(set(token_mean + token_max + token_cls))
    sent_embs  = [e for e in embeddings if (e not in STATIC_EMB) and (e not in token_all)]
    print(f"Groups → static:{len(STATIC_EMB)}  token_mean:{len(token_mean)}  token_max:{len(token_max)}  "
          f"token_cls:{len(token_cls)}  sentence:{len(sent_embs)}")

    def _group_vec(emb_list):
        vec = []
        for m in model_list:
            vals = []
            for e in emb_list:
                try:
                    vals.append(np.median(get_rrmse(e, m)))
                except KeyError:
                    pass
            vec.append(np.nanmedian(vals) if len(vals) else np.nan)
        return np.array(vec, dtype=float)

    vec_static    = _group_vec(list(STATIC_EMB))
    vec_mean      = _group_vec(token_mean)
    vec_max       = _group_vec(token_max)
    vec_cls       = _group_vec(token_cls)
    vec_token_avg = np.nanmedian(np.vstack([vec_mean, vec_max, vec_cls]), axis=0)
    vec_sentence  = _group_vec(sent_embs)

    df = pd.DataFrame({
        "static":     vec_static,
        "token_mean": vec_mean,
        "token_max":  vec_max,
        "token_cls":  vec_cls,
        "token_avg":  vec_token_avg,
        "sentence":   vec_sentence,
    }, index=model_list)
    pre_n = df.shape[0]
    df_clean = df.dropna(how="any", subset=["static", "token_avg", "sentence"])
    post_n = df_clean.shape[0]
    if post_n < pre_n:
        dropped = set(df.index) - set(df_clean.index)
        print(f"Note: dropped {pre_n - post_n} model(s) due to missing combinations → {', '.join(dropped)}")
    df = df_clean

    df_path = REPORT_DIR / "sent_vs_static_vs_token_per_model.csv"
    df.to_csv(df_path)
    print("Per-model medians saved →", df_path)

    print("\nMedian RRMSE across models (lower is better):")
    print(df.median(0).sort_values().round(3).to_string())

    labels = ["static", "token_avg", "sentence"]
    mat = df[labels].to_numpy()
    wilc = wilcoxon_matrix(mat, labels)
    adj, delta = holm_correct_and_effects(wilc, mat, labels)
    wilc_path = REPORT_DIR / "sent_static_token_wilcoxon_raw_p.csv"
    adj_path  = REPORT_DIR / "sent_static_token_wilcoxon_holm_p.csv"
    dlt_path  = REPORT_DIR / "sent_static_token_cliffs_delta.csv"
    wilc.to_csv(wilc_path); adj.to_csv(adj_path); delta.to_csv(dlt_path)

    print("\nRaw Wilcoxon p-values:\n" + wilc.round(4).to_string())
    print("\nHolm–Bonferroni adjusted p-values:\n" + adj.round(4).to_string())
    print("\nCliff’s Δ effect sizes:\n" + delta.round(3).to_string())
    print(f"\nSaved → {wilc_path}\n        {adj_path}\n        {dlt_path}")

    for col in ["token_mean", "token_max", "token_cls"]:
        if df[col].isna().any():
            continue
        p = wilcoxon(df[col], df["sentence"], zero_method="zsplit")[1]
        print(f"Sentence vs {col}: p = {p:.3g} | med(sentence)={df['sentence'].median():.3f}  med({col})={df[col].median():.3f}")

    ax = df.median(0)[["static", "token_avg", "sentence"]].plot.bar(
        figsize=(4,3), title="Median RRMSE: static vs token_avg vs sentence"
    )
    plt.ylabel("Median RRMSE")
    plt.tight_layout()
    bar_path = FIG_DIR / "sent_static_token_bar.png"
    plt.savefig(bar_path, bbox_inches="tight")
    plt.show()
    print("Saved figure →", bar_path)

    # ──────────────────────────────────────────────────────────────
    print(f"\nAll tables are saved in  {REPORT_DIR.resolve()}")
    print(f"Critical-difference diagrams in  {FIG_DIR.resolve()}")

# ──────────────────────────────────────────────────────────────
# 5.  Entry-point
# ──────────────────────────────────────────────────────────────

if __name__=='__main__':
    main()