# MODEL CARDS (Repository-wide)

> **Status:** anonymized for peer review. No datasets or model weights are distributed. This document
> summarizes *how* the models in the repo are built and evaluated, together with scope and limitations.
> Replace “Contact” and any placeholder fields after acceptance.

---

## 1) Traditional Multi‑Target Regressors (MTR)

All MTR models take a **fixed-size text embedding** as input and predict **14 continuous targets**
(8 curricular domains + 6 RIASEC facets). Evaluation uses **LOOCV** and the **RRMSE** metric
(RMSE normalized by a dummy-mean baseline per target). Feature dimensionality is reduced with **PCA** 
where `n_components ∈ {0.7, 0.8, 0.9}` (variance‑retained).

### Model inventory

| key                | family  | paradigm | base estimator                                             | notes |
|--------------------|---------|----------|------------------------------------------------------------|-------|
| `local_mean`       | baseline| local    | `DummyRegressor(strategy="mean")` via `MultiOutputRegressor` | per‑target mean baseline |
| `local_lr`         | linear  | local    | `LinearRegression()` via `MultiOutputRegressor`            | one regressor per target |
| `local_lasso`      | linear  | local    | `Lasso(alpha)` via `MultiOutputRegressor`                  | `alpha ∈ {0.001, 0.01, 0.1, 1.0, 10.0}` |
| `local_rf`         | tree    | local    | `RandomForestRegressor(n_estimators, max_depth)` via `MultiOutputRegressor` | `n_estimators ∈ {50,100}`, `max_depth ∈ {None,5,10}` |
| `global_bag`       | tree    | global   | `BaggingRegressor(DecisionTreeRegressor(max_depth))`       | single model predicts all targets jointly |
| `global_rf`        | tree    | global   | `RandomForestRegressor(n_estimators, max_depth)`           | single model for all targets |
| `chain_ERCcv_mean` | baseline| chain    | `DummyRegressor` inside ERC ensemble                       | OOF CV features propagate along chain |
| `chain_ERCcv_lr`   | linear  | chain    | `LinearRegression()` inside ERC ensemble                   | combined with frozen E5-base embedding **best overall** in our experiments |
| `chain_ERCcv_lasso`| linear  | chain    | `Lasso()` inside ERC ensemble                              |  |
| `chain_ERCcv_rf`   | tree    | chain    | `RandomForestRegressor()` inside ERC ensemble              |  |

**Regressor chains with OOF features.** `RegressorChainCV` produces **out‑of‑fold (OOF) predictions** for each
target in a chain order (randomized inside an **ensemble of chains**, `n_chains ∈ {3,5}` with `cv_splits ∈ {3,5}`).
OOF predictions are appended to features for subsequent targets; final per‑target models are refit on the full data
with the accumulated OOF features.

**Inputs.** Dense sentence vectors from one of the embedding sources (below).  
**Outputs.** A 14‑dimensional float vector of predicted scores.  
**Metric.** Per‑target RRMSE; we report per‑target vectors and aggregated medians across folds/combos as specified in notebooks.

---

## 2) Embedding Sources (features)

We use three families:

### (a) Static word embeddings
- `word2vec` — *Dutch Word2Vec (Coosto)*. Source: “coosto/dutch-word-embeddings” (Word2Vec, 300‑d).
- `fasttext` — *fastText CC nl* (`cc.nl.300.bin.gz`, 300‑d).

> **Note.** We mean‑pool token vectors into one sentence vector.

### (b) Frozen token‑level transformers + pooling
Model IDs (Hugging Face):
- **Dutch encoders**  
  `GroNLP/bert-base-dutch-cased` → `bert_mean`, `bert_max`, `bert_cls`  
  `pdelobelle/robbert-v2-dutch-base` → `robbert_v2_mean`, `robbert_v2_max`, `robbert_v2_cls`  
  `DTAI-KULeuven/robbert-2023-dutch-large` → `robbert2023_mean`, `robbert2023_max`, `robbert2023_cls`
- **English x‑lingual**  
  `microsoft/deberta-v3-large` → `deberta_mean`, `deberta_max`, `deberta_cls`
- **Multilingual**  
  `FacebookAI/xlm-roberta-large` → `xlmr_large_mean`, `xlmr_large_max`, `xlmr_large_cls`  
  `google/rembert` → `rembert_mean`, `rembert_max`, `rembert_cls`

> **Pooling:** mean / max / CLS on final hidden layer to obtain a single sentence vector (no finetuning).

### (c) Frozen sentence‑level encoders
- `jegormeister/bert-base-dutch-cased-snli` → `sbert_bert`
- `NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers` → `sbert_roberta`
- `embaas/sentence-transformers-multilingual-e5-base` → `e5_base`
- `embaas/sentence-transformers-multilingual-e5-large` → `e5_large`
- `sentence-transformers/LaBSE` → `labse`
- `sentence-transformers/paraphrase-xlm-r-multilingual-v1` → `simcse_xlmr_base`

> **Licensing:** consult each upstream model card for license and terms. This repo does **not** redistribute weights.

---

## 3) Fine‑Tuned Transformers (PEFT / LoRA)

We experiment with LoRA adapters on three sentence encoders: `sbert_bert`, `simcse_xlmr_base`, `e5_large`.
Adapters target attention projections; ~80% of lower encoder layers are frozen; a linear **14‑d regression head** is added.
Training uses small grids over learning rate with early stopping; evaluation by LOOCV (precomputed artifacts).

- **Checkpoints:** not distributed. Only summary metrics are provided under `outputs/` to enable review.  
- **Intended use:** research comparison against frozen‑embedding + MTR baselines in tiny‑data settings.  
- **Caveat:** with 95 train instances per fold, PEFT can underperform simpler MTR baselines.

---

## 4) LLM for Synthetic Text Generation (augmentation)

- **Model (local, via Ollama):** `gemma2:9b-instruct-q4_0`  
- **Use:** to generate candidate Dutch activity sentences under taxonomy‑guided prompting; **no model weights or generated text are distributed**.  
- **Labeling:** teacher–student pseudo‑labeling with the best MTR as teacher; students are retrained on augmented data within each CV fold (no leakage).
- **Diversity metric & outputs.** For augmentation analyses we compute diversity as the average \(1 - \cos\) distance in the **E5** embedding space between seed and augmented sentences.

> See `docs/taxonomy-guided-prompting.md` for the schema and prompting strategy. 

---

## 5) Data, Scope & Limitations

- **Data access:** original expert‑annotated dataset **cannot be released**. Review is supported by **precomputed artifacts** (tables/plots/metrics).  
- **Distribution shift:** Activity sentences are short; some targets have skewed coverage, influencing error variance.  
- **Fairness/ethics:** Models are **not** intended for high‑stakes individual recommendations without human oversight. Use aggregated analyses and qualitative validation.  
- **Reproducibility:** CPU‑only re‑runs are supported; see `REPRODUCIBILITY.md`.

---

## 6) Environmental & Dependencies

- **Primary libs:** scikit‑learn, numpy, scipy, matplotlib, scikit‑posthocs, statsmodels, pandas; optional: spaCy.  
- **Transformers:** Hugging Face `transformers`, `datasets`, `peft` (for LoRA).  
- **Execution:** notebooks under `notebooks/`; mirrored scripts under `scripts/`; outputs under `outputs/`.

---

## 7) Intended Use & Misuse

**Intended:** methodological benchmarking of MTR with diverse embeddings; educational research prototyping; figure reproduction for the paper.  
**Out‑of‑scope / Misuse:** automated per‑student advice, profiling, or decisions without expert review; deployment on populations or languages beyond the study’s scope.

---

## 8) Maintenance, Versioning, Contact

- **Version:** `v0.1.0-anon` (for review).  
- **Changelog:** initial public review package.  
- **Contact:** to be added after anonymous review.  
- **Citation:** see `CITATION.cff` (placeholder during review).

