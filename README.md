# CA-MIRT: Contamination-Aware MIRT (NeurIPS 2026 D&B submission code)

Reference implementation of the CA-MIRT 3PL MAP estimator and the
parameter-recovery simulation grid reported in the paper.

This release contains **only** what is needed to reproduce the method
(the response-only contamination-and-ability fit) and the simulation
study. AUC / external-concordance analysis, plotting, and exploratory
scripts are not included.

## Contents

```
code/
├── camirt/
│   ├── __init__.py
│   └── model.py            # CA-MIRT 3PL MAP estimator (L-BFGS-B, multi-start)
├── run_simulation.py       # Simulation grid: parameter recovery
├── run_real_data.py        # Fit a single response-matrix parquet
├── requirements.txt
├── LICENSE
└── README.md
```

## Model

For a binary response `Y_ij` of model `i` on item `j`, CA-MIRT 3PL is

```
P(Y_ij = 1) = c_j + (1 - c_j) * [ alpha_i * beta_j * rho + (1 - alpha_i * beta_j) * f_ij ]
f_ij        = sigmoid( a_j * (theta_i - b_j) )
```

with `c_j` the per-item guessing floor (bounded in `[1/K, 0.5]`),
`alpha_i` the model-side contamination propensity (bounded in `[0, 1]`),
`beta_j` the item exposure (bounded in `[0, 1]`), `rho` the
memorization-conditional accuracy (default `1.0`), and gauge fix
`theta_0 = 0`, `a_0 = 1`. MAP estimate is obtained by L-BFGS-B from
`n_starts` random initializations; the lowest-loss start is returned.

The released estimator uses **uniform Beta(1, 1) priors** on `alpha`
and `beta` (`kappa_alpha = kappa_beta = 1.0`), an `N(0, tau_theta^2)`
prior on `theta` with `tau_theta = 1.0`, and a soft prior on `logit_c`
centered at the K-option chance rate. No model-level anchors, no
soft family priors, no per-pipeline random effects: contamination has
to come from the response matrix alone.

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

JAX is used for autodiff and JIT. CPU is the default backend; the
runtime numbers in the paper are on CPU. To use GPU, install a
matching `jax[cuda12]` build per the official JAX instructions.

## Reproducing the simulation study

The main-paper simulation grid is

* `(I, J) ∈ {2000, 5000} × {5000, 10000, 15000}`,
* `(p_contam, p_expose) ∈ {0.30, 0.50, 0.80}^2`,
* 4-option MCQ data-generating model (`c_j ~ U(0.25, 0.50)`),
* 10 replicates per cell,
* 6 random starts per fit.

To reproduce it:

```bash
python run_simulation.py \
    --replicas 10 --n_starts 6 \
    --out_dir results/simulation
```

This writes `results/simulation/simulation_results.csv` with one row
per fit (Pearson r, Spearman, RMSE, top-10% Jaccard for `alpha`,
plus the analogous metrics for `theta`, `beta`, and `c`), and per-fit
ground-truth and estimated parameters under `results/simulation/raw/`.

A small smoke test (one cell, one replicate, ~1 minute on CPU):

```bash
python run_simulation.py --quick --out_dir results/simulation_smoke
```

To resume an interrupted run, pass `--resume`.

## Reproducing a real-data fit

`run_real_data.py` takes a wide response-matrix parquet (rows are
model ids, columns are item ids, cells are 0/1 for incorrect/correct)
and writes the fitted parameters.

For the four panels reported in the paper:

```bash
# HellaSwag (4-option, K=4)
python run_real_data.py \
    --responses path/to/hellaswag_responses.parquet \
    --out_dir   results/hellaswag \
    --c_floor 0.25 --c_prior_mean 0.28

# MMLU (4-option, K=4)
python run_real_data.py \
    --responses path/to/mmlu_responses.parquet \
    --out_dir   results/mmlu \
    --c_floor 0.25 --c_prior_mean 0.28

# GPQA-Extended (4-option, K=4)
python run_real_data.py \
    --responses path/to/gpqa_responses.parquet \
    --out_dir   results/gpqa \
    --c_floor 0.25 --c_prior_mean 0.28

# MMLU-Pro (10-option, K=10)
python run_real_data.py \
    --responses path/to/mmlu_pro_responses.parquet \
    --out_dir   results/mmlu_pro \
    --c_floor 0.10 --c_prior_mean 0.18
```

Each call produces

```
<out_dir>/
├── models.csv   # model_id, row_mean, theta, alpha
├── items.csv    # item_id, a, b, beta, c
└── fit.json     # hyperparameters, runtime, best -log posterior
```

## Data

The four response matrices analysed in the paper are derived from the
public Open LLM Leaderboard `details` dumps (HellaSwag and MMLU from
v1, MMLU-Pro and GPQA-Extended from v2). Each cell records whether
the model selected the gold MCQ option on a given item. To save space
and to comply with the leaderboard's per-model licensing, we do not
redistribute raw responses here; the response matrices for review are
hosted on Hugging Face under the anonymized release linked in the
paper, with a Croissant metadata file. A small per-panel sample of
the response matrices is included in that release for inspection.

The simulation pipeline (`run_simulation.py`) generates all of its
data internally and depends on no external dataset.

## API

```python
import numpy as np
from camirt import fit_camirt

Y = np.random.randint(0, 2, size=(500, 1000)).astype(np.float32)
fit = fit_camirt(
    Y,
    c_floor=0.25,
    c_prior_mean=0.28,
    n_starts=6,
)
fit["theta"], fit["alpha"], fit["a"], fit["b"], fit["beta"], fit["c"]
```

See the docstring of `fit_camirt` for the full argument list.

## Anonymization

This release contains no author names, institution affiliations, or
non-anonymous URLs. The `LICENSE` file lists the copyright holder as
"Anonymous Authors". Camera-ready replacements will be made after
acceptance.

## License

MIT (see `LICENSE`).
