"""Fit CA-MIRT 3PL on a real-data response matrix.

Input: a wide parquet whose rows are model_ids (string index) and columns
are item ids; cells are 0/1 for incorrect/correct (NaN allowed and treated
as incorrect by default; pass --drop_nan_items to drop incomplete columns).

Output (under --out_dir):
    models.csv   : model_id, row_mean, theta, alpha
    items.csv    : item_id, b, a, beta, c
    fit.json     : run metadata (best -log_post, runtime, hyperparameters)

The four panels reported in the paper are:
    HellaSwag  (K=4):  --c_floor 0.25 --c_prior_mean 0.28
    MMLU       (K=4):  --c_floor 0.25 --c_prior_mean 0.28
    GPQA       (K=4):  --c_floor 0.25 --c_prior_mean 0.28
    MMLU-Pro   (K=10): --c_floor 0.10 --c_prior_mean 0.18
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from camirt import fit_camirt


def load_response_matrix(parquet_path, drop_nan_items=False):
    wide = pd.read_parquet(parquet_path)
    try:
        wide.columns = wide.columns.astype(int)
        wide = wide.reindex(sorted(wide.columns), axis=1)
    except (TypeError, ValueError):
        wide = wide.reindex(sorted(wide.columns.astype(str)), axis=1)

    if drop_nan_items:
        keep = wide.notna().all(axis=0)
        wide = wide.loc[:, keep]
    else:
        wide = wide.fillna(0)

    wide = wide.loc[wide.mean(axis=1).sort_values().index]
    Y = wide.to_numpy(dtype=np.float32)
    return wide, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True,
                   help="Path to wide response parquet (rows=models, cols=items).")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for fitted parameters.")
    p.add_argument("--c_floor", type=float, default=0.25,
                   help="Hard lower bound on c_j (1/K for a K-option MCQ).")
    p.add_argument("--c_prior_mean", type=float, default=0.28,
                   help="Prior mean of c_j; must lie in (c_floor, 0.5).")
    p.add_argument("--c_prior_sigma", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--kappa_alpha", type=float, default=1.0)
    p.add_argument("--kappa_beta", type=float, default=1.0)
    p.add_argument("--tau_theta", type=float, default=1.0)
    p.add_argument("--n_starts", type=int, default=6)
    p.add_argument("--max_iter", type=int, default=5000)
    p.add_argument("--gtol", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop_nan_items", action="store_true",
                   help="Drop columns with any NaN (default: fill NaN with 0).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.responses}", flush=True)
    wide, Y = load_response_matrix(args.responses, drop_nan_items=args.drop_nan_items)
    I, J = Y.shape
    print(f"  Y: I={I} models x J={J} items, mean={Y.mean():.3f}", flush=True)

    t0 = time.time()
    fit = fit_camirt(
        Y,
        c_floor=args.c_floor,
        c_prior_mean=args.c_prior_mean,
        c_prior_sigma=args.c_prior_sigma,
        rho=args.rho,
        kappa_alpha=args.kappa_alpha,
        kappa_beta=args.kappa_beta,
        tau_theta=args.tau_theta,
        n_starts=args.n_starts,
        max_iter=args.max_iter,
        gtol=args.gtol,
        seed=args.seed,
        verbose=True,
    )
    runtime = time.time() - t0
    print(
        f"\nFit done in {runtime:.1f}s, best -log_post={fit['best_loss']:.2f}",
        flush=True,
    )

    models_df = pd.DataFrame({
        "model_id": list(wide.index),
        "row_mean": Y.mean(axis=1),
        "theta": fit["theta"],
        "alpha": fit["alpha"],
    })
    models_df.to_csv(out_dir / "models.csv", index=False)

    items_df = pd.DataFrame({
        "item_id": list(wide.columns),
        "a": fit["a"],
        "b": fit["b"],
        "beta": fit["beta"],
        "c": fit["c"],
    })
    items_df.to_csv(out_dir / "items.csv", index=False)

    meta = dict(
        responses=str(Path(args.responses).resolve()),
        I=int(I), J=int(J),
        c_floor=args.c_floor,
        c_prior_mean=args.c_prior_mean,
        c_prior_sigma=args.c_prior_sigma,
        rho=args.rho,
        kappa_alpha=args.kappa_alpha,
        kappa_beta=args.kappa_beta,
        tau_theta=args.tau_theta,
        n_starts=args.n_starts,
        max_iter=args.max_iter,
        gtol=args.gtol,
        seed=args.seed,
        drop_nan_items=bool(args.drop_nan_items),
        best_loss=float(fit["best_loss"]),
        runtime_s=float(fit["runtime_s"]),
    )
    with open(out_dir / "fit.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
