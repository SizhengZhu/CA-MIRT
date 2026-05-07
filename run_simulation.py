"""Reproduce the CA-MIRT 3PL simulation grid (parameter-recovery experiment).

Generates a 4-option (K=4, c ~ U(0.25, 0.50)) classic-uniform IRT grid:
    (I, J) in {2000, 5000} x {5000, 10000, 15000}
    (p_contam, p_expose) in {0.30, 0.50, 0.80}^2
fits each cell with the released MAP fitter, and reports recovery metrics
(Pearson r, Spearman, RMSE, top-10% Jaccard for alpha) against the
ground-truth parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from camirt import fit_camirt


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _safe_corr(fn, x, y):
    x = np.asarray(x); y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    r = fn(x, y)[0]
    return float(r) if np.isfinite(r) else float("nan")


def _rmse(x, y):
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))


def _top_jaccard(x, y, frac=0.10):
    k = max(1, int(round(len(x) * frac)))
    sx = set(np.argsort(-np.asarray(x))[:k])
    sy = set(np.argsort(-np.asarray(y))[:k])
    return float(len(sx & sy) / len(sx | sy))


def generate_params(I, J, p_contam, p_expose, seed):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=I)
    b = rng.normal(0.0, 1.0, size=J)
    a = np.exp(rng.normal(0.0, 0.5, size=J))
    c = rng.uniform(0.25, 0.50, size=J)

    is_contam = rng.random(I) < p_contam
    alpha = np.zeros(I)
    n_c = int(is_contam.sum())
    if n_c > 0:
        alpha[is_contam] = rng.uniform(0.0, 1.0, size=n_c)

    is_expose = rng.random(J) < p_expose
    beta = np.zeros(J)
    n_e = int(is_expose.sum())
    if n_e > 0:
        beta[is_expose] = rng.uniform(0.0, 1.0, size=n_e)

    return theta, alpha, a, b, beta, c, is_contam, is_expose


def sample_responses(theta, alpha, a, b, beta, c, seed, rho=1.0):
    rng = np.random.default_rng(seed)
    f = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    gamma = alpha[:, None] * beta[None, :]
    p_core = gamma * rho + (1.0 - gamma) * f
    p = c[None, :] + (1.0 - c[None, :]) * p_core
    return (rng.random(p.shape) < p).astype(np.float32)


def run_one(I, J, rep, p_contam, p_expose, n_starts, max_iter, gtol, out_dir):
    seed = (
        20260427
        + rep * 100_003
        + I * 17
        + J
        + int(round(p_contam * 1000)) * 7919
        + int(round(p_expose * 1000)) * 6151
    )
    key = (
        f"I{I}_J{J}_pc{int(round(p_contam * 100)):02d}"
        f"_pe{int(round(p_expose * 100)):02d}_rep{rep:02d}"
    )

    theta, alpha, a, b, beta, c, is_contam, is_expose = generate_params(
        I, J, p_contam, p_expose, seed
    )
    Y = sample_responses(theta, alpha, a, b, beta, c, seed + 1)

    t0 = time.time()
    fit = fit_camirt(
        Y,
        c_floor=0.25,
        c_prior_mean=0.375,
        c_prior_sigma=0.5,
        rho=1.0,
        kappa_alpha=1.0,
        kappa_beta=1.0,
        tau_theta=1.0,
        n_starts=n_starts,
        max_iter=max_iter,
        gtol=gtol,
        seed=seed + 2,
        verbose=True,
    )
    wall = time.time() - t0

    np.savez_compressed(
        out_dir / "raw" / f"{key}.npz",
        theta=theta, alpha=alpha, a=a, b=b, beta=beta, c=c,
        theta_hat=fit["theta"], alpha_hat=fit["alpha"], a_hat=fit["a"],
        b_hat=fit["b"], beta_hat=fit["beta"], c_hat=fit["c"],
        Y_shape=np.array(Y.shape),
    )

    return {
        "key": key,
        "I": I, "J": J, "rep": rep,
        "p_contam": p_contam, "p_expose": p_expose,
        "n_contam": int(is_contam.sum()),
        "n_expose": int(is_expose.sum()),
        "wall_time_s": round(wall, 2),
        "best_loss": fit["best_loss"],
        "n_starts": fit["n_starts"],
        "pearson_alpha": _safe_corr(pearsonr, alpha, fit["alpha"]),
        "spearman_alpha": _safe_corr(spearmanr, alpha, fit["alpha"]),
        "rmse_alpha": _rmse(alpha, fit["alpha"]),
        "top10_jaccard_alpha": _top_jaccard(alpha, fit["alpha"], 0.10),
        "pearson_beta": _safe_corr(pearsonr, beta, fit["beta"]),
        "spearman_beta": _safe_corr(spearmanr, beta, fit["beta"]),
        "rmse_beta": _rmse(beta, fit["beta"]),
        "pearson_theta": _safe_corr(pearsonr, theta, fit["theta"]),
        "spearman_theta": _safe_corr(spearmanr, theta, fit["theta"]),
        "rmse_c": _rmse(c, fit["c"]),
        "mean_true_alpha": float(alpha.mean()),
        "mean_est_alpha": float(fit["alpha"].mean()),
        "mean_true_beta": float(beta.mean()),
        "mean_est_beta": float(fit["beta"].mean()),
        "mean_true_c": float(c.mean()),
        "mean_est_c": float(fit["c"].mean()),
    }


def build_grid(replicas, cells, rates):
    grid = []
    for I, J in cells:
        for p_contam in rates:
            for p_expose in rates:
                for rep in range(replicas):
                    grid.append((I, J, rep, p_contam, p_expose))
    return grid


def load_done(csv_path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["key"].astype(str).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(ROOT / "results" / "simulation"))
    p.add_argument("--replicas", type=int, default=10)
    p.add_argument("--n_starts", type=int, default=6)
    p.add_argument("--max_iter", type=int, default=3000)
    p.add_argument("--gtol", type=float, default=1e-5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="Run a small smoke grid: I=500, J=1000, 1 cell, 1 rep.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "simulation_results.csv"

    if args.quick:
        cells = [(500, 1000)]
        rates = [0.50]
        replicas = 1
        n_starts = max(2, args.n_starts // 3)
    else:
        cells = [(I, J) for I in (2000, 5000) for J in (5000, 10000, 15000)]
        rates = [0.30, 0.50, 0.80]
        replicas = args.replicas
        n_starts = args.n_starts

    grid = build_grid(replicas, cells, rates)
    if args.limit is not None:
        grid = grid[: args.limit]

    done = load_done(csv_path) if args.resume else set()
    first_write = (not csv_path.exists()) or (not args.resume)

    print(
        f"Simulation grid: {len(grid)} fits "
        f"({len(cells)} (I,J) cells x {len(rates) ** 2} rate cells x "
        f"{replicas} reps), {len(done)} already done"
    )
    print(f"Writing to {csv_path}")

    for idx, (I, J, rep, p_contam, p_expose) in enumerate(grid, 1):
        key = (
            f"I{I}_J{J}_pc{int(round(p_contam * 100)):02d}"
            f"_pe{int(round(p_expose * 100)):02d}_rep{rep:02d}"
        )
        if key in done:
            continue
        print(f"[{idx}/{len(grid)}] {key}", flush=True)
        row = run_one(
            I, J, rep, p_contam, p_expose,
            n_starts=n_starts,
            max_iter=args.max_iter,
            gtol=args.gtol,
            out_dir=out_dir,
        )
        print(json.dumps({k: row[k] for k in [
            "wall_time_s", "pearson_alpha", "spearman_alpha",
            "rmse_alpha", "top10_jaccard_alpha", "rmse_c"]}, indent=2),
            flush=True)
        pd.DataFrame([row]).to_csv(
            csv_path, mode="a", header=first_write, index=False
        )
        first_write = False


if __name__ == "__main__":
    main()
