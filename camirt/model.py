"""CA-MIRT 3PL MAP estimator (L-BFGS-B with multiple random starts)."""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_platform_name", "cpu")

_INIT_LOGIT_MEANS = (-4.0, -3.0, -2.0, -1.0, 0.0, 0.5)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _neg_log_posterior(
    raw, Y, I, J,
    rho_val, kappa_alpha, kappa_beta,
    c_prior_mu, c_prior_sigma, tau_theta, c_floor,
):
    idx = 0
    theta = raw[idx:idx + I]; idx += I
    raw_a = raw[idx:idx + J]; idx += J
    b_par = raw[idx:idx + J]; idx += J
    logit_alpha = raw[idx:idx + I]; idx += I
    logit_beta = raw[idx:idx + J]; idx += J
    logit_c = raw[idx:idx + J]

    theta = theta.at[0].set(0.0)
    sig_a = jax.nn.sigmoid(raw_a)
    a = 0.5 + 1.5 * sig_a
    a = a.at[0].set(1.0)
    alpha = jax.nn.sigmoid(logit_alpha)
    beta = jax.nn.sigmoid(logit_beta)
    c = c_floor + (0.5 - c_floor) * jax.nn.sigmoid(logit_c)

    f = jax.nn.sigmoid(a[None, :] * (theta[:, None] - b_par[None, :]))
    gamma = alpha[:, None] * beta[None, :]
    p_core = gamma * rho_val + (1.0 - gamma) * f
    p = c[None, :] + (1.0 - c[None, :]) * p_core
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)

    ll = jnp.sum(Y * jnp.log(p) + (1.0 - Y) * jnp.log(1.0 - p))

    lp_theta = -0.5 * jnp.sum((theta[1:] / tau_theta) ** 2)
    lp_a = jnp.sum(jnp.log(sig_a[1:] + 1e-10) + jnp.log(1.0 - sig_a[1:] + 1e-10))
    lp_b = -0.5 * jnp.sum(b_par ** 2)

    jac_alpha = jnp.sum(logit_alpha - 2.0 * jax.nn.softplus(logit_alpha))
    jac_beta = jnp.sum(logit_beta - 2.0 * jax.nn.softplus(logit_beta))
    shrink_alpha = -(kappa_alpha - 1.0) * jnp.sum(jax.nn.softplus(logit_alpha))
    shrink_beta = -(kappa_beta - 1.0) * jnp.sum(jax.nn.softplus(logit_beta))

    c_resid = (logit_c - c_prior_mu) / c_prior_sigma
    lp_c = -0.5 * jnp.sum(c_resid ** 2)

    return -(ll + lp_theta + lp_a + lp_b
             + jac_alpha + jac_beta
             + shrink_alpha + shrink_beta + lp_c)


def fit_camirt(
    Y: np.ndarray,
    *,
    c_floor: float = 0.25,
    c_prior_mean: float = 0.28,
    c_prior_sigma: float = 0.5,
    rho: float = 1.0,
    kappa_alpha: float = 1.0,
    kappa_beta: float = 1.0,
    tau_theta: float = 1.0,
    n_starts: int = 6,
    max_iter: int = 5000,
    gtol: float = 1e-5,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Fit CA-MIRT 3PL via MAP with L-BFGS-B and multiple random starts.

    Parameters
    ----------
    Y : (I, J) array of binary responses (model i on item j).
    c_floor : hard lower bound on the per-item guessing floor c_j
        (1/K for a K-option MCQ; e.g. 0.25 for K=4, 0.10 for K=10).
    c_prior_mean : prior mean of c_j on the probability scale; must satisfy
        c_floor < c_prior_mean < 0.5.
    c_prior_sigma : prior std of logit_c on the unconstrained scale.
    rho : memorization-conditional accuracy (default 1.0).
    kappa_alpha, kappa_beta : Beta(kappa,1) priors on alpha and beta.
        kappa=1 is uniform; the paper uses 1.0 throughout.
    tau_theta : std of the N(0, tau^2) prior on theta_i (i>=1).
    n_starts : number of random initializations.
    max_iter, gtol : L-BFGS-B options.
    seed : base seed for initialization.
    verbose : print per-start progress.

    Returns
    -------
    dict with keys: theta (I,), alpha (I,), a (J,), b (J,), beta (J,), c (J,),
    best_loss (float), runtime_s (float), n_starts (int).
    """
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError("Y must be 2D (I, J)")
    I, J = Y.shape
    Y_jax = jnp.array(Y)

    if not (c_floor < c_prior_mean < 0.5):
        raise ValueError(
            f"c_prior_mean={c_prior_mean} must lie strictly between "
            f"c_floor={c_floor} and 0.5"
        )
    c_prior_mu = _logit((c_prior_mean - c_floor) / (0.5 - c_floor))

    loss_fn = partial(
        _neg_log_posterior,
        Y=Y_jax, I=I, J=J,
        rho_val=float(rho),
        kappa_alpha=float(kappa_alpha),
        kappa_beta=float(kappa_beta),
        c_prior_mu=float(c_prior_mu),
        c_prior_sigma=float(c_prior_sigma),
        tau_theta=float(tau_theta),
        c_floor=float(c_floor),
    )
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    row_means = Y.mean(axis=1)
    eps = 0.02
    p_skill = np.clip(
        (row_means - c_floor) / max(1.0 - c_floor, 0.5),
        eps, 1.0 - eps,
    )
    theta_warm = np.clip(np.log(p_skill / (1.0 - p_skill)), -6.0, 6.0)

    best_loss = np.inf
    best_x = None
    t0 = time.time()
    for s in range(n_starts):
        rng = np.random.default_rng(seed + s * 1000)
        mu = _INIT_LOGIT_MEANS[s % len(_INIT_LOGIT_MEANS)]
        if s % 2 == 0:
            theta0 = (theta_warm + rng.normal(0.0, 0.3, I)).astype(np.float64)
        else:
            theta0 = rng.normal(0.0, 0.5, I).astype(np.float64)
        raw_a0 = rng.normal(0.0, 0.5, J).astype(np.float64)
        b0 = rng.normal(0.0, 0.5, J).astype(np.float64)
        logit_alpha0 = rng.normal(mu, 0.5, I).astype(np.float64)
        logit_beta0 = rng.normal(mu, 0.5, J).astype(np.float64)
        logit_c0 = rng.normal(c_prior_mu, 0.3, J).astype(np.float64)
        x0 = np.concatenate(
            [theta0, raw_a0, b0, logit_alpha0, logit_beta0, logit_c0]
        )

        def fg(x):
            v, g = loss_and_grad(jnp.asarray(x, dtype=jnp.float32))
            return float(v), np.asarray(g, dtype=np.float64)

        ts = time.time()
        res = minimize(
            fg, x0, jac=True, method="L-BFGS-B",
            options=dict(maxiter=max_iter, gtol=gtol, maxcor=10, disp=False),
        )
        if verbose:
            print(
                f"  start {s} (mu={mu:+.1f}): -log_post={res.fun:.2f}, "
                f"n_iter={res.nit}, max|g|={np.max(np.abs(res.jac)):.3f}, "
                f"t={time.time() - ts:.1f}s",
                flush=True,
            )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_x = res.x.copy()

    runtime_s = time.time() - t0

    idx = 0
    theta = best_x[idx:idx + I].copy(); idx += I
    theta[0] = 0.0
    raw_a = best_x[idx:idx + J]; idx += J
    a = 0.5 + 1.5 / (1.0 + np.exp(-raw_a))
    a[0] = 1.0
    b = best_x[idx:idx + J].copy(); idx += J
    logit_alpha = best_x[idx:idx + I]; idx += I
    alpha = 1.0 / (1.0 + np.exp(-logit_alpha))
    logit_beta = best_x[idx:idx + J]; idx += J
    beta = 1.0 / (1.0 + np.exp(-logit_beta))
    logit_c = best_x[idx:idx + J]
    c = c_floor + (0.5 - c_floor) / (1.0 + np.exp(-logit_c))

    return dict(
        theta=theta, alpha=alpha, a=a, b=b, beta=beta, c=c,
        best_loss=best_loss, runtime_s=runtime_s, n_starts=int(n_starts),
    )
