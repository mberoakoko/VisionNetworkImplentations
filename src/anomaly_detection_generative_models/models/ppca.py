from __future__ import annotations

from enum import Enum
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array
from loguru import logger
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.optim import Adam


class ModelModes(str, Enum):
    WEIGHTS = "w"
    MEAN = "mu"
    SIGMA = "sigma"
    PLATE_OBS = "obs"

class PPCAModelInterface(Protocol):
    obs_dim: int
    n_data: int

    def __call__(self, x: Array) -> None:
        ...

    def extract_params(self, draws: dict) -> tuple[Array, Array, Array]:
        """Extracts (W, mu, sigma) from posterior draws dict."""


class _PPCAModel(eqx.Module):
    latent_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    n_data: int = eqx.field(static=True)

    def __call__(self, x: Array):
        k, d = self.latent_dim, self.obs_dim 

        w = numpyro.sample(ModelModes.WEIGHTS, dist.Normal(0, 1).expand([d, k]).to_event(2))
        mu = numpyro.sample(ModelModes.MEAN, dist.Normal(0, 1).expand([d]).to_event(1))
        sigma = numpyro.sample(ModelModes.SIGMA, dist.HalfNormal(1.0))
        cov = jnp.matmul(w, w.T) + (sigma**2) * jnp.eye(d) + 1e-4 * jnp.eye(d)

        subsample_size = x.shape[0] if x is not None else None
        with numpyro.plate("obs", self.n_data, subsample_size=subsample_size):
            logger.debug(f"subsumple_size := {x.shape[0]}")
            numpyro.sample("x", dist.MultivariateNormal(mu, covariance_matrix=cov), obs=x)

    def extract_params(self, draws: dict) -> tuple[Array, Array, Array]:
        return draws["w"], draws["mu"], draws["sigma"]

class _ARDPPCAModel(eqx.Module):
    latent_dim: int = eqx.field(static=True)  # Set to max capacity k_max
    obs_dim: int = eqx.field(static=True)
    n_data: int = eqx.field(static=True)

    def __call__(self, x: Array):
        k_max, d = self.latent_dim, self.obs_dim

        # 1. Level 2 (Hyper-prior): Per-column relevance variance (HalfNormal)
        alpha_scale = numpyro.sample("alpha_scale", dist.HalfNormal(1.0).expand([k_max]).to_event(1))

        # 2. Level 1 (Prior): Scale W columns by alpha_scale
        # Shape of W is (d, k_max), alpha_scale scales each column k
        w = numpyro.sample(
            "w",
            dist.Normal(0, alpha_scale).expand([d, k_max]).to_event(2)
        )

        mu = numpyro.sample("mu", dist.Normal(0, 1).expand([d]).to_event(1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        cov = jnp.matmul(w, w.T) + (sigma**2) * jnp.eye(d) + 1e-4 * jnp.eye(d)

        subsample_size = x.shape[0] if x is not None else None
        with numpyro.plate("obs", self.n_data, subsample_size=subsample_size):
            numpyro.sample("x", dist.MultivariateNormal(mu, covariance_matrix=cov), obs=x)

    def extract_params(self, draws: dict) -> tuple[Array, Array, Array]:
        return draws["w"], draws["mu"], draws["sigma"]

class PPCADetector:
    name = "ppca"
    score_meaning = "posterior predictive NLL under the marginal Gaussian"

    def __init__(self,
                 model_cls: type = _ARDPPCAModel,
                 latent_dim: int = 3,
                 steps: int = 150,
                 batch_size: int = 256,
                 lr: float = 0.01):
        self.latent_dim = latent_dim
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self._model_cls = model_cls
        self._model: PPCAModelInterface = None
        self._guide = None
        self._params = None
        self.losses: list[float] = []

    @property
    def guide(self): return self._guide

    @property
    def params(self): return self._params


    def fit(self, healthy: np.ndarray, seed: int = 0) -> PPCADetector:
        healthy = np.asarray(healthy, dtype=np.float32)

        # Flatten inputs to (N_samples, obs_dim)
        if healthy.ndim == 3:
            healthy_flat = healthy.reshape(-1, healthy.shape[-1])
        else:
            healthy_flat = healthy
        
        n_data, obs_dim = healthy_flat.shape
        self._model = self._model_cls(self.latent_dim, obs_dim, n_data)
        self._guide = AutoDiagonalNormal(self._model)
        
        svi = SVI(self._model, self._guide, Adam(self.lr), loss=Trace_ELBO(num_particles=1))
        key = jax.random.PRNGKey(seed)

        init_batch = jnp.asarray(healthy_flat[: min(self.batch_size, n_data)])
        state = svi.init(key, init_batch)

        @jax.jit
        def update(state, batch):
            return svi.update(state, batch)

        losses = []
        n_batch = max(1, n_data // self.batch_size)
        for i in range(self.steps):
            start = (i % n_batch) * self.batch_size
            batch = jnp.asarray(healthy_flat[start : start + self.batch_size])
            state, loss = update(state, batch)
            losses.append(float(loss))
            print(f"{PPCADetector.__class__}... Current Loss :== {loss} ===", end="\r")

        self._params = svi.get_params(state)
        self.losses = losses
        self._svi = svi
        return self 

    def score(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim == 3:
            rows = rows.reshape(-1, rows.shape[-1])

        if self._params is None:
            raise RuntimeError("PPCADetector.score called before fit")

        predictive = Predictive(self._guide, params=self._params, num_samples=16)
        # Match sample dict keys to Enum string values
        draws = predictive(jax.random.PRNGKey(1))
        w_draws = draws[ModelModes.WEIGHTS]  # key "w"
        mu_draws = draws[ModelModes.MEAN]     # key "mu"
        sigma_draws = draws[ModelModes.SIGMA] # key "sigma"

        def nll(w, m, s, x):
            eye = jnp.eye(x.shape[-1])
            cov = w @ w.T + (s**2) * eye + 1e-4 * eye
            return -dist.MultivariateNormal(m, covariance_matrix=cov).log_prob(x)

        per_draw = jax.vmap(
            lambda w, m, s: nll(w, m, s, jnp.asarray(rows))
        )(w_draws, mu_draws, sigma_draws)

        return np.asarray(jnp.mean(per_draw, axis=0))


def inspect_effective_k(detector: PPCADetector, threshold: float = 0.05) -> dict:
    """Computes column norms of W to find effective latent dimension k."""
    predictive = Predictive(detector.guide, params=detector.params, num_samples=32)
    draws = predictive(jax.random.PRNGKey(0))

    # W shape: (num_samples, obs_dim, k_max) -> e.g. (32, 30, 15)
    w_draws = draws["w"]

    # Average W over posterior draws -> shape: (30, 15)
    w_mean = jnp.mean(w_draws, axis=0)

    # Compute Euclidean norm of each column (dimension j) -> shape: (15,)
    col_norms = jnp.linalg.norm(w_mean, axis=0)

    # Count dimensions whose column norm exceeds threshold
    active_mask = col_norms > threshold
    effective_k = int(jnp.sum(active_mask))

    return {
        "effective_k": effective_k,
        "column_norms": np.asarray(col_norms),
        "active_mask": np.asarray(active_mask)
    }