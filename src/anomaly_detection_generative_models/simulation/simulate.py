import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from anomaly_detection_generative_models.config.anomaly_detection_config import (
    DEGRADED_MU,
    DEGRADED_NOISE,
    DT,
    HEALTHY_MU,
    HEALTHY_NOISE,
    LATENT_DIM,
    OBS_DIM,
)

_NDARRAY_F32 = NDArray[np.float32]


def simulate_machine(n_steps: int, seed: int = 0) -> dict[str, _NDARRAY_F32]:
    """Noisy Van der Pol oscillator, lifted to 30 sensors, with late wear.
    Hidden state is (position, velocity, slow thermal state). Wear begins at
    85% of the timeline: the nonlinear damping coefficient and process noise
    both increase, which is the friction-drift story from the syllabus.
    """
    key = jax.random.PRNGKey(seed)
    key, k_proj, k_bias, k_init = jax.random.split(key, 4)
    
    projection = jax.random.normal(k_proj, (OBS_DIM, LATENT_DIM)) / jnp.sqrt(LATENT_DIM)
    bias = 0.15 * jax.random.normal(k_bias, (OBS_DIM,))
    state0 = jnp.array([1.2, 0.0, 0.3])

    degrade_at = int(0.85 * n_steps)

    def step(carry, t):
        state, key = carry
        key, k_proc, k_obs = jax.random.split(key, 3)
        worn = t >= degrade_at
        mu = jnp.where(worn, DEGRADED_MU, HEALTHY_MU)
        proc = jnp.where(worn, DEGRADED_NOISE, HEALTHY_NOISE)
        x, v, temp = state
        dx = v
        dv = mu * (1.0 - x**2) * v - x - 0.05 * temp
        dtemp = 0.02 * (x**2 - temp)
        deriv = jnp.stack([dx, dv, dtemp])
        process = jax.random.normal(k_proc, (LATENT_DIM,)) * proc
        state = state + DT * deriv + process
        obs_noise = jnp.where(worn, 0.35, 0.12)
        sensors = projection @ state + bias + jax.random.normal(k_obs, (OBS_DIM,)) * obs_noise
        return (state, key), (state, sensors, worn)

    _, (states, sensors, worn) = jax.lax.scan(
        step, (state0, key), jnp.arange(n_steps)
    )
    return {
        "state": np.asarray(states, dtype=np.float32),
        "sensors": np.asarray(sensors, dtype=np.float32),
        "degraded": np.asarray(worn, dtype=np.float32),
        "projection": np.asarray(projection, dtype=np.float32),
        "bias": np.asarray(bias, dtype=np.float32),
        "degrade_at": np.array([degrade_at]),
    }
