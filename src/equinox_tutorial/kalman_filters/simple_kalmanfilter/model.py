import typing

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp


class SimResult_t(typing.NamedTuple):
    solution_ts: jnp.ndarray 
    true_state: jnp.ndarray 


class ContinousLinearSystem(eqx.Module):
    F: jnp.ndarray
    H: jnp.ndarray
    Diffusion: jnp.ndarray 
    R: jnp.ndarray 

    def simulate(self, x0: jnp.ndarray, t0: float, t1: float, dt0: float, key: jax.random.PRNGKey) -> SimResult_t: 
        drift_key, obs_key = jax.random.split(key)
        
        drit_term = dfx.ODETerm(lambda t, y, args : self.F @ y)
        diffusion = dfx.ControlTerm(
            lambda t, y, args: self.Diffusion,
            dfx.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(self.F.shape[0],), key=key_sde)
        )
