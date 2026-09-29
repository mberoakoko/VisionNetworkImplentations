import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from loguru import logger


def generate_reference_path(
    t0: float, t1: float, num_points: int = 1000, profile: str = "square"
) -> diffrax.CubicInterpolation:
    """Generates a smooth or piecewise differentiable command trajectory path c(t)."""
    ts = jnp.linspace(t0, t1, num_points)
    
    if profile == "square":
        # Square wave swinging between -5.0 and +5.0
        set_val = 5.0
        cs = jnp.where(jnp.sin(2 * jnp.pi * 0.2 * ts) >= 0, set_val, -set_val)
    elif profile == "sine":
        cs = 5.0 * jnp.sin(2 * jnp.pi * 0.5 * ts)
    elif profile == "ramp_step":
        # Ramp up, hold, step down
        cs = jnp.clip(2.0 * ts, 0.0, 5.0) - jnp.where(ts > 3.0, 10.0, 0.0)
    else:
        raise ValueError(f"Unknown profile {profile}")

    # Calculate derivative coefficients for smooth interpolation
    coeffs = diffrax.backward_hermite_coefficients(ts, cs)
    return diffrax.CubicInterpolation(ts, coeffs)



class SystemState(typing.NamedTuple):
    x: Array 
    w_hat: Array 



class PhysicalPlant(eqx.Module):
    """
    Simple plant dynamics 
    """
    omega_true: float 

    def dynamics(self, t: float, x: Array, u: Array) -> Array:
        return self.omega_true * x + u 

class NoisyPhysicalPlant(eqx.Module):
    omega_true: float
    process_noise_std: float  # Process disturbance intensity (sigma_w)
    sensor_noise_std: float   # Measurement noise intensity (sigma_v)

    def drift(self, t: float, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Deterministic dynamics: dx = (omega_true * x + u) dt"""
        return self.omega_true * x + u

    def diffusion(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """Process disturbance gain: G dW_t"""
        return jnp.array([self.process_noise_std])

    def observe(self, x: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Simulate a noisy sensor reading y(t) = x(t) + v(t)"""
        sensor_noise = jax.random.normal(key, shape=x.shape) * self.sensor_noise_std
        return x + sensor_noise


class AdaptiveReferencePointController(eqx.Module):
    """
    Simple Reference Point adaptive controller.
    """
    alpha: float 
    gamma: float 
    
    def compute_control(self, state: SystemState, c: Array) -> Array: 
        x, omega_hat = state 
        u_n = -self.alpha * (x - c) 
        u_a = -omega_hat * x 
        return u_n + u_a 

    def adaptation_law(self, state: SystemState, c: Array) -> Array:
        x, _ = state
        error = x - c 
        return self.gamma * x * error


class ClosedLoopSystem(eqx.Module):
    """Combine the contoller and the plant to create an augmented vector field. """
    plant: PhysicalPlant 
    controller: AdaptiveReferencePointController 


    def drift(self, t: float, state: SystemState, args: typing.Any|dict) -> SystemState:
        
        x, _ = state 
        c_path: diffrax.AbstractPath = args["c_path"]
        c = c_path.evaluate(t)

        u = self.controller.compute_control(state, c)
        dx = self.plant.dynamics(t, x, u)
        d_omega_hat = self.controller.adaptation_law(state, c)  
        return SystemState(dx,d_omega_hat)

    def __call__(self, t: float, state: SystemState, c: Array,  args: typing.Any | dict ) -> SystemState:
        u = self.controller.compute_control(state, c)
        x, _ = state 
        dx = self.plant.dynamics(t, x, u)
        d_omega_hat = self.controller.adaptation_law(state, c)

        return SystemState(
            x=dx,
            w_hat=d_omega_hat 
        )


class NoisyClosedLoopSystem(eqx.Module):
    plant: NoisyPhysicalPlant 
    controller: AdaptiveReferencePointController 

    def drift(self, t: float, state: SystemState, args: dict) -> SystemState:
        """Computes deterministic drift (dt terms) for both Plant and Controller."""
        # 1. Measurement Noise Injection
        # Note: In continuous-time, if measurement noise is high-frequency, 
        # we can sample pseudo-noise via PRNG key or pass an interpolated noise path.
        key = args.get("key", jax.random.PRNGKey(0))
        key_t = jax.random.fold_in(key, jnp.int32(t * 1000))  # Step-varying noise seed
        y_measured = self.plant.observe(state.x, key_t)
        c_path: diffrax.AbstractPath = args["c_path"]

        c = c_path.evaluate(t)


        # 2. Control signal computed from NOISY measurement
        u = self.controller.compute_control(
            state=SystemState(
                x=y_measured,
                w_hat=state.w_hat 
            ),
            c=c 
        )

        # 3. Plant drift & Adaptation drift
        dx_drift = self.plant.drift(t, state.x, u)
        d_omega_hat = self.controller.adaptation_law(
            state=SystemState(
                x=y_measured,
                w_hat=state.w_hat 
            ),
            c=c 
        )

        return SystemState(x=dx_drift, w_hat=d_omega_hat)

    def diffusion(self, t: float, state: SystemState, args: dict) -> SystemState:
        """Computes stochastic diffusion (dW_t terms)."""
        dx_diffusion = self.plant.diffusion(t, state.x)
        
        # Controller/Adaptation parameters usually don't have direct process noise 
        # (their randomness comes purely from the noisy measurement in drift)
        d_omega_hat_diffusion = jnp.zeros_like(state.w_hat) 

        return SystemState(x=dx_diffusion, w_hat=d_omega_hat_diffusion)



def run_adaptive_control_simulation(
    plant: PhysicalPlant,
    controller: AdaptiveReferencePointController,
    x0: float = 2.0,
    omega_hat0: float = 0.0,
    t0: float = 0.0,
    t1: float = 10.0,
    dt0: float = 0.01,
) -> tuple[PyTree, diffrax.Solution]: 
    logger.info("running simulation...")
    closed_loop_system = ClosedLoopSystem(plant, controller)
    z0 = SystemState(
        x=jnp.array([x0]),
        w_hat=jnp.array([omega_hat0]),
    )
    c_path = generate_reference_path(t0, t1)


    term = diffrax.ODETerm(closed_loop_system.drift)
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=z0,
        args={"c_path": c_path},
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
        saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 200)),
    )

    # 3. Evaluate reference array c(t) across saved time steps for plotting
    ts_eval = solution.ts
    c_ref_eval = jax.vmap(c_path.evaluate)(ts_eval)

   
    logger.info("Simulation complete...")
    return c_ref_eval, solution  

def run_noisy_adaptive_control_simulation(
        plant: NoisyPhysicalPlant,
    controller: AdaptiveReferencePointController,
    x0: float = 2.0,
    omega_hat0: float = 0.0,
    t0: float = 0.0,
    t1: float = 10.0,
    dt0: float = 0.01,
    seed: int  = 42
) -> tuple[PyTree, diffrax.Solution]:
    logger.info("running noisy simulation... ")
    
    key = jax.random.PRNGKey(seed)
    key_brownian, key_args = jax.random.split(key)
    
    z0 = SystemState(
        x=jnp.array([x0]),
        w_hat=jnp.array([omega_hat0])
    )
    
    brownian_tree = diffrax.VirtualBrownianTree(
        t0, t1, tol=1e-3, shape=jax.ShapeDtypeStruct(shape=(1, ), dtype=jnp.float32), key=key_brownian 
    )


    closed_loop_system = NoisyClosedLoopSystem(
        plant, controller 
    )
    
    c_path = generate_reference_path(t0, t1)

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(closed_loop_system.drift),
        diffrax.ControlTerm(closed_loop_system.diffusion, brownian_tree)
    )
    solution  = diffrax.diffeqsolve(
        terms=terms,
        solver=diffrax.Euler(),  # Euler acts as Euler-Maruyama for SDEs
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=z0,
        args={"c_path": c_path, "key": key_args},
        saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 200)),
        progress_meter=diffrax.TqdmProgressMeter()
    )
    ts_eval = solution.ts 
    c_ref_eval = jax.vmap(c_path.evaluate)(ts_eval)
    return c_ref_eval, solution 



if __name__ == "__main__":
    plant = PhysicalPlant(omega_true=1.5)
    plant_noisy = NoisyPhysicalPlant(omega_true=1.5, process_noise_std=0.2, sensor_noise_std=0.1)
    controller = AdaptiveReferencePointController(alpha=20.0, gamma=200.0)
    sol = run_noisy_adaptive_control_simulation(plant_noisy, controller, omega_hat0=20)
    
   
