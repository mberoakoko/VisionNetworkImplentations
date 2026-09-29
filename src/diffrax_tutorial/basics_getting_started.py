import typing

import diffrax
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jaxtyping import Array, PyTree, Real
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

plt.style.use("bmh")
plt.rcParams.update({"font.size": 8})
matplotlib.use("TkAgg")

def _display_solution(t_seris: PyTree[Array], x_series: PyTree[Array]) -> None: 
    figure: Figure = plt.figure(figsize=(16, 9))
    ax: Axes = figure.add_subplot()
    sns.lineplot(
        data=pd.DataFrame({"x":jnp.array(t_seris),"y": jnp.array(x_series)}),
        x="x",
        y="y",
        ax=ax 
    )
    plt.tight_layout()
    plt.show()


def plot_diffrax_solution(
    sol: typing.Any,
    state_names: typing.Sequence[str] | None = None,
    title: str = "Diffrax Trajectories",
    figsize: tuple[float, float] = (9.0, 5.0),
    ax: Axes | None = None,
    show_ci: bool = True,
) -> tuple[Figure, Axes]:
    """Plots state trajectories from a Diffrax Solution object.

    Handles 1D states, arbitrary N-dimensional states, and 3D batched ensemble
    solutions (e.g. from jax.vmap across random seeds).

    Args:
        sol: Diffrax solution object containing `.ts` and `.ys`.
        state_names: Optional custom names for state variables.
        title: Plot title.
        figsize: Figure dimensions (width, height).
        ax: Existing Matplotlib Axes to draw on. If None, a new figure is created.
        show_ci: If True and input is an ensemble batch, plots 95% confidence intervals.

    Returns:
        A tuple of (fig, ax).
    """
    ts = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)

    # ------------------------------------------------------------------
    # 1. Standardize Dimensions to (Batch, Time, StateDim)
    # ------------------------------------------------------------------
    if ys.ndim == 1:
        # Scalar state over time -> (1, Time, 1)
        ys = ys[None, :, None]
    elif ys.ndim == 2:
        # Single trajectory: (Time, State) -> (1, Time, State)
        ys = ys[None, :, :]
    elif ys.ndim == 3:
        # Batched trajectory: (Batch, Time, State)
        pass
    else:
        raise ValueError(f"Unsupported array rank for sol.ys: {ys.ndim}")

    num_sims, num_steps, state_dim = ys.shape

    # ------------------------------------------------------------------
    # 2. Assign State Labels
    # ------------------------------------------------------------------
    if state_names is None:
        state_names = [f"$x_{{{i}}}$" for i in range(state_dim)]
    elif len(state_names) != state_dim:
        raise ValueError(
            f"Length of state_names ({len(state_names)}) does not match "
            f"state dimension ({state_dim})."
        )

    # ------------------------------------------------------------------
    # 3. Construct Long-Format DataFrame
    # ------------------------------------------------------------------
    # Reshape (Batch, Time, State) -> (Batch * Time, State)
    flat_ys = ys.reshape(-1, state_dim)
    df = pd.DataFrame(flat_ys, columns=state_names)

    # Broadcast time and simulation index across the batch dimension
    df["Time"] = np.tile(ts, num_sims)
    df["Simulation"] = np.repeat(np.arange(num_sims), num_steps)

    # Melt state columns into long format
    df_long = pd.melt(
        df,
        id_vars=["Time", "Simulation"],
        var_name="State Variable",
        value_name="Value",
    )

    # ------------------------------------------------------------------
    # 4. Render Seaborn Plot
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.set_theme(style="whitegrid")

    # If ensemble, use errorbar; if single run, units='Simulation' avoids aggregation overhead
    errorbar = ("ci", 95) if (num_sims > 1 and show_ci) else None
    units = "Simulation" if num_sims > 1 and not show_ci else None

    sns.lineplot(
        data=df_long,
        x="Time",
        y="Value",
        hue="State Variable",
        style="State Variable",
        units=units,
        estimator="mean" if units is None else None,
        errorbar=errorbar,
        linewidth=1.8,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("State Value")
    fig.tight_layout()

    return fig, ax


def ode_solution():
    logger.info("Starting Simple ODE Solution...")
    def model(t: Real, y: PyTree[Array], args: typing.Any) -> PyTree[Array]:
        return -y 
    term = diffrax.ODETerm(model)
    solver = diffrax.Dopri5()
    save_at = diffrax.SaveAt(ts=jnp.arange(0, 12, 0.01))
    solution = diffrax.diffeqsolve(term, solver, t0=0, t1=13, dt0=0.01, y0=1, saveat=save_at, progress_meter=diffrax.TqdmProgressMeter())
    
    logger.info("Solution complete... ")
    _display_solution(solution.ts, solution.ys)
    return solution 

def stochastic_diff_solution(t0: Real, t1: Real): 
    logger.info("Starting Stochastic Diff eqn...")
    
    def drift(t: Real, y: PyTree[Array], args: typing.Any) -> PyTree[Array]:
        return -y

    def diffusion(t: Real, y: PyTree[Array], args: typing.Any) -> PyTree[Array]:
        return 0.1 * t 

    brownian_motion = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jax.random.PRNGKey(0))
    terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))
    solver = diffrax.Euler()
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, 0.01))
    solution = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=1.0, saveat=saveat, progress_meter=diffrax.TqdmProgressMeter())
    
    _display_solution(solution.ts, solution.ys)

    return solution 


def difF_equation_with_gaussian_noises(t0: Real = 0 , t1: Real = 1):
    logger.info("Starting diff equation with gaussian noise ... ")

    A = jnp.array([[-0.5, 1.0], [-1.0, -0.5]])
    B = jnp.array([[0.0], [1.0]])
    u = lambda t, x: jnp.array([jnp.sin(t)])  # Control input u(t)

    dt0 = 0.01

    # System dimensions: state y is 2D, noise w is 2D
    y0 = jnp.array([1.0, 0.0])

    # Noise Intensity (sigma) such that continuous covariance rate Q = sigma @ sigma.T
    sigma = jnp.array([[0.1, 0.0], [0.0, 0.2]])

    def drift(t: Real, x: PyTree[Array], args: typing.Any ) -> Array: 
        return A@x + B@u(t, x)

    def diffusion(t: Real, x: PyTree[Array], args: typing.Any) -> Array: 
        return sigma 
    # 3. Brownian Motion Generator
    key = jax.random.PRNGKey(42)
    brownian_motion = diffrax.VirtualBrownianTree(
        t0=t0, t1=t1, tol=1e-3, shape=(2,), key=key
    )

    # 4. Construct SDE terms: dy = drift*dt + diffusion*dW
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian_motion),
    )

    # 5. Solve using Euler-Maruyama (Euler solver handles MultiTerm SDEs)
    sol = diffrax.diffeqsolve(
        terms=terms,
        solver=diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100)),
    )
    fig, ax = plot_diffrax_solution(
    sol, 
    state_names=["Position", "Velocity"],
    title="Harmonic Oscillator Trajectory"
    )
    plt.show()

if __name__ == "__main__": 
    difF_equation_with_gaussian_noises(t0 = 0, t1=12)
