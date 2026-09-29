import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

plt.rcParams.update({"font.size": 8})
matplotlib.use("TkAgg")
plt.style.use("bmh")


def plot_adaptive_control_comparison(
    sol_1: typing.Any,
    sol_2: typing.Any | None = None,
    c_ref: np.ndarray | float | None = None,
    true_omega: float | None = None,
    labels: tuple[str, str] = ("Deterministic", "Stochastic"),
    figsize: tuple[float, float] = (12.0, 8.0),
) -> tuple[Figure, np.ndarray]:
    """Plots state x(t) and parameter estimate w_hat(t) trajectories in a 2x2 grid.

    Top row (Axes [0,0], [0,1]): State trajectory x(t) vs Reference c(t).
    Bottom row (Axes [1,0], [1,1]): Parameter estimate w_hat(t) vs True Omega.

    Args:
        sol_1: Primary Diffrax solution (contains .ts and PyTree .ys with .x and .omega_hat).
        sol_2: Optional secondary Diffrax solution for side-by-side comparison.
        c_ref: Array or scalar reference setpoint values aligned with time steps.
        true_omega: Optional scalar representing the ground truth physical parameter.
        labels: Tuple naming the two solution modalities (e.g. ("Nominal", "Noisy")).
        figsize: Figure dimensions (width, height).

    Returns:
        Tuple of (fig, axes) where axes is a 2x2 numpy array of Matplotlib Axes.
    """
    sns.set_theme(style="whitegrid")

    # ------------------------------------------------------------------
    # 1. Helper to extract and format PyTree solutions into DataFrame
    # ------------------------------------------------------------------
    def extract_df(sol: typing.Any, modality_label: str) -> pd.DataFrame:
        ts = np.asarray(sol.ts)

        # Unpack PyTree state attributes
        x_vals = np.asarray(sol.ys.x).squeeze()
        w_vals = np.asarray(sol.ys.w_hat).squeeze()

        df_mod = pd.DataFrame(
            {
                "Time": ts,
                "State x": x_vals,
                "w_hat": w_vals,
                "Modality": modality_label,
            }
        )
        return df_mod

    # Build combined long-format DataFrame
    df_1 = extract_df(sol_1, labels[0])
    if sol_2 is not None:
        df_2 = extract_df(sol_2, labels[1])
        df_all = pd.concat([df_1, df_2], ignore_index=True)
        modalities = [labels[0], labels[1]]
    else:
        df_all = df_1
        modalities = [labels[0]]

    # ------------------------------------------------------------------
    # 2. Setup 2x2 Plot Grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True)

    # ------------------------------------------------------------------
    # 3. Top Row: State Trajectories x(t)
    # ------------------------------------------------------------------
    for col_idx, mod_label in enumerate(modalities):
        ax = axes[0, col_idx]
        sub_df = df_all[df_all["Modality"] == mod_label]

        # Plot state x(t)
        sns.lineplot(
            data=sub_df,
            x="Time",
            y="State x",
            ax=ax,
            color="tab:blue",
            linewidth=2.0,
            label="State $x(t)$",
        )

        # Overlay reference c(t)
        if c_ref is not None:
            ts_mod = sub_df["Time"].to_numpy()
            if np.isscalar(c_ref):
                c_vals = np.full_like(ts_mod, c_ref)
            else:
                c_vals = np.asarray(c_ref)

            ax.plot(
                ts_mod,
                c_vals,
                color="tab:red",
                linestyle="--",
                linewidth=1.8,
                label="Reference $c(t)$",
            )

        ax.set_title(f"State Dynamics [{mod_label}]")
        ax.set_ylabel("State Value $x$")
        ax.legend(loc="best")

    # If sol_2 is omitted, turn off the top-right subplot
    if sol_2 is None:
        axes[0, 1].axis("off")

    # ------------------------------------------------------------------
    # 4. Bottom Row: Parameter Adaptation w_hat(t)
    # ------------------------------------------------------------------
    for col_idx, mod_label in enumerate(modalities):
        ax = axes[1, col_idx]
        sub_df = df_all[df_all["Modality"] == mod_label]

        # Plot parameter estimate w_hat(t)
        sns.lineplot(
            data=sub_df,
            x="Time",
            y="w_hat",
            ax=ax,
            color="tab:green",
            linewidth=2.0,
            label=r"Estimate $\hat{\omega}(t)$",
        )

        # Overlay true omega value if provided
        if true_omega is not None:
            ax.axhline(
                y=true_omega,
                color="black",
                linestyle=":",
                linewidth=1.8,
                label=r"True $\omega^*$",
            )

        ax.set_title(f"Parameter Adaptation [{mod_label}]")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel(r"Parameter $\hat{\omega}$")
        ax.legend(loc="best")

    # If sol_2 is omitted, turn off the bottom-right subplot
    if sol_2 is None:
        axes[1, 1].axis("off")

    fig.tight_layout()
    return fig, axes
