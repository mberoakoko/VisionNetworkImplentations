import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import Predictive
import matplotlib

matplotlib.use("TkAgg")
plt.style.use("dark_background")

def plot_ard_column_norms(
    detector,
    threshold: float = 0.05,
    num_samples: int = 64,
    figsize: tuple[int, int] = (10, 5),
) -> dict:
    """Plots the column norms of the weight matrix W across posterior draws

    to visualize effective latent dimension discovery and pruning.

    Parameters
    ----------
    detector : PPCADetector
        Fitted PPCADetector instance initialized with ARDPPCAModel.
    threshold : float, optional
        Threshold below which a column norm is considered pruned (default 0.05).
    num_samples : int, optional
        Number of posterior draws to average over (default 64).
    figsize : tuple[int, int], optional
        Matplotlib figure size.

    Returns
    -------
    dict
        Dictionary containing effective_k, column_norms, and active_mask.
    """
    if detector._params is None or detector._guide is None:
        raise RuntimeError(
            "Detector must be fitted before plotting column norms."
        )

    # 1. Sample posterior draws for W
    predictive = Predictive(
        detector._guide, params=detector._params, num_samples=num_samples
    )
    draws = predictive(jax.random.PRNGKey(0))

    if "w" not in draws:
        raise KeyError(
            "Sample 'w' not found in predictive draws. Ensure detector uses ARDPPCAModel."
        )

    # w_draws shape: (num_samples, obs_dim, k_max)
    w_draws = draws["w"]

    # 2. Compute column norms per draw: shape (num_samples, k_max)
    # axis=1 computes Euclidean norm across feature dimension (obs_dim)
    draw_col_norms = jnp.linalg.norm(w_draws, axis=1)

    # Mean and standard deviation across posterior draws
    mean_norms = np.asarray(jnp.mean(draw_col_norms, axis=0))
    std_norms = np.asarray(jnp.std(draw_col_norms, axis=0))

    k_max = len(mean_norms)
    k_indices = np.arange(1, k_max + 1)

    # Determine active vs pruned dimensions
    active_mask = mean_norms > threshold
    effective_k = int(np.sum(active_mask))

    # 3. Create Visualization
    fig, ax = plt.subplots(figsize=figsize)

    colors = [
        "#1f77b4" if is_active else "#d62728" for is_active in active_mask
    ]

    # Bar chart of mean column norms with error bars (uncertainty)
    bars = ax.bar(
        k_indices,
        mean_norms,
        yerr=std_norms,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    # Pruning threshold line
    ax.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Pruning Threshold ({threshold})",
    )

    # Formatting and annotation
    ax.set_xticks(k_indices)
    ax.set_xlabel("Latent Dimension ($k$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        r"Column Norm $\|W_{:, k}\|_2$", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        f"ARD Latent Dimension Pruning (Discovered $k = {effective_k}$ out of $k_{{max}} = {k_max}$)",
        fontsize=12,
        pad=12,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Add legend items for Active vs Pruned
    active_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        fc="#1f77b4",
        edgecolor="black",
        label=f"Active Dimensions ({effective_k})",
    )
    pruned_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        fc="#d62728",
        edgecolor="black",
        label=f"Pruned Dimensions ({k_max - effective_k})",
    )
    ax.legend(handles=[active_patch, pruned_patch, ax.get_legend_handles_labels()[0][0]])

    plt.tight_layout()
    plt.show()

    return {
        "effective_k": effective_k,
        "column_norms": mean_norms,
        "active_mask": active_mask,
    }