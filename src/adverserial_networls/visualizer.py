import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import matplotlib
matplotlib.use("TkAgg")
plt.style.use("bmh")


# 1. Pure Helper Functions
def _compute_sample_metrics(
    model, image: jnp.ndarray, pert: jnp.ndarray, label: int
) -> tuple[int, float, int, float]:
    """Computes predictions and max confidence probabilities for clean vs perturbed image."""
    logits_clean = model(image)
    logits_adv = model(image + pert)

    clean_pred = int(jnp.argmax(logits_clean))
    adv_pred = int(jnp.argmax(logits_adv))

    clean_prob = float(jax.nn.softmax(logits_clean)[clean_pred])
    adv_prob = float(jax.nn.softmax(logits_adv)[adv_pred])

    return clean_pred, clean_prob, adv_pred, adv_prob


def _setup_figure() -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Creates a 4-column plot grid for Clean, Perturbation, Adversarial, and Histogram views."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.canvas.manager.set_window_title("Adversarial Sample Batch Visualizer")
    plt.subplots_adjust(bottom=0.22, wspace=0.3)  # Reserve bottom space for controls
    return fig, axes


# 2. Individual Panel Renderers
def _render_image_panel(ax: plt.Axes, img_2d: jnp.ndarray, title: str, color: str = "black"):
    """Renders a single 2D image panel."""
    ax.clear()
    ax.imshow(img_2d, cmap="gray")
    ax.set_title(title, color=color, fontsize=10)
    ax.axis("off")


def _render_pert_panel(ax: plt.Axes, pert_2d: jnp.ndarray):
    """Renders the 2D spatial perturbation map."""
    ax.clear()
    ax.imshow(pert_2d, cmap="coolwarm")
    ax.set_title(f"Perturbation (δ)\nRange: [{pert_2d.min():.2f}, {pert_2d.max():.2f}]", fontsize=10)
    ax.axis("off")


def _render_histogram_panel(ax: plt.Axes, pert_flat: jnp.ndarray):
    """Renders the pixel perturbation noise distribution histogram."""
    ax.clear()
    ax.hist(pert_flat, bins=25, color="steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Perturbation Noise Hist", fontsize=10)
    ax.set_xlabel("Value (δ)", fontsize=8)
    ax.set_ylabel("Pixel Count", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)


# 3. Panel Compositor
def _render_panels(
    axes: tuple[plt.Axes, ...],
    clean_img: jnp.ndarray,
    pert_img: jnp.ndarray,
    metrics: tuple[int, float, int, float],
    true_label: int,
):
    """Renders all 4 panels onto the grid axes."""
    ax_clean, ax_pert, ax_adv, ax_hist = axes
    clean_pred, clean_prob, adv_pred, adv_prob = metrics

    # Squeeze channel axis: (1, H, W) -> (H, W)
    clean_2d = jnp.squeeze(clean_img)
    pert_2d = jnp.squeeze(pert_img)
    adv_2d = jnp.squeeze(clean_img + pert_img)

    # 1. Clean Panel
    clean_title = f"Clean Image\nTrue: {true_label} | Pred: {clean_pred}\nConf: {clean_prob:.1%}"
    clean_color = "green" if clean_pred == true_label else "red"
    _render_image_panel(ax_clean, clean_2d, clean_title, clean_color)

    # 2. Perturbation Spatial Panel
    _render_pert_panel(ax_pert, pert_2d)

    # 3. Adversarial Panel
    adv_title = f"Adversarial Sample\nPred: {adv_pred}\nConf: {adv_prob:.1%}"
    adv_color = "green" if adv_pred == true_label else "red"
    _render_image_panel(ax_adv, adv_2d, adv_title, adv_color)

    # 4. Noise Histogram Panel
    _render_histogram_panel(ax_hist, pert_2d.flatten())


# 4. Interactive Controller
def plot_interactive_batch(
    model,
    images: jnp.ndarray,
    pert: jnp.ndarray,
    labels: jnp.ndarray,
):
    """Spawns an interactive Matplotlib GUI to browse through a batch of adversarial samples."""
    batch_size = images.shape[0]
    fig, axes = _setup_figure()

    # Define Slider and Radio Button Widgets
    ax_slider = plt.axes([0.15, 0.08, 0.45, 0.04])
    slider = Slider(ax_slider, "Sample", 0, batch_size - 1, valinit=0, valfmt="%d")

    ax_radio = plt.axes([0.68, 0.02, 0.20, 0.12])
    radio = RadioButtons(ax_radio, ["All", "Misclassified", "Correct"], active=0)

    # Filter state
    state = {"indices": list(range(batch_size))}

    def _update_indices(_=None):
        mode = radio.value_selected
        indices = []
        for i in range(batch_size):
            _, _, adv_pred, _ = _compute_sample_metrics(model, images[i], pert[i], int(labels[i]))
            true_lbl = int(labels[i])

            if mode == "All":
                indices.append(i)
            elif mode == "Misclassified" and adv_pred != true_lbl:
                indices.append(i)
            elif mode == "Correct" and adv_pred == true_lbl:
                indices.append(i)

        state["indices"] = indices if indices else [0]
        _draw_sample(int(slider.val))

    def _draw_sample(idx: int):
        valid_indices = state["indices"]
        target_idx = valid_indices[idx % len(valid_indices)]

        metrics = _compute_sample_metrics(model, images[target_idx], pert[target_idx], int(labels[target_idx]))
        _render_panels(axes, images[target_idx], pert[target_idx], metrics, int(labels[target_idx]))
        fig.canvas.draw_idle()

    # Callbacks
    slider.on_changed(lambda val: _draw_sample(int(val)))
    radio.on_clicked(_update_indices)

    # Initial Draw
    _draw_sample(0)
    plt.show()