
    # Clear previous frames
    for ax in axes:
        ax.clear()

    # Clean Image Panel
    ax_clean.imshow(clean_2d, cmap="gray")
    ax_clean.set_title(
        f"Clean Image\nTrue: {true_label} | Pred: {clean_pred}\nConf: {clean_prob:.1%}",
        color="green" if clean_pred == true_label else "red",
    )

    # Perturbation Panel
    ax_pert.imshow(pert_2d, cmap="coolwarm")
    ax_pert.set_title(f"Perturbation (δ)\nRange: [{pert_2d.min():.2f}, {pert_2d.max():.2f}]")

    # Adversarial Image Panel
    ax_adv.imshow(adv_2d, cmap="gray")
    ax_adv.set_title(
        f"Adversarial Sample\nPred: {adv_pred}\nConf: {adv_prob:.1%}",
        color="green" if adv_pred == true_label else "red",
    )

    for ax in axes:
        ax.axis("off")


# 3. Interactive Controller
def plot_interactive_batch(
    model,
    images: jnp.ndarray,
    pert: jnp.ndarray,
    labels: jnp.ndarray,
):
    """Spawns an interactive Matplotlib GUI to browse through a batch of adversarial samples.

    Args:
        model: Equinox CNN model instance.
        images: Clean batch tensor of shape (B, 1, H, W).
        pert: Perturbation batch tensor of shape (B, 1, H, W).
        labels: Ground-truth integer labels array of shape (B,).
    """
    batch_size = images.shape[0]
    fig, axes = _setup_figure()

    # Define Slider and Radio Button Widgets
    ax_slider = plt.axes([0.15, 0.08, 0.5, 0.04])
    slider = Slider(ax_slider, "Sample", 0, batch_size - 1, valinit=0, valfmt="%d")

    ax_radio = plt.axes([0.72, 0.02, 0.18, 0.12])
    radio = RadioButtons(ax_radio, ["All", "Misclassified", "Correct"], active=0)

    # State holder for interactive filter selection
    state = {"indices": list(range(batch_size))}

    def _update_indices(_=None):
        mode = radio.value_selected
        indices = []
        for i in range(batch_size):
            metrics = _compute_sample_metrics(model, images[i], pert[i], int(labels[i]))
            _, _, adv_pred, _ = metrics
            true_lbl = int(labels[i])

            if mode == "All":
                indices.append(i)
            elif mode == "Misclassified" and adv_pred != true_lbl:
                indices.append(i)
            elif mode == "Correct" and adv_pred == true_lbl:
                indices.append(i)

        state["indices"] = indices if indices else [0]
        # Re-trigger drawing
        _draw_sample(int(slider.val))

    def _draw_sample(idx: int):
        valid_indices = state["indices"]
        # Map slider value to filtered index
        target_idx = valid_indices[idx % len(valid_indices)]

        metrics = _compute_sample_metrics(model, images[target_idx], pert[target_idx], int(labels[target_idx]))
        _render_panels(axes, images[target_idx], pert[target_idx], metrics, int(labels[target_idx]))
        fig.canvas.draw_idle()

    # Attach event callbacks
    slider.on_changed(lambda val: _draw_sample(int(val)))
    radio.on_clicked(_update_indices)

    # Initial Draw
    _draw_sample(0)
    plt.show()