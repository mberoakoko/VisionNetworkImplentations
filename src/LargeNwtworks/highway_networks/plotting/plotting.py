import matplotlib
import matplotlib.pyplot as plt
from models.plain_vs_highway_no_pytree import ExperimentConfig

matplotlib.use("TkAgg")
plt.style.use("bmg")
plt.rcParams.update({"font.size": 8})

def plot_results(
    plain_hist: dict[str, list[float]], 
    highway_hist: dict[str, list[float]], 
    config: ExperimentConfig
):
    """Generates comparison visualizations for training dynamics."""
    epochs = range(1, config.epochs + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Training Loss Comparison
    axes[0].plot(epochs, plain_hist["train_loss"], 'o-', label="Plain Network", color="#E63946", linewidth=2)
    axes[0].plot(epochs, highway_hist["train_loss"], 's-', label="Highway Network", color="#1D3557", linewidth=2)
    axes[0].set_title(f"Training Loss ({config.num_layers} Layers Deep)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].legend()

    # 2. Test Accuracy Comparison
    axes[1].plot(epochs, [a * 100 for a in plain_hist["test_acc"]], 'o-', label="Plain Network", color="#E63946", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in highway_hist["test_acc"]], 's-', label="Highway Network", color="#1D3557", linewidth=2)
    axes[1].set_title("Test Accuracy (%)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy %")
    axes[1].legend()

    # 3. Gradient Norm Stability
    axes[2].plot(epochs, plain_hist["grad_norms"], 'o-', label="Plain Network", color="#E63946", linewidth=2)
    axes[2].plot(epochs, highway_hist["grad_norms"], 's-', label="Highway Network", color="#1D3557", linewidth=2)
    axes[2].set_title("Global Gradient Norm (Gradient Flow)", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 Gradient Norm")
    axes[2].set_yscale('log')
    axes[2].legend()

    plt.tight_layout()
    output_filename = "highway_vs_plain_results.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n[Visualization] Comparison plot saved to '{output_filename}'.")
