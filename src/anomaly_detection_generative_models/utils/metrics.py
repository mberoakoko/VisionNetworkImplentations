import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from anomaly_detection_generative_models.config.anomaly_detection_config import FIG_DIR

matplotlib.use("TkAgg")
plt.rcParams.update({"font.size": 8})


def binary_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_plots(states, sensors, degraded, losses, nll, threshold, metrics):
    t = np.arange(len(states))
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, states[:, 0], lw=0.6, color="#1f4e79")
    ax[0].set_ylabel("position")
    ax[1].plot(t, states[:, 1], lw=0.6, color="#b85c38")
    ax[1].set_ylabel("velocity")
    ax[2].plot(t, sensors[:, 0], lw=0.4, color="#2f6f4e")
    ax[2].set_ylabel("sensor 0")
    ax[2].set_xlabel("time index")
    cut = int(np.argmax(degraded))
    for a in ax:
        a.axvline(cut, color="crimson", ls="--", lw=1)
    fig.suptitle("Van der Pol machine: healthy regime, then injected wear")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simulation.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#1f4e79")
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO (mini-batch, subsample-scaled)")
    ax.set_title("JIT-compiled SVI on marginal PPCA")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "elbo.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(nll, lw=0.7, color="#1f4e79")
    ax.axhline(threshold, color="crimson", ls="--", label=f"99th pct threshold {threshold:.1f}")
    ax.axvline(cut, color="gray", ls=":", label="wear onset")
    ax.set_xlabel("time index")
    ax.set_ylabel("posterior predictive NLL")
    ax.set_title(
        f"Anomaly score  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "anomaly.png", dpi=140)
    plt.close(fig)
