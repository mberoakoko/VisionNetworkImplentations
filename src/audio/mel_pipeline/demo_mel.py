"""Demo: synthesize a speech-like signal and plot the mel pipeline stages."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mel_pipeline import MelConfig, hz_to_mel, mel_spectrogram, mel_to_hz

OUT = Path(__file__).parent.parent / "workspace/artifacts"
OUT.mkdir(parents=True, exist_ok=True)
assert OUT.exists(), "file has to exist! "


def synthesize(config: MelConfig, duration_s: float = 3.525, seed: int = 0) -> jax.Array:
    """Vowel-like harmonic stack with a moving formant and amplitude envelope.

    Length 3.525 s at 8 kHz is 28,200 samples, the example on the slide.
    """
    sr = config.sample_rate
    n = int(duration_s * sr)
    t = jnp.arange(n) / sr
    key = jax.random.PRNGKey(seed)
    f0 = 140.0 + 25.0 * jnp.sin(2 * jnp.pi * 1.7 * t)
    phase = jnp.cumsum(2 * jnp.pi * f0 / sr)
    harmonics = jnp.arange(1, 18)
    # Two formants that drift, so the mel bands are not static.
    f1 = 500.0 + 180.0 * jnp.sin(2 * jnp.pi * 0.6 * t)
    f2 = 1500.0 + 400.0 * jnp.sin(2 * jnp.pi * 0.35 * t + 1.0)
    partials = []
    for h in harmonics:
        freq = f0 * h
        env = jnp.exp(-0.5 * ((freq - f1) / 180.0) ** 2) + 0.55 * jnp.exp(
            -0.5 * ((freq - f2) / 280.0) ** 2
        )
        partials.append(env * jnp.sin(h * phase) / h)
    harmonic = jnp.sum(jnp.stack(partials, axis=0), axis=0)
    gate = 0.15 + 0.85 * (0.5 + 0.5 * jnp.sin(2 * jnp.pi * 3.0 * t)) ** 2
    noise = 0.04 * jax.random.normal(key, (n,))
    wave = (harmonic * gate + noise).astype(jnp.float32)
    return wave / jnp.max(jnp.abs(wave)) * 0.8


def main() -> None:
    config = MelConfig()
    wave = synthesize(config)
    feats = mel_spectrogram(wave, config)
    feats = jax.tree.map(lambda x: np.asarray(x), feats)
    wave_np = np.asarray(wave)

    n_frames = feats.power.shape[0]
    times = np.arange(n_frames) * config.frame_step / config.sample_rate
    print(f"waveform {wave_np.shape} sr={config.sample_rate}")
    print(f"frames {feats.frames.shape} power {feats.power.shape}")
    print(f"filterbank {feats.filterbank.shape} log_mel {feats.log_mel.shape} mfcc {feats.mfcc.shape}")
    print(f"mel centers Hz: {np.array2string(feats.mel_centers_hz, precision=1)}")

    # --- Figure 1: waveform, STFT cartoon numbers, filterbank ---
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    fig.suptitle("Mel filterbank and spectrogram pipeline (JAX)", fontsize=14)

    t = np.arange(wave_np.shape[0]) / config.sample_rate
    axes[0, 0].plot(t, wave_np, color="#1f4e79", lw=0.4)
    axes[0, 0].set_title("Waveform  x  (8 kHz, 3.525 s)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    fb = feats.filterbank
    for row in fb:
        axes[0, 1].plot(feats.freqs_hz, row, lw=1.0)
    axes[0, 1].set_xlim(0, config.fmax + 200)
    axes[0, 1].set_title(f"Mel-spaced filterbank  T ∈ R^{{{config.n_mels}×{fb.shape[1]}}}")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Weight")
    axes[0, 1].axvline(config.fmin, color="0.4", ls="--", lw=0.7)
    axes[0, 1].axvline(config.fmax, color="0.4", ls="--", lw=0.7)

    log_power = np.log(np.maximum(feats.power, 1e-10)).T
    im = axes[1, 0].imshow(
        log_power,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], feats.freqs_hz[0], feats.freqs_hz[-1]],
        cmap="magma",
    )
    axes[1, 0].set_ylim(0, config.sample_rate / 2)
    axes[1, 0].set_title(r"Periodogram  $\log P$,  $P \in \mathbb{R}^{%d \times %d}$" % feats.power.shape[::-1])
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(
        feats.log_mel.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, config.n_mels - 1],
        cmap="magma",
    )
    axes[1, 1].set_title(r"Log mel spectrogram  $\log E \in \mathbb{R}^{%d \times %d}$" % feats.log_mel.shape[::-1])
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Mel filter index")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.savefig(OUT / "mel_pipeline_overview.png", dpi=140)
    plt.close(fig)

    # --- Figure 2: DCT / MFCCs and one frame through the bank ---
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    mid = n_frames // 2
    axes[0].plot(feats.freqs_hz, feats.power[mid], color="0.55", lw=0.8, label="periodogram")
    axes[0].set_xlim(0, config.fmax + 200)
    axb = axes[0].twinx()
    axb.plot(feats.mel_centers_hz, feats.mel_energy[mid], "o-", color="#b85c38", ms=4, label="mel energy")
    axes[0].set_title(f"One frame (t = {times[mid]:.2f} s) through T")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power")
    axb.set_ylabel("Filterbank energy")

    im = axes[1].imshow(
        feats.log_mel.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, config.n_mels - 1],
        cmap="inferno",
    )
    axes[1].set_title("Mel spectrogram (log energy)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel band")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im = axes[2].imshow(
        feats.mfcc.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, config.n_mfcc - 1],
        cmap="coolwarm",
    )
    axes[2].set_title(f"DCT → MFCC  ({config.n_mfcc} coeffs)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Cepstral index")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.savefig(OUT / "mel_dct.png", dpi=140)
    plt.close(fig)

    # --- Figure 3: mel warping, the thing the filterbank is built on ---
    hz = np.linspace(0, config.sample_rate / 2, 500)
    mel = np.asarray(hz_to_mel(jnp.asarray(hz)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(hz, mel, color="#1f4e79", lw=2)
    ax.scatter(feats.mel_centers_hz, np.asarray(hz_to_mel(jnp.asarray(feats.mel_centers_hz))),
               color="#b85c38", zorder=3, label="26 filter peaks")
    ax.axvspan(config.fmin, config.fmax, color="#1f4e79", alpha=0.08, label="filterbank span")
    ax.set_xlabel("Hz")
    ax.set_ylabel("Mel  =  1125 ln(1 + f/700)")
    ax.set_title("Why the triangles are packed at low frequency")
    ax.legend(frameon=False)
    fig.savefig(OUT / "mel_scale.png", dpi=140)
    plt.close(fig)

    np.savez(
        OUT / "mel_features.npz",
        waveform=wave_np,
        power=feats.power,
        filterbank=feats.filterbank,
        log_mel=feats.log_mel,
        mfcc=feats.mfcc,
        freqs_hz=feats.freqs_hz,
        mel_centers_hz=feats.mel_centers_hz,
        times=times,
    )
    print("wrote plots and mel_features.npz")


if __name__ == "__main__":
    main()