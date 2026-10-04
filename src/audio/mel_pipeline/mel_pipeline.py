"""Functional, production-style mel spectrogram and MFCC pipeline in JAX.

Matches the CU Boulder slide conventions:
  - 25 ms frames, 10 ms hop (at 8 kHz: 200 / 80 samples)
  - zero-padded DFT so the one-sided spectrum has K = 257 bins
  - HTK mel scale: M(f) = 1125 * ln(1 + f/700)
  - 26 triangular filters from 300 Hz to 4 kHz, peak-normalized to 1
  - log filterbank energy, then type-II DCT (MFCCs)

Every public function is pure. Shapes are documented. Arrays stay on the
caller-chosen JAX device; no global state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class MelConfig(NamedTuple):
    """Immutable analysis configuration.

    ``n_fft`` must be even. ``n_freq = n_fft // 2 + 1``. The slide uses a
    200-sample window and K = 257 bins, which implies zero-padding to 512.
    """

    sample_rate: int = 8000
    frame_length: int = 200
    frame_step: int = 80
    n_fft: int = 512
    n_mels: int = 26
    fmin: float = 300.0
    fmax: float = 4000.0
    n_mfcc: int = 13
    pre_emphasis: float = 0.97
    log_floor: float = 1e-10
    # Peak-1 triangles match the slide. Set True for Slaney equal-area weights.
    slaney_norm: bool = False


class MelFeatures(NamedTuple):
    """Intermediates a caller usually wants to inspect or plot."""

    frames: Array          # (n_frames, frame_length)
    window: Array          # (frame_length,)
    power: Array           # (n_frames, n_freq) periodogram
    filterbank: Array      # (n_mels, n_freq)
    mel_energy: Array      # (n_frames, n_mels)
    log_mel: Array         # (n_frames, n_mels)
    mfcc: Array            # (n_frames, n_mfcc)
    freqs_hz: Array        # (n_freq,)
    mel_centers_hz: Array  # (n_mels,)


def hz_to_mel(freq_hz: Array) -> Array:
    """HTK mel scale used on the slide: M(f) = 1125 * ln(1 + f/700)."""
    return 1125.0 * jnp.log(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: Array) -> Array:
    """Inverse HTK mel: M^{-1}(m) = 700 * (exp(m/1125) - 1)."""
    return 700.0 * (jnp.exp(mel / 1125.0) - 1.0)


def hamming_window(frame_length: int) -> Array:
    """Periodic Hamming window, length ``frame_length``."""
    n = jnp.arange(frame_length)
    return 0.54 - 0.46 * jnp.cos(2.0 * jnp.pi * n / frame_length)


def preemphasize(samples: Array, coeff: float) -> Array:
    """First-order high-pass: y[n] = x[n] - coeff * x[n-1]."""
    return jnp.concatenate(
        [samples[:1], samples[1:] - coeff * samples[:-1]], axis=0
    )


def frame_signal(samples: Array, frame_length: int, frame_step: int) -> Array:
    """Slice ``samples`` into overlapping frames, shape (n_frames, frame_length).

    Trailing samples that do not fill a frame are dropped, matching the slide
    count ``(n - frame_length) / frame_step`` when that division is exact.
    """
    n = samples.shape[0]
    n_frames = (n - frame_length) // frame_step + 1
    starts = jnp.arange(n_frames) * frame_step
    idx = starts[:, None] + jnp.arange(frame_length)[None, :]
    return samples[idx]


def periodogram(frames: Array, window: Array, n_fft: int) -> Array:
    """Window, zero-pad, and return the power spectrum |X|^2 / N.

    The slide writes P_i(k) = (1/N) |X~(k)|^2 with N the window length, not the
    FFT length. ``n_freq = n_fft // 2 + 1`` (257 when n_fft is 512).
    """
    windowed = frames * window
    spec = jnp.fft.rfft(windowed, n=n_fft, axis=-1)
    n = frames.shape[-1]
    return (jnp.abs(spec) ** 2) / n


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    slaney_norm: bool = False,
) -> tuple[Array, Array]:
    """Build the mel-spaced triangular filterbank.

    Returns
    -------
    weights : (n_mels, n_freq)
        Each row is one triangle. Peaks are 1 unless ``slaney_norm`` is set,
        in which case each triangle has unit area in Hz (Slaney / librosa).
    centers_hz : (n_mels,)
        Hz frequency of each triangle peak.

    Construction follows the slide:
      1. Map [fmin, fmax] through M(f) = 1125 ln(1 + f/700).
      2. Place ``n_mels + 2`` points evenly in mel (the two extras are the
         left and right feet of the first and last triangles).
      3. Map those points back with M^{-1}, then onto FFT bin centers.
      4. Raise a linear ramp from the left foot to the peak and lower a
         linear ramp from the peak to the right foot.
    """
    n_freq = n_fft // 2 + 1
    mel_lo = hz_to_mel(jnp.asarray(fmin))
    mel_hi = hz_to_mel(jnp.asarray(fmax))
    mel_points = jnp.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Bin index of a frequency: k = f * n_fft / sample_rate, k in [0, n_freq).
    bins = hz_points * n_fft / sample_rate
    fft_freqs = jnp.arange(n_freq) * (sample_rate / n_fft)

    def one_triangle(left: Array, center: Array, right: Array) -> Array:
        rising = (fft_freqs - left) / jnp.maximum(center - left, 1e-6)
        falling = (right - fft_freqs) / jnp.maximum(right - center, 1e-6)
        tri = jnp.maximum(0.0, jnp.minimum(rising, falling))
        if slaney_norm:
            # Equal-area in Hz: divide by the triangle width.
            tri = tri * (2.0 / jnp.maximum(right - left, 1e-6))
        return tri

    weights = jax.vmap(one_triangle)(bins[:-2], bins[1:-1], bins[2:])
    return weights, hz_points[1:-1]


def dct_matrix(n_mfcc: int, n_mels: int) -> Array:
    """Orthogonal type-II DCT matrix, shape (n_mfcc, n_mels).

    ```
    c[i] = sum_j logE[j] * cos(pi * i * (j + 0.5) / n_mels) * scale[i]
    with scale[0] = sqrt(1/n_mels) and scale[i>0] = sqrt(2/n_mels).
    ```
    This is the basis used by librosa and by most speech MFCC front ends.

    """
    i = jnp.arange(n_mfcc)[:, None]
    j = jnp.arange(n_mels)[None, :]
    basis = jnp.cos(jnp.pi * i * (j + 0.5) / n_mels)
    scale = jnp.sqrt(2.0 / n_mels) * jnp.ones((n_mfcc, 1))
    scale = scale.at[0].set(jnp.sqrt(1.0 / n_mels))
    return basis * scale


def apply_filterbank(power: Array, weights: Array, log_floor: float) -> tuple[Array, Array]:
    """E = P @ T^T, then log. ``power`` is (frames, freq), ``weights`` is (mels, freq)."""
    energy = power @ weights.T
    return energy, jnp.log(jnp.maximum(energy, log_floor))


def mfcc_from_log_mel(log_mel: Array, n_mfcc: int) -> Array:
    """Project log mel energies onto the low-order DCT basis."""
    basis = dct_matrix(n_mfcc, log_mel.shape[-1])
    return log_mel @ basis.T


def mel_spectrogram(samples: Array, config: MelConfig) -> MelFeatures:
    """Full front end: pre-emphasis, STFT periodogram, mel filterbank, DCT.

    ``samples`` is a 1-D waveform. Output frames are time-major.
    """
    emphasized = preemphasize(samples, config.pre_emphasis)
    window = hamming_window(config.frame_length)
    frames = frame_signal(emphasized, config.frame_length, config.frame_step)
    power = periodogram(frames, window, config.n_fft)
    weights, centers = mel_filterbank(
        config.sample_rate,
        config.n_fft,
        config.n_mels,
        config.fmin,
        config.fmax,
        config.slaney_norm,
    )
    energy, log_mel = apply_filterbank(power, weights, config.log_floor)
    coeffs = mfcc_from_log_mel(log_mel, config.n_mfcc)
    freqs = jnp.arange(config.n_fft // 2 + 1) * (config.sample_rate / config.n_fft)
    return MelFeatures(
        frames=frames,
        window=window,
        power=power,
        filterbank=weights,
        mel_energy=energy,
        log_mel=log_mel,
        mfcc=coeffs,
        freqs_hz=freqs,
        mel_centers_hz=centers,
    )


# JIT the pure pipeline. Config is a pytree (NamedTuple of scalars/bools).
mel_spectrogram_jit = jax.jit(mel_spectrogram, static_argnames=())
