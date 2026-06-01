"""Feature extraction for classical baselines."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def extract_bandpower_features(
    X: np.ndarray,
    sfreq: float = 125.0,
    mu_band: tuple[float, float] = (8.0, 12.0),
    beta_band: tuple[float, float] = (13.0, 30.0),
) -> np.ndarray:
    """
    Extract log-variance, mu and beta bandpower per channel.

    X: (n_samples, n_time, n_channels)
    Returns: (n_samples, n_channels * 3)
    """
    n, _, n_ch = X.shape
    feats = np.zeros((n, n_ch * 3), dtype=np.float64)
    for i in range(n):
        trial = X[i]
        for ch in range(n_ch):
            sig = trial[:, ch]
            log_var = np.log(np.var(sig) + 1e-12)
            mu_bp = _bandpower(sig, sfreq, mu_band)
            beta_bp = _bandpower(sig, sfreq, beta_band)
            feats[i, ch * 3] = log_var
            feats[i, ch * 3 + 1] = mu_bp
            feats[i, ch * 3 + 2] = beta_bp
    return feats


def _bandpower(
    sig: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
) -> float:
    freqs, psd = welch(sig, fs=sfreq, nperseg=min(256, len(sig)))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[idx], freqs[idx]))
