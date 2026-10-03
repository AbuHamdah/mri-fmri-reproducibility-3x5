"""Recalculate subject metrics; actual out-of-fold results stay distinct from memorization."""
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import norm, binomtest
ROOT = Path(__file__).resolve().parent
NAMES = ['accuracy', 'f1', 'kappa', 'auc', 'precision', 'recall', 'specificity', 'brier', 'ece']

def metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    h = p >= threshold
    tp = ((y == 1) & h).sum()
    tn = ((y == 0) & ~h).sum()
    fp = ((y == 0) & h).sum()
    fn = ((y == 1) & ~h).sum()
    n = len(y)
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    pos = p[y == 1]
    neg = p[y == 0]
    auc = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)) if len(pos) * len(neg) else np.nan
    ece = 0
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return np.array([acc, 2 * tp / max(2 * tp + fp + fn, 1), (acc - pe) / (1 - pe) if pe != 1 else np.nan, auc, tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1), np.mean((p - y) ** 2), ece])

def bca(y, p, n=10000):
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(y), (n, len(y)))
    boot = np.array([metrics(y[ix], p[ix]) for ix in indices])
    point = metrics(y, p)
    jack = np.array([metrics(np.delete(y, i), np.delete(p, i)) for i in range(len(y))])
    out = {}
    for k, name in enumerate(NAMES):
        b = boot[:, k]
        b = b[np.isfinite(b)]
        j = jack[:, k]
        j = j[np.isfinite(j)]
        d = j.mean() - j
        den = 6 * np.sum(d * d) ** 1.5
        a = np.sum(d ** 3) / den if den else 0
        frac = ((b < point[k]).sum() + 0.5 * (b == point[k]).sum()) / len(b)
        z0 = norm.ppf(np.clip(frac, 0.5 / len(b), 1 - 0.5 / len(b)))
        z = norm.ppf([0.025, 0.975])
        q = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out[name] = {'low': float(np.quantile(b, q[0])), 'high': float(np.quantile(b, q[1])), 'method': 'BCa subject bootstrap', 'resamples': n, 'valid_resamples': len(b), 'interpretation': 'descriptive within-dataset OOF; previously examined cohort; not external validation'}
    correct = int(((p >= 0.5) == y).sum())
    nsub = len(y)
    ph = correct / nsub
    z = norm.ppf(0.975)
    den = 1 + z * z / nsub
    center = (ph + z * z / (2 * nsub)) / den
    half = z * math.sqrt(ph * (1 - ph) / nsub + z * z / (4 * nsub * nsub)) / den
    out['accuracy'] = {'low': center - half, 'high': center + half, 'method': 'Wilson 95%', 'correct': correct, 'n': nsub, 'interpretation': 'descriptive within-dataset OOF on a previously examined cohort'}
    return out
