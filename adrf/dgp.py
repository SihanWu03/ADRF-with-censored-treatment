"""Simulation truth. Estimators receive ObservedData, never latent A/C/Y."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ObservedData:
    x: np.ndarray
    u: np.ndarray
    delta: np.ndarray
    y: np.ndarray

    def subset(self, index):
        return ObservedData(*(getattr(self, k)[index] for k in ('x', 'u', 'delta', 'y')))

    def __len__(self):
        return len(self.x)


def theta(a, setting='linear'):
    value = 2.0 + 1.5 * a
    if setting == 'nonlinear':
        value = value + 6.0 * (a - 1.2) ** 2
    elif setting != 'linear':
        raise ValueError(f'Unknown setting {setting}')
    return value


def rates(x):
    return np.exp(-1.0 + 0.2*x), np.exp(-0.9 + 1.2*x)


def generate(n, seed, setting='linear', T=3.0, noise_sd=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    ra, rc = rates(x)
    a = -np.log1p(-rng.random(n) * (-np.expm1(-ra*T))) / ra
    c = rng.exponential(1.0/rc)
    y = theta(a, setting) + (4.0+a)*x + rng.normal(0, noise_sd, n)
    delta = a <= c
    data = ObservedData(x, np.minimum(a, c), delta, np.where(delta, y, np.nan))
    diagnostics = dict(censoring_rate=float(1-delta.mean()), n_complete=int(delta.sum()))
    return data, diagnostics
