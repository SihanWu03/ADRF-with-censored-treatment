"""Multivariate adapters for the validated simulation nuisance algorithms.

Feature handling extends ``adrf.nuisance`` to the NHANES covariates.
Neural likelihood training, Bernstein formulae and Nyström
linear algebra are imported from that module. All covariate preprocessing is
fitted on the supplied outer training fold. Predictions accept raw covariates
with shape ``[..., p]`` and dose arrays broadcast against the leading dimensions.
For a Cartesian grid use ``X[:, None, :]`` and ``a[None, :]``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from adrf import nuisance as core


DTYPE = torch.float64


def _tensor(value, device):
    return torch.as_tensor(value, device=device, dtype=DTYPE)


def _broadcast_x_a(x, a):
    if x.ndim < 1:
        raise ValueError('Predicted X must retain its final covariate dimension')
    shape = torch.broadcast_shapes(x.shape[:-1], a.shape)
    return x.expand(*shape, x.shape[-1]), a.expand(shape)


def _features(x, a, T):
    x, a = _broadcast_x_a(x, a)
    return torch.cat((x.reshape(-1, x.shape[-1]),
                      (2. * a.reshape(-1, 1) / T - 1.)), dim=1)


class _CoefficientNet(nn.Module):
    def __init__(self, p, outputs, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(p, width), nn.Tanh(),
                                     nn.Linear(width, outputs))
        nn.init.normal_(self.network[-1].weight, std=.02)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return self.network(x)


class _NeuralTreatment(core._NeuralTreatment):
    def __init__(self, p, T, degree, width, floor):
        nn.Module.__init__(self)
        self.T, self.degree, self.floor = T, degree, floor
        self.net = _CoefficientNet(p, degree + 1, width)


class _NeuralCensoring(core._NeuralCensoring):
    def __init__(self, p, T, degree, width, hazard_min, hazard_max):
        nn.Module.__init__(self)
        self.T, self.degree = T, degree
        self.hazard_min, self.hazard_max = hazard_min, hazard_max
        self.net = _CoefficientNet(p, degree + 1, width)
        initial = min(.95, max(.05, (.4 - hazard_min) / (hazard_max - hazard_min)))
        nn.init.constant_(self.net.network[-1].bias,
                          math.log(initial / (1. - initial)))


class _KernelMu:
    def __init__(self, T, landmarks, length, rotation, feature_mean, coef,
                 y_mean, chunk_size=8192):
        self.T, self.landmarks, self.length = T, landmarks, length
        self.dual_coef = rotation @ coef
        self.offset = y_mean - feature_mean @ coef
        self.chunk_size = chunk_size

    def __call__(self, x, a):
        shape = torch.broadcast_shapes(x.shape[:-1], a.shape)
        if x.ndim == 3 and a.ndim == 2 and x.shape[1] == 1 and a.shape[0] == 1:
            # Gaussian factorization is exact for any number of covariates.
            kx = core._kernel(x[:, 0, :], self.landmarks[:, :-1], self.length)
            scaled_a = 2. * a.T / self.T - 1.
            ka = core._kernel(scaled_a, self.landmarks[:, -1:], self.length)
            return (kx * self.dual_coef[None, :]) @ ka.T + self.offset
        z = _features(x, a, self.T)
        if not len(z):
            return x.new_empty(shape)
        values = [core._kernel(part, self.landmarks, self.length) @ self.dual_coef
                  + self.offset for part in z.split(self.chunk_size)]
        return torch.cat(values).reshape(shape)


def _fit_kernel_mu(x, a, y, T, seed, config):
    """Validate and refit the outcome model with multivariate features."""
    n = len(y)
    tr, va = core._split(n, x.device, seed)
    landmarks_n = int(config.get('landmarks', 192))
    if landmarks_n < 2:
        raise ValueError('At least two Nyström landmarks are required')
    lengths = [float(v) for v in config.get('kernel_lengths', [1., 2., 4.])]
    ridges = [float(v) for v in config.get('kernel_ridges', [1e-7, 1e-6, 1e-5])]
    if not lengths or not ridges or not all(math.isfinite(v) and v > 0 for v in lengths + ridges):
        raise ValueError('Kernel lengths and ridges must be positive and finite')
    chunk_size = int(config.get('prediction_chunk_size', 8192))
    if chunk_size < 1:
        raise ValueError('prediction_chunk_size must be positive')
    z = _features(x, a, T)
    order = torch.randperm(len(tr), generator=core._generator(x.device, seed + 1), device=x.device)
    landmarks = z[tr[order[:min(landmarks_n, len(tr))]]]
    best = (float('inf'), lengths[0], ridges[0])
    with torch.no_grad():
        for length in lengths:
            phi, rotation = core._nystrom_basis(z[tr], landmarks, length)
            pv = core._kernel(z[va], landmarks, length) @ rotation
            for ridge in ridges:
                mean, coef, y_mean = core._ridge(phi, y[tr], ridge)
                mse = float(((pv - mean) @ coef + y_mean - y[va]).square().mean())
                if not math.isfinite(mse):
                    raise FloatingPointError('Non-finite kernel outcome validation loss')
                if mse < best[0]:
                    best = (mse, length, ridge)
        order = torch.randperm(n, generator=core._generator(x.device, seed + 2), device=x.device)
        landmarks = z[order[:min(landmarks_n, n)]]
        phi, rotation = core._nystrom_basis(z, landmarks, best[1])
        mean, coef, y_mean = core._ridge(phi, y, best[2])
        model = _KernelMu(T, landmarks, best[1], rotation, mean, coef, y_mean, chunk_size)
    return model


@dataclass
class NuisanceBundle:
    outcome: Any
    treatment: Any
    censoring: Any
    device: torch.device
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    diagnostics: dict  # Required by the shared estimator cache interface.

    def _inputs(self, x, a):
        x, a = _tensor(x, self.device), _tensor(a, self.device)
        if x.ndim < 1 or x.shape[-1] != len(self.x_mean):
            raise ValueError('Prediction X must end in the fitted covariate dimension')
        return (x - self.x_mean) / self.x_scale, a

    @torch.no_grad()
    def mu(self, x, a):
        return self.outcome(*self._inputs(x, a))

    @torch.no_grad()
    def density(self, x, a):
        return self.treatment.density(*self._inputs(x, a))

    @torch.no_grad()
    def survival_a(self, x, a):
        return self.treatment.survival(*self._inputs(x, a))

    @torch.no_grad()
    def survival_c(self, x, a):
        return self.censoring.survival(*self._inputs(x, a))

    @torch.no_grad()
    def hazard_c(self, x, a):
        return self.censoring.hazard(*self._inputs(x, a))


def fit_nuisance(X, u, delta, y, T=3., seed=0, device='cuda', config=None):
    """Fit flexible nuisances using only supplied outer-training-fold rows."""
    config = dict(config or {})
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but unavailable; choose device="cpu" explicitly')
    if not math.isfinite(T) or T <= 0:
        raise ValueError('T must be finite and positive')
    x = _tensor(X, device)
    u, delta, y = [_tensor(v, device).reshape(-1) for v in (u, delta, y)]
    if x.ndim != 2 or x.shape[1] < 1 or not len(x) == len(u) == len(delta) == len(y) or len(x) < 12:
        raise ValueError('X must have shape [n,p]; all arrays need n >= 12 rows')
    observed = delta == 1
    if not bool(((delta == 0) | observed).all()) or int(observed.sum()) < 8:
        raise ValueError('delta must be binary and at least eight outcomes must be observed')
    if not bool(torch.isfinite(x).all() & torch.isfinite(u).all() & torch.isfinite(y[observed]).all()):
        raise ValueError('X, u, and uncensored outcomes must be finite')
    if not bool(((u >= 0.) & (u <= T)).all()):
        raise ValueError('Observed u must lie in [0,T]')
    # No evaluation observations enter these moments. Constant columns remain
    # zero after centering, with unit divisor to avoid division by zero.
    x_mean = x.mean(0)
    raw_scale = x.std(0, correction=0)
    x_scale = torch.where(raw_scale > 1e-12, raw_scale, torch.ones_like(raw_scale))
    x = (x - x_mean) / x_scale
    p = x.shape[1]
    degree = int(config.get('density_degree', max(8, min(12, round(5 + math.log2(len(x) / 100.))))))
    hazard_degree = int(config.get('hazard_degree', 3))
    width = int(config.get('hidden_width', 16))
    floor = float(config.get('density_uniform_floor', .01))
    hmin = float(config.get('hazard_min', .02))
    hmax = float(config.get('hazard_max', 3.))
    if degree < 2 or hazard_degree < 1 or width < 1 or not 0 < floor < 1 or not 0 < hmin < hmax:
        raise ValueError('Invalid neural density or hazard configuration')
    treatment, _ = core._fit_neural(
        lambda: _NeuralTreatment(p, T, degree, width, floor),
        x, u, delta, seed + 11, config)
    censoring, _ = core._fit_neural(
        lambda: _NeuralCensoring(p, T, hazard_degree, width, hmin, hmax),
        x, u, delta, seed + 23, config)
    outcome = _fit_kernel_mu(x[observed], u[observed], y[observed], T, seed + 37, config)
    return NuisanceBundle(outcome, treatment, censoring, device, x_mean, x_scale, {})
