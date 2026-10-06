"""Censoring-aware nuisance fits, using only each cross-fitting training fold.

The nonparametric fits run in float64 on the requested torch device.  The
treatment density is a neural mixture of beta densities (a Bernstein density,
not a spline); its survivor is analytical.  The censoring hazard is a positive
Bernstein polynomial with coefficients learned by a neural network.  Its
cumulative integral is analytical too.  Outcome regression is Nyström Gaussian
kernel ridge, with no DGP-specific outcome features.  Internal validation is
confined to the supplied training data, followed by a full-training refit.

All prediction methods broadcast ``x`` and ``a``; for an evaluation grid pass
``x[:, None], a[None, :]``.  A one-dimensional x denotes one covariate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DTYPE = torch.float64


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE, device=device)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _split(n: int, device: torch.device, seed: int, fraction: float = .2):
    order = torch.randperm(n, generator=_generator(device, seed), device=device)
    nv = max(1, min(n - 2, int(round(n * fraction))))
    return order[nv:], order[:nv]


def _bernstein(t: torch.Tensor, degree: int) -> torch.Tensor:
    """Bernstein polynomial basis, evaluated without log(0) at endpoints."""
    k = torch.arange(degree + 1, dtype=DTYPE, device=t.device)
    choose = t.new_tensor([math.comb(degree, j) for j in range(degree + 1)])
    t = t.clamp(0., 1.)[..., None]
    return choose * t.pow(k) * (1. - t).pow(degree - k)


def _beta_survivors(t: torch.Tensor, degree: int) -> torch.Tensor:
    # Beta(k+1, degree-k+1) survivor = binomial(degree+1, t) CDF at k.
    return _bernstein(t, degree + 1).cumsum(dim=-1)[..., :degree + 1]


class _CoefficientNet(nn.Module):
    def __init__(self, outputs: int, width: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                     nn.Linear(width, outputs))
        # Start near simple, well-supported marginal models.
        nn.init.normal_(self.network[-1].weight, std=.02)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return self.network(x[..., None])


class _NeuralTreatment(nn.Module):
    def __init__(self, T, degree, width, floor):
        super().__init__()
        self.T, self.degree, self.floor = T, degree, floor
        self.net = _CoefficientNet(degree + 1, width)

    def weights(self, x):
        return ((1. - self.floor) * self.net(x).softmax(dim=-1)
                + self.floor / (self.degree + 1))

    def density(self, x, a):
        value = (self.weights(x) * _bernstein(a / self.T, self.degree)).sum(-1)
        return value * ((self.degree + 1) / self.T) * ((a >= 0.) & (a <= self.T))

    def survival(self, x, a):
        return (self.weights(x) * _beta_survivors(a / self.T, self.degree)).sum(-1)

    def loss(self, x, u, delta):
        # This is the observed right-censored treatment likelihood, not a
        # regression on imputed or latent treatment values.
        w = self.weights(x)
        f = (w * _bernstein(u / self.T, self.degree)).sum(-1) * ((self.degree + 1) / self.T)
        s = (w * _beta_survivors(u / self.T, self.degree)).sum(-1)
        return -(delta * f.clamp_min(1e-14).log()
                 + (1. - delta) * s.clamp_min(1e-14).log()).mean()

    def roughness(self, x):
        w = self.weights(x)
        return (w[..., 2:] - 2 * w[..., 1:-1] + w[..., :-2]).square().mean()


class _NeuralCensoring(nn.Module):
    def __init__(self, T, degree, width, hazard_min, hazard_max):
        super().__init__()
        self.T, self.degree = T, degree
        self.hazard_min, self.hazard_max = hazard_min, hazard_max
        self.net = _CoefficientNet(degree + 1, width)
        # An initial hazard of .4 avoids starting at the upper-bound midpoint.
        initial = min(.95, max(.05, (.4 - hazard_min) / (hazard_max - hazard_min)))
        nn.init.constant_(self.net.network[-1].bias, math.log(initial / (1. - initial)))

    def coefficients(self, x):
        return self.hazard_min + (self.hazard_max - self.hazard_min) * self.net(x).sigmoid()

    def hazard(self, x, a):
        return (self.coefficients(x) * _bernstein(a / self.T, self.degree)).sum(-1)

    def cumulative(self, x, a):
        coef = self.coefficients(x)
        integral = (1. - _beta_survivors(a / self.T, self.degree)) * (self.T / (self.degree + 1))
        # Constant terminal hazard extends the model past T; the estimator
        # itself only queries [0,T].  The integral is zero for negative a.
        return (coef * integral).sum(-1) + coef[..., -1] * (a - self.T).clamp_min(0.)

    def survival(self, x, a):
        return (-self.cumulative(x, a)).exp()

    def loss(self, x, u, delta):
        coef = self.coefficients(x)
        hazard = (coef * _bernstein(u / self.T, self.degree)).sum(-1)
        cumulative = (coef * (1. - _beta_survivors(u / self.T, self.degree))).sum(-1) * (self.T / (self.degree + 1))
        return (cumulative - (1. - delta) * hazard.log()).mean()

    def roughness(self, x):
        coef = self.coefficients(x)
        return (coef[..., 1:] - coef[..., :-1]).square().mean()


def _new_model(factory, device, seed):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return factory().to(device=device, dtype=DTYPE)


def _fit_neural(factory, x, u, delta, seed, config):
    device = x.device
    tr, va = _split(len(x), device, seed)
    epochs = int(config.get('epochs', 450))
    min_epochs = min(epochs, int(config.get('min_epochs', 80)))
    patience = int(config.get('patience', 50))
    check_every = int(config.get('validation_every', 5))
    lr = float(config.get('learning_rate', .015))
    roughness = float(config.get('roughness_penalty', .005))
    weight_decay = float(config.get('weight_decay', 1e-4))
    if epochs < 1 or check_every < 1:
        raise ValueError('epochs and validation_every must be positive')

    def train_step(model, optimizer, indices):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x[indices], u[indices], delta[indices])
        objective = loss + roughness * model.roughness(x[indices])
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite censored neural likelihood')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()

    model = _new_model(factory, device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_epoch, last_improvement = float('inf'), 1, 0
    for epoch in range(1, epochs + 1):
        train_step(model, optimizer, tr)
        if epoch % check_every == 0 or epoch == epochs:
            with torch.no_grad():
                value = float(model.loss(x[va], u[va], delta[va]))
            if epoch >= min_epochs and value < best_loss - 1e-6:
                best_loss, best_epoch, last_improvement = value, epoch, epoch
            if epoch >= min_epochs and epoch - last_improvement >= patience:
                break
    selected = best_epoch
    # Fresh initialization and full training fold, including its internal
    # validation subset. No outer validation observation is ever seen here.
    model = _new_model(factory, device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    full = torch.arange(len(x), device=device)
    for _ in range(selected):
        train_step(model, optimizer, full)
    model.eval()
    with torch.no_grad():
        full_loss = float(model.loss(x, u, delta))
    if not math.isfinite(full_loss):
        raise FloatingPointError('Non-finite refitted neural likelihood')
    return model, {}


def _features(x, a, T):
    x, a = torch.broadcast_tensors(x, a)
    return torch.stack((x.reshape(-1), 2. * a.reshape(-1) / T - 1.), dim=1)


def _kernel(z, landmarks, length):
    dist = (z.square().sum(1, keepdim=True)
            + landmarks.square().sum(1)[None, :] - 2. * z @ landmarks.T)
    return (-.5 * dist.clamp_min(0.) / length ** 2).exp()


class _KernelMu:
    def __init__(self, T, landmarks, length, rotation, feature_mean, coef, y_mean, chunk_size=8192):
        self.T, self.landmarks, self.length = T, landmarks, length
        self.rotation, self.feature_mean, self.coef, self.y_mean = rotation, feature_mean, coef, y_mean
        # Collapse the Nyström projection once. Prediction then needs a single
        # kernel-vector multiplication even on dense quadrature grids.
        self.dual_coef = rotation @ coef
        self.offset = y_mean - feature_mean @ coef
        self.chunk_size = chunk_size

    def __call__(self, x, a):
        shape = torch.broadcast_shapes(x.shape, a.shape)
        if x.ndim == 2 and a.ndim == 2 and x.shape[1] == 1 and a.shape[0] == 1:
            # On a Cartesian grid the Gaussian kernel factors exactly into
            # one kernel for x and one for scaled a. Avoid materializing the
            # (n * doses, landmarks) matrix used by generic paired inputs.
            # Training, landmarks, dual coefficients and regularization are
            # unchanged; only floating-point summation order can differ.
            kx = (-.5 * (x - self.landmarks[None, :, 0]).square() / self.length ** 2).exp()
            scaled_a = 2. * a.T / self.T - 1.
            ka = (-.5 * (scaled_a - self.landmarks[None, :, 1]).square() / self.length ** 2).exp()
            return (kx * self.dual_coef[None, :]) @ ka.T + self.offset
        z = _features(x, a, self.T)
        if not len(z):
            return x.new_empty(shape)
        output = []
        for part in z.split(self.chunk_size):
            output.append(_kernel(part, self.landmarks, self.length) @ self.dual_coef + self.offset)
        return torch.cat(output).reshape(shape)


def _nystrom_basis(z, landmarks, length):
    gram = _kernel(landmarks, landmarks, length)
    eigval, eigvec = torch.linalg.eigh(gram)
    keep = eigval > max(1e-10, float(eigval[-1]) * 1e-12)
    rotation = eigvec[:, keep] / eigval[keep].sqrt()[None, :]
    return _kernel(z, landmarks, length) @ rotation, rotation


def _ridge(phi, y, ridge):
    mean, y_mean = phi.mean(0), y.mean()
    centered = phi - mean
    gram = centered.T @ centered
    gram.diagonal().add_(len(y) * ridge)
    coef = torch.linalg.solve(gram, centered.T @ (y - y_mean))
    return mean, coef, y_mean


def _fit_kernel_mu(x, a, y, T, seed, config):
    n = len(y)
    tr, va = _split(n, x.device, seed)
    landmarks_n = int(config.get('landmarks', 192))
    if landmarks_n < 2:
        raise ValueError('At least two Nyström landmarks are required')
    lengths = [float(v) for v in config.get('kernel_lengths', [1., 2., 4.])]
    ridges = [float(v) for v in config.get('kernel_ridges', [1e-7, 1e-6, 1e-5])]
    if not lengths or not ridges or min(lengths + ridges) <= 0:
        raise ValueError('Kernel lengths and ridges must be positive')
    z = _features(x, a, T)
    order = torch.randperm(len(tr), generator=_generator(x.device, seed + 1), device=x.device)
    landmarks = z[tr[order[:min(landmarks_n, len(tr))]]]
    scores, best = [], (float('inf'), lengths[0], ridges[0])
    with torch.no_grad():
        for length in lengths:
            phi, rotation = _nystrom_basis(z[tr], landmarks, length)
            pv = _kernel(z[va], landmarks, length) @ rotation
            for ridge in ridges:
                mean, coef, y_mean = _ridge(phi, y[tr], ridge)
                mse = float(((pv - mean) @ coef + y_mean - y[va]).square().mean())
                scores.append(mse)
                if mse < best[0]:
                    best = (mse, length, ridge)
        order = torch.randperm(n, generator=_generator(x.device, seed + 2), device=x.device)
        landmarks = z[order[:min(landmarks_n, n)]]
        phi, rotation = _nystrom_basis(z, landmarks, best[1])
        mean, coef, y_mean = _ridge(phi, y, best[2])
        model = _KernelMu(T, landmarks, best[1], rotation, mean, coef, y_mean,
                          int(config.get('prediction_chunk_size', 8192)))
        train_mse = float(((phi - mean) @ coef + y_mean - y).square().mean())
    if not all(math.isfinite(score) for score in scores) or not math.isfinite(train_mse):
        raise FloatingPointError('Non-finite kernel outcome fit')
    return model


class _ParametricMu:
    def __init__(self, coef, omit_x):
        self.coef, self.omit_x = coef, omit_x

    def design(self, x, a):
        x, a = torch.broadcast_tensors(x, a)
        parts = [torch.ones_like(a), a, a.square()]
        if not self.omit_x:
            parts += [x, a * x]
        return torch.stack(parts, dim=-1)

    def __call__(self, x, a):
        return self.design(x, a) @ self.coef


class _ParametricTime:
    def __init__(self, beta, T, treatment, omit_x):
        self.beta, self.T, self.treatment, self.omit_x = beta, T, treatment, omit_x

    def rate(self, x):
        lograte = self.beta[0] + (0. * x if self.omit_x else self.beta[1] * x)
        return lograte.clamp(-12., 6.).exp()

    def density(self, x, a):
        r = self.rate(x)
        f = r * (-r * a.clamp(0., self.T)).exp() / (-torch.expm1(-r * self.T))
        return f * ((a >= 0.) & (a <= self.T))

    def survival(self, x, a):
        r = self.rate(x)
        if self.treatment:
            t = a.clamp(0., self.T)
            return (-r * t).exp() * (-torch.expm1(-r * (self.T - t))) / (-torch.expm1(-r * self.T))
        return (-r * a.clamp_min(0.)).exp()

    def hazard(self, x, a):
        return torch.broadcast_tensors(self.rate(x), a)[0]


def _fit_parametric_time(x, u, delta, T, treatment, omit_x, config):
    beta = torch.zeros(1 if omit_x else 2, dtype=DTYPE, device=x.device, requires_grad=True)
    with torch.no_grad():
        beta[0] = -.9
    model = _ParametricTime(beta, T, treatment, omit_x)
    iterations = int(config.get('parametric_max_iter', 100))
    optimizer = torch.optim.LBFGS([beta], lr=1., max_iter=iterations,
                                tolerance_grad=1e-9, tolerance_change=1e-12,
                                line_search_fn='strong_wolfe')
    def objective():
        r = model.rate(x)
        if treatment:
            z = (-torch.expm1(-r * T)).log()
            logf = r.log() - r * u - z
            logs = -r * u + (-torch.expm1(-r * (T - u))).clamp_min(1e-15).log() - z
            return -(delta * logf + (1. - delta) * logs).mean()
        return (r * u - (1. - delta) * r.log()).mean()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite parametric censored likelihood')
        loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    return _ParametricTime(beta.detach(), T, treatment, omit_x)


@dataclass
class NuisanceBundle:
    outcome: Any
    treatment: Any
    censoring: Any
    device: torch.device
    diagnostics: dict

    @torch.no_grad()
    def mu(self, x, a):
        return self.outcome(_tensor(x, self.device), _tensor(a, self.device))

    @torch.no_grad()
    def density(self, x, a):
        return self.treatment.density(_tensor(x, self.device), _tensor(a, self.device))

    @torch.no_grad()
    def survival_a(self, x, a):
        return self.treatment.survival(_tensor(x, self.device), _tensor(a, self.device))

    @torch.no_grad()
    def survival_c(self, x, a):
        return self.censoring.survival(_tensor(x, self.device), _tensor(a, self.device))

    @torch.no_grad()
    def hazard_c(self, x, a):
        return self.censoring.hazard(_tensor(x, self.device), _tensor(a, self.device))


def fit_nuisance(x, u, delta, y, kind='neural', misspec='none', T=3., seed=0,
                 device='cuda', config=None) -> NuisanceBundle:
    """Fit nuisances to an observed training fold; censored y may be NaN.

    ``kind='parametric'`` supports one misspecification at a time: ``mu``
    removes x and a*x from outcome regression; ``a`` or ``c`` omits x from
    that time model. ``neural`` intentionally supports only ``misspec='none'``.
    CUDA requests fail clearly if CUDA is unavailable; use ``device='cpu'``
    explicitly for a CPU run. config keys are documented in this module.
    """
    config = dict(config or {})
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but unavailable; choose device="cpu" explicitly')
    if T <= 0 or kind not in ('neural', 'parametric') or misspec not in ('none', 'mu', 'a', 'c'):
        raise ValueError('Invalid T, nuisance kind, or misspecification')
    if kind == 'neural' and misspec != 'none':
        raise ValueError('Controlled misspecification is defined only for parametric fits')
    x, u, delta, y = [_tensor(v, device).reshape(-1) for v in (x, u, delta, y)]
    if not len(x) == len(u) == len(delta) == len(y) or len(x) < 12:
        raise ValueError('Training arrays must have equal lengths and at least 12 rows')
    observed = delta == 1
    if not bool(((delta == 0) | observed).all()) or int(observed.sum()) < 8:
        raise ValueError('delta must be binary and at least eight outcomes must be observed')
    if not bool(torch.isfinite(x).all() & torch.isfinite(u).all() & torch.isfinite(y[observed]).all()):
        raise ValueError('x, u, and uncensored outcomes must be finite')
    if not bool(((u >= 0.) & (u <= T)).all()):
        raise ValueError('Observed follow-up u must lie in [0,T]')
    complete_case = bool(config.get('complete_case', False))
    if complete_case and not bool(observed.all()):
        raise ValueError('complete_case=True requires an already filtered uncensored training sample')
    if kind == 'neural':
        degree = int(config.get('density_degree', max(8, min(12, round(5 + math.log2(len(x) / 100.))))))
        hazard_degree = int(config.get('hazard_degree', 3))
        width = int(config.get('hidden_width', 16))
        floor = float(config.get('density_uniform_floor', .01))
        hmin = float(config.get('hazard_min', .02))
        hmax = float(config.get('hazard_max', 3.))
        if degree < 2 or hazard_degree < 1 or width < 1 or not 0 < floor < 1 or not 0 < hmin < hmax:
            raise ValueError('Invalid neural density or hazard configuration')
        treatment, _ = _fit_neural(lambda: _NeuralTreatment(T, degree, width, floor),
                                   x, u, delta, seed + 11, config)
        if complete_case:
            censoring = None
        else:
            censoring, _ = _fit_neural(lambda: _NeuralCensoring(T, hazard_degree, width, hmin, hmax),
                                       x, u, delta, seed + 23, config)
        outcome = _fit_kernel_mu(x[observed], u[observed], y[observed], T, seed + 37, config)
    else:
        outcome = _ParametricMu(None, misspec == 'mu')
        design = outcome.design(x[observed], u[observed])
        gram = design.T @ design
        gram.diagonal().add_(1e-10)
        outcome.coef = torch.linalg.solve(gram, design.T @ y[observed])
        treatment = _fit_parametric_time(x, u, delta, T, True, misspec == 'a', config)
        censoring = _fit_parametric_time(x, u, delta, T, False, misspec == 'c', config)
    return NuisanceBundle(outcome, treatment, censoring, device, {})
