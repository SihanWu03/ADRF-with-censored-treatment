"""A frozen NHANES-calibrated law; its truth is not a real causal estimate.

X is sampled independently from the fixed empirical source population. A has
a truncated lognormal / uniform mixture density on [0, 3]. The outcome mean
is a cubic regression spline in A, including effect modification. Separate
innovations generate A, the potential-outcome error, and censoring given X.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from sklearn.preprocessing import SplineTransformer

from adrf.dgp import ObservedData

T = 3.0
DOSE_SCALE = 20.0  # a = dietary fiber (g) / 20
UNIFORM_MIXTURE = 0.10


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def _ridge(design, response, penalty):
    mean, scale = design.mean(0), design.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    z = (design - mean) / scale
    center = float(response.mean())
    coef = np.linalg.solve(z.T @ z / len(z) + penalty * np.eye(z.shape[1]),
                           z.T @ (response - center) / len(z))
    return mean, scale, coef, center


def _predict(design, model):
    mean, scale, coef, center = model
    return (design - mean) / scale @ coef + center


class Generator:
    def __init__(self, arrays, meta):
        self.w = {name: np.array(value, copy=True) for name, value in arrays.items()}
        for value in self.w.values():
            value.setflags(write=False)
        self.meta = json.loads(json.dumps(meta, allow_nan=False))
        for name, current in (("T", T), ("dose_scale", DOSE_SCALE), ("uniform_mixture", UNIFORM_MIXTURE)):
            if meta[name] != current:
                raise ValueError(f"Frozen {name} differs from the implementation")
        for name in ("sigma_log_a", "sigma_y"):
            if name in meta and not (np.isfinite(meta[name]) and meta[name] > 0):
                raise ValueError(f"{name} must be finite and strictly positive")
        self.x_columns = list(meta["x_columns"])
        self.continuous = [self.x_columns.index(name) for name in meta["smooth_x_columns"]]
        self.modifiers = [self.x_columns.index(name) for name in meta["modifier_columns"]]
        self.spline = SplineTransformer(
            degree=3, knots=np.linspace(0.0, T, 6)[:, None],
            include_bias=False, extrapolation="linear").fit(np.array([[0.0], [T]]))

    @property
    def source_n(self):
        return len(self.w["x_pool"])

    def _z(self, x):
        return (np.asarray(x) - self.w["x_mean"]) / self.w["x_scale"]

    def _base(self, x):
        z = self._z(x)
        return np.column_stack((z, z[:, self.continuous] ** 2))

    def _outcome_design(self, x, a):
        z = self._z(x)
        a = np.broadcast_to(np.asarray(a, dtype=float), (len(x),))
        basis = self.spline.transform(a[:, None])
        return np.column_stack((self._base(x), basis,
                                *(basis * z[:, j, None] for j in self.modifiers)))

    def _model(self, prefix):
        return tuple(self.w[f"{prefix}_{name}"] for name in ("mean", "scale", "coef", "center"))

    def mu(self, x, a):
        return _predict(self._outcome_design(x, a), self._model("mu"))

    def theta(self, targets):
        # Exact expectation under the fixed empirical X law and mean-zero error.
        targets = np.asarray(targets, dtype=float).reshape(-1)
        base_mean = self._base(self.w["x_pool"]).mean(0)
        z_mean = self._z(self.w["x_pool"]).mean(0)
        basis = self.spline.transform(targets[:, None])
        mean_design = np.column_stack((np.broadcast_to(base_mean, (len(targets), len(base_mean))),
                                       basis, *(basis * z_mean[j] for j in self.modifiers)))
        return _predict(mean_design, self._model("mu"))

    def log_a_mean(self, x):
        return _predict(self._base(x), self._model("a"))

    def draw_a(self, x, rng):
        mean = self.log_a_mean(x)
        sd = float(self.meta["sigma_log_a"])
        upper = ndtr((np.log(T) - mean) / sd)
        if not np.all(np.isfinite(upper) & (upper > 0)):
            raise FloatingPointError("Invalid treatment truncation probability")
        u = np.maximum(rng.random(len(x)) * upper, np.finfo(float).tiny)
        a = np.exp(mean + sd * ndtri(u))
        uniform = rng.random(len(x)) < UNIFORM_MIXTURE
        a[uniform] = rng.uniform(0.0, T, uniform.sum())
        if not np.all(np.isfinite(a) & (a > 0.0) & (a < T)):
            raise FloatingPointError("Invalid draw from the continuous treatment law")
        return a

    def censor_rate(self, x):
        j = self.x_columns.index("age")
        return float(self.meta["censor_scale"]) * np.exp(0.3 * self._z(x)[:, j])

    def generate(self, n, seed):
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
        source_index = streams[0].integers(self.source_n, size=n)
        x = self.w["x_pool"][source_index].copy()
        a = self.draw_a(x, streams[1])
        y = self.mu(x, a) + float(self.meta["sigma_y"]) * streams[2].normal(size=n)
        c = streams[3].exponential(1.0 / self.censor_rate(x))
        delta = a <= c
        data = ObservedData(x, np.minimum(a, c), delta, np.where(delta, y, np.nan))
        # The estimator receives only data. Latent values are deliberately not returned.
        return data, {"n": n, "seed": seed, "n_complete": int(delta.sum()),
                      "n_censored": int((~delta).sum()),
                      "censoring_rate": float((~delta).mean())}

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "weights.npz", **self.w)
        save_json(folder / "generator.json", self.meta)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        with np.load(folder / "weights.npz", allow_pickle=False) as values:
            arrays = dict(values)
        return cls(arrays, json.loads((folder / "generator.json").read_text(encoding="utf-8")))


def fit(source_csv, x_columns, seed=917541):
    frame = pd.read_csv(source_csv)
    x = frame[x_columns].to_numpy(float)
    a = frame["dose"].to_numpy(float) / DOSE_SCALE
    y = frame["y"].to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(a).all() and np.isfinite(y).all()):
        raise ValueError("Source preprocessing must explicitly handle missing values")
    if not np.all((a > 0.0) & (a < T)) or len(x) < 500:
        raise ValueError("Need at least 500 source rows and 0 < fiber < 60 g")
    order = np.random.default_rng(seed).permutation(len(x))
    tr, va, _ = np.split(order, [int(0.6 * len(x)), int(0.8 * len(x))])
    dev = np.concatenate((tr, va))
    xmean, xscale = x[tr].mean(0), x[tr].std(0)
    xscale = np.where(xscale > 1e-10, xscale, 1.0)
    smooth = ["age", "bmi", "log_energy", "pir"]
    modifiers = ["age", "sex", "bmi", "log_energy"]
    arrays = {"x_pool": x, "x_mean": xmean, "x_scale": xscale}
    meta = {"format_version": 1, "x_columns": list(x_columns),
            "smooth_x_columns": smooth, "modifier_columns": modifiers,
            "T": T, "dose_scale": DOSE_SCALE, "uniform_mixture": UNIFORM_MIXTURE,
            "calibration_seed": seed, "source_n": len(x),
            "dose_relation": "nonlinear_cubic_spline_with_effect_modification",
            "target_population": "fixed unweighted empirical source-X distribution",
            "interpretation": "Constructed-world ADRF; not an identified NHANES causal effect",
            "fit_scope": "60% training / 20% validation / 20% outer holdout; refit selected models on 80% development",
            "outcome": "LDL cholesterol", "outcome_unit": "mg/dL",
            "dose": "day-1 dietary fiber in grams, a = grams / 20",
            "censoring": "artificial exposure observation threshold, not NHANES follow-up",
            "selected_penalties": {}}
    generator = Generator(arrays, meta)
    for prefix, design, response in (("a", generator._base(x), np.log(a)),
                                      ("mu", generator._outcome_design(x, a), y)):
        candidates = []
        for penalty in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            model = _ridge(design[tr], response[tr], penalty)
            mse = float(np.mean((_predict(design[va], model) - response[va]) ** 2))
            candidates.append({"penalty": penalty, "validation_mse": mse})
        chosen = min(candidates, key=lambda v: v["validation_mse"])
        model = _ridge(design[dev], response[dev], chosen["penalty"])
        for name, value in zip(("mean", "scale", "coef", "center"), model):
            arrays[f"{prefix}_{name}"] = np.asarray(value)
        meta["selected_penalties"][prefix] = chosen["penalty"]
        meta["sigma_log_a" if prefix == "a" else "sigma_y"] = float(np.sqrt(chosen["validation_mse"]))
    generator = Generator(arrays, meta)
    # Calibrate expected censoring deterministically, not by trying analysis seeds.
    nodes, weights = leggauss(80)
    q, weights = (nodes + 1.0) / 2.0, weights / 2.0
    means = generator.log_a_mean(x)
    sd = meta["sigma_log_a"]
    upper = ndtr((np.log(T) - means) / sd)
    if not np.all(np.isfinite(upper) & (upper > 0)):
        raise FloatingPointError("Invalid calibration truncation probability")
    lognormal_a = np.exp(means[:, None] + sd * ndtri(upper[:, None] * q))
    tilt = np.exp(0.3 * generator._z(x)[:, x_columns.index("age")])

    def retention(scale):
        rate = scale * tilt
        lognormal = np.exp(-rate[:, None] * lognormal_a) @ weights
        uniform = -np.expm1(-rate * T) / (rate * T)
        return float(((1.0 - UNIFORM_MIXTURE) * lognormal + UNIFORM_MIXTURE * uniform).mean())

    meta["censoring_target"] = 0.35
    meta["censor_scale"] = float(brentq(lambda v: retention(v) - 0.65, 1e-8, 100.0))
    meta["expected_censoring_rate"] = float(1.0 - retention(meta["censor_scale"]))
    meta["density_lower_bound"] = UNIFORM_MIXTURE / T
    meta["censor_survival_lower_bound"] = float(np.exp(-meta["censor_scale"] * tilt.max() * T))
    meta["sigma_source"] = "inner-validation RMSE; includes predictive model error"
    return Generator(arrays, meta)
