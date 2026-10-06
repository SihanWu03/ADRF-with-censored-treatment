"""Aggregate saved repetitions into the simulation statistics used in the paper."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import seed_for

META = ["experiment", "setting", "n", "seed", "rep", "misspec"]
CELL = ["experiment", "setting", "n", "misspec"]
GROUP = CELL + ["bandwidth", "method"]
POINT_COLUMNS = GROUP + ["a", "h", "b", "bias", "rmse", "coverage", "mean_ci_length", "denominator"]
BAND_COLUMNS = GROUP + ["h", "b", "grid_band_coverage", "mean_band_width", "denominator"]


def _read_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _check_metadata(meta, config):
    if (meta["experiment"] != config["experiment"] or meta["setting"] not in config["settings"]
            or meta["n"] not in config["sample_sizes"] or meta["misspec"] not in config["misspecifications"]
            or not isinstance(meta["rep"], int) or not 0 <= meta["rep"] < config["repetitions"]
            or meta["seed"] != seed_for(config, meta["setting"], meta["n"], meta["rep"])):
        raise ValueError("Saved metadata does not match the configured experiment and seed")


def _finite(frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values in {columns}")
    return values


def _validate_layout(points: pd.DataFrame, curves: pd.DataFrame, meta: dict, config: dict) -> None:
    """Validate saved result structure; also used by the runner before resume."""
    _check_metadata(meta, config)
    methods = ["MR", "MRDB"]
    if config["include_conventional"]:
        methods += ["Conventional", "ConventionalDB"]
    expected = {(bw, method) for bw in config["bandwidths"] for method in methods}
    if config["experiment"] == "main":
        expected.add(("none", "PI"))
    for frame in (points, curves):
        actual = set(frame[["bandwidth", "method"]].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError("Saved job does not contain every configured method/bandwidth")
    if curves.duplicated(["bandwidth", "method"]).any():
        raise ValueError("Saved job repeats a curve record")
    grid = np.linspace(config["a_min"], config["a_max"], config["grid_size"])
    for key, frame in points.groupby(["bandwidth", "method"]):
        recorded = np.sort(_finite(frame, ["a"]).ravel())
        if len(recorded) != len(grid) or not np.allclose(recorded, grid, rtol=0, atol=1e-12):
            raise ValueError(f"Saved job has an incomplete/duplicated dose grid: {key}")
    _finite(points, ["truth", "estimate"])
    interval = points.loc[points.method.ne("PI")]
    _finite(interval, ["h", "b", "ci_low", "ci_high", "covered"])
    if not interval.covered.isin([0, 1]).all() or not interval.ci_high.gt(interval.ci_low).all():
        raise ValueError("Invalid interval coverage or interval limits")
    for frame in (points, curves):
        pi = frame.loc[frame.method.eq("PI")]
        if not pi.bandwidth.eq("none").all():
            raise ValueError("PI must appear only under bandwidth='none'")
        for name in ("h", "b", "se", "ci_low", "ci_high", "covered", "mean_ci_length",
                     "band_low", "band_high", "band_covered", "band_covered_point", "mean_band_width", "critical"):
            if name in pi and pi[name].notna().any():
                raise ValueError(f"PI has no bandwidth or confidence inference: {name} must be null")
    if config["confidence_bands"]:
        bands = curves.loc[curves.method.eq("MRDB")]
        _finite(bands, ["band_covered", "mean_band_width"])
        if not bands.band_covered.isin([0, 1]).all() or not bands.mean_band_width.gt(0).all():
            raise ValueError("Invalid MRDB band coverage or width")


class _Moments:
    """Welford means and squared errors, with memory independent of repetitions."""

    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)

    def add(self, values):
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)


def aggregate(out: Path):
    """Summarize every valid saved repetition; retain missing/failed accounting."""
    out = Path(out).resolve()
    config = _read_json(out / "run_config.json")
    grid = np.linspace(config["a_min"], config["a_max"], config["grid_size"])
    point_acc, band_acc, success = {}, {}, set()
    recorded_grids, bandwidths = {}, {}
    for path in sorted((out / "jobs").glob("*.json")):
        try:
            job = _read_json(path)
            meta = job["metadata"]
            points, curves = pd.DataFrame(job["points"]), pd.DataFrame(job["curves"])
            _validate_layout(points, curves, meta, config)
            identity = tuple(meta[name] for name in CELL) + (meta["rep"],)
            if identity in success:
                raise ValueError("Duplicate saved repetition")
            success.add(identity)
            for (bandwidth, method), frame in points.groupby(["bandwidth", "method"], sort=False):
                frame = frame.sort_values("a")
                error = frame.estimate.to_numpy(dtype=float) - frame.truth.to_numpy(dtype=float)
                pi = method == "PI"
                coverage = np.zeros(len(grid)) if pi else frame.covered.to_numpy(dtype=float)
                length = np.zeros(len(grid)) if pi else (frame.ci_high - frame.ci_low).to_numpy(dtype=float)
                values = np.column_stack((error, coverage, length))
                with np.errstate(over="ignore"):
                    if not np.isfinite(values).all() or not np.isfinite(error ** 2).all():
                        raise ValueError("Invalid estimation error or squared error")
                key = tuple(meta[name] for name in CELL) + (bandwidth, method)
                if key not in point_acc:
                    point_acc[key] = _Moments(values.shape)
                    recorded_grids[key] = frame.a.to_numpy(dtype=float)
                    bandwidths[key] = (np.nan, np.nan) if pi else (float(frame.h.iloc[0]), float(frame.b.iloc[0]))
                elif not np.array_equal(recorded_grids[key], frame.a.to_numpy(dtype=float)):
                    raise ValueError("Saved repetitions use different evaluation grids")
                point_acc[key].add(values)
            if config["confidence_bands"]:
                for row in curves.loc[curves.method.eq("MRDB")].to_dict("records"):
                    key = tuple(meta[name] for name in CELL) + (row["bandwidth"], "MRDB")
                    values = np.array([row["band_covered"], row["mean_band_width"]], dtype=float)
                    if key not in band_acc:
                        band_acc[key] = _Moments(values.shape)
                    band_acc[key].add(values)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Cannot aggregate saved job {path.name}: {error}") from error

    failed, failure_rows = set(), []
    for path in sorted((out / "failures").glob("*.json")):
        failure = _read_json(path)
        meta = failure["metadata"]
        _check_metadata(meta, config)
        identity = tuple(meta[name] for name in CELL) + (meta["rep"],)
        failed.add(identity)
        failure_rows.append({**{name: meta[name] for name in META}, "file": path.name,
                             "resolved": identity in success, "error": failure["error"]})
    status_rows = []
    for setting, n, misspec in product(config["settings"], config["sample_sizes"], config["misspecifications"]):
        cell = (config["experiment"], setting, n, misspec)
        planned = {cell + (rep,) for rep in range(config["repetitions"])}
        complete = len(planned & success)
        unresolved = len(planned & (failed - success))
        pending = len(planned) - complete - unresolved
        status_rows.append({**dict(zip(CELL, cell)), "expected": len(planned), "success": complete,
                            "failed": unresolved, "pending": pending,
                            "state": "complete" if complete == len(planned) else "incomplete"})
    status = pd.DataFrame(status_rows)

    point_parts = []
    for key, acc in point_acc.items():
        h, b = bandwidths[key]
        pi = key[-1] == "PI"
        point_parts.append(pd.DataFrame({**dict(zip(GROUP, key)), "a": recorded_grids[key], "h": h, "b": b,
            "bias": acc.mean[:, 0], "rmse": np.sqrt(acc.mean[:, 0] ** 2 + acc.m2[:, 0] / acc.count),
            "coverage": np.nan if pi else acc.mean[:, 1],
            "mean_ci_length": np.nan if pi else acc.mean[:, 2], "denominator": acc.count}))
    point_summary = (pd.concat(point_parts, ignore_index=True).sort_values(GROUP + ["a"])
                     if point_parts else pd.DataFrame(columns=POINT_COLUMNS))
    band_rows = []
    for key, acc in band_acc.items():
        h, b = bandwidths[key]
        band_rows.append({**dict(zip(GROUP, key)), "h": h, "b": b, "grid_band_coverage": acc.mean[0],
                          "mean_band_width": acc.mean[1], "denominator": acc.count})

    outputs = {"pointwise_summary.csv": point_summary, "experiment_status.csv": status,
               "failures.csv": pd.DataFrame(failure_rows, columns=META + ["file", "resolved", "error"])}
    if config["confidence_bands"]:
        outputs["mrdb_band_summary.csv"] = pd.DataFrame(band_rows, columns=BAND_COLUMNS).sort_values(GROUP)
    for name, frame in outputs.items():
        frame.astype(object).where(pd.notna(frame), "").to_csv(out / name, index=False)
    counts = {name: int(status[name].sum()) for name in ("expected", "success", "failed", "pending")}
    manifest = {"repetitions": counts, "outputs": list(outputs),
                "interpretation": "All saved repetitions enter the statistics. PI has one estimate per dose/repetition and no interval or band statistics. MRDB band coverage is evaluated over the recorded dose grid."}
    (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
