"""Regenerate the LDL sample, fit MR/MRDB, and export the manuscript figure."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from adrf.estimators import cache_fold, inference, solve, split_folds, tensor
from nhanes.generator import DOSE_SCALE, Generator, T, fit, save_json
from nhanes.nuisance import fit_nuisance
from nhanes.prepare import sha256


def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed or partial outputs: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def calibrate(source_dir, output_dir, cfg):
    source_dir = Path(source_dir)
    source = source_dir / "source.csv"
    manifest = json.loads((source_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Prepared source differs from its manifest")
    output_dir = fresh(output_dir)
    generator = fit(source, manifest["x_columns"], cfg["calibration_seed"])
    generator.meta["source_sha256"] = manifest["source_sha256"]
    generator.save(output_dir)
    print(f"Calibrated LDL generator: sigma_y={generator.meta['sigma_y']:.8f}, expected censoring={generator.meta['expected_censoring_rate']:.3f}", flush=True)


def plot(output_dir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / "pointwise_results.csv", float_precision="round_trip")
    lower, upper = cfg["display_dose_range"]
    display = frame.loc[frame.dose_g.between(lower-1e-10, upper+1e-10)].copy()
    if display.empty:
        raise ValueError("No estimated doses fall inside the display interval")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8})
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    colors = {"MR": "#2675B5", "MRDB": "#DE7E26"}
    for method in ("MR", "MRDB"):
        part = display.loc[display.method.eq(method)].sort_values("a")
        ax.fill_between(part.dose_g, part.ci_low, part.ci_high, color=colors[method], alpha=.13, linewidth=0)
        ax.plot(part.dose_g, part.estimate, color=colors[method], linestyle="-." if method == "MR" else "-",
                lw=1.7 if method == "MR" else 2.1, label=method, zorder=3 if method == "MR" else 4)
    ax.plot(part.dose_g, part.truth, color="#171717", linestyle="--", lw=1.5,
            label="Generating-model truth", zorder=5)
    ax.set_xlim(lower, upper)
    if [lower, upper] == [16., 27.5]:
        ax.set_xticks([16, 18, 20, 22, 24, 26, 27.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Dietary fiber dose (g/day)", labelpad=8)
    ax.set_ylabel("Mean LDL cholesterol (mg/dL)", labelpad=8)
    ax.set_title(r"NHANES-calibrated example: $h=2n^{-1/5}$", fontsize=13, pad=13)
    ax.grid(color="#E4E5E7", linewidth=.6, alpha=.65)
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=.9, edgecolor="#dddddd", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "nhanes_ldl_adrf.png", dpi=250)
    fig.savefig(output_dir / "nhanes_ldl_adrf.pdf")
    plt.close(fig)


def analyze(calibration_dir, output_dir, cfg):
    calibration_dir = Path(calibration_dir)
    n, seed, device = cfg["n"], cfg["analysis_seed"], cfg["device"]
    if n % cfg["folds"]:
        raise ValueError("Equal fold sizes are required")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    torch.set_num_threads(cfg["torch_threads"])
    h = cfg["c_h"] * n ** (-cfg["bandwidth_power"])
    b = cfg["b_over_h"] * h
    largest = max(h, b)
    if not (cfg["a_min"]-largest > 0 and cfg["a_max"]+largest < T):
        raise ValueError("Local windows must be inside [0,3]")
    output_dir = fresh(output_dir)
    started = time.perf_counter()
    generator = Generator.load(calibration_dir)
    data, generation = generator.generate(n, seed)
    if not np.array_equal(np.isnan(data.y), ~data.delta):
        raise AssertionError("Censored outcomes must be masked")
    observed = pd.DataFrame(data.x, columns=generator.x_columns)
    observed.insert(0, "generated_id", np.arange(n))
    observed["u"], observed["delta"], observed["y"] = data.u, data.delta.astype(int), data.y
    observed.to_csv(output_dir / "generated_observed.csv", index=False)
    package = Path(__file__).resolve().parents[1]
    code_paths = [package / "adrf" / name for name in ("estimators.py", "nuisance.py", "dgp.py")]
    code_paths += [Path(__file__), Path(__file__).with_name("nuisance.py"), Path(__file__).with_name("generator.py")]
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
                    n=n, generation=generation, methods=["MR", "MRDB"], h=h, b=b,
                    source_n=generator.source_n, new_sample_generated=True,
                    source_sha256=generator.meta["source_sha256"],
                    calibration_weights_sha256=sha256(calibration_dir / "weights.npz"),
                    observed_data_sha256=sha256(output_dir / "generated_observed.csv"),
                    code_sha256={str(path.relative_to(package)).replace("\\", "/"): sha256(path) for path in code_paths},
                    numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
                    device=torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "cpu",
                    pointwise_ci_level=1-cfg["alpha"], confidence_bands=False,
                    interpretation="Known truth under the frozen NHANES-calibrated model; not a real NHANES causal curve. Pointwise intervals condition on calibration. MR intervals do not correct smoothing bias.")
    save_json(output_dir / "analysis_manifest.json", manifest)
    print(f"Generated sample: {generation}", flush=True)
    try:
        targets_np = np.linspace(cfg["a_min"], cfg["a_max"], cfg["grid_size"])
        targets = tensor(targets_np, device)
        grid = tensor(np.linspace(0., cfg["a_max"]+largest, cfg["integration_grid"]), device)
        caches = []
        for fold, (evaluation, training) in enumerate(split_folds(n, cfg["folds"], seed+3901)):
            print(f"Fitting nuisance fold {fold+1}/{cfg['folds']} from observed data only", flush=True)
            train = data.subset(training)
            model = fit_nuisance(train.x, train.u, train.delta, train.y, T=T,
                                 seed=seed+100*fold+7103, device=device, config=cfg["nuisance"])
            cache = cache_fold(data, evaluation, training, model, grid, False, cfg)
            caches.append(cache)
            del model
        solved, _ = solve(caches, grid, targets, h, b, n)
        # Known truth is evaluated only after fitting and never enters estimation.
        truth = generator.theta(targets_np)
        rows = []
        for method in ("MR", "MRDB"):
            result = inference(*solved[method], draws=0, alpha=cfg["alpha"], seed=seed+17389, confidence_bands=False)
            for j, a in enumerate(targets_np):
                rows.append(dict(method=method, bandwidth="regular", a=a, dose_g=a*DOSE_SCALE,
                                 truth=float(truth[j]), h=h, b=b, n=n,
                                 **{name: float(values[j]) for name, values in result.items()}))
        frame = pd.DataFrame(rows)
        if not np.isfinite(frame.select_dtypes("number").to_numpy()).all():
            raise FloatingPointError("Nonfinite estimator output")
        if not ((frame.ci_low < frame.estimate) & (frame.estimate < frame.ci_high)).all():
            raise AssertionError("Invalid pointwise interval ordering")
        frame.to_csv(output_dir / "pointwise_results.csv", index=False)
        plot(output_dir, cfg)
        manifest.update(status="completed", elapsed_seconds=time.perf_counter()-started)
        save_json(output_dir / "analysis_manifest.json", manifest)
        print(f"Completed MR/MRDB and figure in {manifest['elapsed_seconds']:.1f}s: {output_dir}", flush=True)
    except Exception as error:
        manifest.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        save_json(output_dir / "analysis_manifest.json", manifest)
        raise
