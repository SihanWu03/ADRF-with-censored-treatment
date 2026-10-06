"""Paper simulation layouts; data validation is handled by make_paper_outputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, MaxNLocator
import numpy as np
import pandas as pd


SAMPLE_SIZES = (2000, 4000, 8000)
SETTINGS = ("linear", "nonlinear")
BANDWIDTHS = ("regular", "undersmooth")
MAIN_METHODS = ("Conventional", "ConventionalDB", "MR", "MRDB")
MAIN_PLOT_METHODS = MAIN_METHODS + ("PI",)
ROBUST_METHODS = ("MR", "MRDB")
DOSES = np.linspace(0.8, 1.8, 201)
METRICS = ("bias", "rmse", "coverage")
MAIN_METRICS = METRICS + ("mean_ci_length",)
REPORT_NOTE = "Each displayed cell uses 1000 repetitions."
COLORS = {
    "Conventional": "#737980",
    "ConventionalDB": "#A67C38",
    "MR": "#315D87",
    "MRDB": "#AB493E",
    "PI": "#507B67",
}
LINESTYLES = {"regular": "-", "undersmooth": (0, (4.2, 2.2))}
METHOD_STYLES = {
    "Conventional": (0, (6.0, 2.5)),
    "ConventionalDB": (0, (4.5, 1.9, 1.0, 1.9)),
    "MR": "-",
    "MRDB": (0, (2.7, 1.7)),
    "PI": (0, (1.0, 1.8)),
}
MARKERS = {"Conventional": "^", "ConventionalDB": "s", "MR": "o", "MRDB": "D", "PI": "v"}
METHOD_MARK_OFFSETS = {"Conventional": 3, "ConventionalDB": 11, "MR": 19, "MRDB": 27, "PI": 35}
METRIC_LABELS = {"bias": "Bias", "rmse": "RMSE (log scale)", "coverage": "Coverage (%)", "mean_ci_length": "Mean CI length"}
MISSPEC_LABELS = {"mu": r"$\mu$ misspecified", "a": r"$f_A$ misspecified", "c": r"$S_C$ misspecified"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9.5,
        "mathtext.fontset": "stix",
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "axes.titlepad": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def limits(values: np.ndarray, metric: str) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    require(len(values) > 0, "Missing plot data")
    if metric == "coverage":
        # A small display margin makes genuine zero-coverage curves visible.
        return (-2.0, 101.5)
    if metric == "rmse":
        # Identical multiplicative padding for every compared sample size.
        require(np.all(values > 0), "RMSE must be positive for the log scale")
        return (float(values.min()) / 1.16, float(values.max()) * 1.16)
    if metric == "mean_ci_length":
        return (0.0, float(values.max() * 1.08))
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    padding = max((high - low) * 0.075, 0.001)
    return (low - padding, high + padding)


def configure_axis(axis, metric: str, ylim: tuple[float, float]) -> None:
    axis.set_xlim(0.79, 1.81)
    axis.set_ylim(*ylim)
    axis.xaxis.set_major_locator(FixedLocator([0.8, 1.0, 1.2, 1.4, 1.6, 1.8]))
    axis.tick_params(axis="both", pad=3, labelbottom=True, labelleft=True)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E8E8E8", linewidth=0.4)
    if metric == "rmse":
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(which="minor", axis="y", left=False)
    elif metric == "coverage":
        ticks = ([0, 25, 50, 75, 100] if ylim[0] < 0 else
                 [85, 90, 95, 100] if ylim[0] == 85 else
                 [70, 80, 90, 100] if ylim[0] == 65 else [20, 40, 60, 80, 100])
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.axhline(95, color="#444444", linestyle=(0, (1.0, 2.0)), linewidth=0.65, zorder=1)
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:g}"))
        if metric == "bias":
            axis.axhline(0, color="#444444", linestyle=(0, (1.0, 2.0)), linewidth=0.65, zorder=1)
    axis.set_xlabel(r"Dose $a$", labelpad=3.5)
    axis.set_ylabel(METRIC_LABELS[metric], labelpad=4)


def draw_curves(axis, frame: pd.DataFrame, metric: str, methods: tuple[str, ...], *, bandwidth: str | None = None, detail: bool = False) -> None:
    for method in methods:
        if method == "PI" and metric in ("coverage", "mean_ci_length"):
            continue
        for rule in (("none",) if method == "PI" else (bandwidth,) if bandwidth else BANDWIDTHS):
            values = frame.loc[frame["method"].eq(method) & frame["bandwidth"].eq(rule)].sort_values("a")
            require(len(values) == len(DOSES), f"Internal plotting error: expected one complete curve for {method}/{rule}")
            response = values[metric].to_numpy() * (100 if metric == "coverage" else 1)
            offset = METHOD_MARK_OFFSETS[method] if bandwidth else (4 + 10 * ROBUST_METHODS.index(method) + 20 * BANDWIDTHS.index(rule))
            axis.plot(values["a"], response, color=COLORS[method],
                      linestyle=METHOD_STYLES[method] if bandwidth else LINESTYLES[rule],
                      linewidth=0.85 if detail else 1.05,
                      marker=MARKERS[method], markevery=(offset, 40),
                      markersize=2.4 if detail else 3.5, markerfacecolor="white",
                      markeredgewidth=0.65, solid_capstyle="butt", dash_capstyle="butt",
                      clip_on=True, zorder=3)


def legend(figure, *, stability: bool = False, y: float = 0.897, fontsize: float = 8.5) -> None:
    if stability:
        handles = [Line2D([0], [0], color=COLORS[method], linewidth=1.05,
                          linestyle=LINESTYLES[rule], marker=MARKERS[method],
                          markersize=3.5, markerfacecolor="white", markeredgewidth=0.65,
                          label=f"{method}, {'regular' if rule == 'regular' else 'undersmoothed'}")
                   for method in ROBUST_METHODS for rule in BANDWIDTHS]
    else:
        handles = [Line2D([0], [0], color=COLORS[name], linewidth=1.05,
                          linestyle=METHOD_STYLES[name], marker=MARKERS[name],
                          markersize=3.5, markerfacecolor="white", markeredgewidth=0.65,
                          label=name) for name in MAIN_PLOT_METHODS]
    figure.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, y), ncol=4 if stability else 5,
                  handlelength=2.5, columnspacing=1.4, handletextpad=0.55, borderaxespad=0,
                  fontsize=fontsize)


def add_detail(axis, frame: pd.DataFrame, metric: str, ylim: tuple[float, float], *, bandwidth: str | None = None,
               methods: tuple[str, ...] = ROBUST_METHODS, title: str | None = None,
               show_title: bool = True, tick_size: float = 7.8, coverage_bottom: float = 0.56) -> None:
    """An inset repeats original values; it never offsets or smooths them."""
    # Coverage insets occupy a verified empty region above the low-coverage
    # MR curve, never hide the full-range data behind a white inset patch.
    bounds = [0.12, coverage_bottom, 0.72, 0.26] if metric == "coverage" else [0.27, 0.49, 0.69, 0.24]
    inset = axis.inset_axes(bounds)
    inset.set_facecolor("white")
    inset.set_xlim(0.78, 1.82)
    inset.set_ylim(*ylim)
    inset.xaxis.set_major_locator(FixedLocator([0.8, 1.3, 1.8]))
    inset.yaxis.set_major_locator(FixedLocator([90, 95, 100]) if metric == "coverage" else MaxNLocator(nbins=3))
    inset.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:g}"))
    inset.tick_params(axis="both", labelsize=tick_size, pad=1.5, length=1.8, width=0.45)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#A0A0A0")
        spine.set_linewidth(0.45)
    inset.axhline(95 if metric == "coverage" else 0, color="#777777", linewidth=0.5, linestyle=(0, (1, 2)), zorder=1)
    draw_curves(inset, frame, metric, methods, bandwidth=bandwidth, detail=True)
    if not show_title:
        return
    if metric == "bias":
        # The full nonlinear bias curve crosses the space above the inset.
        # Label the window in the empty left margin rather than over that curve.
        axis.text(0.015, 0.60, "Detail", transform=axis.transAxes,
                  fontsize=7.5, va="center", ha="left", color="#444444")
    else:
        inset.set_title(title or "90–100% detail", fontsize=7.5, pad=3)


def save_figure(figure, output: Path, stem: str, description: str) -> list[dict]:
    paths = []
    figure.text(0.5, 0.007, REPORT_NOTE, ha="center", fontsize=7.5)
    for extension in ("pdf", "png"):
        target = output / f"{stem}.{extension}"
        metadata = {"Title": description, "Author": "", "Subject": REPORT_NOTE, "Creator": "make_paper_outputs.py", "CreationDate": None, "ModDate": None} if extension == "pdf" else {"Title": description, "Software": "make_paper_outputs.py"}
        figure.savefig(target, dpi=300, metadata=metadata)
        paths.append({"path": target.name, "bytes": target.stat().st_size, "sha256": sha256(target)})
    plt.close(figure)
    return paths


def main_figure(frame: pd.DataFrame, setting: str, bandwidth: str, output: Path) -> list[dict]:
    """One setting/bandwidth: three sample-size rows by four metric columns."""
    figure, axes = plt.subplots(3, 4, figsize=(9.6, 8.4), squeeze=False)
    figure.subplots_adjust(left=0.083, right=0.990, bottom=0.12, top=0.840, hspace=0.48, wspace=0.30)
    subset = frame.loc[frame["setting"].eq(setting)]
    number = SETTINGS.index(setting) + 1
    title = f"Setting {number}: {setting.capitalize()} dose–response curve"
    figure.suptitle(title, x=0.53, y=0.987, fontsize=16.0)
    legend(figure, y=0.931, fontsize=11.0)

    # Full-panel limits are shared across n and across the two bandwidth figures.
    # Insets retain original coordinates; only the axis window changes.
    near_zero = subset.loc[subset["method"].isin(("MR", "MRDB", "PI"))]
    if setting == "nonlinear":
        near_zero = near_zero.loc[~(near_zero["method"].eq("MR") & near_zero["bandwidth"].eq("regular"))]
    bias_detail_limits = limits(near_zero["bias"].to_numpy(), "bias")
    for col, metric in enumerate(MAIN_METRICS):
        ylim = limits(subset[metric].to_numpy(), metric)
        methods = MAIN_PLOT_METHODS if metric in ("bias", "rmse") else MAIN_METHODS
        for row, n in enumerate(SAMPLE_SIZES):
            axis = axes[row, col]
            values = subset.loc[subset["n"].eq(n)]
            configure_axis(axis, metric, ylim)
            axis.tick_params(axis="both", labelsize=11.0)
            axis.xaxis.label.set_size(11.0)
            axis.set_ylabel("")
            draw_curves(axis, values, metric, methods, bandwidth=bandwidth)
            if metric == "bias":
                detail_methods = ("MRDB", "PI") if setting == "nonlinear" and bandwidth == "regular" else ("MR", "MRDB", "PI")
                add_detail(axis, values, metric, bias_detail_limits, bandwidth=bandwidth,
                           methods=detail_methods, show_title=False, tick_size=9.5)
            elif metric == "coverage":
                add_detail(axis, values, metric, (90, 100), bandwidth=bandwidth,
                           show_title=False, tick_size=9.5, coverage_bottom=0.60)
        position = axes[0, col].get_position()
        figure.text((position.x0 + position.x1) / 2, position.y1 + 0.025,
                    METRIC_LABELS[metric], ha="center", va="bottom", fontsize=12.0)
    for row, n in enumerate(SAMPLE_SIZES):
        position = axes[row, 0].get_position()
        figure.text(0.020, (position.y0 + position.y1) / 2, rf"$n={n}$",
                    ha="center", va="center", rotation=90, fontsize=12.0)
    return save_figure(figure, output, f"main_{setting}_{bandwidth}", title + "; " + bandwidth)


def stability_figure(frame: pd.DataFrame, setting: str, misspec: str, output: Path) -> list[dict]:
    subset = frame.loc[frame["setting"].eq(setting) & frame["misspec"].eq(misspec)]
    figure, axes = plt.subplots(3, 3, figsize=(7.35, 8.65), squeeze=False)
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.081, top=0.839, hspace=0.53, wspace=0.37)
    for row, metric in enumerate(METRICS):
        ylim = limits(subset[metric].to_numpy(), metric)
        if metric == "coverage" and setting == "linear":
            ylim = (85, 100)
        elif metric == "coverage":
            ylim = (65, 100) if misspec == "mu" else (10, 100)
        for col, n in enumerate(SAMPLE_SIZES):
            axis = axes[row, col]
            values = subset.loc[subset["n"].eq(n)]
            configure_axis(axis, metric, ylim)
            draw_curves(axis, values, metric, ROBUST_METHODS)
            axis.set_title(f"({chr(97 + len(SAMPLE_SIZES) * row + col)})  " + rf"$n={n:,}$")
            if metric == "coverage" and setting == "nonlinear" and misspec != "mu":
                add_detail(axis, values, metric, (90, 100))
    number = SETTINGS.index(setting) + 1
    figure.suptitle("Stability under nuisance-model misspecification", x=0.54, y=0.984, fontsize=13.5)
    figure.text(0.54, 0.95, f"Setting {number}: {setting}" + "  |  " + MISSPEC_LABELS[misspec], ha="center", fontsize=11.0)
    legend(figure, stability=True, y=0.912)
    figure.text(0.54, 0.882, r"Regular: $h=2n^{-1/5}$     |     Undersmoothing: $h=n^{-1/4}$", ha="center", fontsize=9.0, color="#555555")
    note = ("Coverage axes: 85-100%." if setting == "linear" else
            "Coverage axes: 65-100%." if misspec == "mu" else "Coverage axes: 10-100%; insets: 90-100%.")
    figure.text(0.105, 0.027, "Dotted lines: zero bias and 95% coverage. RMSE uses a log scale. " + note, ha="left", fontsize=7.8, color="#555555")
    return save_figure(figure, output, f"stability_{setting}_{misspec}", f"Stability experiment: {setting}, {misspec} misspecified")

