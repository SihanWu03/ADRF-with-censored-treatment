"""Create paper simulation figures/tables from RESULTS/main and RESULTS/stability.

Run report.py first if the summaries need rebuilding.
PI enters only Bias/RMSE panels, once per input dose/repetition.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUP = ["experiment", "setting", "n", "misspec", "bandwidth", "method"]
SETTINGS = ("linear", "nonlinear")
BANDWIDTHS = ("regular", "undersmooth")
METHODS = ("Conventional", "ConventionalDB", "PI", "MR", "MRDB")
LABELS = {"Conventional": "Conv.", "ConventionalDB": "Conv.DB", "PI": "PI", "MR": "MR", "MRDB": "MRDB"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(path):
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    labels = set(GROUP) - {"n"} | {"experiment_state"}
    for column in frame:
        if column not in labels:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def expected_groups(config):
    methods = ("MR", "MRDB") if config["experiment"] == "stability" else tuple(m for m in METHODS if m != "PI")
    groups = set(itertools.product((config["experiment"],), config["settings"], config["sample_sizes"],
                                   config["misspecifications"], BANDWIDTHS, methods))
    if config["experiment"] == "main":
        groups.update(("main", s, n, "none", "none", "PI") for s in config["settings"] for n in config["sample_sizes"])
    return groups


def load_inputs(root):
    frames, configs, audit = {}, {}, {}
    for experiment in ("main", "stability"):
        directory = root / experiment
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8-sig"))
        require(config["experiment"] == experiment, f"{directory}: experiment mismatch")
        require(set(config["settings"]) == set(SETTINGS) and set(config["sample_sizes"]) == {2000, 4000, 8000},
                "Paper output requires both settings and n=2000,4000,8000")
        require(config["alpha"] == .05 and config["b_over_h"] == 1.25, "The paper uses alpha=.05 and b=1.25h")
        require(config["bandwidths"] == {"regular": .2, "undersmooth": .25}
                and config["c_h"] == {"regular": 2., "undersmooth": 1.}, "Unexpected paper bandwidth rules")
        require(config["a_min"] == .8 and config["a_max"] == 1.8, "Expected evaluation interval [0.8,1.8]")
        require(set(config["misspecifications"]) == ({"none"} if experiment == "main" else {"mu", "a", "c"}),
                f"{directory}: unexpected misspecification scenarios")
        require(config["confidence_bands"] == (experiment == "main"), "Only main MRDB uses confidence bands")
        require(config["repetitions"] == 1000 and config["grid_size"] == 201
                and config["integration_grid"] == 1601 and config["folds"] == 2,
                "Paper output requires the complete R=1000, 201-dose, 1601-node, two-fold design")
        require(config["seed_start"] == (200000 if experiment == "main" else 400000), "Unexpected formal seed namespace")
        require(config["bootstrap_draws"] == (999 if experiment == "main" else 0), "Unexpected multiplier count")
        status = pd.read_csv(directory / "experiment_status.csv", keep_default_na=False)
        cell = GROUP[:4]
        expected_cells = set(itertools.product((experiment,), SETTINGS, (2000, 4000, 8000), config["misspecifications"]))
        require(not status.duplicated(cell).any()
                and set(status[cell].itertuples(index=False, name=None)) == expected_cells,
                f"{directory}: missing or duplicate status cells")
        require(status.expected.eq(1000).all() and status.success.eq(1000).all()
                and status.failed.eq(0).all() and status.pending.eq(0).all()
                and status.state.eq("complete").all(), f"{directory}: unfinished or failed repetitions")
        path = directory / "pointwise_summary.csv"
        frame = read_frame(path)
        needed = set(GROUP) | {"a", "h", "b", "bias", "rmse", "coverage", "mean_ci_length", "denominator"}
        require(needed <= set(frame), f"{path}: missing columns {sorted(needed - set(frame))}")
        require(not frame.duplicated(GROUP + ["a"]).any(), f"{path}: duplicate summary rows")
        require(set(frame[GROUP].itertuples(index=False, name=None)) == expected_groups(config),
                f"{path}: missing or unexpected paper groups")
        require(frame.denominator.eq(1000).all(), f"{path}: expected 1000 repetitions per group")
        require(np.isfinite(frame[["a", "bias", "rmse"]]).all().all(), f"{path}: nonfinite metrics")
        require((frame.rmse ** 2 + 1e-12 >= frame.bias ** 2).all(), f"{path}: RMSE smaller than absolute bias")
        require(frame.rmse.gt(0).all(), f"{path}: nonpositive RMSE cannot be plotted on the log scale")
        grid = np.linspace(.8, 1.8, 201)
        for key, cell_frame in frame.groupby(GROUP, sort=False):
            doses = np.sort(cell_frame.a.to_numpy())
            require(len(doses) == 201 and np.allclose(doses, grid, rtol=0, atol=1e-12),
                    f"{path}: incomplete grid for {key}")
        pi = frame.method.eq("PI")
        require(frame.loc[pi, ["h", "b", "coverage", "mean_ci_length"]].isna().all().all(),
                "PI has no bandwidth or confidence intervals")
        interval = frame.loc[~pi]
        require(np.isfinite(interval[["h", "b", "coverage", "mean_ci_length"]]).all().all(), "Invalid interval statistics")
        require(interval.coverage.between(0, 1).all() and interval.mean_ci_length.gt(0).all(), "Invalid coverage/CI length")
        for bandwidth, cells in interval.groupby("bandwidth"):
            h = config["c_h"][bandwidth] * cells.n ** (-config["bandwidths"][bandwidth])
            require(np.allclose(cells.h, h) and np.allclose(cells.b, 1.25 * h), "Recorded bandwidth mismatch")
        frames[experiment], configs[experiment] = frame.sort_values(GROUP + ["a"]), config
        for filename in ("pointwise_summary.csv", "run_config.json", "manifest.json", "experiment_status.csv"):
            source = directory / filename
            audit[str(source.relative_to(root))] = digest(source)
    path = root / "main/mrdb_band_summary.csv"
    bands = read_frame(path)
    require(not bands.duplicated(GROUP).any(), "Duplicate MRDB band groups")
    expected = frames["main"].loc[frames["main"].method.eq("MRDB"), GROUP].drop_duplicates()
    require(set(bands[GROUP].itertuples(index=False, name=None))
            == set(expected.itertuples(index=False, name=None)), "Band/pointwise groups disagree")
    require(bands.denominator.eq(1000).all(), "MRDB bands require 1000 repetitions")
    require(np.isfinite(bands[["grid_band_coverage", "mean_band_width"]]).all().all()
            and bands.grid_band_coverage.between(0, 1).all() and bands.mean_band_width.gt(0).all(), "Invalid band metrics")
    audit[str(path.relative_to(root))] = digest(path)
    return frames, configs, bands, audit


def fmt(value, percent=False):
    if value is None or pd.isna(value):
        return r"\textemdash"
    text = f"{float(value) * (100 if percent else 1):.{1 if percent else 3}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def metrics(row):
    return [fmt(row.get("bias")), fmt(row.get("rmse")), fmt(row.get("coverage"), True), fmt(row.get("mean_ci_length"))]


def table_text(caption, label, columns, heading, rows, note, long=False):
    env = "longtable" if long else "tabular"
    lines = [r"\begingroup", r"\fontsize{8.5}{10}\selectfont" if long else r"\small", r"\setlength{\tabcolsep}{3pt}"]
    if not long:
        lines += [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}"]
    lines.append(r"\begin{" + env + "}{" + columns + "}")
    if long:
        lines += [r"\caption{" + caption + r"}\label{" + label + r"}\\"]
    lines += [r"\toprule", " & ".join(heading) + r" \\", r"\midrule"]
    if long:
        lines += [r"\endfirsthead", r"\toprule", " & ".join(heading) + r" \\", r"\midrule", r"\endhead"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{" + env + "}", r"\par\smallskip", r"\noindent\parbox{\textwidth}{\footnotesize " + note + "}"]
    if not long:
        lines.append(r"\end{table}")
    lines += [r"\endgroup", ""]
    return "\n".join(lines)


def export_tables(frames, configs, bands, output, note):
    exc, data = output / "exc", output / "tables"
    exc.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(.8, 1.8, 201)
    doses = [.8, 1.3, 1.8]
    methods_note = (r" Bias is signed Monte Carlo bias; RMSE is root mean squared error. CP is nominal 95\% interval coverage (\%). "
                    "Length is mean interval length. PI has no confidence intervals or second-stage bandwidth; its same estimates appear in both bandwidth panels.")
    for experiment in ("main", "stability"):
        frame = frames[experiment]
        selected = frame.loc[np.isclose(frame.a.to_numpy()[:, None], np.array(doses)[None, :]).any(axis=1)]
        selected.to_csv(data / f"{experiment}_representative_results.csv", index=False)
    for setting in configs["main"]["settings"]:
        selected = frames["main"].loc[frames["main"].setting.eq(setting)]
        rows = []
        for bandwidth, n, dose, method in itertools.product(BANDWIDTHS, configs["main"]["sample_sizes"], doses, METHODS):
            actual_bw = "none" if method == "PI" else bandwidth
            cell = selected.loc[selected.bandwidth.eq(actual_bw) & selected.n.eq(n) & np.isclose(selected.a, dose) & selected.method.eq(method)]
            row = cell.iloc[0].to_dict()
            rows.append([bandwidth, str(n), f"{dose:g}", LABELS[method], *metrics(row)])
        text = table_text(f"Main experiment: {setting}.", f"tab:sim-main-{setting}", "llrlrrrr",
                          ["Bandwidth", "$n$", "$a$", "Method", "Bias", "RMSE", r"CP (\%)", "Length"], rows, note + methods_note, long=True)
        (exc / f"main_{setting}.tex").write_text(text, encoding="utf-8")
    for bandwidth in BANDWIDTHS:
        selected = frames["stability"].loc[frames["stability"].bandwidth.eq(bandwidth)]
        rows = []
        for setting, n, dose, misspec in itertools.product(configs["stability"]["settings"], configs["stability"]["sample_sizes"], doses, ("mu", "a", "c")):
            values = []
            for method in ("MR", "MRDB"):
                cell = selected.loc[selected.setting.eq(setting) & selected.n.eq(n) & np.isclose(selected.a, dose) & selected.misspec.eq(misspec) & selected.method.eq(method)]
                values.extend(metrics(cell.iloc[0].to_dict()))
            rows.append([setting, str(n), f"{dose:g}", {"mu": r"$\mu$", "a": r"$f_A$", "c": r"$S_C$"}[misspec], *values])
        text = table_text(f"Stability experiment: {bandwidth} bandwidth.", f"tab:sim-stability-{bandwidth}", "llrlrrrrrrrr",
                          ["Setting", "$n$", "$a$", "Wrong", "MR Bias", "RMSE", "CP", "Length", "MRDB Bias", "RMSE", "CP", "Length"],
                          rows, note + r" Wrong identifies the single misspecified nuisance. CP is coverage (\%); the other nuisance models are correctly specified.", long=True)
        (exc / f"stability_{bandwidth}.tex").write_text(text, encoding="utf-8")
    rows = []
    for setting, n in itertools.product(configs["main"]["settings"], configs["main"]["sample_sizes"]):
        values = []
        for bandwidth in BANDWIDTHS:
            cell = bands.loc[bands.setting.eq(setting) & bands.n.eq(n) & bands.bandwidth.eq(bandwidth)]
            row = cell.iloc[0].to_dict()
            values.extend([fmt(row.get("grid_band_coverage"), True), fmt(row.get("mean_band_width"))])
        rows.append([setting, str(n), *values])
    text = table_text("MRDB simultaneous confidence bands.", "tab:sim-bands", "lrrrrr",
                      ["Setting", "$n$", "Regular SCP", "Width", "Undersmooth SCP", "Width"], rows,
                      note + f" SCP is coverage (\\%) over all {len(grid)} evaluation doses. Width is averaged over doses and replications.")
    (exc / "mrdb_bands.tex").write_text(text, encoding="utf-8")
    bands.to_csv(data / "mrdb_band_results.csv", index=False)
    return doses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.results_root.resolve(), args.out.resolve()
    for source in (root / "main", root / "stability"):
        require(output != source and source not in output.parents and output not in source.parents,
                "Output must be separate from the main/stability input directories")
    frames, configs, bands, audit = load_inputs(root)
    note = "Each displayed cell uses 1000 repetitions."
    output.mkdir(parents=True, exist_ok=True)
    doses = export_tables(frames, configs, bands, output, note)
    files = []
    from paper_reporting import figures
    figures.style()
    art = output / "art"
    art.mkdir(exist_ok=True)
    for setting, bandwidth in itertools.product(SETTINGS, BANDWIDTHS):
        files.extend(figures.main_figure(frames["main"], setting, bandwidth, art))
    for setting, misspec in itertools.product(SETTINGS, ("mu", "a", "c")):
        files.extend(figures.stability_figure(frames["stability"], setting, misspec, art))
    accounting = {
        name: pd.read_csv(root / name / "experiment_status.csv", keep_default_na=False).to_dict("records")
        for name in frames
    }
    record = dict(input_root=str(root), input_sha256=audit,
                  status="complete_formal_design", note=note,
                  accounting=accounting, configurations=configs, representative_doses=doses,
                  figures=files, pi="One input estimate per dose/repetition; reused in both bandwidth figures. No PI confidence intervals or band statistics.")
    for relative, expected in audit.items():
        require(digest(root / relative) == expected, f"Input changed during export: {relative}")
    (output / "paper_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "out": str(output), "figure_files": len(files)}))
    return record


if __name__ == "__main__":
    main()
