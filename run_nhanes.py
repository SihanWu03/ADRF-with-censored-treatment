"""Reproduce the NHANES LDL example from public source files, without saved results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nhanes.analysis import analyze, calibrate, plot
from nhanes.prepare import prepare


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "nhanes.json")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw" / "nhanes")
    parser.add_argument("--out", type=Path, default=root / "results" / "nhanes")
    parser.add_argument("--download", action="store_true", help="Download missing public CDC/NCHS XPT tables; verify pinned hashes")
    parser.add_argument("--device", help="For example cpu or cuda:0; default uses CUDA when available, otherwise CPU")
    parser.add_argument("--stage", choices=("all", "prepare", "calibrate", "analyze", "plot"), default="all")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["device"] = args.device or (cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(cfg["torch_threads"])
    if args.stage in ("all", "prepare"):
        prepare(args.raw_dir, args.out / "source", download=args.download)
    if args.stage in ("all", "calibrate"):
        calibrate(args.out / "source", args.out / "calibration", cfg)
    if args.stage in ("all", "analyze"):
        analyze(args.out / "calibration", args.out / "analysis", cfg)
    if args.stage == "plot":
        # Plot saved output under the settings that produced that output.
        saved = json.loads((args.out / "analysis" / "analysis_manifest.json").read_text(encoding="utf-8"))
        plot(args.out / "analysis", saved["config"])


if __name__ == "__main__":
    main()
