"""Run the manuscript workflows from this directory or any other working directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def build_commands(args):
    output = args.output_dir.resolve()
    prefix = [sys.executable, "-B"]
    commands = []
    experiments = {
        "all": ("main", "stability"), "simulations": ("main", "stability"),
        "nhanes": (),
    }[args.scope]
    for experiment in experiments:
        config = experiment + ".json"
        command = prefix + [str(ROOT / "run.py"), "--config", str(ROOT / "configs" / config),
                            "--out", str(output / experiment)]
        if args.device:
            command += ["--device", args.device]
        if args.retry_failed:
            command += ["--retry-failed"]
        commands.append(command)
    if args.scope in ("all", "simulations"):
        command = prefix + [str(ROOT / "make_paper_outputs.py"), "--results-root", str(output),
                            "--out", str(output / "paper")]
        commands.append(command)
    if args.scope in ("all", "nhanes"):
        raw = args.raw_dir.resolve() if args.raw_dir else output / "raw" / "nhanes"
        command = prefix + [str(ROOT / "run_nhanes.py"), "--raw-dir", str(raw),
                            "--out", str(output / "nhanes")]
        if not args.raw_dir:
            command.append("--download")
        if args.device:
            command += ["--device", args.device]
        commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "simulations", "nhanes"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", help="For example cuda, cuda:0, or cpu; otherwise use each config")
    parser.add_argument("--raw-dir", type=Path, help="Existing verified NHANES XPT cache; omit to download")
    parser.add_argument("--retry-failed", action="store_true", help="Retry recorded failed simulation jobs")
    args = parser.parse_args()
    if args.raw_dir and args.scope not in ("all", "nhanes"):
        parser.error("--raw-dir applies only to all or nhanes")
    commands = build_commands(args)
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
