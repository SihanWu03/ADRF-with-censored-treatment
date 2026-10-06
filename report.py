"""Rebuild paper simulation summaries from saved repetitions without refitting."""
import argparse
import json
from pathlib import Path

from adrf.reporting import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Simulation output directory containing jobs/.")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.out), indent=2))


if __name__ == "__main__":
    main()
