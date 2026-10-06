# Reproduction

## Installation

Use Python 3.12 and install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

GPU execution requires a CUDA-compatible PyTorch installation. Use `--device cpu` for CPU execution.

## Run all experiments

```bash
python reproduce.py --device cuda --output-dir results
```

This runs the main simulations, misspecification experiments, and NHANES analysis, then produces the corresponding figures and tables. The simulation design uses 1,000 repetitions per cell: 6,000 main-experiment tasks and 18,000 misspecification tasks.

To run the two parts separately:

```bash
python reproduce.py --scope simulations --device cuda --output-dir results
python reproduce.py --scope nhanes --device cuda --output-dir results
```

Repeat a simulation command to resume an interrupted run; completed tasks are skipped. Add `--retry-failed` to rerun failed tasks with the same seeds. Use a new output directory when changing the configuration or numerical environment.

## NHANES data

The NHANES workflow downloads 15 public CDC/NCHS files for the 2003-2008 cycles, verifies their hashes, constructs the LDL cohort, and calibrates the semi-synthetic model before estimation. File URLs and hashes are recorded in `nhanes/source_files.json`.

To use an existing directory containing these files:

```bash
python reproduce.py --scope nhanes --raw-dir /path/to/nhanes/xpt --device cuda --output-dir results
```

NHANES stages require empty output directories. Individual stages are available through `run_nhanes.py --stage prepare`, `--stage calibrate`, `--stage analyze`, and `--stage plot`, with a shared `--out results/nhanes` directory.

## Regenerate figures and tables

```bash
python report.py --out results/main
python report.py --out results/stability
python make_paper_outputs.py --results-root results --out results/paper
```

| Output | Directory |
| --- | --- |
| Main simulation results | `results/main/` |
| Misspecification results | `results/stability/` |
| Simulation figures | `results/paper/art/` |
| LaTeX tables | `results/paper/exc/` |
| Table data | `results/paper/tables/` |
| NHANES estimates and figure | `results/nhanes/analysis/` |

## Configuration

| File | Experiment |
| --- | --- |
| `configs/main.json` | Main simulation |
| `configs/stability.json` | Nuisance-model misspecification |
| `configs/nhanes.json` | NHANES-calibrated analysis |

Both simulation bandwidth rules use the same data, cross-fitting folds, and nuisance fits. PI is evaluated for point estimation only. Simultaneous bands are computed for main-experiment MRDB; the misspecification and NHANES analyses use pointwise intervals.

For multiple GPUs, run `run.py` with distinct `--shard-index` values, a common `--shards` count, and `--no-report`. After all workers finish, run `report.py` on their shared output directory.
