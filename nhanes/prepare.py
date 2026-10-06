"""Download public NHANES inputs and build the manuscript's LDL cohort.

The cohort is constructed from 15 public XPT tables whose contents are
verified against pinned SHA-256 hashes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from nhanes.generator import save_json

CYCLES = ((2003, "C"), (2005, "D"), (2007, "E"))
LIPID_MODULE = {2003: "L13AM", 2005: "TRIGLY", 2007: "TRIGLY"}
EXPECTED_N = 4797
EXPECTED_P = 17


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_specs():
    return json.loads(Path(__file__).with_name("source_files.json").read_text(encoding="utf-8-sig"))


def validate_source(record, data):
    if not data.startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!") or len(data) % 80:
        raise ValueError(f"{record['file']}: response is not a valid SAS XPORT table")
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"{record['file']}: source differs from the manuscript's pinned public file")


def ensure_sources(raw_dir, download=False):
    raw_dir = Path(raw_dir)
    records = source_specs()
    missing = [item["file"] for item in records if not (raw_dir / item["file"]).is_file()]
    if missing and not download:
        raise FileNotFoundError("Missing NHANES XPT inputs. Use --download or --raw-dir with a complete cache: " + ", ".join(missing))
    if missing:
        raw_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = raw_dir / record["file"]
        if path.exists():
            validate_source(record, path.read_bytes())
            continue
        print(f"Downloading {record['file']} from CDC/NCHS", flush=True)
        for attempt in range(3):
            try:
                request = Request(record["url"], headers={"User-Agent": "NHANES academic reproducibility"})
                with urlopen(request, timeout=60) as response:
                    data = response.read()
                validate_source(record, data)
                break
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        # Never overwrite an existing raw table, including concurrent downloads.
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            validate_source(record, path.read_bytes())
    return records


def read_module(raw_dir, module, suffix, columns):
    frame = pd.read_sas(Path(raw_dir) / f"{module}_{suffix}.xpt", format="xport")
    absent = sorted(set(columns) - set(frame.columns))
    if absent or frame.SEQN.duplicated().any():
        raise ValueError(f"{module}_{suffix}: missing columns {absent} or duplicate respondent IDs")
    frame = frame.loc[:, columns].copy()
    # Some pandas XPORT readers decode SAS zero as approximately 5.4e-79.
    # No selected variable has genuine measurements at this magnitude.
    for column in columns:
        magnitude = frame[column].abs()
        mask = magnitude.gt(0) & magnitude.lt(1e-12)
        frame.loc[mask, column] = 0.0
    return frame


def prepare_cycle(raw_dir, year, suffix):
    frame = read_module(raw_dir, "DEMO", suffix,
                        ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "INDFMPIR", "RIDEXPRG"])
    for module, columns in (
        ("BMX", ["SEQN", "BMXBMI"]),
        ("SMQ", ["SEQN", "SMQ020", "SMQ040"]),
        ("DR1TOT", ["SEQN", "DR1DRSTZ", "DR1TFIBE", "DR1TKCAL"]),
        (LIPID_MODULE[year], ["SEQN", "WTSAF2YR", "LBDLDL", "LBXTR"]),
    ):
        frame = frame.merge(read_module(raw_dir, module, suffix, columns),
                            how="left", on="SEQN", validate="one_to_one")

    def retain(mask):
        nonlocal frame
        frame = frame.loc[mask.fillna(False)].copy()

    retain(frame.RIDAGEYR.between(20, 79) & frame.RIAGENDR.isin([1, 2]))
    pregnancy_max = 44 if year == 2007 else 59
    eligible = frame.RIAGENDR.eq(2) & frame.RIDAGEYR.le(pregnancy_max)
    retain(frame.RIDEXPRG.ne(1) & (~eligible | frame.RIDEXPRG.eq(2)))
    retain(frame.DR1DRSTZ.eq(1))
    retain(frame.DR1TFIBE.gt(0) & frame.DR1TFIBE.lt(60))
    frame["smoking"] = np.nan
    frame.loc[frame.SMQ020.eq(2), "smoking"] = 0
    frame.loc[frame.SMQ020.eq(1) & frame.SMQ040.eq(3), "smoking"] = 1
    frame.loc[frame.SMQ020.eq(1) & frame.SMQ040.isin([1, 2]), "smoking"] = 2
    retain(frame.BMXBMI.gt(0) & np.isfinite(frame.BMXBMI))
    retain(frame.DR1TKCAL.gt(0) & np.isfinite(frame.DR1TKCAL))
    retain(frame.DMDEDUC2.isin([1, 2, 3, 4, 5]))
    retain(frame.INDFMPIR.ge(0) & np.isfinite(frame.INDFMPIR))
    retain(frame.smoking.notna())
    retain(frame.RIDRETH1.isin([1, 2, 3, 4, 5]))
    retain(frame.WTSAF2YR.gt(0) & np.isfinite(frame.WTSAF2YR))
    retain(frame.LBDLDL.gt(0) & np.isfinite(frame.LBDLDL))
    retain(frame.LBXTR.gt(0) & frame.LBXTR.lt(400))
    out = pd.DataFrame({"subject_id": frame.SEQN.astype("int64"), "dose": frame.DR1TFIBE,
                        "y": frame.LBDLDL, "age": frame.RIDAGEYR, "sex": frame.RIAGENDR.eq(2).astype(int),
                        "bmi": frame.BMXBMI, "log_energy": np.log(frame.DR1TKCAL), "pir": frame.INDFMPIR})
    for category in (2, 3, 4, 5):
        out[f"x_education_{category}"] = frame.DMDEDUC2.eq(category).astype(int)
    out["x_smoking_former"] = frame.smoking.eq(1).astype(int)
    out["x_smoking_current"] = frame.smoking.eq(2).astype(int)
    for category in (1, 2, 4, 5):
        out[f"x_race_{category}"] = frame.RIDRETH1.eq(category).astype(int)
    out["x_cycle_2005"] = int(year == 2005)
    out["x_cycle_2007"] = int(year == 2007)
    return out


def prepare(raw_dir, output_dir, download=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace a prepared cohort: {output_dir}")
    records = ensure_sources(raw_dir, download)
    cycles = [prepare_cycle(raw_dir, year, suffix) for year, suffix in CYCLES]
    source = pd.concat(cycles, ignore_index=True).sort_values("subject_id")
    x_columns = [column for column in source if column not in ("subject_id", "dose", "y")]
    if (len(source), len(x_columns)) != (EXPECTED_N, EXPECTED_P):
        raise ValueError(f"Expected the manuscript cohort {(EXPECTED_N, EXPECTED_P)}, got {(len(source), len(x_columns))}")
    if source.subject_id.duplicated().any() or not np.isfinite(source.to_numpy()).all():
        raise ValueError("Prepared cohort has duplicate IDs or nonfinite values")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.csv"
    # Store the calibration inputs with 12 significant digits.
    source.to_csv(source_path, index=False, float_format="%.12g")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), n=len(source), p=len(x_columns),
                    x_columns=x_columns, source_sha256=sha256(source_path), source_files=records,
                    treatment="DR1TFIBE: day-1 dietary fiber, g/day; normalized a=dose/20",
                    outcome="LBDLDL: calculated LDL cholesterol, mg/dL",
                    fasting="WTSAF2YR>0 identifies eligibility; weights are not used for fitting or averaging",
                    target="Fixed unweighted empirical covariate distribution, not the US population",
                    interpretation="Public real measurements calibrate a synthetic law; no real causal effect is identified")
    save_json(output_dir / "source_manifest.json", manifest)
    print(f"Prepared LDL cohort: n={len(source)}, p={len(x_columns)}", flush=True)
    return manifest
