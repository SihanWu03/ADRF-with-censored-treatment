#!/usr/bin/env python3
"""Portable resumable simulation entry point (Windows/Linux, CUDA/CPU)."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

from adrf.config import read_config,validate,fingerprint,seed_for


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)



def is_fatal_cuda_error(exc):
    """Classify a poisoned CUDA context using CPU-side exception text only.

    Ordinary local-design failures and recoverable CUDA OOM are not fatal.
    Follow exception chaining because an outer fit error may wrap CUDA.
    """
    signatures = (
        'illegal memory access', 'illegal address', 'device-side assert',
        'unspecified launch failure', 'launch failure',
        'launch timed out', 'launch timeout', 'misaligned address',
        'illegal instruction', 'hardware stack error',
        'context is destroyed', 'context was destroyed',
        'context has been destroyed', 'context destroyed',
        'context is lost', 'context was lost', 'context has been lost', 'context lost',
    )
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower().replace('_', ' ')
        if 'cuda' in message and any(marker in message for marker in signatures):
            return True
        current = current.__cause__ or current.__context__
    return False


def archive_previous_failure(failure):
    """Preserve the exact previous JSON bytes before replacing an attempt.

    The content hash deduplicates the same already-archived record. History is
    nested beneath failures/history, outside the reporter's current-file glob.
    """
    if not failure.exists():
        return None
    payload = failure.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    archived = failure.parent / 'history' / failure.stem / (digest + '.json')
    archived.parent.mkdir(parents=True, exist_ok=True)
    if not archived.exists():
        temporary = archived.with_name(archived.name + f'.{os.getpid()}.tmp')
        temporary.write_bytes(payload)
        try:
            os.link(temporary, archived)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    return archived


def write_failure(failure, metadata, exc, traceback_text, device):
    archive_previous_failure(failure)
    fatal = is_fatal_cuda_error(exc)
    atomic_json(failure, dict(metadata=metadata, error=str(exc),
                             traceback=traceback_text,
                             failed_utc=datetime.now(timezone.utc).isoformat(),
                             process_id=os.getpid(), device=device,
                             fatal_cuda=fatal))
    return fatal


def unresolved_failure_count(out):
    # A successful retry resolves the current failure without deleting its
    # historical evidence. Nested attempt history never counts as a failure.
    return sum(not (out / 'jobs' / path.name).exists()
               for path in (out / 'failures').glob('*.json'))


def validate_saved_job(value,meta,cfg):
    """Validate a saved result before treating a repetition as complete."""
    import numpy as np
    import pandas as pd
    from adrf.reporting import _validate_layout
    if not isinstance(value,dict) or value.get('metadata')!=meta:
        raise ValueError('Saved job identity differs from its expected seed/setting')
    if not isinstance(value.get('diagnostics'),dict):
        raise ValueError('Saved job is missing its diagnostics')
    for name in ('points','curves'):
        rows=value.get(name)
        if not isinstance(rows,list) or not rows or not all(isinstance(row,dict) for row in rows):
            raise ValueError(f'Saved job has no complete {name} records')
    points,curves=pd.DataFrame(value['points']),pd.DataFrame(value['curves'])
    _validate_layout(points,curves,meta,cfg)
    if curves.duplicated(['bandwidth','method']).any():
        raise ValueError('Saved job repeats a curve record')
    for kind in ('points','curves'):
        for row in value[kind]:
            pi=row['method']=='PI'
            fields=['a','truth','estimate'] if kind=='points' else []
            if not pi:
                fields+=['h','b']
                if kind=='points':
                    fields+=['se','ci_low','ci_high']
            if cfg['confidence_bands'] and row['method']=='MRDB':
                fields+=['band_low','band_high','critical'] if kind=='points' else ['mean_band_width']
            for name in fields:
                number=row.get(name)
                if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number):
                    raise ValueError(f'Saved {row["method"]} {kind} record has invalid {name}')
            if kind=='points' and not pi and not isinstance(row.get('covered'),bool):
                raise ValueError('Saved interval record has no coverage indicator')
            if not isinstance(row.get('n_used'),int) or not 0<row['n_used']<=meta['n']:
                raise ValueError('Saved record has an invalid analysis sample size')
    if not np.isfinite(float(value.get('elapsed_seconds',float('nan')))):
        raise ValueError('Saved job has an invalid elapsed time')


def archive_invalid_job(path):
    """Move exact corrupt bytes outside jobs/*.json before an explicit retry."""
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    archived=path.parent/'history'/path.stem/(digest+'.json')
    archived.parent.mkdir(parents=True,exist_ok=True)
    os.replace(path,archived)
    return archived


def source_fingerprint():
    root=Path(__file__).resolve().parent
    files=[root/'run.py']+sorted((root/'adrf').glob('*.py'))
    digest=hashlib.sha256()
    for path in files:
        if path.name=='reporting.py':
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def initialize_output(out,cfg):
    import torch
    import numpy as np
    import scipy
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(config_hash=fingerprint(cfg),source_hash=source_fingerprint())
    runtime=dict(python_version=platform.python_version(),numpy=np.__version__,
                 scipy=scipy.__version__,torch=torch.__version__,cuda=torch.version.cuda,
                 device_type=torch.device(cfg['device']).type)
    manifest=out/'manifest.json'
    if not manifest.exists():
        proposed=dict(identity,created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
                        platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(torch.device(cfg['device'])) if cfg['device'].startswith('cuda') and torch.cuda.is_available() else None,
                        config=cfg,runtime=runtime)
        candidate=out/f'manifest.{os.getpid()}.candidate'
        atomic_json(candidate,proposed)
        try:
            # Atomic no-overwrite publication: two GPU shards cannot install
            # different configurations in the same newly created output.
            os.link(candidate,manifest)
        except FileExistsError:
            pass
        finally:
            candidate.unlink(missing_ok=True)
    previous=json.loads(manifest.read_text(encoding='utf-8'))
    if any(previous.get(k)!=v for k,v in identity.items()):
        raise RuntimeError('Output belongs to a different configuration/code revision. Use a new --out directory; existing results are preserved.')
    if previous.get('runtime')!=runtime:
        raise RuntimeError('Numerical environment or CPU/CUDA device category differs from this run. Use a new --out directory; existing results are preserved. GPU index changes alone are allowed.')
    atomic_json(out/'run_config.json',cfg)
    root=Path(__file__).resolve().parent
    snapshot=out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in [root/'run.py']+sorted((root/'adrf').glob('*.py')):
        target=snapshot/path.relative_to(root)
        content=path.read_bytes()
        if not target.exists() or target.read_bytes()!=content:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            temporary.write_bytes(content)
            os.replace(temporary,target)
    return identity


def jobs(cfg):
    for setting in cfg['settings']:
        for n in cfg['sample_sizes']:
            for rep in range(cfg['repetitions']):
                for misspec in cfg['misspecifications']:
                    meta=dict(experiment=cfg['experiment'],setting=setting,n=n,rep=rep,
                              seed=seed_for(cfg,setting,n,rep),misspec=misspec)
                    yield f'{setting}_n{n}_{misspec}_seed{meta["seed"]}',meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/main.json')
    parser.add_argument('--out',required=True)
    parser.add_argument('--device')
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--no-report',action='store_true')
    args=parser.parse_args()
    cfg=read_config(args.config)
    if args.device is not None:
        cfg['device']=args.device
    validate(cfg)
    if args.shards<1 or not 0<=args.shard_index<args.shards:
        parser.error('Need 0 <= shard-index < shards')
    if not args.no_report and args.shards>1:
        parser.error('Automatic reports require separate output directories; for shared-output shards use --no-report and report.py')
    selected=[job for j,job in enumerate(jobs(cfg)) if j%args.shards==args.shard_index]
    # Import the numerical implementation after validating the run configuration.
    from adrf.simulation import run_job
    out=Path(args.out).resolve()
    initialize_output(out,cfg)
    successes=failures=skipped=0
    for index,(jobid,meta) in enumerate(selected):
        result=out/'jobs'/f'{jobid}.json'
        failure=out/'failures'/f'{jobid}.json'
        valid_saved=False
        corrupt_saved=False
        if result.exists():
            try:
                validate_saved_job(json.loads(result.read_text(encoding='utf-8')),meta,cfg)
                valid_saved=True
            except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
                if args.retry_failed:
                    archived=archive_invalid_job(result)
                    print(f'Archived invalid saved job: {archived}; retrying the same seed',flush=True)
                else:
                    corrupt_saved=True
                    error=ValueError(f'Invalid saved job {result}: {exc}. Use --retry-failed to archive it and rerun the same seed, or use a new --out directory.')
                    write_failure(failure,meta,error,traceback.format_exc(),cfg['device'])
                    failures+=1
                    print(str(error),file=sys.stderr,flush=True)
        if valid_saved or (failure.exists() and not args.retry_failed and not corrupt_saved):
            skipped+=1
        elif corrupt_saved:
            pass
        else:
            try:
                value=run_job(cfg,meta)
                atomic_json(result,value)
                # Retain the previous failure JSON after a successful retry.
                successes+=1
                print(f'[{index+1}/{len(selected)}] OK {jobid} {value["elapsed_seconds"]:.1f}s',flush=True)
            except Exception as exc:
                fatal=write_failure(failure,meta,exc,traceback.format_exc(),cfg['device'])
                failures+=1
                print(f'[{index+1}/{len(selected)}] FAILED {jobid}: {exc}',file=sys.stderr,flush=True)
                if fatal:
                    remaining=len(selected)-index-1
                    print(f'Fatal CUDA context error; stopping this process after {jobid}. '
                          f'{remaining} later selected jobs were not attempted in this invocation. '
                          'Restart in a fresh process with --retry-failed; the seed is unchanged.',
                          file=sys.stderr,flush=True)
                    return 2
    if not args.no_report:
        from adrf.reporting import aggregate
        aggregate(out)
    print(f'Done: success={successes}, failed={failures}, skipped={skipped}; {out}',flush=True)
    unresolved=unresolved_failure_count(out)
    return 1 if failures or unresolved else 0


if __name__=='__main__':
    sys.exit(main())
