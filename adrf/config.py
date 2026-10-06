"""Simulation configuration, validation, and stable paper seed namespaces."""
import copy
import hashlib
import json
import math

DEFAULT = dict(
    experiment='main', settings=['linear', 'nonlinear'], sample_sizes=[2000,4000,8000],
    misspecifications=['none'], repetitions=1000, seed_start=200000,
    bandwidths={'undersmooth': 0.25, 'regular': 0.2},
    c_h={'undersmooth':1.0,'regular':2.0}, b_over_h=1.25,
    T=3.0, noise_sd=1.0, a_min=0.8, a_max=1.8, grid_size=201,
    folds=2, integration_grid=1601, bootstrap_draws=999, confidence_bands=True,
    include_conventional=True,
    alpha=0.05,
    device='cuda', torch_threads=2,
    nuisance={'epochs':400, 'patience':45, 'learning_rate':0.01},
)


def bandwidth_scale(cfg, bandwidth):
    """Return the fixed positive C_h for one prespecified bandwidth rule."""
    if bandwidth not in cfg['bandwidths']:
        raise ValueError(f'Unknown bandwidth rule: {bandwidth}')
    scale = cfg['c_h']
    if not isinstance(scale,dict) or set(scale) != set(cfg['bandwidths']):
        raise ValueError('c_h mapping must match all bandwidth names exactly')
    for value in scale.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('c_h must contain finite positive numbers')
        try:
            finite_positive = math.isfinite(value) and value > 0
        except OverflowError:
            finite_positive = False
        if not finite_positive:
            raise ValueError('c_h must contain finite positive numbers')
    return float(scale[bandwidth])


def validate(cfg):
    if cfg['experiment'] not in ('main','stability'):
        raise ValueError('experiment must be main or stability')
    if cfg['experiment']=='stability' and any(m not in ('mu','a','c') for m in cfg['misspecifications']):
        raise ValueError('stability requires one misspecified nuisance: mu, a, or c')
    if cfg['experiment']=='main' and cfg['misspecifications'] != ['none']:
        raise ValueError('main must fit all nuisances flexibly')
    if cfg['folds'] < 2 or cfg['repetitions'] < 1:
        raise ValueError('Need at least two folds and one repetition')
    if not 0 < cfg['alpha'] < 1 or cfg['b_over_h'] <= 0:
        raise ValueError('Invalid alpha or bandwidth scale')
    if cfg['grid_size'] < 3 or cfg['integration_grid'] < 50:
        raise ValueError('Numerical grids are too small')
    if not isinstance(cfg['confidence_bands'],bool):
        raise ValueError('confidence_bands must be a boolean')
    if not isinstance(cfg['include_conventional'],bool):
        raise ValueError('include_conventional must be a boolean')
    if cfg['experiment']=='stability' and cfg['confidence_bands']:
        raise ValueError('The stability experiment uses pointwise inference only')
    if cfg['confidence_bands'] and cfg['bootstrap_draws'] < 99:
        raise ValueError('Use at least 99 multiplier draws (999 recommended)')
    if not cfg['settings'] or set(cfg['settings'])-{'linear','nonlinear'} or len(set(cfg['settings']))!=len(cfg['settings']):
        raise ValueError('settings must contain distinct linear/nonlinear entries')
    if not cfg['sample_sizes'] or len(set(cfg['sample_sizes']))!=len(cfg['sample_sizes']):
        raise ValueError('sample_sizes must be nonempty and distinct')
    if not 0 < cfg['a_min'] < cfg['a_max'] < cfg['T'] or cfg['noise_sd'] <= 0:
        raise ValueError('Invalid target interval, support, or outcome noise')
    if not isinstance(cfg['bandwidths'], dict) or not cfg['bandwidths']:
        raise ValueError('bandwidths must be a nonempty mapping of named rules')
    for bandwidth in cfg['bandwidths']:
        bandwidth_scale(cfg, bandwidth)
    for n in cfg['sample_sizes']:
        if n < 100 or n % cfg['folds']:
            raise ValueError('Sample sizes must be >=100 and divisible by folds')
        for bandwidth, exponent in cfg['bandwidths'].items():
            if exponent not in (0.2, 0.25):
                raise ValueError('Only the prespecified n^-1/5 and n^-1/4 rules are allowed')
            r = bandwidth_scale(cfg, bandwidth) * n**(-exponent) * max(1,cfg['b_over_h'])
            if cfg['a_min']-r <= 0 or cfg['a_max']+r >= cfg['T']:
                raise ValueError(f'Kernel window leaves (0,T) at n={n}; change a range or bandwidth')
    return cfg


def read_config(path):
    cfg = copy.deepcopy(DEFAULT)
    with open(path, encoding='utf-8') as f:
        supplied = json.load(f)
    unknown = set(supplied)-set(cfg)
    if unknown:
        raise ValueError(f'Unknown config fields {sorted(unknown)}')
    cfg.update(supplied)
    if cfg['experiment']=='stability' and 'confidence_bands' not in supplied:
        cfg['confidence_bands']=False
    return validate(cfg)


def fingerprint(cfg):
    # Runtime placement does not change an experiment's identity.
    scientific = {k:v for k,v in cfg.items() if k not in ('device','torch_threads')}
    return hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()[:16]


def seed_for(cfg, setting, n, rep):
    # Same observed data for both bandwidths and all misspecification rounds.
    key=[int(cfg['seed_start']),cfg['experiment'],setting,int(n),int(rep)]
    digest=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest()
    # Wide deterministic seed namespace; independent of task ordering, sample
    # size subsetting, sharding, and the misspecified component.
    return int.from_bytes(digest[:8],'big') & ((1<<62)-1)
