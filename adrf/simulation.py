"""One seed fits each nuisance once per fold, shared across both bandwidths."""
import time
import numpy as np
import torch
from .config import bandwidth_scale
from .dgp import generate,theta
from .estimators import tensor,split_folds,cache_fold,solve,inference,predict_grid
from .nuisance import fit_nuisance


def fit_caches(data,cfg,seed,misspec,grid,complete_case=False,pi_targets=None):
    caches=[]
    device=grid.device
    pi_total=torch.zeros_like(pi_targets) if pi_targets is not None else None
    membership=np.zeros(len(data),dtype=int) if pi_targets is not None else None
    for k,(ev,tr) in enumerate(split_folds(len(data),cfg['folds'],seed+1009)):
        nc=dict(cfg['nuisance'],complete_case=complete_case)
        model=fit_nuisance(tensor(data.x[tr],device),tensor(data.u[tr],device),
                           tensor(data.delta[tr],device),tensor(data.y[tr],device),
                           kind='neural' if cfg['experiment']=='main' else 'parametric',
                           misspec=misspec,T=cfg['T'],seed=seed+7919*(k+1),device=str(device),config=nc)
        cached=cache_fold(data,ev,tr,model,grid,complete_case,cfg)
        caches.append(cached)
        if pi_targets is not None:
            # Reuse exactly the outcome fit used by MR. PI averages fold-out
            # predictions over all individuals, including censored records;
            # the training-fold marginal m in cached.m is a different quantity.
            prediction=predict_grid(model,'mu',tensor(data.x[ev],device),pi_targets)
            if prediction.shape!=(len(ev),len(pi_targets)) or not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError('Invalid or nonfinite PI predictions')
            pi_total+=prediction.sum(0)
            membership[ev]+=1
            del prediction
        del model
    if membership is not None and not np.all(membership==1):
        raise ValueError('Cross-fitting must predict every individual exactly once')
    pi_estimate=None if pi_total is None else (pi_total/len(data)).cpu().numpy()
    return caches,pi_estimate


def prepare_job(cfg,meta):
    """Fit once and cache equations for both prespecified bandwidth rules."""
    start=time.perf_counter()
    n,seed,setting,misspec=meta['n'],meta['seed'],meta['setting'],meta['misspec']
    device=torch.device(cfg['device'])
    torch.set_num_threads(cfg['torch_threads'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly use --device cpu for a CPU run')
    data,diagnostics=generate(n,seed,setting,cfg['T'],cfg['noise_sd'])
    largest_h=max(bandwidth_scale(cfg,name)*n**(-exponent)
                  for name,exponent in cfg['bandwidths'].items())
    stop=cfg['a_max']+largest_h*max(1,cfg['b_over_h'])
    if stop >= cfg['T']:
        raise ValueError('Integration endpoint must remain below T')
    grid=torch.linspace(0,stop,cfg['integration_grid'],device=device,dtype=torch.float64)
    targets=torch.linspace(cfg['a_min'],cfg['a_max'],cfg['grid_size'],device=device,dtype=torch.float64)
    a=targets.cpu().numpy()
    truth=theta(a,setting)
    caches,pi_estimate=fit_caches(data,cfg,seed,misspec,grid,
                                pi_targets=targets if cfg['experiment']=='main' else None)
    groups=[('MR',caches,n)]
    if cfg['experiment']=='main' and cfg['include_conventional']:
        cc=data.subset(data.delta)
        cc_caches,_=fit_caches(cc,cfg,seed+104729,'none',grid,True)
        groups.append(('Conventional',cc_caches,len(cc)))
    return dict(metadata=dict(meta),grid=grid,targets=targets,truth=truth,groups=groups,
                pi_estimate=pi_estimate,
                diagnostics=diagnostics,integration_stop=stop,
                elapsed_seconds=time.perf_counter()-start)


def evaluate_prepared(cfg,meta,prepared):
    """Evaluate both prespecified bandwidth rules with the same fitted caches."""
    start=time.perf_counter()
    if prepared['metadata']!=meta:
        raise ValueError('Prepared data belong to a different simulation seed or setting')
    n,seed=meta['n'],meta['seed']
    grid,targets=prepared['grid'],prepared['targets']
    truth,groups=prepared['truth'],prepared['groups']
    device=grid.device
    a=targets.cpu().numpy()
    largest_h=max(bandwidth_scale(cfg,name)*n**(-exponent)
                  for name,exponent in cfg['bandwidths'].items())
    if cfg['a_max']+largest_h*max(1,cfg['b_over_h']) > prepared['integration_stop']+1e-12:
        raise ValueError('Prepared integration grid does not contain all kernel windows')
    diagnostics=dict(prepared['diagnostics'])
    points=[]
    curves=[]
    for bw_index,(bandwidth,exponent) in enumerate(cfg['bandwidths'].items()):
        h=bandwidth_scale(cfg,bandwidth)*n**(-exponent)
        b=cfg['b_over_h']*h
        for family,cache,n_used in groups:
            estimates,_=solve(cache,grid,targets,h,b,n_used)
            for method,(est,phi) in estimates.items():
                method=method.replace('MR','Conventional') if family=='Conventional' else method
                # Only MRDB has simultaneous inference in the manuscript.
                # All target points share multipliers; MR and both complete-case
                # comparators always receive pointwise intervals only.
                bands=cfg['confidence_bands'] and method=='MRDB'
                infer=inference(est,phi,cfg['bootstrap_draws'],cfg['alpha'],seed+65537+bw_index*997,bands)
                covered=(infer['ci_low']<=truth)&(truth<=infer['ci_high'])
                if bands:
                    bandcovered=(infer['band_low']<=truth)&(truth<=infer['band_high'])
                common=dict(bandwidth=bandwidth,h=h,b=b,method=method,n_used=n_used)
                for j,dose in enumerate(a):
                    row=dict(common,a=float(dose),truth=float(truth[j]),covered=bool(covered[j]))
                    fields=['estimate','se','ci_low','ci_high']
                    if bands:
                        row.update(band_covered_point=bool(bandcovered[j]),critical=infer['critical'])
                        fields+=['band_low','band_high']
                    for field in fields:
                        row[field]=float(infer[field][j])
                    points.append(row)
                curve=dict(common)
                if bands:
                    curve.update(band_covered=bool(bandcovered.all()),
                                 mean_band_width=float(np.mean(infer['band_high']-infer['band_low'])))
                curves.append(curve)
    if prepared['pi_estimate'] is not None:
        # PI has no second-stage bandwidth and no evaluated confidence interval.
        # Store it once; presentation code may repeat this same curve beside
        # each bandwidth rule without fitting or generating another estimate.
        estimate=prepared['pi_estimate']
        common=dict(bandwidth='none',h=None,b=None,method='PI',n_used=n)
        for j,dose in enumerate(a):
            points.append(dict(common,a=float(dose),truth=float(truth[j]),
                               estimate=float(estimate[j]),se=None,ci_low=None,
                               ci_high=None,covered=None))
        curves.append(dict(common))
        diagnostics['pi_averaging']='out_of_fold_predictions_over_all_individuals'
        diagnostics['pi_outcome_fit']='same_fitted_MR_outcome_model'
        diagnostics['pi_inference']='not_evaluated'
    if device.type=='cuda':
        torch.cuda.synchronize(device)
    diagnostics['inference']='full_equation_sandwich_with_empirical_mu_and_p_marginals'
    diagnostics['confidence_bands']=cfg['confidence_bands']
    diagnostics['confidence_band_methods']=['MRDB'] if cfg['confidence_bands'] else []
    diagnostics['integration_stop']=prepared['integration_stop']
    return dict(metadata=meta,diagnostics=diagnostics,points=points,curves=curves,
                elapsed_seconds=prepared['elapsed_seconds']+time.perf_counter()-start)


def run_job(cfg,meta):
    return evaluate_prepared(cfg,meta,prepare_job(cfg,meta))
