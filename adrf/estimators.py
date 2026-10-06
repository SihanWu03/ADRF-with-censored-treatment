"""GPU cross-fitted MR equations and complete-case T&W adaptation.

The Fubini form implements manuscript sections 3--4. Integration uses linear
finite-element weights on a dense uniform grid, including exact partial-cell
tail weights. Censored outcomes never enter a regression residual.
"""
from dataclasses import dataclass
import numpy as np
import torch


DTYPE = torch.float64


def tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=DTYPE)


def kernel(z):
    return torch.where(z.abs() < 1, .75*(1-z*z), 0.)


def interp_rows(values, t, grid):
    """Paired row interpolation, or interpolation of one common curve."""
    pos = (t.clamp(grid[0],grid[-1])-grid[0])/(grid[1]-grid[0])
    index = pos.floor().long().clamp(0,len(grid)-2)
    frac = pos-index
    if values.ndim == 1:
        return values[index]*(1-frac)+values[index+1]*frac
    row = torch.arange(len(values),device=values.device)
    return values[row,index]*(1-frac)+values[row,index+1]*frac


def interpolation_scatter(t, values, grid):
    """Adjoint of interpolation: sum_i values_i * interpolated curve(t_i)."""
    pos = (t.clamp(grid[0],grid[-1])-grid[0])/(grid[1]-grid[0])
    idx = pos.floor().long().clamp(0,len(grid)-2)
    f = (pos-idx)[:,None]
    out = values.new_zeros((len(grid), values.shape[1]))
    out.index_add_(0,idx,values*(1-f))
    out.index_add_(0,idx+1,values*f)
    return out


def trapezoid_weights(grid):
    weights = torch.full_like(grid,grid[1]-grid[0])
    weights[0] *= .5
    weights[-1] *= .5
    return weights


def tail_weights(u, grid):
    """Integral from u to grid[-1] of each piecewise-linear nodal basis."""
    dt = grid[1]-grid[0]
    z = (u[:,None].clamp(grid[0],grid[-1])-grid[None,:])/dt
    def primitive(v):
        return torch.where(v < -1,0.,torch.where(v<0,.5*(v+1)**2,
                           torch.where(v<1,1-.5*(1-v)**2,1.)))
    lower = primitive((grid[0]-grid)/dt)
    prefix = dt*(primitive(z)-lower[None,:])
    return (trapezoid_weights(grid)[None,:]-prefix).clamp_min(0)


@torch.no_grad()
def predict_grid(model, method, x, grid, chunk=64):
    """Evaluate predictions in chunks to limit temporary memory use."""
    output = []
    for start in range(0,len(x),chunk):
        output.append(getattr(model,method)(x[start:start+chunk,None],grid[None,:]))
    return torch.cat(output,dim=0)


@dataclass
class FoldCache:
    evaluation: np.ndarray
    training: np.ndarray
    u: torch.Tensor
    delta: torch.Tensor
    ipcw: torch.Tensor
    xi: torch.Tensor
    residual_over_density: torch.Tensor
    w: torch.Tensor
    mu_reference: torch.Tensor
    f_reference: torch.Tensor
    m: torch.Tensor
    p: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def cache_fold(data, evaluation, training, model, grid, complete_case=False, cfg=None):
    device=grid.device
    x=tensor(data.x[evaluation],device)
    u=tensor(data.u[evaluation],device)
    delta=tensor(data.delta[evaluation],device)
    xref=tensor(data.x[training],device)
    mu_ref=predict_grid(model,'mu',xref,grid)
    f_ref=predict_grid(model,'density',xref,grid)
    m=mu_ref.mean(0)
    p=f_ref.mean(0)
    fu=model.density(x,u)
    scu=torch.ones_like(u) if complete_case else model.survival_c(x,u)
    if not torch.isfinite(mu_ref).all() or (fu<=0).any() or (scu<=0).any():
        raise FloatingPointError('Nonfinite nuisance or nonpositive density/survival')
    ipcw=delta/scu
    residual=torch.zeros_like(u)
    event=delta.bool()
    # Do not even evaluate Y or mu at censored outcomes.
    residual[event]=(tensor(data.y[evaluation][data.delta[evaluation]],device)
                     -model.mu(x[event],u[event]))/fu[event]
    xi=interp_rows(m,u,grid)+residual*interp_rows(p,u,grid)
    if complete_case:
        w=u.new_zeros((len(u),len(grid)))
    else:
        fg=predict_grid(model,'density',x,grid)
        sa=predict_grid(model,'survival_a',x,grid)
        sc=predict_grid(model,'survival_c',x,grid)
        hazard=predict_grid(model,'hazard_c',x,grid)
        if (sa<=0).any() or (sc<=0).any():
            raise FloatingPointError('Integration window reaches a zero-survival region')
        integrand=hazard/(sa*sc)
        increments=.5*(integrand[:,1:]+integrand[:,:-1])*(grid[1]-grid[0])
        J=torch.cat([u.new_zeros((len(u),1)),torch.cumsum(increments,1)],1)
        ju=interp_rows(J,u,grid)
        jmin=torch.where(grid[None,:] <= u[:,None], J, ju[:,None])
        jump=torch.zeros_like(u)
        censored=(~event)&(u<grid[-1])
        jump[censored]=1/(scu[censored]*model.survival_a(x[censored],u[censored]))
        w=fg*(jump[:,None]*tail_weights(u,grid)-jmin*trapezoid_weights(grid)[None,:])
    return FoldCache(evaluation,training,u,delta,ipcw,xi,residual,w,mu_ref,f_ref,m,p,{})


def split_folds(n, folds, seed):
    rng=np.random.default_rng(seed)
    groups=np.array_split(rng.permutation(n),folds)
    indices=np.arange(n)
    return [(ev,indices[~np.isin(indices,ev)]) for ev in groups]


@torch.no_grad()
def moment_equations(caches, grid, targets, width, n, order):
    """Return per-person moments and all empirical-marginal derivatives."""
    g=len(targets)
    powers=2*order+1
    num=grid.new_zeros((n,g,order+1))
    moments=grid.new_zeros((n,g,max(powers,4)))
    extra=grid.new_zeros((n,g,order+1))
    z=(grid[:,None]-targets[None,:])/width
    kg=kernel(z)/width
    Q=torch.stack([kg*z**j for j in range(max(powers,4))],-1)
    Qflat=Q.reshape(len(grid),-1)
    Qnum=Q[:,:,:order+1].reshape(len(grid),-1)
    for fold in caches:
        zu=(fold.u[:,None]-targets[None,:])/width
        ku=kernel(zu)/width
        H=fold.ipcw[:,None,None]*torch.stack([ku*zu**j for j in range(max(powers,4))],-1)
        hm=H[:,:,:order+1].reshape(len(fold.u),-1)
        ev=torch.as_tensor(fold.evaluation,device=grid.device)
        tr=torch.as_tensor(fold.training,device=grid.device)
        moments[ev]=H+(fold.w@Qflat).reshape(len(ev),g,-1)
        num[ev]=H[:,:,:order+1]*fold.xi[:,None,None]+(fold.w@(Qnum*fold.m[:,None])).reshape(len(ev),g,-1)
        # Frozen conditional fits, exact derivative of empirical m and p in
        # this discretized estimator. Membership/scale account for fold reuse.
        coeff_m=interpolation_scatter(fold.u,hm,grid)+fold.w.sum(0)[:,None]*Qnum
        coeff_p=interpolation_scatter(fold.u,hm*fold.residual_over_density[:,None],grid)
        delta_m=(fold.mu_reference-fold.m[None,:])@coeff_m
        delta_p=(fold.f_reference-fold.p[None,:])@coeff_p
        extra[tr]+=(delta_m+delta_p).reshape(len(tr),g,order+1)/len(tr)
    return num,moments,extra


def design_from_moments(moments,order):
    return torch.stack([torch.stack([moments[:,:,i+j] for j in range(order+1)],-1)
                        for i in range(order+1)],-2)


def invert_design(d):
    eigen=torch.linalg.eigvalsh(d)
    condition=torch.linalg.cond(d)
    if not torch.isfinite(d).all() or (eigen[:,0]<=1e-9).any() or (condition>1e8).any():
        raise FloatingPointError(f'Invalid local design: min eigenvalue={float(eigen.min()):.3g}, max condition={float(condition.max()):.3g}')
    # No ridge, pseudoinverse, clipping, or nan_to_num that could mask failure.
    return torch.linalg.inv(d),float(condition.max())


@torch.no_grad()
def solve(caches,grid,targets,h,b,n):
    nh,mh,ah=moment_equations(caches,grid,targets,h,n,1)
    nb,mb,ab=moment_equations(caches,grid,targets,b,n,2)
    dhi=design_from_moments(mh,1)
    dbi=design_from_moments(mb,2)
    ih,_=invert_design(dhi.mean(0))
    ib,_=invert_design(dbi.mean(0))
    beta=torch.einsum('gij,gj->gi',ih,nh.mean(0))
    gamma=torch.einsum('gij,gj->gi',ib,nb.mean(0))
    bias_i=mh[:,:,2:4]
    d=torch.einsum('gij,gj->gi',ih,bias_i.mean(0))
    c=d[:,0]
    sbeta=torch.einsum('gij,ngj->ngi',ih,nh-torch.einsum('ngij,gj->ngi',dhi,beta)+ah)
    sgamma=torch.einsum('gij,ngj->ngi',ib,nb-torch.einsum('ngij,gj->ngi',dbi,gamma)+ab)
    sc=torch.einsum('gj,ngj->ng',ih[:,0,:],bias_i-torch.einsum('ngij,gj->ngi',dhi,d))
    ratio=(h/b)**2
    phi=sbeta[:,:,0]
    phidb=phi-ratio*(c[None,:]*sgamma[:,:,2]+gamma[None,:,2]*sc)
    phi=phi-phi.mean(0)
    phidb=phidb-phidb.mean(0)
    estimate=beta[:,0]
    estimate_db=estimate-ratio*gamma[:,2]*c
    return {'MR':(estimate,phi),'MRDB':(estimate_db,phidb)},{}


@torch.no_grad()
def inference(estimate,phi,draws,alpha,seed,confidence_bands=True):
    n=len(phi)
    norm=torch.linalg.vector_norm(phi,dim=0)
    if not torch.isfinite(phi).all() or (norm<=0).any():
        raise FloatingPointError('Invalid influence function / zero variance')
    se=norm/n
    z=torch.distributions.Normal(0.,1.).icdf(torch.tensor(1-alpha/2,dtype=DTYPE)).item()
    result=dict(estimate=estimate.cpu().numpy(),se=se.cpu().numpy(),
                ci_low=(estimate-z*se).cpu().numpy(),ci_high=(estimate+z*se).cpu().numpy())
    if not confidence_bands:
        return result
    if draws < 1:
        raise ValueError('Positive multiplier draws required for confidence bands')
    gen=torch.Generator(device=phi.device).manual_seed(seed)
    maxima=[]
    for start in range(0,draws,128):
        e=torch.randn((min(128,draws-start),n),dtype=phi.dtype,device=phi.device,generator=gen)
        maxima.append(((e@phi)/norm[None,:]).abs().amax(1))
    critical=torch.quantile(torch.cat(maxima),1-alpha,interpolation='higher')
    result.update(band_low=(estimate-critical*se).cpu().numpy(),
                  band_high=(estimate+critical*se).cpu().numpy(),critical=float(critical))
    return result
