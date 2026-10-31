"""Exact training primitives extracted from the recorded experiment implementation."""

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

def fit_linear(a,y,penalty):
    xt=torch.tensor(a-a.mean(axis=1,keepdims=True),dtype=torch.float64)
    target=torch.tensor(y/y.sum(axis=1,keepdims=True),dtype=torch.float64)
    w=torch.zeros(a.shape[-1],dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([w],max_iter=300,line_search_fn='strong_wolfe',tolerance_grad=1e-6,tolerance_change=1e-9)
    def closure():
        optimizer.zero_grad();loss=-(target*(xt@w).log_softmax(dim=1)).sum()+penalty*w.square().sum()
        loss.backward();return loss
    optimizer.step(closure)
    return w.detach().numpy()

def stratified_folds(votes, seed, k=3):
    rng=np.random.default_rng(seed); folds=[[] for _ in range(k)]
    for value in np.unique(votes.max(axis=1)):
        ix=np.flatnonzero(votes.max(axis=1)==value); rng.shuffle(ix)
        for j,part in enumerate(np.array_split(ix,k)):folds[j].extend(part.tolist())
    return [np.array(sorted(f),dtype=int) for f in folds]

class Ranker(torch.nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden'];self.architecture=c['architecture']
        self.fc1=torch.nn.Linear(1026,h);self.fc2=torch.nn.Linear(4*h+2 if self.architecture=='set' else h,64);self.fc3=torch.nn.Linear(64,1);self.drop=torch.nn.Dropout(c['dropout'])
    def forward(self,z,same,raw):
        h=self.fc1(z).relu()
        if self.architecture=='set':
            group=same@h;context=h.mean(dim=1,keepdim=True).expand_as(h)
            h=torch.cat([h,context,group,h*group,raw[:,:,-2:]],dim=-1)
        h=self.drop(h);h=self.drop(self.fc2(h).relu())
        return self.fc3(h).squeeze(-1)

def loss_fn(s,y,objective):
    if objective=='uniform':return -((y/y.sum(dim=1,keepdim=True))*s.log_softmax(dim=1)).sum(dim=1).mean()
    if objective=='marginal':return (s.logsumexp(dim=1)-s.masked_fill(y==0,-torch.inf).logsumexp(dim=1)).mean()
    mask=y[:,:,None]*(1-y[:,None,:]);delta=s[:,:,None]-s[:,None,:]
    return ((torch.nn.functional.softplus(-delta)*mask).sum(dim=(1,2))/mask.sum(dim=(1,2)).clamp_min(1)).mean()

def train_member(a,same,y,mean,scale,config,seed,epochs):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    xt=torch.tensor((a-mean)/scale,device='cuda',dtype=torch.float32)
    st=torch.tensor(same,device='cuda',dtype=torch.float32)
    raw=torch.tensor(a,device='cuda',dtype=torch.float32)
    yt=torch.tensor(y,device='cuda',dtype=torch.float32)
    model=Ranker(config).cuda();optim=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
    for epoch in range(epochs):
        model.train()
        for ix in torch.randperm(len(xt),device='cuda').split(256):
            score=model(xt[ix],st[ix],raw[ix]);loss=loss_fn(score,yt[ix],config['objective'])
            optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
    state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
    return dict(mean=mean,scale=scale,w1=state['fc1.weight'].T,b1=state['fc1.bias'],w2=state['fc2.weight'].T,b2=state['fc2.bias'],w3=state['fc3.weight'].T,b3=state['fc3.bias'])

def fit_residual(a,y,base,penalty):
 mask=y[:,:,None]*(1-y[:,None,:]);mask/=np.maximum(mask.sum((1,2),keepdims=True),1);offset=np.log(np.clip(base,1e-9,1))
 def f(w):
  score=(a@w).reshape(-1,4)+offset;delta=score[:,:,None]-score[:,None,:];g=-expit(-delta)*mask;grad=(g.sum(2)-g.sum(1)).ravel();return float((np.logaddexp(0,-delta)*mask).sum()+penalty*(w@w)),a.T@grad+2*penalty*w
 result=minimize(f,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9});assert result.success,result.message
 return result.x
