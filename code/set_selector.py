"""NumPy-only CPU inference for learned candidate-set comparisons."""
import json
import numpy as np
from common import parse_answer
from submission import prepare,choose
from arithmetic_features import bank_features


def set_inputs(bank,arrays):
    x,valid,votes=prepare(bank,arrays)
    features=np.concatenate([x,votes[...,None]/4,valid[...,None].astype(float)],axis=-1).astype(np.float32)
    same=[]
    for q in bank:
        aa=[parse_answer(c['text']) for c in q['candidates']]
        same.append([[a is not None and a==b for b in aa] for a in aa])
    same=np.asarray(same,dtype=np.float32)
    same/=np.maximum(same.sum(axis=-1,keepdims=True),1.)
    return features,same,valid

def neural_scores(features,same,w,architecture):
    z=(features-w['mean'])/w['scale']
    h=np.maximum(z@w['w1']+w['b1'],0)
    if architecture=='set':
        global_mean=np.broadcast_to(h.mean(axis=1,keepdims=True),h.shape)
        group=same@h
        h=np.concatenate([h,global_mean,group,h*group,features[:,:,-2:]],axis=-1)
    h=np.maximum(h@w['w2']+w['b2'],0)
    return (h@w['w3']+w['b3']).squeeze(-1)

def probability(scores,valid):
    scores=np.where(valid | ~valid.any(axis=1,keepdims=True),scores,-np.inf)
    scores=scores-scores.max(axis=1,keepdims=True)
    exp=np.exp(scores)
    return exp/exp.sum(axis=1,keepdims=True)

def predict(bank,arrays,checkpoint):
    if not bank:return []
    config=json.loads((checkpoint/'config.json').read_text())
    features,same,valid=set_inputs(bank,arrays)
    probabilities=[]
    for member in config['members']:
        with np.load(checkpoint/member,allow_pickle=False) as w:
            probabilities.append(probability(neural_scores(features,same,w,config['architecture']),valid))
    neural=np.mean(probabilities,axis=0)
    alpha=config['neural_fraction']
    if alpha<1:
        x,v,votes=prepare(bank,arrays)
        a=np.concatenate([x,votes[...,None]/4,v[...,None].astype(float)],axis=-1)
        _,quality=bank_features(bank)
        with np.load(checkpoint/'baseline.npz',allow_pickle=False) as w:base=a@w['weights']
        base-=config['arithmetic_penalty']*quality[:,:,0]
        neural=alpha*neural+(1-alpha)*probability(base,valid)
    return choose(neural,valid)
