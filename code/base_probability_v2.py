import json
import numpy as np
from set_selector import set_inputs,neural_scores,probability
from submission import prepare
from arithmetic_features import bank_features

def base_probability(bank,arrays,checkpoint):
 cfg=json.loads((checkpoint/'config.json').read_text());a,same,valid=set_inputs(bank,arrays);ps=[]
 for name in cfg['members']:
  with np.load(checkpoint/name) as w:ps.append(probability(neural_scores(a,same,w,cfg['architecture']),valid))
 x,valid,votes=prepare(bank,arrays);f=np.concatenate([x,votes[...,None]/4,valid[...,None]],2)
 with np.load(checkpoint/'baseline.npz') as w:s=f@w['weights']
 s-=cfg['arithmetic_penalty']*bank_features(bank)[1][:,:,0]
 return (1-cfg['neural_fraction'])*probability(s,valid)+cfg['neural_fraction']*np.mean(ps,0),valid
