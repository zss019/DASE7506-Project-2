import json
import numpy as np
from answer_ensemble_features_v2 import matrices
from base_probability_v2 import base_probability
from submission import choose

def predict(bank,arrays,checkpoint):
 if not bank:return []
 cfg=json.loads((checkpoint/'config.json').read_text());p,valid=base_probability(bank,arrays,checkpoint/'base');s=np.log(np.maximum(p,1e-9));fs=matrices(bank,cfg['fractions'])
 for mode,weight in cfg['fractions'].items():
  with np.load(checkpoint/f'{mode}.npz') as w:s+=weight*(fs[mode]@w['weights']).reshape(-1,4)
 return choose(s,valid)
