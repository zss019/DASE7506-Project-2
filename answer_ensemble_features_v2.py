"""Exact shared preprocessing for the four numeric relationship feature views."""
import re,hashlib
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from common import parse_answer
from answer_structure_v2 import flags,D
from target_alignment_features import target_clause

def matrices(bank,modes):
 outputs={mode:([],[],[]) for mode in modes};cache={}
 for i,row in enumerate(bank):
  fs=flags(row['question'],[parse_answer(c['text']) for c in row['candidates']]);terms={}
  for name,text in [('full',row['question']),('target',target_clause(row['question']))]:
   words=re.findall('[a-z]+',text.lower());terms[name]=sorted(set(words+[' '.join(words[k:k+2]) for k in range(len(words)-1)]))
  for mode in modes:
   rr,cc,vv=outputs[mode]
   for j,ff in enumerate(fs):
    features=ff
    if mode=='no_formula':features=[f for f in ff if not f.startswith('reachable:')]
    elif mode=='formula_only':features=[f for f in ff if f.startswith('reachable:')]
    words=terms['target' if mode=='target' else 'full'];keys=features+([a+'|'+b for a in features for b in words] if mode!='no_text' else []);counts=Counter()
    for key in keys:
     if key not in cache:
      h=int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),'little');cache[key]=(h%D,1 if h>>63 else -1)
     col,sign=cache[key];counts[col]+=sign
    norm=np.sqrt(sum(v*v for v in counts.values()))
    for col,v in counts.items():
     if v:rr.append(4*i+j);cc.append(col);vv.append(v/max(norm,1))
 return {m:csr_matrix((v,(r,c)),shape=(4*len(bank),D),dtype=float) for m,(r,c,v) in outputs.items()}
