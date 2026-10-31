"""Answer/question arithmetic relationship features, never reference answers."""
import re,hashlib
from collections import Counter
from fractions import Fraction
import numpy as np
from scipy.sparse import csr_matrix
from common import parse_answer
from program_prior import question_values
D=32768

def apply(a,b,op):
 with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
  return [lambda:a+b,lambda:a-b,lambda:a*b,lambda:a/b][op]()
def flags(question,answers):
 values=np.array([float(v) for v in question_values(question)[:8]],float);out=[[] for _ in answers]
 if not len(values):return out
 # Enumerate small formula families without storing any question/answer lookup.
 a=values[:,None];b=values[None,:];families={'literal':values,'sum_all':np.array([values.sum()]),'mean_all':np.array([values.mean()])}
 for op,s in enumerate(['+','-','*','/']):
  pair=apply(a,b,op);families[s]=pair.ravel()
  for op2,t in enumerate(['+','-','*','/']):
   families[f'({s}){t}']=apply(pair[:,:,None],values[None,None,:],op2).ravel()
  for const in [2.,100.,60.,7.]:families[f'{s}const{int(const)}']=apply(values,const,op).ravel()
 for j,answer in enumerate(answers):
  if answer is None:continue
  value=float(Fraction(answer));out[j].append('positive' if value>0 else 'zero' if value==0 else 'negative');out[j].append('integer' if Fraction(answer).denominator==1 else 'fraction')
  for key,v in families.items():
   ok=np.isfinite(v)&np.isclose(v,value,rtol=1e-8,atol=1e-9)
   if ok.any():out[j].append('reachable:'+key)
  out[j].append('magnitude:'+str(int(np.clip(np.floor(np.log10(max(abs(value),1e-9)/max(np.max(abs(values)),1e-9))),-4,4))))
  out[j].append('relative_rank:'+str(int((values<value).sum()*4/len(values))))
 return out

def matrix(bank,interaction=True):
 rr=[];cc=[];vv=[];coverage=0
 for i,row in enumerate(bank):
  answers=[parse_answer(c['text']) for c in row['candidates']];fs=flags(row['question'],answers)
  words=re.findall('[a-z]+',row['question'].lower());terms=sorted(set(words+[' '.join(words[k:k+2]) for k in range(len(words)-1)]))
  for j,features in enumerate(fs):
   coverage+=any(s.startswith('reachable') for s in features);keys=features+([a+'|'+b for a in features for b in terms] if interaction else []);counts=Counter()
   for key in keys:
    h=int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),'little');counts[h%D]+=(1 if h>>63 else -1)
   norm=np.sqrt(sum(v*v for v in counts.values()))
   for col,v in counts.items():
    if v:rr.append(4*i+j);cc.append(col);vv.append(v/max(norm,1))
 return csr_matrix((vv,(rr,cc)),shape=(4*len(bank),D),dtype=float),coverage
