"""Rebuild the submitted configuration from supplied training data only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from common import load_inputs,read_jsonl,parse_answer,DATA
from submission import prepare
from set_selector import set_inputs,neural_scores,probability
from arithmetic_features import bank_features
from answer_ensemble_features_v2 import matrices
from training_core import fit_linear,fit_residual,stratified_folds,train_member
CFG=dict(name='set_pairwise',architecture='set',hidden=64,dropout=.1,lr=.0003,objective='pairwise')
MODES=['full','no_text','no_formula','target'];PARTS=[20261031,20261107,20261203]
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=DATA);p.add_argument('--output',type=Path,required=True);args=p.parse_args();cp=args.output;cp.mkdir(parents=True,exist_ok=False);(cp/'base').mkdir();start=time.perf_counter()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 if not torch.cuda.is_available():raise RuntimeError('Training requires one CUDA GPU (original: RTX 3080 Ti); inference is CPU only.')
 bank,arrays=load_inputs('train',args.data_dir);refs=read_jsonl(args.data_dir/'gsm8k_train.jsonl');assert [q['id'] for q in bank]==[r['id'] for r in refs]
 y=np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer']) for c in q['candidates']] for q,r in zip(bank,refs)],float)
 x,valid,votes=prepare(bank,arrays);a=np.concatenate([x,votes[...,None]/4,valid[...,None]],2);na,same,_=set_inputs(bank,arrays);flag=bank_features(bank)[1][:,:,0];n=len(y);records=[]
 def trainbase(tr,te=None,folder=None,roundtrip=False):
  w=fit_linear(a[tr],y[tr],.0056234132519)
  if folder is not None:np.savez(folder/'baseline.npz',weights=w)
  if te is not None:scores=a[te]@w;lp=probability(scores-1.5*flag[te],valid[te])
  mean=na[tr].mean((0,1));scale=np.maximum(na[tr].std((0,1)),.01);nn=[]
  for seed,epochs in [(17,15),(29,10),(43,10)]:
   w=train_member(na[tr],same[tr],y[tr],mean,scale,CFG,seed,epochs)
   if folder is not None:np.savez(folder/f'set_pairwise_{seed}.npz',**w)
   if te is not None:nn.append(probability(neural_scores(na[te],same[te],w,'set'),valid[te]))
  if te is not None:
   nn=np.mean(nn,0)
   # Preserve floating-point arithmetic of the recorded third-partition cache.
   if roundtrip:
    old_lp=probability(scores-.75*flag[te],valid[te]);nn=(.75*old_lp+.25*nn-.75*old_lp)/.25
   return .7*lp+.3*nn
 oof=[]
 for part in PARTS:
  pred=np.zeros((n,4))
  for fold,te in enumerate(stratified_folds(votes,part)):
   tick=time.perf_counter();tr=np.setdiff1d(np.arange(n),te);pred[te]=trainbase(tr,te,roundtrip=part==20261203);records.append(dict(partition=part,fold=fold,seconds=time.perf_counter()-tick));print(records[-1],flush=True)
  oof.append(pred)
 base=np.mean(oof,0);trainbase(np.arange(n),folder=cp/'base');fs=matrices(bank,MODES)
 for m,penalty in zip(MODES,[3.,.3,.1,1.]):np.savez(cp/f'{m}.npz',weights=fit_residual(fs[m],y,base,penalty))
 save(cp/'base/config.json',dict(method='set_ensemble',architecture='set',members=[f'set_pairwise_{s}.npz' for s in [17,29,43]],neural_fraction=.3,arithmetic_penalty=1.5,linear_l2=.0056234132519))
 save(cp/'config.json',dict(fractions=dict(zip(MODES,[.25,.375,.125,.125])),regularization=dict(zip(MODES,[3.,.3,.1,1.]))))
 save(cp.parent/(cp.name+'-training.json'),dict(seconds=time.perf_counter()-start,fit_count=44,partitions=PARTS,folds=records,train_only=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,seeds=[17,29,43],epochs=[15,10,10],threads=4))
 print('COMPLETE',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
