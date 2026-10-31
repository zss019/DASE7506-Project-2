"""Fixed-weight validation ablations; no test access or refitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from common import load_inputs,read_jsonl,parse_answer
from base_probability_v2 import base_probability
from answer_ensemble_features_v2 import matrices
from submission import choose
p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoint');p.add_argument('--output',type=Path,required=True);a=p.parse_args();bank,arr=load_inputs('validation',a.data_dir);refs=read_jsonl(a.data_dir/'gsm8k_validation.jsonl');assert [q['id'] for q in bank]==[r['id'] for r in refs]
y=np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer']) for c in q['candidates']] for q,r in zip(bank,refs)],float);hard=np.array([len({parse_answer(c['text']) for c in q['candidates']}-{None})>=2 for q in bank]);prob,valid=base_probability(bank,arr,a.checkpoint/'base');cfg=json.loads((a.checkpoint/'config.json').read_text());fs=matrices(bank,cfg['fractions']);delta={}
for m in fs:
 with np.load(a.checkpoint/f'{m}.npz') as w:delta[m]=cfg['fractions'][m]*(fs[m]@w['weights']).reshape(-1,4)
rows=[]
for name,modes in [('base_only',[]),('full_model',list(delta))]+[(f'without_{m}',[k for k in delta if k!=m]) for m in delta]:
 score=np.log(np.maximum(prob,1e-9))
 for m in modes:score+=delta[m]
 choices=choose(score,valid);right=y[np.arange(len(y)),choices].astype(bool);rows.append(dict(method=name,correct=int(right.sum()),n=len(bank),hard_correct=int(right[hard].sum()),hard_n=int(hard.sum())))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(split='validation',fixed_final_weights=True,no_refitting=True,no_test_access=True,rows=rows),indent=2)+'\n');print(json.dumps(rows))
