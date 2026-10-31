"""Compare checkpoint arrays without accessing any bank or reference answers."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);a=p.parse_args();result={}
for f in sorted(a.first.rglob('*.npz')):
 name=str(f.relative_to(a.first))
 with np.load(f) as x,np.load(a.second/name) as y:
  assert set(x.files)==set(y.files);result[name]={k:dict(exact=bool(np.array_equal(x[k],y[k])),max_absolute_difference=float(np.max(abs(x[k]-y[k])))) for k in x.files}
print(json.dumps(result,indent=2))
