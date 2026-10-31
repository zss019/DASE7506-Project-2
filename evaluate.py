"""Run the unchanged official predictor and scorer in fresh CPU processes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoint');p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='');tick=time.perf_counter();logs=[]
 commands=[[sys.executable,str(ROOT/'code/predict.py'),'--method',str(ROOT/'code/answer_ensemble_fast_v2.py'),'--checkpoint',str(a.checkpoint.resolve()),'--split',a.split,'--data-dir',str(a.data_dir.resolve()),'--output',str((a.output/'predictions.json').resolve())],[sys.executable,str(ROOT/'code/score_predictions.py'),'--split',a.split,'--data-dir',str(a.data_dir.resolve()),'--predictions',str((a.output/'predictions.json').resolve()),'--output',str((a.output/'score.json').resolve())]]
 for cmd in commands:
  r=subprocess.run(cmd,env=env,capture_output=True,text=True);logs.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));(a.output/'execution.json').write_text(json.dumps(logs,indent=2)+'\n');r.check_returncode()
 result=dict(fresh_prediction_plus_scoring_seconds=time.perf_counter()-tick,threads=4,cuda_disabled=True,python=sys.version,score=json.loads((a.output/'score.json').read_text()));(a.output/'runtime.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
