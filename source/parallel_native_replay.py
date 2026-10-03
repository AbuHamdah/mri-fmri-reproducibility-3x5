import os,json,time,hashlib,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
P=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(job):
 import numpy as np,tensorflow as tf,gc
 try:
  tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
 except RuntimeError:pass
 tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
 m=tf.keras.models.load_model(P/job['model'],compile=False);probs=[]
 for paths in job['input_groups']:
  if job.get('batch_runs'):
   value=m(np.stack([np.load(P/path) for path in paths]),training=False).numpy().ravel().tolist()
  else:
   values=[]
   for path in paths:
    x=np.load(P/path);x=x[None] if x.shape==tuple(m.input_shape[1:]) else x
    values.append(float(m(x,training=False).numpy().ravel()[0]))
   value=float(np.mean(values))
  probs.append(value)
 delta=float(np.max(np.abs(np.asarray(probs)-np.asarray(job['expected']))))
 assert delta<=1e-4,(job['model'],delta)
 result=dict(model=job['model'],model_sha256=sha(P/job['model']),subject_ids=job['subject_ids'],input_groups=job['input_groups'],input_sha256=[[sha(P/x) for x in g] for g in job['input_groups']],batch_runs=job.get('batch_runs',False),probabilities=probs,maximum_probability_difference=delta,tf_intra_threads=2,tf_inter_threads=2)
 del m;tf.keras.backend.clear_session();gc.collect()
 return result
def main():
 started=time.time();out=P/'reproduction_run/native_parallel';out.mkdir(parents=True,exist_ok=True)
 inputs=read(P/'INPUTS.json');ids=sorted({r['subject_id'] for r in inputs});bykind={k:{s:sorted(r['path'] for r in inputs if r['subject_id']==s and r['kind']==k) for s in ids} for k in ['mri','fmri']};jobs=[]
 for seed in [42,43,44]:
  for fold in range(1,6):
   d=P/'selected/mri_baseline/adopted_models'/f'seed{seed}_fold{fold}';rows=read(P/'selected/mri_baseline/calibration'/f'seed{seed}_fold{fold}.json')['predictions']
   jobs.append(dict(model=(d/'model.keras').relative_to(P).as_posix(),subject_ids=[r['subject_id'] for r in rows],input_groups=[bykind['mri'][r['subject_id']] for r in rows],expected=[r['probability'] for r in rows]))
 for series in ['mri_baseline','mri_agentic']:
  for seed in [42,43,44]:
   for fold in range(1,6):
    d=P/'selected'/series/'branches'/series/f'seed{seed}_fold{fold}';by={r['subject_id']:r for r in read(d/'ALL_SUBJECT_PREDICTIONS.json')}
    jobs.append(dict(model=(d/'model.keras').relative_to(P).as_posix(),subject_ids=ids,input_groups=[bykind['mri'][s] for s in ids],expected=[by[s]['probability'] for s in ids]))
 for series in ['fmri_baseline','fmri_agentic']:
  ca=read(P/'selected'/series/'cache'/series/'SOURCE_RUN_PROBABILITIES.json')
  jobs.append(dict(model=f'parent_models/{series}.keras',subject_ids=ids,input_groups=[bykind['fmri'][s] for s in ids],expected=ca['three_run_probabilities'],batch_runs=True))
 for series_dir in sorted((P/'comparisons/branches').iterdir()):
  for seed in [42,43,44]:
   for fold in range(1,6):
    d=series_dir/f'seed{seed}_fold{fold}';rows=read(d/'test_predictions.json')
    jobs.append(dict(model=(d/'model.keras').relative_to(P).as_posix(),subject_ids=[r['subject_id'] for r in rows],input_groups=[r['input_paths'] for r in rows],expected=[r['probability'] for r in rows]))
 assert len(jobs)==197
 results=[]
 with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as executor:
  fs={executor.submit(run,j):j for j in jobs}
  for f in as_completed(fs):
   r=f.result();results.append(r);name=hashlib.sha256(r['model'].encode()).hexdigest()+'.json';(out/name).write_text(json.dumps(r,indent=2),encoding='utf-8')
   (out/'progress.json').write_text(json.dumps(dict(completed=len(results),total=len(jobs),elapsed_seconds=time.time()-started)),encoding='utf-8');print('Native model',len(results),'of',len(jobs),r['model'],r['maximum_probability_difference'],flush=True)
 cache=dict(passed=True,models={r['model']:r for r in results},elapsed_seconds=time.time()-started,scope='Actual full saved-model inference from copied processed arrays. Eight independent processes, each with original TensorFlow 2/2 threads. No training, label changes or prediction editing.')
 (P/'reproduction_run/NATIVE_FULL_MODEL_CACHE.json').write_text(json.dumps(cache,indent=2),encoding='utf-8');print('ALL197 FULL MODELS VERIFIED',flush=True)
if __name__=='__main__':main()
