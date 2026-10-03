import subprocess,sys,json,time,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1];out=P/'reproduction_run';out.mkdir(exist_ok=True)
started=time.time()
for name in ['REPRODUCTION_AUDIT.json','NATIVE_FULL_MODEL_CACHE.json','PIPELINE_AUDIT.json','PIPELINE_FAILURE.json','FAILURE.json']:
 p=out/name
 if p.exists():
  archive=out/'preserved_attempts'/('before_rerun_'+str(time.time_ns()));archive.mkdir(parents=True);shutil.copy2(p,archive/name);p.unlink()
with (out/'PIPELINE_EXECUTION.log').open('w',encoding='utf-8') as log:
 for script in ['parallel_native_replay.py','replay.py']:
  print('Starting',script,flush=True)
  proc=subprocess.Popen([sys.executable,str(P/'source'/script)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  for line in proc.stdout:sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
  result=proc.wait()
  if result:
   (out/'PIPELINE_FAILURE.json').write_text(json.dumps(dict(script=script,exit_code=result,elapsed_seconds=time.time()-started)),encoding='utf-8');sys.exit(result)
(out/'PIPELINE_AUDIT.json').write_text(json.dumps(dict(passed=True,elapsed_seconds=time.time()-started,phases=['fresh parallel full-model inference','downstream checkpoint replay and33 metric sets']),indent=2),encoding='utf-8')
print('Complete reproduction pipeline passed',flush=True)
