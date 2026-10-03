import json, hashlib, shutil, re, html, os
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'source' else Path('D:/MRI_FMRI_ADOPTED_RESULTS_AND_COMPARISONS_V1')

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
audit = read(P / 'reproduction_run/REPRODUCTION_AUDIT.json')
assert audit['passed'] and audit['result_rows'] == 33
sources = read(P / 'COPY_MANIFEST.json') + read(P / 'EXTRA_COPY_MANIFEST.json')
assert len({v['path'] for v in sources}) == len(sources)
for i, v in enumerate(sources):
    assert sha(P / v['path']) == v['sha256'], v['path']
    if i % 1000 == 0:
        print('Verified delivery assets:', i, 'of', len(sources), flush=True)
save(P / 'VERIFIED_COPY_MANIFEST.json', sources)
data = read(P / 'RESULTS.json')
errorcases = []
confidence = []
for r in read(P / 'reproduction_run/POOLED_REGENERATED_PREDICTIONS.json'):
    rows = r['predictions']
    yy = np.array([v['true_label'] for v in rows])
    hh = np.array([v['prediction'] for v in rows])
    n = len(rows)
    correct = int(sum(hh == yy))
    z = 1.959963984540054
    ph = correct / n
    den = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    rng = np.random.default_rng(r['seed'])
    draw = np.concatenate([rng.choice(np.flatnonzero(yy == c), size=(5000, sum(yy == c)), replace=True) for c in [0, 1]], axis=1)
    Y = yy[draw]
    H = hh[draw]
    tp = ((Y == 1) & (H == 1)).sum(1)
    fp = ((Y == 0) & (H == 1)).sum(1)
    fn = ((Y == 1) & (H == 0)).sum(1)
    ff = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    confidence.append(dict(result=r['result'], seed=r['seed'], children=n, accuracy_95CI=[float(center - half), float(center + half)], accuracy_method='Wilson', f1_95CI=list(map(float, np.quantile(ff, [0.025, 0.975]))), f1_method='5000 class-stratified child Bootstrap percentile resamples', scope='Descriptive within the examined cohort; not a generalization guarantee. Each seed repeats the same42 children.'))
    errorcases.append(dict(result=r['result'], seed=r['seed'], TP=int(sum((yy == 1) & (hh == 1))), TN=int(sum((yy == 0) & (hh == 0))), FP=int(sum((yy == 0) & (hh == 1))), FN=int(sum((yy == 1) & (hh == 0))), errors=[dict(v, error_type='FN' if v['true_label'] == 1 else 'FP') for v in rows if v['prediction'] != v['true_label']]))
save(P / 'results/CONFIDENCE_INTERVALS.json', confidence)
save(P / 'results/ERROR_CASES_AND_CONFUSION.json', errorcases)
recs = read(P / 'reproduction_run/MODEL_RESULT_INDEX.json')
tested = {}
for pooled in read(P / 'reproduction_run/POOLED_REGENERATED_PREDICTIONS.json'):
    for v in pooled['predictions']:
        tested.setdefault((pooled['result'], pooled['seed'], v['test_fold']), set()).add(v['subject_id'])
for r in recs:
    d = Path(r['model']).parent
    deps = []
    split_dir = P / d
    if split_dir.name in ['mri', 'fmri']:
        split_dir = split_dir.parent
    split_file = next((split_dir / x for x in ['SPLIT.json', 'SPLIT_DETAILS.json'] if (split_dir / x).is_file()), None)
    if split_file:
        split = read(split_file)
        r['split_path'] = split_file.relative_to(P).as_posix()
        r['nominal_train_ids'] = split.get('train', split.get('nominal_train', []))
        r['validation_ids'] = split.get('validation', split.get('nominal_validation', []))
        r['test_ids'] = split.get('test', split.get('nominal_test', []))
        r['fitting_metadata'] = {k: v for k, v in split.items() if k.startswith('actual')}
        tr = set(r['nominal_train_ids'])
        va = set(r['validation_ids'])
        te = set(r['test_ids'])
        r['nominal_partition_overlap'] = dict(train_validation=sorted(tr & va), train_test=sorted(tr & te), validation_test=sorted(va & te))
        assert not tr & va and (not tr & te) and (not va & te), (r['result'], r['seed'], r['fold'], 'nominal split overlap')
        assert te == tested[r['result'], r['seed'], r['fold']], (r['result'], r['seed'], r['fold'], 'test predictions do not match the split')
        r['test_prediction_subject_ids_match_split'] = True
    r['protocol_evidence_files'] = [(split_dir / x).relative_to(P).as_posix() for x in ['protocol.json', 'PROTOCOL.json', 'RESULT.json'] if (split_dir / x).is_file()]
    if r['result'] == 'mri_baseline':
        r['threshold_selection_file'] = f"selected/mri_baseline/calibration/seed{r['seed']}_fold{r['fold']}.json"
    if r['result'] in ['Warm-adapter Soft Voting', 'Stochastic Neural Stacking']:
        run = f"seed{r['seed']}_fold{r['fold']}"
        deps = [f'fusion/warm_adapters/runs/{run}/mri/MODEL.keras', f'fusion/warm_adapters/runs/{run}/fmri/MODEL.keras']
        if r['result'] == 'Stochastic Neural Stacking':
            deps.append(f'fusion/neural_stacking/runs/{run}/MODEL.keras')
        deps += ['extractors/mri_features128.keras', 'extractors/fmri_features128.keras']
    run = f"seed{r['seed']}_fold{r['fold']}"
    if r['result'] in ['fmri_baseline', 'fmri_agentic']:
        deps.append(f"parent_models/{r['result']}.keras")
    if r['result'] == 'Agent-controlled Feature Fusion':
        deps += ['extractors/mri_features128.keras', 'extractors/fmri_features128.keras']
    if r['result'] in ['Warm-adapter Soft Voting', 'Stochastic Neural Stacking', 'Agent-controlled Feature Fusion']:
        deps += [f'selected/mri_agentic/branches/mri_agentic/{run}/model.keras', f'selected/fmri_agentic/branches/fmri_agentic/{run}/model.keras', 'parent_models/fmri_agentic.keras']
    if ' / ' in r['result']:
        mode = r['result'].split(' / ')[0]
        deps += [f'selected/mri_{mode}/branches/mri_{mode}/{run}/model.keras', f'selected/fmri_{mode}/branches/fmri_{mode}/{run}/model.keras', f'parent_models/fmri_{mode}.keras']
    deps = list(dict.fromkeys(deps))
    r['input_index'] = 'INPUTS.json'
    r['replay_entrypoint'] = 'source/run_reproduction.py'
    r['additional_model_dependencies'] = [dict(path=x, sha256=sha(P / x)) for x in deps]
    r['current_model_path'] = r['model']
    r['original_provenance_absolute_paths_in_evidence_only'] = True
save(P / 'MODEL_TO_RESULT_INDEX.json', recs)
runtime = dict(docker_image='dyslexia-pipeline-tf212-py310:latest', docker_image_id='sha256:1afede9925046727d60d3a374ecbe32fe6984326a7aafc08167832a7d9fb307b', python='3.10.12', packages=dict(tensorflow='2.12.0', numpy='1.23.5', scipy='1.10.1', h5py='3.16.0'), network='disabled for verification', only_host_mount='This delivery folder mapped to /package', tf_intra_threads=2, tf_inter_threads=2, deterministic_ops=True, cuda_disabled=True)
runtime['portable_image_archive'] = 'environment/tf212_py310_image.tar'
runtime['portable_image_archive_sha256'] = sha(P / runtime['portable_image_archive'])
save(P / 'REPRODUCTION_ENVIRONMENT.json', runtime)
runtime['native_replay_parallel_processes'] = 8
runtime['docker_cpu_limit'] = 16
runtime['docker_memory_limit'] = '24g'
save(P / 'REPRODUCTION_ENVIRONMENT.json', runtime)
(P / 'REPLAY.ps1').write_text('$ErrorActionPreference = "Stop"\n$taskRoot = $PSScriptRoot\n$taskImage = "sha256:1afede9925046727d60d3a374ecbe32fe6984326a7aafc08167832a7d9fb307b"\n& docker image inspect $taskImage *> $null\nif ($LASTEXITCODE -ne 0) {\n    & docker image load --input (Join-Path $taskRoot "environment/tf212_py310_image.tar")\n    if ($LASTEXITCODE -ne 0) { throw "Unable to load the included reproduction environment." }\n}\n& docker run --rm --network none --cpus 16 --memory 24g -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 -e PYTHONPATH=/package/source --mount "type=bind,source=$taskRoot,target=/package" $taskImage python /package/source/run_reproduction.py\nif ($LASTEXITCODE -ne 0) { throw "Reproduction failed. Inspect reproduction_run/FAILURE.json and EXECUTION.log." }\nWrite-Host "Reproduction passed. Read reproduction_run/REPRODUCTION_AUDIT.json."\n', encoding='utf-8')
failure = P / 'reproduction_run/FAILURE.json'
if failure.exists():
    shutil.copy2(failure, P / 'evidence/earlier_failed_replay.json')
    failure.unlink()
save(P / 'reproduction_run/progress.json', dict(stage='complete', result_rows=33, result_model_records=audit['result_model_records'], elapsed_seconds=audit['elapsed_seconds']))
e = P / 'evidence/first_successful_reproduction'
if not e.exists():
    shutil.copytree(P / 'reproduction_run', e)
columns = ['accuracy', 'recall', 'f1', 'precision', 'specificity', 'auc', 'kappa', 'brier', 'ece']
sections = []
for key, title in [('adopted', 'Current adopted results'), ('source_comparisons', 'Historical source comparisons'), ('previous_fusion_comparisons', 'Earlier fixed-threshold fusion comparisons')]:
    rows = ''
    for r in data[key]:
        vals = []
        for c in columns:
            v = r['metrics'][c]
            scale = 100 if c in ['accuracy', 'recall', 'f1', 'precision', 'specificity'] else 1
            digits = 2 if scale == 100 else 4
            vals.append('<td>' + f"{v['mean'] * scale:.{digits}f} ± {v['sd'] * scale:.{digits}f}" + '</td>')
        rows += '<tr><td>' + html.escape(r['name']) + '</td>' + ''.join(vals) + '</tr>'
    sections.append('<h2>' + title + '</h2><div class="scroll"><table><tr><th>Result</th>' + ''.join(('<th>' + c + (' %' if c in columns[:5] else '') + '</th>' for c in columns)) + '</tr>' + rows + '</table></div>')
(P / 'START_RESULTS_EN.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Curated verified results and comparisons</title><style>body{font-family:Arial;margin:24px;line-height:1.5;max-width:1450px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;white-space:nowrap}th{background:#eef}.scroll{overflow:auto}.note{padding:12px;background:#fff3dd}</style><h1>Curated results and comparisons — checkpoint replay verified</h1><p>33 requested results. Mean ± sample SD across three pooled seeds,42 children per seed. Original source packages preserved.</p><p class="note">' + html.escape(data['scope']) + ' No raw NIfTI preprocessing or training rerun is claimed.</p><p><a href="README_EN.txt">Scope and reproduction instructions</a> | <a href="REPLAY.ps1">Replay script</a> | <a href="RESULTS.json">All33 result records</a> | <a href="MODEL_TO_RESULT_INDEX.json">Checkpoint-to-result index</a> | <a href="reproduction_run/REPRODUCTION_AUDIT.json">Actual replay audit</a> | <a href="results/ERROR_CASES_AND_CONFUSION.json">Errors and confusion counts</a> | <a href="results/CONFIDENCE_INTERVALS.json">Descriptive95% intervals</a></p>' + ''.join(sections) + '</html>', encoding='utf-8')
bad = []
for p in P.rglob('*.html'):
    for href in re.findall('href="([^"]+)"', p.read_text(encoding='utf-8')):
        if not (p.parent / href).is_file():
            bad.append(dict(file=str(p.relative_to(P)), link=href))
assert not bad, bad
scope = read(P / 'DELIVERY_SCOPE.json')
scope['status'] = 'verified_checkpoint_inference_delivery_complete'
scope['verification_date'] = '2026-10-03'
save(P / 'DELIVERY_SCOPE.json', scope)
inventory = []
for i, p in enumerate(P.rglob('*')):
    if p.is_file() and p.name not in ['FINAL_PACKAGE_INVENTORY.json', 'FINAL_DELIVERY_AUDIT.json']:
        inventory.append(dict(path=p.relative_to(P).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
save(P / 'FINAL_PACKAGE_INVENTORY.json', inventory)
save(P / 'FINAL_DELIVERY_AUDIT.json', dict(passed=True, result_rows=33, adopted_results=7, source_comparisons=10, previous_fusion_comparisons=16, copied_assets=len(sources), copied_source_hashes_match=False, files=len(inventory), bytes=sum((v['bytes'] for v in inventory)), broken_links=bad, model_result_records=len(recs), keras_files=sum((1 for p in P.rglob('*.keras'))), h5_files=sum((1 for p in P.rglob('*.h5'))), previous_isolated_offline_reproduction_passed=True, raw_preprocessing_repeated=False, training_repeated=False, originals_preserved=True))
print(json.dumps(read(P / 'FINAL_DELIVERY_AUDIT.json')), flush=True)
