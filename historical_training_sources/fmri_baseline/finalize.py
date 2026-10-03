import json, hashlib, re
from pathlib import Path
O = Path(__file__).resolve().parents[1]
R = Path('D:\\MRI_FMRI_FMRI_FOLLOWUP_REVIEW_V1')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')
assert read(O / 'COEFFICIENT_AUDIT.json')['passed'] and read(O / 'NATIVE_MODEL_REPLAY_AUDIT.json')['passed']
summary = read(O / 'SUMMARY.json')
native = read(O / 'NATIVE_MODEL_REPLAY_AUDIT.json')
lookup = {(x['branch'], x['seed'], x['fold']): x for x in native['model_records']}
index = read(O / 'MODEL_RESULT_INDEX.json')
for r in index:
    r['model_sha256'] = lookup[r['series'], r['seed'], r['fold']]['model_sha256']
write(O / 'MODEL_RESULT_INDEX.json', index)
check = {'agentic_above_baseline': summary[1]['metrics']['accuracy']['mean'] > summary[0]['metrics']['accuracy']['mean'], 'baseline_accuracy': summary[0]['metrics']['accuracy'], 'agentic_accuracy': summary[1]['metrics']['accuracy'], 'superiority_or_variability_enforced': False}
write(O / 'OBJECTIVE_CHECK.json', check)
names = ['accuracy', 'recall', 'f1', 'precision', 'specificity', 'kappa', 'auc', 'brier', 'ece']
percent = {'accuracy', 'recall', 'f1', 'precision', 'specificity'}
table = '<table><tr><th>Branch</th>' + ''.join(('<th>' + k + (' %' if k in percent else '') + '</th>' for k in names)) + '</tr>'
for r in summary:
    table += '<tr><td>' + r['series'] + '</td>' + ''.join((f"<td>{r['metrics'][k]['mean'] * (100 if k in percent else 1):.4f} ± {r['metrics'][k]['sd'] * (100 if k in percent else 1):.4f}</td>" for k in names)) + '</tr>'
section = '<h2>Multiscale nonlinear RBF residual training</h2><p>Thirty selected current-partition protocols;1680 fitted candidate coefficient sets and30 unfitted identity controls retained. Training-only normalization and RBF centers; true-label BCE and regularization toward zero residual; fixed56-candidate multiscale grid plus identity. Selection: validation accuracy, minimum validation BCE, stronger regularization, smaller gamma. Both branches share the protocol. Threshold remains0.5. No new Qwen calls or encoder updates.</p>' + table + '</table><p>   Superiority and nonzeroSD were not enforced; Agentic has a higher descriptive mean than Baseline in this follow-up, but both are below earlier fixed-threshold results. This is not established improvement or evidence of a new Qwen benefit. All models and results remain available. Do not select this or a preceding attempt as a confirmed improvement using examined test outcomes.</p>'
section += '<p>Paired mean accuracy difference: +0.7937 percentage points. Descriptive paired child bootstrap95% interval: -7.9365 to +9.5238 points, including zero. Agentic is lower for seed44; ordering is not uniform. No statistically established or independent superiority is claimed.</p><p><a href="PAIRED_COMPARISON.json">Paired child comparison</a></p><img src="PER_SEED_METRICS.png" style="max-width:100%" alt="Actual per-seed accuracy recall and F1"><img src="OPTIMIZER_CURVES.png" style="max-width:100%" alt="Selected optimization traces">'
body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Nonlinear residual follow-up</title><style>body{font:16px Arial;max-width:1450px;margin:30px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}</style><h1>fMRI nonlinear residual follow-up:3 seeds ×5 partitions</h1>' + section + '<p><a href="PLAN.json">Protocol</a> | <a href="SUMMARY.json">Metrics</a> | <a href="POOLED_PER_SEED.json">All predictions and95% intervals</a> | <a href="MODEL_RESULT_INDEX.json">Models/splits</a> | <a href="COEFFICIENT_AUDIT.json">Fit and selection audit</a> | <a href="NATIVE_MODEL_REPLAY_AUDIT.json">All30 saved heads replay</a> | <a href="OBJECTIVE_CHECK.json">Measured outcome</a> | <a href="training.log">Training log</a> | <a href="native_save_replay.log">Native replay log</a></p><p>Saved Keras heads consume four cached-input values: three ordered source-run probabilities plus frozen source child logit. They do not accept raw images. Native parent replay from processed arrays is inherited from the immutable source run-reliability package and referenced by hashes. No new raw NIfTI replay is claimed. Per-candidate coefficient checkpoints and optimizer histories are retained; selected30 heads have native Keras files.</p></html>'
(O / 'REPORT_EN.html').write_text(body, encoding='utf-8')
(O / 'README_EN.txt').write_text(' \nBoth branches use the same declared grids and selection rules. \nInputs: three ordered native parent batch3 run probabilities, plus logit(float32 mean), clipped1e-12. Native selected Keras heads require four cached-input values, not images. Training-only standardization and RBF centers are embedded in each selected model.\nsource/train.py fits residual coefficients on true training labels only. source/audit.py independently checks normalization, centers, coefficient gradients, validation selection, labels and metrics. source/save_replay.py creates selected native Keras heads and verifies all42 cached-input predictions for all30 heads. All selected model classifications matched. Native parent source replay is inherited from the immutable prior package and linked by SHA256; this is not raw-data or from-scratch retraining.\n1680 fitted candidate coefficient checkpoints and optimizer traces,30 identity controls,30 selected Keras models, actual splits, probabilities, confusion counts, error IDs, metrics,95% descriptive intervals and native replay logs are retained.\nHost Python requires NumPy/SciPy; native model serialization used TensorFlow2.12 Docker image dyslexia-pipeline-tf212-py310:latest, CPU deterministic operations, two intra/inter-op threads. Parent/read-only source packages are external dependencies, stated explicitly rather than claiming standalone delivery.\nOutcome: Baseline92.0635% +/-1.3746 points; Agentic92.8571% +/-2.3810 points. Both are below earlier fixed-threshold results; this is not established improvement. No guaranteed ordering or enforced variation. All attempts remain in the combined review.\n', encoding='utf-8')
rows = read(R / 'ALL_METHOD_RESULTS.json')
rows.extend(({'method': 'Batch3 multiscale RBF residual (exploratory)', 'series': z['series'], 'metrics': z['metrics'], 'evidence': '../' + O.name + '/REPORT_EN.html'} for z in summary))
write(R / 'ALL_METHOD_RESULTS.json', rows)
rp = R / 'REPORT_EN.html'
rb = rp.read_text(encoding='utf-8')
rp.write_text(rb.replace('</html>', section.replace('href="PAIRED_COMPARISON.json"', 'href="../' + O.name + '/PAIRED_COMPARISON.json"').replace('src="PER_SEED_METRICS.png"', 'src="../' + O.name + '/PER_SEED_METRICS.png"').replace('src="OPTIMIZER_CURVES.png"', 'src="../' + O.name + '/OPTIMIZER_CURVES.png"') + '<a href="../' + O.name + '/REPORT_EN.html">All nonlinear residual evidence</a></html>'), encoding='utf-8')
status = read(R / 'CURRENT_STATUS.json')
status['latest_followup'] = 'multiscale_rbf_residual'
status['latest_followup_completed'] = True
status['latest_objective_check'] = check
status['latest_saved_model_replay_passed'] = True
write(R / 'CURRENT_STATUS.json', status)
for root in [O, R]:
    bad = []
    for f in root.rglob('*.html'):
        for link in re.findall('(?:href|src)=["\\\']([^"\\\']+)', f.read_text(encoding='utf-8'), re.I):
            if link.startswith(('http:', 'https:', 'mailto:', 'data:', '#')):
                continue
            if not (f.parent / link.split('#')[0].split('?')[0]).exists():
                bad.append({'file': str(f), 'link': link})
    assert not bad, bad
    inv = [{'path': f.relative_to(root).as_posix(), 'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in root.rglob('*') if f.is_file() and f.name not in ['FINAL_INVENTORY.json', 'FINAL_DELIVERY_AUDIT.json', 'COMPLETE.json']]
    write(root / 'FINAL_INVENTORY.json', inv)
    write(root / 'FINAL_DELIVERY_AUDIT.json', {'passed': True, 'broken_html_links': bad, 'inventory_files': len(inv)})
write(O / 'COMPLETE.json', {'completed': True, 'selected_protocols': 30, 'selected_native_models': 30, 'saved_model_replay_passed': True, 'coefficient_audit_passed': True})
print(json.dumps(check))
