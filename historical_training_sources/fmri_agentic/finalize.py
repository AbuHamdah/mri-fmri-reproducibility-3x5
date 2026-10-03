import json, hashlib, re
from pathlib import Path
from extract import generate
O = Path(__file__).resolve().parents[1]
R = Path('D:\\MRI_FMRI_FMRI_FOLLOWUP_REVIEW_V1')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')
assert generate()['records'] == read(O / 'INPUT_FEATURES.json')['records']
audit = read(O / 'NATIVE_MODEL_REPLAY_AUDIT.json')
assert audit['passed']
summary = read(O / 'SUMMARY.json')
index = []
for r in audit['records']:
    d = O / 'branches' / r['series'] / f"seed{r['seed']}_fold{r['fold']}"
    r = dict(r)
    r.update({'model': (d / 'model.keras').relative_to(O).as_posix(), 'split': (d / 'SPLIT.json').relative_to(O).as_posix(), 'selected_checkpoint': (d / 'SELECTED_CHECKPOINT.json').relative_to(O).as_posix()})
    index.append(r)
    test = read(d / 'test_predictions.json')
    write(d / 'ERRORS.json', [z for z in test if z['true_label'] != z['predicted_label']])
write(O / 'MODEL_RESULT_INDEX.json', index)
check = {'agentic_above_baseline': summary[1]['metrics']['accuracy']['mean'] > summary[0]['metrics']['accuracy']['mean'], 'baseline_accuracy': summary[0]['metrics']['accuracy'], 'agentic_accuracy': summary[1]['metrics']['accuracy'], 'ordering_or_variation_enforced': False}
write(O / 'OBJECTIVE_CHECK.json', check)
names = ['accuracy', 'recall', 'precision', 'f1', 'specificity', 'kappa', 'auc', 'brier', 'ece']
pct = set(names[:5])
table = '<table><tr><th>Branch</th>' + ''.join(('<th>' + k + (' %' if k in pct else '') + '</th>' for k in names)) + '</tr>'
for r in summary:
    table += '<tr><td>' + r['series'] + '</td>' + ''.join((f"<td>{r['metrics'][k]['mean'] * (100 if k in pct else 1):.4f} ± {r['metrics'][k]['sd'] * (100 if k in pct else 1):.4f}</td>" for k in names)) + '</tr>'
section = '<h2>Class-balanced boosted decision-stump residual heads</h2><p>Thirty selected model protocols. Nineteen inputs: three source-run probabilities and16 numerical image descriptors. Newton gradient-boosted residual decision stumps fit true current training labels on frozen parent logits. Class weights n_train/(2*n_train_class) are computed inside each current training partition and multiply training gradients and Hessians. Validation selection metrics remain unweighted. The same class-balancing rule applies to both branches; no test-derived weighting is used. Split thresholds derive only from training values, with at least3 training children per leaf and maximum absolute leaf correction2. Three learning rates, two leaf regularization strengths and three staged checkpoints (5/10/20 rounds) plus identity were declared. Validation balanced accuracy, minimum BCE, fewer trees, stronger regularization and smaller learning rate select the model. Both branches share the rules; threshold fixed0.5. All540 fitted candidate checkpoints and30 identity controls retained. No new encoder training or Qwen calls occurred.</p>' + table + '</table><p>Actual mean±sampleSD over3 pooled42-child seed evaluations.    Accuracy is within one percentage point of95%, but Agentic superiority was not achieved. All attempts remain available.</p>'
body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Class-balanced class-balanced boosted-stump residual follow-up</title><style>body{font:16px Arial;margin:30px}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>Class-balanced class-balanced boosted-stump residual follow-up:3 seeds ×5 current partitions</h1>' + section + '<p><a href="PLAN.json">Protocol</a> | <a href="SUMMARY.json">All nine metrics</a> | <a href="POOLED_PER_SEED.json">Predictions per seed</a> | <a href="MODEL_RESULT_INDEX.json">Models, splits, fitting and selection IDs</a> | <a href="INPUT_FEATURES.json">Processed-input descriptors and input hashes</a> | <a href="NATIVE_MODEL_REPLAY_AUDIT.json">Complete saved-head replay and independent coefficient/selection checks</a> | <a href="OBJECTIVE_CHECK.json">Measured outcome</a> | <a href="training.log">Training log</a> | <a href="native_save_replay.log">Replay log</a></p><p>All126 processed arrays were re-extracted and feature values/input hashes matched. Descriptors are generic numerical summaries, not verified BOLD/motion measures. Selected Keras heads take20 inputs: three source-run probabilities,16 descriptors, and frozen parent mean logit. They do not take raw images. Parent native replay is inherited from the immutable original-batch run-reliability package; no raw NIfTI replay is claimed.</p></html>'
(O / 'REPORT_EN.html').write_text(body, encoding='utf-8')
(O / 'README_EN.txt').write_text('Exploratory class-balanced boosted-stump residual heads. source/train.py fits true current training labels only; source/save_audit.py independently regenerates all permitted training-only split thresholds, every candidate tree fit and validation choice, saves selected native Keras heads and replays all1260 head outputs. Three learning rates, two leaf regularization strengths and three staged checkpoints were declared; all540 fitted checkpoints and30 identity controls remain available. source/finalize.py re-extracts all126 processed arrays, verifying features and input hashes. Parent/source dependencies are D:/MRI_FMRI_LAST_LAYER_ALL_3X5_V1 and D:/MRI_FMRI_FMRI_BATCH3_RUN_RELIABILITY_3X5_V1. Selected Keras heads need20 input values, not images. This is not a standalone raw/from-scratch training package.  All actual metrics, labels, candidate checkpoints, optimizer histories, splits and source input hashes are retained. No guaranteed target and no new Qwen calls.', encoding='utf-8')
ar = read(R / 'ALL_METHOD_RESULTS.json')
ar.extend(({'method': 'Class-balanced boosted decision-stump residual (exploratory)', 'series': z['series'], 'metrics': z['metrics'], 'evidence': '../' + O.name + '/REPORT_EN.html'} for z in summary))
write(R / 'ALL_METHOD_RESULTS.json', ar)
rp = R / 'REPORT_EN.html'
rp.write_text(rp.read_text(encoding='utf-8').replace('</html>', section + '<a href="../' + O.name + '/REPORT_EN.html">All class-balanced boosted-stump residual evidence</a></html>'), encoding='utf-8')
status = read(R / 'CURRENT_STATUS.json')
status['latest_followup'] = 'balanced_stump_residual'
status['latest_followup_completed'] = True
status['latest_objective_check'] = check
status['requested_joint_target_achieved'] = False
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
write(O / 'COMPLETE.json', {'completed': True, 'selected_native_models': 30, 'complete_saved_head_replay_passed': True, 'all_processed_input_features_reextracted_verified': True})
print(json.dumps(check))
