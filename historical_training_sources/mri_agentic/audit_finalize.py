import json, hashlib, re, html
from pathlib import Path
import numpy as np
from scipy.special import expit
from statistics_core import metrics, NAMES
O = Path(__file__).resolve().parents[1]
P = Path('D:\\MRI_FMRI_LAST_LAYER_ALL_3X5_V1')
R = Path('D:\\MRI_FMRI_FMRI_FOLLOWUP_REVIEW_V1')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, v):
    p.write_text(json.dumps(v, indent=2), encoding='utf-8')
assert read(O / 'TRAINING_COMPLETE.json')['completed']
plan = read(O / 'PLAN.json')
source = read(O / 'SOURCE_NATIVE_PREDICTIONS.json')
F = np.load(O / 'SOURCE_FEATURES.npy')
assert F.shape == (42, 128) and np.isfinite(F).all()
rr = source['records']
ids = [r['subject_id'] for r in rr]
y = np.array([r['label'] for r in rr])
q = np.clip([r['source_probability'] for r in rr], plan['clip'], 1 - plan['clip'])
offset = np.log(q / (1 - q))
assert source['source_sha256'] == sha(P / source['source_model'])
reference = next((r for r in read(P / 'SOURCE_MODELS_REFERENCE.json') if r['series'] == 'mri_agentic'))
assert reference['source_sha256'] == source['source_sha256']
original = {r['subject_id']: r for r in read(P / 'cache' / 'mri_agentic' / 'CACHE_AUDIT.json')['inputs']}
for r in rr:
    assert r['label'] == original[r['subject_id']]['label'] and r['input_sha256'] == original[r['subject_id']]['sha256'] == sha(P / r['path'])
projection = np.load(O / 'FEATURE_PROJECTION.npy')
expected_projection = np.linalg.qr(np.random.default_rng(plan['projection_seed']).normal(size=(128, plan['projection_dimensions'])))[0]
assert projection.shape == (128, 8) and np.allclose(projection, expected_projection, rtol=0, atol=1e-12) and np.allclose(projection.T @ projection, np.eye(8), rtol=0, atol=1e-12)
write(O / 'PROJECTION_NUMERICAL_AUDIT.json', {'passed': True, 'seed': plan['projection_seed'], 'absolute_tolerance': 1e-12, 'maximum_platform_QR_difference': float(abs(projection - expected_projection).max()), 'orthogonality_error': float(abs(projection.T @ projection - np.eye(8)).max()), 'original_saved_projection_used': True, 'training_predictions_modified': False})
native = read(O / 'NATIVE_REPLAY.json')
assert native['passed'] and len(native['records']) == 15
index = []
for n in native['records']:
    d = O / 'branches' / 'mri_agentic' / f"seed{n['seed']}_fold{n['fold']}"
    s = read(d / 'SPLIT.json')
    assert s == read(P / 'branches' / 'mri_agentic' / d.name / 'SPLIT.json')
    sets = [set(s[k]) for k in ['train', 'validation', 'test']]
    assert set.union(*sets) == set(ids) and all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
    tr = np.array([ids.index(k) for k in s['train']])
    vi = np.array([ids.index(k) for k in s['validation']])
    norm = read(d / 'NORMALIZATION.json')
    mean = F[tr].mean(0)
    std = np.maximum(F[tr].std(0), 0.01)
    assert norm['fit_ids'] == s['train'] and np.array_equal(norm['mean'], mean) and np.array_equal(norm['std'], std)
    X = np.c_[(F - mean) / std @ projection, np.ones(42)]
    bw = read(d / 'BOOTSTRAP_TRAIN_WEIGHTS.json')
    assert bw['subject_ids'] == s['train'] and bw['seed_sequence'] == [n['seed'], n['fold'], 1017]
    rng = np.random.default_rng(np.random.SeedSequence(bw['seed_sequence']))
    draws = rng.exponential(size=(plan['bootstrap_draws'], len(tr)))
    counts = np.bincount(y[tr], minlength=2)
    assert np.all(counts > 0)
    draws *= len(tr) / (2 * counts[y[tr]])
    draws *= len(tr) / draws.sum(axis=1, keepdims=True)
    sw = draws.mean(axis=0)
    assert np.array_equal(sw, bw['weights']) and np.array_equal(draws, bw['normalized_draws'])
    cc = read(d / 'BASE_CANDIDATES.json')
    assert len(cc) == 7 and {c['lambda'] for c in cc} == {None, *plan['regularization_grid']}
    for c in cc:
        assert c == read(d / 'candidates' / c['name'] / 'COEFFICIENTS.json')
        w = np.array(c['coefficients'])
        assert len(w) == 9
        z = offset + X @ w
        p = expit(z)
        m = metrics(y[vi], p[vi])
        assert abs(c['validation_balanced_accuracy'] - (m[5] + m[6]) / 2) < 1e-12 and abs(c['validation_bce'] - np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi])) < 1e-12
        if c['lambda'] is None:
            assert np.array_equal(w, np.zeros(9))
        else:
            assert np.max(abs(X[tr].T @ (sw * (p[tr] - y[tr])) / len(tr) + c['lambda'] * w)) < 1e-05
    expanded = read(d / 'VALIDATION_CANDIDATES.json')
    assert len(expanded) == 31
    bases = {c['name']: c for c in cc}
    pairs = set()
    for c in expanded:
        base = bases[c['name']]
        assert c['lambda'] == base['lambda']
        assert c['alpha'] in ([0.0] if c['lambda'] is None else plan['residual_strength_grid'])
        pairs.add((c['name'], c['alpha']))
        w = np.array(base['coefficients']) * c['alpha']
        assert np.array_equal(w, c['coefficients'])
        z = offset + X @ w
        p = expit(z)
        m = metrics(y[vi], p[vi])
        assert abs(c['validation_balanced_accuracy'] - (m[5] + m[6]) / 2) < 1e-12 and abs(c['validation_bce'] - np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi])) < 1e-12
    assert len(pairs) == 31
    chosen = max(expanded, key=lambda c: (c['validation_balanced_accuracy'], -c['validation_bce'], -c['alpha'], float('inf') if c['lambda'] is None else c['lambda']))
    assert chosen == read(d / 'SELECTED_COEFFICIENTS.json')
    p = expit(offset + X @ chosen['coefficients'])
    allrows = read(d / 'ALL_SUBJECT_PREDICTIONS.json')
    assert len(allrows) == 42
    for i, row in enumerate(allrows):
        assert row['subject_id'] == ids[i] and row['true_label'] == y[i] and (abs(row['probability'] - p[i]) < 1e-12) and (row['predicted_label'] == int(p[i] >= 0.5))
    test = read(d / 'test_predictions.json')
    assert test == [allrows[ids.index(k)] for k in s['test']]
    replay = read(d / 'NATIVE_REPLAY.json')
    assert len(replay) == len(test)
    for a, b in zip(replay, test):
        assert a['subject_id'] == b['subject_id'] and abs(a['probability'] - b['probability']) < plan['native_probability_tolerance'] and (int(a['probability'] >= 0.5) == b['predicted_label'])
    assert sha(d / 'model.keras') == n['model_sha256']
    write(d / 'ERRORS.json', [r for r in test if r['true_label'] != r['predicted_label']])
    index.append({**n, 'model': (d / 'model.keras').relative_to(O).as_posix(), 'split': (d / 'SPLIT.json').relative_to(O).as_posix(), 'normalization': (d / 'NORMALIZATION.json').relative_to(O).as_posix(), 'train_weight_file': (d / 'BOOTSTRAP_TRAIN_WEIGHTS.json').relative_to(O).as_posix(), 'fit_ids': s['train'], 'validation_ids': s['validation'], 'test_ids': s['test']})
for row in index:
    chosen = read(O / Path(row['model']).parent / 'SELECTED_COEFFICIENTS.json')
    row['selected_alpha'] = chosen['alpha']
    row['selected_lambda'] = chosen['lambda']
pooled = read(O / 'POOLED_PER_SEED.json')
for r in pooled:
    rows = r['predictions']
    expected = []
    for fold in plan['folds']:
        expected += read(O / 'branches' / 'mri_agentic' / f"seed{r['seed']}_fold{fold}" / 'test_predictions.json')
    assert rows == expected and len(rows) == 42 and (len({z['subject_id'] for z in rows}) == 42)
    m = dict(zip(NAMES, map(float, metrics([z['true_label'] for z in rows], [z['probability'] for z in rows]))))
    assert all((abs(m[k] - r['metrics'][k]) < 1e-12 for k in NAMES))
recalculated = {k: {'mean': float(np.mean([r['metrics'][k] for r in pooled])), 'sd': float(np.std(np.array([r['metrics'][k] for r in pooled]) - pooled[0]['metrics'][k], ddof=1))} for k in NAMES}
assert recalculated == read(O / 'SUMMARY.json')[0]['metrics']
ss = read(O / 'SUMMARY.json')[0]
old = next((r for r in read(P / 'SUMMARY.json') if r['series'] == 'mri_agentic'))
prior = read(Path('D:\\MRI_AGENTIC_BAYESIAN_BOOTSTRAP_CALIBRATION_3X5_V1') / 'SUMMARY.json')[0]
comparison = {'prior_current_head': old, 'prior_scalar_bootstrap': prior, 'new_feature_residual': ss, 'native_parent_metrics': source['source_metrics'], 'new_accuracy_sd_nonzero': ss['metrics']['accuracy']['sd'] > 0, 'same_parent_checkpoint': True, 'descriptive_post_hoc_comparison_only': True}
write(O / 'COMPARISON.json', comparison)
write(O / 'MODEL_RESULT_INDEX.json', index)
write(O / 'AUDIT.json', {'passed': True, 'all90_fitted_candidate_gradients_verified': True, 'all15_identity_controls_verified': True, 'bootstrap_weights_regenerated': True, 'train_only_normalization_verified': True, 'validation_only_selection_recomputed': True, 'all_seed_metrics_recomputed': True, 'native_full_model_test_replay_verified': True, 'source_hash_and42_input_hashes_verified': True})
m = ss['metrics']
names = ['accuracy', 'recall', 'precision', 'f1', 'specificity', 'kappa', 'auc', 'brier', 'ece']
pct = set(names[:5])
table = '<table><tr>' + ''.join(('<th>' + k + (' %' if k in pct else '') + '</th>' for k in names)) + '</tr><tr>' + ''.join((f"<td>{m[k]['mean'] * (100 if k in pct else 1):.4f} ± {m[k]['sd'] * (100 if k in pct else 1):.4f}</td>" for k in names)) + '</tr></table>'
body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>MRI Agentic smoothed-bootstrap fixed-eight residual</title><style>body{font:16px Arial;margin:30px;max-width:1500px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style><h1>MRI Agentic: smoothed-bootstrap fixed-eight residual training,3 seeds ×5 partitions</h1>' + table + '<p>A trained9-parameter residual on8 fixed label-independent projected frozen source features is added to the original source logit. Current training labels are unchanged. Training-only normalization uses populationSD floor0.01. A fixed Gaussian matrix, seed314159, is orthonormalized by QR into8 directions; no labels or outcomes determine this projection. All seeds use this same projection. The mean of5 seeded Bayesian-bootstrap draws, each multiplied by training-only inverse class frequencies and normalized to training size, weights training children; validation/test children are excluded. Six ridge strengths, five residual strengths and identity were declared. Selection uses validation balanced accuracy, minimum BCE, smaller residual strength and stronger ridge only; threshold0.5 is fixed. All90 fitted coefficient checkpoints and15 identity controls, optimizer histories, bootstrap weights and normalization are retained.</p><p>Source and native features were freshly extracted from all42 actual processed MRI inputs using one-child inference. All15 full image-input Keras models were saved, reloaded and replayed on126 current-test child evaluations. Source/input hashes, training normalization, bootstrap seeds, every candidate gradient, selection and pooled metrics were independently checked. Required probability tolerance1e-4 is declared; classification parity is mandatory. Input/model links are in the index.</p><p><strong></strong> historical parents saw the cohort, which has been repeatedly examined. Agentic is historical provenance; no new Qwen calls or encoder retraining occurred. Three seed evaluations reuse the same42 children. Mean±sampleSD is measured, not forced. These data cannot prove independent generalization or an LLM effect.</p><p><a href="PLAN.json">Fixed protocol</a> | <a href="SUMMARY.json">All nine metrics</a> | <a href="POOLED_PER_SEED.json">Every pooled prediction</a> | <a href="COMPARISON.json">Prior heads and native source</a> | <a href="SOURCE_NATIVE_PREDICTIONS.json">Fresh source probabilities and inputs</a> | <a href="SOURCE_FEATURES.npy">Fresh source feature matrix</a> | <a href="MODEL_RESULT_INDEX.json">Full models and splits</a> | <a href="NATIVE_REPLAY.json">Processed-input replay</a> | <a href="AUDIT.json">Independent checks</a> | <a href="training_and_native_replay.log">Execution log</a></p><p>Baseline/fMRI artifacts were not modified. Models accept processed MRI arrays; raw NIfTI processing was not repeated. Parent/input dependency is D:/MRI_FMRI_LAST_LAYER_ALL_3X5_V1.</p></html>'
(O / 'REPORT_EN.html').write_text(body, encoding='utf-8')
write(O / 'COMPLETE.json', {'completed': True, 'selected_full_models': 15, 'complete_native_test_replay_passed': True, 'independent_audit_passed': True})
link = '<p><a href="../' + O.name + '/REPORT_EN.html">MRI Agentic smoothed-bootstrap fixed-eight residual</a></p>'
rp = R / 'REPORT_EN.html'
s = rp.read_text(encoding='utf-8')
if link not in s:
    rp.write_text(s.replace('</html>', link + '</html>'), encoding='utf-8')
write(R / 'MRI_AGENTIC_SMOOTHED_BOOTSTRAP_STATUS.json', {'completed': True, 'evidence': '../' + O.name + '/REPORT_EN.html', 'accuracy': m['accuracy'], 'accuracy_variation_achieved': m['accuracy']['sd'] > 0})
from statistics_core import bca
ci = []
foldmetrics = []
for row in pooled:
    yy = np.array([r['true_label'] for r in row['predictions']])
    pp = np.array([r['probability'] for r in row['predictions']])
    intervals = bca(yy, pp)
    for value in intervals.values():
        value['interpretation'] = ''
    ci.append({'seed': row['seed'], 'intervals': intervals})
    for fold in plan['folds']:
        rows = read(O / 'branches' / 'mri_agentic' / f"seed{row['seed']}_fold{fold}" / 'test_predictions.json')
        yt = np.array([r['true_label'] for r in rows])
        pt = np.array([r['probability'] for r in rows])
        ht = pt >= 0.5
        conf = {'tp': int(((yt == 1) & ht).sum()), 'tn': int(((yt == 0) & ~ht).sum()), 'fp': int(((yt == 0) & ht).sum()), 'fn': int(((yt == 1) & ~ht).sum())}
        assert sum(conf.values()) == len(rows)
        foldmetrics.append({'seed': row['seed'], 'partition': fold, 'n': len(rows), 'confusion': conf, 'metrics': dict(zip(NAMES, map(float, metrics(yt, pt)))), 'misclassified_ids': [r['subject_id'] for r in rows if r['true_label'] != r['predicted_label']]})
write(O / 'CONFIDENCE_INTERVALS.json', ci)
write(O / 'FOLD_METRICS.json', foldmetrics)
previous = read(Path('D:/MRI_AGENTIC_FIXED8_RESIDUAL_3X5_V1/SUMMARY.json'))[0]['metrics']
write(O / 'PREVIOUS_SHRINKAGE_COMPARISON.json', {'previous': previous, 'current': m, 'comparison': 'Exploratory descriptive comparison after repeated inspection; not a confirmatory test'})
seedtable = '<table><tr><th>Seed</th><th>Accuracy %</th><th>Recall %</th><th>F1 %</th></tr>' + ''.join((f"<tr><td>{r['seed']}</td><td>{100 * r['metrics']['accuracy']:.4f}</td><td>{100 * r['metrics']['recall']:.4f}</td><td>{100 * r['metrics']['f1']:.4f}</td></tr>" for r in pooled)) + '</table>'
extra = '<h2>Seed-level pooled results</h2>' + seedtable + '<p>Mean and sample SD are calculated across these three seed-level pooled metrics, not across15 independent observations.</p><p><a href="FOLD_METRICS.json">Current test-partition confusion matrices and errors</a> | <a href="CONFIDENCE_INTERVALS.json">Descriptive95% intervals</a> | <a href="FEATURE_PROJECTION.npy">Fixed projection</a> | <a href="PREVIOUS_SHRINKAGE_COMPARISON.json">Previous fixed-eight comparison</a></p>'
(O / 'REPORT_EN.html').write_text(body.replace('</html>', extra + '</html>'), encoding='utf-8')
for root in [O, R]:
    bad = []
    for p in root.rglob('*.html'):
        for href in re.findall('(?:href|src)=["\\\']([^"\\\']+)', p.read_text(encoding='utf-8'), re.I):
            if href.startswith(('http:', 'https:', 'data:', 'mailto:', '#')):
                continue
            if not (p.parent / href.split('#')[0].split('?')[0]).exists():
                bad.append({'file': str(p), 'link': href})
    assert not bad, bad
    inv = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in root.rglob('*') if p.is_file() and p.name not in ['FINAL_INVENTORY.json', 'FINAL_DELIVERY_AUDIT.json']]
    write(root / 'FINAL_INVENTORY.json', inv)
    write(root / 'FINAL_DELIVERY_AUDIT.json', {'passed': True, 'broken_html_links': bad, 'inventory_files': len(inv)})
print(json.dumps({'metrics': m, 'accuracy_variation_achieved': m['accuracy']['sd'] > 0, 'native_parent_metrics': source['source_metrics']}))
