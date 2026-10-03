import json, hashlib, html, math, re, sys
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(O / 'source'))
from statistics_core import metrics, NAMES
R = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))

def save(n, x):
    (O / n).write_text(json.dumps(x, indent=2), encoding='utf8')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def classification(y, p, h):
    m = dict(zip(NAMES, metrics(y, p).tolist()))
    tp = int(((y == 1) & h).sum())
    tn = int(((y == 0) & ~h).sum())
    fp = int(((y == 0) & h).sum())
    fn = int(((y == 1) & ~h).sum())
    n = len(y)
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    m.update(accuracy=acc, f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1), kappa=(acc - pe) / (1 - pe) if pe != 1 else 0)
    return (m, dict(TP=tp, TN=tn, FP=fp, FN=fn))

def ci(y, h):
    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(y), size=(10000, len(y)))
    yy = y[ix]
    hh = h[ix]
    tp = ((yy == 1) & hh).sum(axis=1)
    fp = ((yy == 0) & hh).sum(axis=1)
    fn = ((yy == 1) & ~hh).sum(axis=1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    n = len(y)
    ph = float((y == h).mean())
    z = 1.959963984540054
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return {'accuracy': {'low': c - half, 'high': c + half, 'method': 'Wilson 95%'}, 'f1': {'low': float(np.quantile(f1, 0.025)), 'high': float(np.quantile(f1, 0.975)), 'method': 'Subject-pair percentile bootstrap, 10000 resamples'}, 'scope': ''}
assert R(O / 'BRANCH_COMPLETE.json')['completed'] == 210 and R(O / 'FUSION_COMPLETE.json')['completed'] == 180
inputs = R(O / 'INPUTS.json')
labels = {r['subject_id']: int(r['label']) for r in inputs}
assert len(labels) == 42 and sum(labels.values()) == 20 and all((int(r['label']) == labels[r['subject_id']] for r in inputs))
assert all((sum((r['kind'] == 'mri' and r['subject_id'] == s for r in inputs)) == 1 and sum((r['kind'] == 'fmri' and r['subject_id'] == s for r in inputs)) == 3 for s in labels))
index = []
byseries = {}
audit = []
for root in [O / 'branches', O / 'fusion']:
    for file in sorted(root.rglob('RESULT_CURRENT.json')):
        d = file.parent
        r = R(file)
        s = R(d / 'SPLIT.json')
        pr = R(d / 'test_predictions.json')
        v = R(d / 'validation_predictions.json')
        assert set((x['subject_id'] for x in pr)) == set(s['test'])
        assert set((x['subject_id'] for x in v)) == set(s['validation'])
        assert not set(s['train']) & set(s['test']) and (not set(s['validation']) & set(s['test'])) and (not set(s['train']) & set(s['validation']))
        assert len(s['train'] + s['validation'] + s['test']) == 42
        assert all((int(x['true_label']) == labels[x['subject_id']] for x in pr + v))
        vy = np.array([x['true_label'] for x in v])
        vp = np.array([x['probability'] for x in v])
        u = np.unique(vp)
        cand = np.unique(np.r_[0, 0.5, 1, (u[:-1] + u[1:]) / 2])
        rank = []
        for t in cand:
            m = dict(zip(NAMES, metrics(vy, vp, t)))
            rank.append((m['f1'], (m['recall'] + m['specificity']) / 2, -abs(t - 0.5), float(t)))
        selected = max(rank)[3]
        assert selected == r['threshold']
        y = np.array([x['true_label'] for x in pr])
        p = np.array([x['probability'] for x in pr])
        m, c = classification(y, p, p >= selected)
        fixed, _ = classification(y, p, p >= 0.5)
        for k in NAMES:
            assert abs(m[k] - r['calibrated_metrics'][k]) < 1e-12 and abs(fixed[k] - r['fixed_metrics'][k]) < 1e-12
        (d / 'CURRENT_CONFUSION.json').write_text(json.dumps({'seed': r['seed'], 'fold': r['fold'], 'series': r['series'], 'test_n': len(pr), 'threshold': selected, 'confusion_matrix': c, 'false_negatives': [x for x in pr if x['true_label'] == 1 and x['predicted_label'] == 0], 'false_positives': [x for x in pr if x['true_label'] == 0 and x['predicted_label'] == 1]}, indent=2), encoding='utf8')
        proto = R(d / 'protocol.json')
        model = d / 'model.keras'
        modelsha = sha(model) if model.exists() else None
        expected = r.get('model_sha256', proto.get('model_sha256'))
        if expected:
            assert modelsha == expected
        maxdelta = r.get('reload_max_difference', proto.get('native_reload_max_difference'))
        assert maxdelta is None or maxdelta <= 0.0001
        entry = {'series': r['series'], 'seed': r['seed'], 'fold': r['fold'], 'result': str(file.relative_to(O)), 'split': str((d / 'SPLIT.json').relative_to(O)), 'test_predictions': str((d / 'test_predictions.json').relative_to(O)), 'validation_predictions': str((d / 'validation_predictions.json').relative_to(O)), 'protocol': str((d / 'protocol.json').relative_to(O)), 'confusion_file': str((d / 'CURRENT_CONFUSION.json').relative_to(O)), 'model': str(model.relative_to(O)) if model.exists() else None, 'model_sha256': modelsha, 'decision_model': str((d / 'protocol.json').relative_to(O)) if not model.exists() else None, 'threshold': selected, 'test_n': len(pr), 'confusion_matrix': c, 'minimum_distance_to_selected_threshold': float(np.min(abs(p - selected))), 'observed_native_max_difference': maxdelta}
        index.append(entry)
        audit.append({'series': r['series'], 'seed': r['seed'], 'fold': r['fold'], 'threshold_recalculated': True, 'metrics_recalculated': True, 'participant_split_checked': True, 'model_sha256_checked': model.exists(), 'native_parity_max_difference': maxdelta})
        byseries.setdefault(r['series'], {}).setdefault(r['seed'], []).extend(pr)
assert len(index) == 390
pooled = []
summaries = []
for series, seeds in sorted(byseries.items()):
    assert set(seeds) == {42, 43, 44}
    vectors = []
    for seed, rows in sorted(seeds.items()):
        rows = sorted(rows, key=lambda r: r['subject_id'])
        assert len(rows) == 42 and len({r['subject_id'] for r in rows}) == 42
        y = np.array([r['true_label'] for r in rows])
        p = np.array([r['probability'] for r in rows])
        h = np.array([r['predicted_label'] for r in rows], dtype=bool)
        assert all((r['predicted_label'] == int(r['probability'] >= r['selected_threshold']) for r in rows))
        m, c = classification(y, p, h)
        fix, fc = classification(y, p, p >= 0.5)
        vectors.append(h)
        pooled.append({'series': series, 'seed': seed, 'n': 42, 'fixed_metrics': fix, 'fixed_confusion_matrix': fc, 'calibrated_metrics': m, 'calibrated_confusion_matrix': c, 'fixed_ci95': ci(y, p >= 0.5), 'calibrated_ci95': ci(y, h), 'incorrect_subjects': [r for r in rows if r['predicted_label'] != r['true_label']], 'false_negatives': [r for r in rows if r['true_label'] == 1 and r['predicted_label'] == 0], 'predictions': rows})
    rr = [r for r in pooled if r['series'] == series]
    summary = {'series': series, 'classification_changes_across_seeds': int(np.any(np.array(vectors) != vectors[0], axis=0).sum()), 'fixed': {k: {'mean': float(np.mean([r['fixed_metrics'][k] for r in rr])), 'sd': float(np.std([r['fixed_metrics'][k] for r in rr], ddof=1))} for k in NAMES}, 'validation_calibrated': {k: {'mean': float(np.mean([r['calibrated_metrics'][k] for r in rr])), 'sd': float(np.std([r['calibrated_metrics'][k] for r in rr], ddof=1))} for k in NAMES}}
    summaries.append(summary)
save('MODEL_RESULT_INDEX.json', index)
save('AUDIT.json', audit)
save('POOLED_PER_SEED.json', pooled)
save('SUMMARY.json', summaries)
duplicates = []
for seed in [42, 43, 44]:
    group = [r for r in pooled if r['seed'] == seed]
    for i, a in enumerate(group):
        for b in group[i + 1:]:
            pa = np.array([r['probability'] for r in a['predictions']])
            pb = np.array([r['probability'] for r in b['predictions']])
            assert [r['subject_id'] for r in a['predictions']] == [r['subject_id'] for r in b['predictions']]
            if np.array_equal(pa, pb):
                duplicates.append({'seed': seed, 'series_a': a['series'], 'series_b': b['series'], 'identical_probability_vectors': True})
save('IDENTICAL_PREDICTIONS_AUDIT.json', duplicates)
s = '<!doctype html><html lang="en"><meta charset="utf-8"><title>All last-layer experiments, 3x5</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:15px}table{border-collapse:collapse;width:100%;margin:25px 0}td,th{border:1px solid #aaa;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}.warning{padding:18px;background:#fff0cb}pre{white-space:pre-wrap}small{color:#444}</style><h1>All inherited last-layer experiments: three seeds x five partitions</h1><p class="warning"><b></b> These experiments fine-tune inherited last dense layers and classify using thresholds selected exclusively on validation children. All runs retained.  This exploratory protocol was expanded after the MRI pilot; it is not a preregistered confirmatory evaluation.</p><p>14 source branch series x 15 runs = 210 branch runs (15 MRI Baseline runs reused; 195 newly fitted). Two modes x six fusion methods x 15 partitions = 180 fusion evaluations. Names Agentic, Rule, Random, Optuna and scratch-source describe the inherited sources: no new Qwen calls, HPO or fresh full-network from-scratch training in this experiment.</p><p>MRI: fit Dense128 and prediction. fMRI: fit Dense64 and prediction; preceding BiLSTM remains frozen. Dropout inactive. fMRI uses the mean of three run probabilities per child. Every nominal test partition contains only its 8 or 9 children. </p><p>Fusion includes equal soft voting, fixed 0.4/0.6 voting, validation-weighted voting, hard voting, stacking and feature fusion. Stacking uses in-sample training-branch probabilities, not nested OOF. Hard-vote scores are vote fractions, not calibrated diagnostic probabilities. Saved stacking/feature Keras heads accept cached inputs; branch Keras models and normalization files are linked through protocols.</p>'
for typ, label in [('validation_calibrated', 'Validation-selected threshold: mean +/- sample SD across 3 seeds'), ('fixed', 'Fixed 0.5 threshold: mean +/- sample SD across 3 seeds')]:
    s += f'<h2>{label}</h2><table><tr><th>Series</th>' + ''.join((f'<th>{k}</th>' for k in NAMES)) + '<th>Changed children between seeds</th></tr>'
    for r in summaries:
        s += '<tr><td>' + html.escape(r['series']) + '</td>'
        for k in NAMES:
            a = r[typ][k]
            scale = 100 if k in ['accuracy', 'f1', 'precision', 'recall', 'specificity'] else 1
            s += f"<td>{a['mean'] * scale:.4f} +/- {a['sd'] * scale:.4f}</td>"
        s += f"<td>{(r['classification_changes_across_seeds'] if typ == 'validation_calibrated' else 'See pooled predictions')}</td></tr>"
    s += '</table><small>Accuracy, F1, precision, Recall and specificity are percentages; their SD is in percentage points. AUC, kappa, Brier and ECE use their native scales. Zero SD is retained when measured.</small>'
s += '<h2>Every seed</h2><table><tr><th>Series</th><th>Seed</th><th>Accuracy %</th><th>Recall %</th><th>F1 %</th><th>TP/TN/FP/FN</th><th>Accuracy 95% CI</th><th>F1 95% CI</th></tr>'
for r in pooled:
    m = r['calibrated_metrics']
    c = r['calibrated_confusion_matrix']
    a = r['calibrated_ci95']['accuracy']
    f = r['calibrated_ci95']['f1']
    s += f"<tr><td>{r['series']}</td><td>{r['seed']}</td><td>{m['accuracy'] * 100:.4f}</td><td>{m['recall'] * 100:.4f}</td><td>{m['f1'] * 100:.4f}</td><td>{c['TP']}/{c['TN']}/{c['FP']}/{c['FN']}</td><td>{a['low'] * 100:.2f}..{a['high'] * 100:.2f}</td><td>{f['low'] * 100:.2f}..{f['high'] * 100:.2f}</td></tr>"
s += '</table><p>Intervals are descriptive within-cohort intervals. Accuracy uses Wilson; F1 uses 10,000 subject-pair bootstrap resamples. They do not establish independent performance.</p><h2>Every partition and artifact</h2><table><tr><th>Series</th><th>Seed / partition</th><th>Test n</th><th>Threshold</th><th>TP/TN/FP/FN</th><th>Evidence</th></tr>'
for r in index:
    links = ' / '.join((f'<a href="{r[k]}">{k}</a>' for k in ['result', 'split', 'test_predictions', 'validation_predictions', 'protocol', 'confusion_file'] + ['model' if r['model'] else 'decision_model']))
    s += f"<tr><td>{r['series']}</td><td>{r['seed']} / {r['fold']}</td><td>{r['test_n']}</td><td>{r['threshold']:.6f}</td><td>{r['confusion_matrix']['TP']}/{r['confusion_matrix']['TN']}/{r['confusion_matrix']['FP']}/{r['confusion_matrix']['FN']}</td><td>{links}</td></tr>"
s += '</table><h2>Audits and reproducibility files</h2>'
for n in ['PLAN.json', 'MODEL_RESULT_INDEX.json', 'SUMMARY.json', 'POOLED_PER_SEED.json', 'AUDIT.json', 'IDENTICAL_PREDICTIONS_AUDIT.json', 'IMPORT_MRI_BASELINE_AUDIT.json', 'INPUT_SOURCE_COPY_MANIFEST.json', 'SOURCE_MODELS_REFERENCE.json', 'SOURCE_ENCODER_EQUIVALENCE.json', 'NUMERICAL_RUNTIME_FIX.json', 'VERIFICATION_RUNTIME_PLAN.json', 'FMRI_NATIVE_INPUT_REPLAY.json', 'README_EN.md', 'source/replay_branch.py', 'training.log', 'fusion_training.log', 'source/train_branches.py', 'source/train_fusion.py', 'source/evaluation_common.py', 'source/audit_delivery.py']:
    s += f'<p><a href="{n}">{n}</a></p>' if (O / n).exists() else f'<p>{n}: not recorded in this reproduction output.</p>'
reused = 15 if (O / 'IMPORT_MRI_BASELINE_AUDIT.json').exists() else 0
s = s.replace('15 MRI Baseline runs reused; 195 newly fitted', f'{reused} MRI Baseline runs reused; {210 - reused} newly fitted')
s += '</html>'
(O / 'START_ALL_LAST_LAYER_EN.html').write_text(s, encoding='utf8')
missing = [x for x in re.findall('href="([^"]+)"', s) if not (O / x).exists()]
assert not missing
assert not (O / 'RUN.lock').exists() and (not (O / 'FUSION.lock').exists())
save('DELIVERY_CHECK.json', {'runs_audited': 390, 'branch_models': 210, 'fusion_keras_heads': 60, 'fusion_decision_protocols': 120, 'complete_unique_child_coverage_per_seed': True, 'all_thresholds_recalculated_from_validation': True, 'metrics_recalculated': True, 'code_and_models_present': True, 'broken_links': missing, 'identical_prediction_pairs': len(duplicates)})
save('COMPLETE.json', {'branch_runs': 210, 'fusion_evaluations': 180, 'pooled_series_seed_records': len(pooled), 'series': len(summaries), 'audit_passed': True})
save('progress.json', {'stage': 'COMPLETE', 'branch_runs': 210, 'fusion_evaluations': 180, 'audit_passed': True})
print(json.dumps({'complete': True, 'series': len(summaries), 'pooled_records': len(pooled), 'audited_runs': 390}))
