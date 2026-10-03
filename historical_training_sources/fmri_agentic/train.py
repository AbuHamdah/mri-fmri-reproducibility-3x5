raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from statistics_core import metrics, NAMES
O = Path(__file__).resolve().parents[1]
A = Path('D:\\MRI_FMRI_FMRI_BATCH3_RUN_RELIABILITY_3X5_V1')
P = Path('D:\\MRI_FMRI_LAST_LAYER_ALL_3X5_V1')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')
plan = {'branches': ['fmri_baseline', 'fmri_agentic'], 'seeds': [42, 43, 44], 'folds': [1, 2, 3, 4, 5], 'learning_rates': [0.05, 0.15, 0.3], 'leaf_l2': [0.1, 1.0], 'staged_rounds': [5, 10, 20], 'minimum_train_children_per_leaf': 3, 'maximum_absolute_leaf_correction': 2.0, 'class_weight_rule': 'n_train/(2*n_train_class), computed separately inside each current training partition; validation metrics remain unweighted', 'method': 'Class-balanced Newton gradient-boosted decision stumps for true-label train BCE, residual to frozen parent logit,19 numeric input features', 'threshold_candidates': 'Midpoints of current training feature values only', 'selection': 'Validation balanced accuracy, minimum validation BCE, fewer trees, stronger leaf regularization, smaller learning rate', 'threshold': 0.5, 'current_test_fitting': False, 'repeatedly_examined_cohort': True, 'target_guarantee': False}
assert not (O / 'PLAN.json').exists()
write(O / 'PLAN.json', plan)
features = read(O / 'INPUT_FEATURES.json')['records']
pooled = []
summary = []
completed = 0
for branch in plan['branches']:
    ca = read(A / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json')
    write(O / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json', ca)
    ids = ca['subject_ids']
    assert ids == [r['subject_id'] for r in features] and ca['labels'] == [r['true_label'] for r in features]
    y = np.array(ca['labels'])
    source = np.array(ca['three_run_probabilities'])
    x = np.c_[source, np.array([r['features'] for r in features])]
    q = np.clip(source.astype('float32').mean(1).astype('float64'), 1e-12, 1 - 1e-12)
    offset = np.log(q / (1 - q))
    records = []
    for seed in plan['seeds']:
        whole = []
        for fold in plan['folds']:
            d = O / 'branches' / branch / f'seed{seed}_fold{fold}'
            s = read(P / 'branches' / branch / f'seed{seed}_fold{fold}' / 'SPLIT.json')
            sets = [set(s[k]) for k in ['train', 'validation', 'test']]
            assert all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
            tr = np.array([ids.index(z) for z in s['train']])
            vi = np.array([ids.index(z) for z in s['validation']])
            counts = np.bincount(y[tr], minlength=2)
            assert np.all(counts > 0)
            sample_weights = len(tr) / (2 * counts[y[tr]])
            cuts = []
            for j in range(19):
                values = np.unique(x[tr, j])
                thresholds = (values[:-1] + values[1:]) / 2
                for t in thresholds:
                    left = x[tr, j] < t
                    if left.sum() >= 3 and (~left).sum() >= 3:
                        cuts.append({'feature': j, 'threshold': float(t)})
            assert cuts
            mask = np.stack([x[:, c['feature']] < c['threshold'] for c in cuts], axis=1)
            M = mask[tr].astype('float64')
            candidates = []

            def add(name, lr, l2, trees, z, trace):
                p = expit(z)
                m = metrics(y[vi], p[vi])
                c = {'name': name, 'learning_rate': lr, 'leaf_l2': l2, 'trees': [dict(t) for t in trees], 'validation_balanced_accuracy': float((m[5] + m[6]) / 2), 'validation_bce': float(np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi]))}
                write(d / 'candidates' / name / 'CHECKPOINT.json', c)
                write(d / 'candidates' / name / 'TRAINING_CURVE.json', trace)
                candidates.append(c)
            add('identity', 0.0, None, [], offset, [])
            for lr in plan['learning_rates']:
                for l2 in plan['leaf_l2']:
                    z = offset.copy()
                    trees = []
                    trace = []
                    for epoch in range(1, 21):
                        p = expit(z[tr])
                        g = sample_weights * (y[tr] - p)
                        h = sample_weights * p * (1 - p)
                        GL = g @ M
                        HL = h @ M
                        GR = g.sum() - GL
                        HR = h.sum() - HL
                        gain = GL * GL / (HL + l2) + GR * GR / (HR + l2) - g.sum() ** 2 / (h.sum() + l2)
                        k = int(np.argmax(gain))
                        left = float(np.clip(GL[k] / (HL[k] + l2), -2, 2))
                        right = float(np.clip(GR[k] / (HR[k] + l2), -2, 2))
                        tree = {**cuts[k], 'left_correction': left, 'right_correction': right, 'gain': float(gain[k]), 'train_left_children': int(M[:, k].sum()), 'train_right_children': int(len(tr) - M[:, k].sum())}
                        trees.append(tree)
                        z += lr * np.where(mask[:, k], left, right)
                        trace.append({'round': epoch, 'training_bce': float(np.mean(sample_weights * (np.logaddexp(0, z[tr]) - y[tr] * z[tr])))})
                        if epoch in plan['staged_rounds']:
                            add(f'lr{lr}_l2{l2}_rounds{epoch}', lr, l2, trees, z, trace.copy())
            chosen = max(candidates, key=lambda c: (c['validation_balanced_accuracy'], -c['validation_bce'], -len(c['trees']), float('inf') if c['leaf_l2'] is None else c['leaf_l2'], -c['learning_rate']))
            z = offset.copy()
            for t in chosen['trees']:
                z += chosen['learning_rate'] * np.where(x[:, t['feature']] < t['threshold'], t['left_correction'], t['right_correction'])
            p = expit(z)
            rows = [{'subject_id': sid, 'true_label': int(y[i]), 'probability': float(p[i]), 'predicted_label': int(p[i] >= 0.5), 'seed': seed, 'test_fold': fold, 'threshold': 0.5} for i, sid in enumerate(ids)]
            by = {r['subject_id']: r for r in rows}
            write(d / 'SPLIT.json', s)
            write(d / 'ALLOWED_TRAIN_SPLITS.json', {'fit_ids': s['train'], 'cuts': cuts})
            write(d / 'ALL_SUBJECT_PREDICTIONS.json', rows)
            write(d / 'test_predictions.json', [by[z] for z in s['test']])
            write(d / 'validation_predictions.json', [by[z] for z in s['validation']])
            write(d / 'VALIDATION_CANDIDATES.json', candidates)
            write(d / 'SELECTED_CHECKPOINT.json', chosen)
            whole += [by[z] for z in s['test']]
            completed += 1
            write(O / 'progress.json', {'stage': 'FIT_COMPLETE', 'completed': completed, 'total': 30})
            print(json.dumps({'completed': completed, 'branch': branch, 'seed': seed, 'fold': fold, 'chosen': chosen['name']}), flush=True)
        assert len(whole) == 42 and len({r['subject_id'] for r in whole}) == 42
        r = {'series': branch, 'seed': seed, 'metrics': dict(zip(NAMES, map(float, metrics([r['true_label'] for r in whole], [r['probability'] for r in whole])))), 'predictions': whole}
        pooled.append(r)
        records.append(r)
    summary.append({'series': branch, 'metrics': {k: {'mean': float(np.mean([r['metrics'][k] for r in records])), 'sd': float(np.std(np.array([r['metrics'][k] for r in records]) - records[0]['metrics'][k], ddof=1))} for k in NAMES}})
write(O / 'SUMMARY.json', summary)
write(O / 'POOLED_PER_SEED.json', pooled)
write(O / 'COEFFICIENTS_COMPLETE.json', {'selected_protocols': 30, 'retained_fitted_candidate_checkpoints': 540})
print(json.dumps(summary))
