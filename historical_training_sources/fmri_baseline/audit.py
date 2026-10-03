import json, hashlib, math, re
from pathlib import Path
import numpy as np
from scipy.special import expit
from statistics_core import metrics, NAMES
O = Path(__file__).resolve().parents[1]
A = Path('D:\\MRI_FMRI_FMRI_BATCH3_RUN_RELIABILITY_3X5_V1')
R = Path('D:\\MRI_FMRI_FMRI_FOLLOWUP_REVIEW_V1')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(O / 'PLAN.json')
summary = []
pooled = []
index = []
identity = 0
for branch in plan['branches']:
    ca = read(O / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json')
    assert ca == read(A / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json')
    ids = ca['subject_ids']
    raw = np.array(ca['three_run_probabilities'])
    y = np.array(ca['labels'])
    q = np.clip(raw.astype('float32').mean(1).astype('float64'), 1e-12, 1 - 1e-12)
    offset = np.log(q / (1 - q))
    seed_records = []
    for seed in plan['seeds']:
        whole = []
        for fold in plan['folds']:
            d = O / 'branches' / branch / f'seed{seed}_fold{fold}'
            s = read(d / 'SPLIT.json')
            protocol = read(d / 'protocol.json')
            norm = read(d / 'NORMALIZATION.json')
            candidates = read(d / 'VALIDATION_CANDIDATES.json')
            chosen = read(d / 'SELECTED_COEFFICIENTS.json')
            tr = np.array([ids.index(z) for z in s['train']])
            vi = np.array([ids.index(z) for z in s['validation']])
            sets = [set(s[k]) for k in ['train', 'validation', 'test']]
            assert all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
            assert set.union(*sets) == set(ids)
            mean = raw[tr].mean(0)
            std = np.maximum(raw[tr].std(0), 1e-06)
            z = (raw - mean) / std
            centers = z[tr]
            assert norm['fit_ids'] == norm['center_subject_ids'] == protocol['actual_fitting_ids'] == s['train']
            assert np.array_equal(mean, norm['mean']) and np.array_equal(std, norm['std']) and np.array_equal(centers, norm['normalized_centers'])
            assert len(candidates) == 57
            assert {(c['gamma'], c['lambda']) for c in candidates} == {(None, None)} | {(g, l) for g in plan['gamma_grid'] for l in plan['lambda_grid']}
            for c in candidates:
                g = c['gamma']
                K = np.zeros((42, len(tr) + 1)) if g is None else np.c_[np.exp(-g * np.mean((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)), np.ones(42)]
                w = np.array(c['coefficients'])
                logit = offset + K @ w
                p = expit(logit)
                m = metrics(y[vi], p[vi])
                bce = float(np.mean(np.logaddexp(0, logit[vi]) - y[vi] * logit[vi]))
                assert abs(c['validation_accuracy'] - m[0]) < 1e-12 and abs(c['validation_bce'] - bce) < 1e-12
                if g is not None:
                    assert np.max(abs(K[tr].T @ (p[tr] - y[tr]) / len(tr) + c['lambda'] * w)) < 1e-05
            expected = max(candidates, key=lambda c: (c['validation_accuracy'], -c['validation_bce'], float('inf') if c['lambda'] is None else c['lambda'], 0 if c['gamma'] is None else -c['gamma']))
            assert chosen == expected and chosen['name'] == protocol['selected_candidate']
            identity += chosen['gamma'] is None
            g = chosen['gamma']
            K = np.zeros((42, len(tr) + 1)) if g is None else np.c_[np.exp(-g * np.mean((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)), np.ones(42)]
            p = expit(offset + K @ chosen['coefficients'])
            rows = read(d / 'ALL_SUBJECT_PREDICTIONS.json')
            by = {r['subject_id']: r for r in rows}
            for r in rows:
                i = ids.index(r['subject_id'])
                assert r['true_label'] == y[i] and abs(r['probability'] - p[i]) < 1e-12 and (r['predicted_label'] == int(p[i] >= 0.5))
            test = read(d / 'test_predictions.json')
            assert test == [by[z] for z in s['test']]
            m = dict(zip(NAMES, map(float, metrics([r['true_label'] for r in test], [r['probability'] for r in test]))))
            assert m == read(d / 'RESULT.json')['metrics']
            yy = np.array([r['true_label'] for r in test])
            hh = np.array([r['predicted_label'] for r in test])
            conf = {k: int(v.sum()) for k, v in [('tp', (yy == 1) & (hh == 1)), ('tn', (yy == 0) & (hh == 0)), ('fp', (yy == 0) & (hh == 1)), ('fn', (yy == 1) & (hh == 0))]}
            write(d / 'CONFUSION.json', {'counts': conf, 'n': len(test), 'errors': [r for r in test if r['true_label'] != r['predicted_label']]})
            index.append({'series': branch, 'seed': seed, 'fold': fold, 'selected_candidate': chosen['name'], 'split': (d / 'SPLIT.json').relative_to(O).as_posix(), 'model': (d / 'model.keras').relative_to(O).as_posix(), 'source_model': ca['source_model'], 'source_sha256': ca['source_model_sha256']})
            whole += test
        assert len(whole) == 42 and len({r['subject_id'] for r in whole}) == 42
        y0 = np.array([r['true_label'] for r in whole])
        p0 = np.array([r['probability'] for r in whole])
        h = p0 >= 0.5
        m = dict(zip(NAMES, map(float, metrics(y0, p0))))
        n = 42
        ph = m['accuracy']
        zz = 1.959963984540054
        den = 1 + zz * zz / n
        c = (ph + zz * zz / (2 * n)) / den
        hw = zz * math.sqrt(ph * (1 - ph) / n + zz * zz / (4 * n * n)) / den
        rng = np.random.default_rng(42)
        ii = rng.integers(0, n, (10000, n))
        tp = ((y0[ii] == 1) & h[ii]).sum(1)
        fp = ((y0[ii] == 0) & h[ii]).sum(1)
        fn = ((y0[ii] == 1) & ~h[ii]).sum(1)
        f = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        record = {'series': branch, 'seed': seed, 'metrics': m, 'confidence95': {'accuracy': {'low': c - hw, 'high': c + hw, 'method': 'Wilson95%'}, 'f1': {'low': float(np.quantile(f, 0.025)), 'high': float(np.quantile(f, 0.975)), 'method': 'Paired child percentile bootstrap,10000 resamples seed42'}, 'scope': ''}, 'predictions': whole}
        pooled.append(record)
        seed_records.append(record)
    summary.append({'series': branch, 'metrics': {k: {'mean': float(np.mean([r['metrics'][k] for r in seed_records])), 'sd': float(np.std(np.array([r['metrics'][k] for r in seed_records]) - seed_records[0]['metrics'][k], ddof=1))} for k in NAMES}})
write(O / 'SUMMARY.json', summary)
write(O / 'POOLED_PER_SEED.json', pooled)
write(O / 'MODEL_RESULT_INDEX.json', index)
write(O / 'COEFFICIENT_AUDIT.json', {'passed': True, 'selected_protocols': 30, 'selected_identity_controls': identity, 'retained_fitted_candidate_coefficients': 1680, 'train_only_normalization_and_centers_verified': True, 'training_gradient_verified': True, 'validation_choice_verified': True, 'current_test_excluded_from_new_fitting': True})
print(json.dumps(summary))
