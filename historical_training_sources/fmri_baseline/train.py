raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import json, hashlib, time, traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from statistics_core import metrics, NAMES
O = Path(__file__).resolve().parents[1]
A = Path('D:\\MRI_FMRI_FMRI_BATCH3_RUN_RELIABILITY_3X5_V1')
P = Path('D:\\MRI_FMRI_LAST_LAYER_ALL_3X5_V1')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
plan = {'method': 'Train-only RBF residual correction to frozen parent child logit', 'branches': ['fmri_baseline', 'fmri_agentic'], 'seeds': [42, 43, 44], 'folds': [1, 2, 3, 4, 5], 'gamma_grid': [0.001, 0.01, 0.1, 0.5, 2.0, 20.0, 200.0], 'lambda_grid': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], 'normalization': 'Training children only; std floor1e-6', 'centers': 'Every current training child normalized three-run probability vector', 'objective': 'True-label train BCE plus lambda/2 squared residual coefficients, including intercept; frozen native parent logit offset', 'selection': 'Validation accuracy, then minimum validation BCE, stronger regularization, smaller gamma; unfitted identity is strongest', 'threshold': 0.5, 'current_test_fitting': False, 'exploratory_examined_cohort': True, 'target_guarantee': False, 'all_candidates_retained': True}
assert not (O / 'PLAN.json').exists(), 'Refuse accidental repeat'
write(O / 'PLAN.json', plan)
lock = O / 'RUN.lock'
lock.open('x').write('host_train')
start = time.time()
completed = 0
try:
    assert read(A / 'AUDIT.json')['passed'] and read(A / 'FULL_RAW_REPLAY_AUDIT.json')['passed']
    for branch in plan['branches']:
        ca = read(A / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json')
        assert sha(P / ca['source_model']) == ca['source_model_sha256']
        write(O / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json', ca)
        ids = ca['subject_ids']
        raw = np.array(ca['three_run_probabilities'])
        y = np.array(ca['labels'])
        p = raw.astype('float32').mean(1).astype('float64')
        q = np.clip(p, 1e-12, 1 - 1e-12)
        offset = np.log(q / (1 - q))
        for seed in plan['seeds']:
            for fold in plan['folds']:
                tick = time.time()
                d = O / 'branches' / branch / f'seed{seed}_fold{fold}'
                s = read(P / 'branches' / branch / f'seed{seed}_fold{fold}' / 'SPLIT.json')
                sets = [set(s[k]) for k in ['train', 'validation', 'test']]
                assert all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
                tr = np.array([ids.index(z) for z in s['train']])
                vi = np.array([ids.index(z) for z in s['validation']])
                te = np.array([ids.index(z) for z in s['test']])
                mean = raw[tr].mean(0)
                std = np.maximum(raw[tr].std(0), 1e-06)
                norm = (raw - mean) / std
                centers = norm[tr]
                write(d / 'SPLIT.json', s)
                write(d / 'NORMALIZATION.json', {'mean': mean.tolist(), 'std': std.tolist(), 'fit_ids': s['train'], 'normalized_centers': centers.tolist(), 'center_subject_ids': s['train']})
                candidates = []
                for gamma, lam in [(None, None)] + [(g, l) for g in plan['gamma_grid'] for l in plan['lambda_grid']]:
                    name = 'identity' if gamma is None else f'gamma{gamma}_lambda{lam}'
                    cd = d / 'candidates' / name
                    cd.mkdir(parents=True, exist_ok=True)
                    trace = []
                    w = np.zeros(len(tr) + 1)
                    if gamma is None:
                        K = np.zeros((42, len(tr) + 1))
                        success = True
                        message = 'Unfitted identity'
                    else:
                        K = np.c_[np.exp(-gamma * np.mean((norm[:, None, :] - centers[None, :, :]) ** 2, axis=2)), np.ones(42)]

                        def objective(w):
                            z = offset[tr] + K[tr] @ w
                            loss = float(np.mean(np.logaddexp(0, z) - y[tr] * z) + 0.5 * lam * np.dot(w, w))
                            gradient = K[tr].T @ (expit(z) - y[tr]) / len(tr) + lam * w
                            return (loss, gradient)

                        def callback(w):
                            trace.append({'iteration': len(trace) + 1, 'train_objective': objective(w)[0]})
                        fit = minimize(objective, w, jac=True, method='L-BFGS-B', callback=callback, options={'maxiter': 1000, 'gtol': 1e-09, 'ftol': 1e-12})
                        if not fit.success:
                            write(cd / 'INITIAL_OPTIMIZER_FAILURE.json', {'message': str(fit.message), 'coefficients': fit.x.tolist()})
                            fit = minimize(objective, fit.x, jac=True, method='BFGS', callback=callback, options={'maxiter': 1000, 'gtol': 1e-07})
                        assert np.isfinite(fit.x).all() and np.max(abs(objective(fit.x)[1])) < 1e-05, (fit.message, objective(fit.x)[1])
                        w = fit.x
                        success = bool(fit.success)
                        message = str(fit.message)
                    z = offset + K @ w
                    pp = expit(z)
                    vp = pp[vi]
                    vm = dict(zip(NAMES, metrics(y[vi], vp)))
                    bce = float(np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi]))
                    co = {'name': name, 'gamma': gamma, 'lambda': lam, 'coefficients': w.tolist(), 'validation_accuracy': float(vm['accuracy']), 'validation_bce': bce, 'optimizer_success': success, 'optimizer_message': message}
                    write(cd / 'COEFFICIENTS.json', co)
                    write(cd / 'OPTIMIZER_HISTORY.json', trace)
                    candidates.append(co)
                chosen = max(candidates, key=lambda c: (c['validation_accuracy'], -c['validation_bce'], float('inf') if c['lambda'] is None else c['lambda'], 0 if c['gamma'] is None else -c['gamma']))
                g = chosen['gamma']
                K = np.zeros((42, len(tr) + 1)) if g is None else np.c_[np.exp(-g * np.mean((norm[:, None, :] - centers[None, :, :]) ** 2, axis=2)), np.ones(42)]
                pp = expit(offset + K @ chosen['coefficients'])
                rows = [{'subject_id': sid, 'true_label': int(y[i]), 'probability': float(pp[i]), 'predicted_label': int(pp[i] >= 0.5), 'source_run_probabilities': raw[i].tolist(), 'parent_child_logit': float(offset[i]), 'seed': seed, 'test_fold': fold, 'model_id': f'{branch}_seed{seed}_fold{fold}_rbf_residual', 'selected_threshold': 0.5} for i, sid in enumerate(ids)]
                by = {r['subject_id']: r for r in rows}
                write(d / 'ALL_SUBJECT_PREDICTIONS.json', rows)
                write(d / 'test_predictions.json', [by[z] for z in s['test']])
                write(d / 'validation_predictions.json', [by[z] for z in s['validation']])
                write(d / 'VALIDATION_CANDIDATES.json', candidates)
                write(d / 'SELECTED_COEFFICIENTS.json', chosen)
                write(d / 'protocol.json', {'source_model': ca['source_model'], 'source_sha256': ca['source_model_sha256'], 'actual_fitting_ids': s['train'], 'normalization_fit_ids': s['train'], 'validation_ids': s['validation'], 'test_ids': s['test'], 'threshold': 0.5, 'selected_candidate': chosen['name'], 'head_input': 'Three source native batch3 run probabilities plus logit of float32 mean, clipped1e-12', 'source_child_batch_size': 3, 'new_test_label_fitting': False, 'seconds': time.time() - tick})
                write(d / 'RESULT.json', {'series': branch, 'seed': seed, 'fold': fold, 'metrics': dict(zip(NAMES, map(float, metrics(y[te], pp[te])))), 'selected_candidate': chosen['name']})
                completed += 1
                write(O / 'progress.json', {'stage': 'COEFFICIENTS_FIT', 'completed': completed, 'total': 30})
                print(json.dumps({'completed': completed, 'branch': branch, 'seed': seed, 'fold': fold, 'chosen': chosen['name']}), flush=True)
    write(O / 'COEFFICIENTS_COMPLETE.json', {'completed': 30, 'actual_seconds': time.time() - start})
except Exception as e:
    write(O / 'FAILED.json', {'error': repr(e), 'traceback': traceback.format_exc()})
    raise
finally:
    lock.unlink(missing_ok=True)
