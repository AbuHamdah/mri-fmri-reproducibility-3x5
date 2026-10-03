raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import tensorflow as tf
from statistics_core import metrics, NAMES
P = Path('/package')
O = Path('/out')
read = lambda p: json.loads(p.read_text())
plan = read(O / 'PLAN.json')
start = time.time()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2))
projection = np.linalg.qr(np.random.default_rng(plan['projection_seed']).normal(size=(128, plan['projection_dimensions'])))[0]
np.save(O / 'FEATURE_PROJECTION.npy', projection)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
parent = tf.keras.models.load_model(P / plan['source_model'], compile=False)
parent.trainable = False
encoder = tf.keras.Model(parent.inputs, [parent.output, parent.get_layer('features128').output])
encoder.trainable = False
inputs = sorted([r for r in read(P / 'INPUTS.json') if r['kind'] == 'mri'], key=lambda r: r['subject_id'])
assert len(inputs) == 42
source = []
F = []
for r in inputs:
    a = np.load(P / r['path'])
    assert np.isfinite(a).all()
    p, f = encoder(a[None], training=False)
    source.append({**r, 'input_sha256': sha(P / r['path']), 'source_probability': float(p.numpy().ravel()[0])})
    F.append(f.numpy().ravel().astype('float64'))
    print(json.dumps({'source_children': len(source), 'total': 42}), flush=True)
F = np.array(F)
assert F.shape == (42, 128) and np.isfinite(F).all()
np.save(O / 'SOURCE_FEATURES.npy', F)
write(O / 'SOURCE_NATIVE_PREDICTIONS.json', {'source_model': plan['source_model'], 'source_sha256': sha(P / plan['source_model']), 'records': source, 'source_metrics': dict(zip(NAMES, map(float, metrics([r['label'] for r in source], [r['source_probability'] for r in source]))))})
ids = [r['subject_id'] for r in source]
y = np.array([r['label'] for r in source])
q = np.clip([r['source_probability'] for r in source], plan['clip'], 1 - plan['clip'])
offset = np.log(q / (1 - q))
pooled = []
native = []
completed = 0
for seed in plan['seeds']:
    whole = []
    for fold in plan['folds']:
        d = O / 'branches' / 'mri_agentic' / f'seed{seed}_fold{fold}'
        s = read(P / 'branches' / 'mri_agentic' / f'seed{seed}_fold{fold}' / 'SPLIT.json')
        sets = [set(s[k]) for k in ['train', 'validation', 'test']]
        assert all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
        tr = np.array([ids.index(k) for k in s['train']])
        vi = np.array([ids.index(k) for k in s['validation']])
        mean = F[tr].mean(0)
        std = np.maximum(F[tr].std(0), 0.01)
        X = np.c_[(F - mean) / std @ projection, np.ones(42)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 1017]))
        draws = rng.exponential(size=(plan['bootstrap_draws'], len(tr)))
        counts = np.bincount(y[tr], minlength=2)
        assert np.all(counts > 0)
        draws *= len(tr) / (2 * counts[y[tr]])
        draws *= len(tr) / draws.sum(axis=1, keepdims=True)
        sw = draws.mean(axis=0)
        write(d / 'SPLIT.json', s)
        write(d / 'NORMALIZATION.json', {'fit_ids': s['train'], 'mean': mean.tolist(), 'std': std.tolist(), 'std_floor': 0.01})
        write(d / 'BOOTSTRAP_TRAIN_WEIGHTS.json', {'seed_sequence': [seed, fold, 1017], 'subject_ids': s['train'], 'weights': sw.tolist(), 'normalized_draws': draws.tolist()})
        cc = []
        for lam in [None] + plan['regularization_grid']:
            w = np.zeros(9)
            trace = []
            if lam is not None:

                def objective(w):
                    z = offset[tr] + X[tr] @ w
                    return (float(np.mean(sw * (np.logaddexp(0, z) - y[tr] * z)) + lam / 2 * np.dot(w, w)), X[tr].T @ (sw * (expit(z) - y[tr])) / len(tr) + lam * w)

                def callback(w):
                    trace.append({'iteration': len(trace) + 1, 'training_objective': objective(w)[0]})
                opt = minimize(objective, w, jac=True, method='L-BFGS-B', callback=callback, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-09})
                w = opt.x
                assert np.max(abs(objective(w)[1])) < 1e-05
            z = offset + X @ w
            p = expit(z)
            m = metrics(y[vi], p[vi])
            name = 'identity' if lam is None else str(lam)
            c = {'name': name, 'lambda': lam, 'coefficients': w.tolist(), 'validation_balanced_accuracy': float((m[5] + m[6]) / 2), 'validation_bce': float(np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi]))}
            write(d / 'candidates' / name / 'COEFFICIENTS.json', c)
            write(d / 'candidates' / name / 'OPTIMIZER_HISTORY.json', trace)
            cc.append(c)
        write(d / 'BASE_CANDIDATES.json', cc)
        expanded = []
        for base in cc:
            for alpha in [0.0] if base['lambda'] is None else plan['residual_strength_grid']:
                w = np.array(base['coefficients']) * alpha
                z = offset + X @ w
                p = expit(z)
                m = metrics(y[vi], p[vi])
                expanded.append({'name': base['name'], 'lambda': base['lambda'], 'alpha': alpha, 'coefficients': w.tolist(), 'validation_balanced_accuracy': float((m[5] + m[6]) / 2), 'validation_bce': float(np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi]))})
        cc = expanded
        chosen = max(cc, key=lambda c: (c['validation_balanced_accuracy'], -c['validation_bce'], -c['alpha'], float('inf') if c['lambda'] is None else c['lambda']))
        write(d / 'VALIDATION_CANDIDATES.json', cc)
        write(d / 'SELECTED_COEFFICIENTS.json', chosen)
        w = np.array(chosen['coefficients'])
        inp = tf.keras.Input(parent.input_shape[1:], dtype='float32', name='actual_processed_mri')
        raw, features = encoder(inp, training=False)
        norm = tf.keras.layers.Dense(128, dtype='float64', trainable=False, name='training_feature_normalization')
        normalized = norm(features)
        norm.set_weights([np.diag(1 / std), -mean / std])
        proj = tf.keras.layers.Dense(8, dtype='float64', use_bias=False, trainable=False, name='fixed_label_independent_projection')
        projected = proj(normalized)
        proj.set_weights([projection])
        res = tf.keras.layers.Dense(1, dtype='float64', trainable=False, name='trained_residual')
        delta = res(projected)
        res.set_weights([w[:-1, None], w[-1:]])
        raw = tf.cast(raw, tf.float64)
        clipped = tf.clip_by_value(raw, plan['clip'], 1 - plan['clip'])
        comp = tf.keras.layers.Dense(1, dtype='float64', trainable=False, name='probability_complement')
        cp = comp(clipped)
        comp.set_weights([np.array([[-1.0]]), np.array([1.0])])
        logit = tf.keras.layers.Subtract(dtype='float64')([tf.math.log(clipped), tf.math.log(cp)])
        out = tf.keras.layers.Activation('sigmoid', dtype='float64')(tf.keras.layers.Add(dtype='float64')([logit, delta]))
        full = tf.keras.Model(inp, out)
        full.trainable = False
        full.save(d / 'model.keras', save_format='keras_v3')
        loaded = tf.keras.models.load_model(d / 'model.keras', compile=False)
        p = expit(offset + X @ w)
        rows = [{'subject_id': sid, 'true_label': int(y[i]), 'probability': float(p[i]), 'predicted_label': int(p[i] >= 0.5), 'input_path': source[i]['path'], 'input_sha256': source[i]['input_sha256'], 'seed': seed, 'test_fold': fold, 'threshold': 0.5} for i, sid in enumerate(ids)]
        by = {r['subject_id']: r for r in rows}
        write(d / 'ALL_SUBJECT_PREDICTIONS.json', rows)
        write(d / 'test_predictions.json', [by[k] for k in s['test']])
        write(d / 'validation_predictions.json', [by[k] for k in s['validation']])
        replays = []
        for sid in s['test']:
            row = by[sid]
            actual = float(loaded(np.load(P / row['input_path'])[None], training=False).numpy().ravel()[0])
            err = abs(actual - row['probability'])
            assert err < plan['native_probability_tolerance'] and int(actual >= 0.5) == row['predicted_label']
            replays.append({'subject_id': sid, 'probability': actual, 'difference': err})
        write(d / 'NATIVE_REPLAY.json', replays)
        native.append({'seed': seed, 'fold': fold, 'model_sha256': sha(d / 'model.keras'), 'maximum_processed_input_replay_difference': max((r['difference'] for r in replays)), 'test_children_replayed': len(replays), 'selected_candidate': chosen['name']})
        whole += [by[k] for k in s['test']]
        completed += 1
        write(O / 'progress.json', {'completed': completed, 'total': 15, 'stage': 'FEATURE_RESIDUAL_NATIVE_REPLAY'})
        print(json.dumps({'completed': completed, 'seed': seed, 'fold': fold, 'selected': chosen['name']}), flush=True)
        del loaded, full
    assert len(whole) == 42 and len({r['subject_id'] for r in whole}) == 42
    pooled.append({'series': 'mri_agentic', 'seed': seed, 'metrics': dict(zip(NAMES, map(float, metrics([r['true_label'] for r in whole], [r['probability'] for r in whole])))), 'predictions': whole})
summary = [{'series': 'mri_agentic', 'metrics': {k: {'mean': float(np.mean([r['metrics'][k] for r in pooled])), 'sd': float(np.std(np.array([r['metrics'][k] for r in pooled]) - pooled[0]['metrics'][k], ddof=1))} for k in NAMES}}]
write(O / 'SUMMARY.json', summary)
write(O / 'POOLED_PER_SEED.json', pooled)
write(O / 'NATIVE_REPLAY.json', {'passed': True, 'saved_full_models': 15, 'actual_processed_test_children_replayed': 126, 'source_children_replayed': 42, 'records': native})
write(O / 'TRAINING_COMPLETE.json', {'completed': True, 'selected_models': 15, 'actual_seconds': time.time() - start, 'new_qwen_calls': False})
print(json.dumps(summary))
