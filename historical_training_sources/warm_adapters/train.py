raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import json, hashlib, time, html
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.special import expit
from scipy.optimize import minimize
from statistics_core import metrics, NAMES
R = Path('/experiment')

def read(p):
    return json.loads(p.read_text())

def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
a = np.load(R / 'inputs/FEATURES.npz')
ids = a['subject_ids'].tolist()
y = a['labels'].astype('float32')
plan = read(R / 'PLAN.json')
locks = []
runs = []
start = time.time()
for seed in [42, 43, 44]:
    for fold in range(1, 6):
        d = R / 'runs' / f'seed{seed}_fold{fold}'
        s = read(d / 'SPLIT.json')
        ix = {k: np.array([ids.index(v) for v in s[k]]) for k in ['train', 'validation', 'test']}
        tr = ix['train']
        va = ix['validation']
        assert not (set(tr) & set(va) or set(tr) & set(ix['test']) or set(va) & set(ix['test']))
        p = np.load(d / 'SOURCE_PROBABILITIES.npz')
        assert p['subject_ids'].tolist() == ids and np.array_equal(p['labels'], y)
        pair = p['pairs'].astype('float32')
        full = []
        for j, kind in enumerate(['mri', 'fmri']):
            tf.keras.backend.clear_session()
            runseed = seed * 1000 + fold * 10 + j
            tf.keras.utils.set_random_seed(runseed)
            rng = np.random.default_rng(runseed)
            boot = np.concatenate([rng.choice(tr[y[tr] == c], size=sum(y[tr] == c), replace=True) for c in [0, 1]])
            rng.shuffle(boot)
            X = a[kind].astype('float32')
            mean = X[tr].mean(0)
            std = np.maximum(X[tr].std(0), 0.01)
            prior = np.clip(pair[:, j], 1e-07, 1 - 1e-07)
            base = (np.log(prior) - np.log1p(-prior))[:, None]
            inp = tf.keras.Input((128,), name='frozen_features')
            b = tf.keras.Input((1,), name='frozen_parent_logit')
            v = tf.keras.layers.Normalization(mean=mean, variance=std ** 2)(inp)
            v = tf.keras.layers.Dense(16, activation='relu')(v)
            v = tf.keras.layers.Dropout(0.3)(v)
            v = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(v)
            res = tf.keras.layers.Rescaling(3.0)(tf.keras.layers.Activation('tanh')(v))
            out = tf.keras.layers.Activation('sigmoid')(tf.keras.layers.Add()([b, res]))
            m = tf.keras.Model([inp, b], out)
            m.add_loss(0.01 * tf.reduce_mean(tf.square(res)))
            m.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy')
            dest = d / kind
            dest.mkdir()
            init = m([X, base], training=False).numpy().ravel()
            diff = float(np.max(abs(init - pair[:, j])))
            assert diff < 1e-06
            m.save(dest / 'INITIAL_MODEL.keras')
            t = time.time()
            h = m.fit([X[boot], base[boot]], y[boot], epochs=25, batch_size=8, shuffle=True, verbose=0, validation_data=([X[va], base[va]], y[va]))
            m.save(dest / 'MODEL.keras')
            save(dest / 'TRAINING_HISTORY.json', {k: list(map(float, v)) for k, v in h.history.items()})
            save(dest / 'TRAINING_EVIDENCE.json', dict(seed=runseed, epochs=25, training_seconds=time.time() - t, bootstrap_ids=[ids[i] for i in boot], initial_parent_probability_difference=diff, optimizer_iterations=int(m.optimizer.iterations.numpy()), initial_model_sha256=sha(dest / 'INITIAL_MODEL.keras'), final_model_sha256=sha(dest / 'MODEL.keras')))
            pp = m([X, base], training=False).numpy().ravel()
            loaded = tf.keras.models.load_model(dest / 'MODEL.keras', compile=False)
            again = loaded([X, base], training=False).numpy().ravel()
            err = float(np.max(abs(pp - again)))
            assert err < 1e-06
            full.append(pp)
            save(dest / 'ALL_PREDICTIONS.json', [dict(subject_id=k, true_label=int(y[i]), probability=float(pp[i])) for i, k in enumerate(ids)])
            locks.append(dict(run=d.name, kind=kind, path=f'runs/{d.name}/{kind}/MODEL.keras', sha256=sha(dest / 'MODEL.keras'), reload_difference=err))
        newpair = np.column_stack(full)
        np.save(d / 'NEW_MODALITY_PROBABILITIES.npy', newpair)
        q = np.clip(newpair, 1e-07, 1 - 1e-07)
        z = np.log(q / (1 - q))
        Z = np.c_[z, np.ones(42)]
        w0 = np.array([0.5, 0.5, 0.0])
        rng = np.random.default_rng(seed * 1000 + fold * 10 + 2)
        draw = np.concatenate([rng.choice(tr[y[tr] == c], size=sum(y[tr] == c), replace=True) for c in [0, 1]])
        rng.shuffle(draw)

        def obj(w):
            v = Z[draw] @ w
            delta = w - w0
            return (float(np.mean(np.logaddexp(0, v) - y[draw] * v) + 0.05 * np.dot(delta, delta)), Z[draw].T @ (expit(v) - y[draw]) / len(draw) + 0.1 * delta)
        opt = minimize(obj, w0, jac=True, method='L-BFGS-B', options=dict(maxiter=1000, gtol=1e-09, ftol=1e-12))
        assert max(abs(obj(opt.x)[1])) < 1e-05
        save(d / 'STACKER.json', dict(coefficients=opt.x.tolist(), initial_coefficients=w0.tolist(), ridge=0.1, bootstrap_ids=[ids[i] for i in draw], fit_ids=s['train'], training_inputs='In-sample updated branch probabilities; not nested OOF', objective=float(opt.fun), gradient_max=float(max(abs(obj(opt.x)[1])))))
        runs.append((seed, fold, d, s))
        save(R / 'PROGRESS.json', dict(stage='training', completed=len(runs), total=15))
        print('Completed training', len(runs), 'of15', flush=True)
save(R / 'ALL_MODELS_FROZEN_BEFORE_TEST.json', dict(models=locks, stackers=[dict(run=d.name, sha256=sha(d / 'STACKER.json')) for _, _, d, _ in runs], current_test_metrics_generated=False))
pool = {}
foldmetrics = []
for seed, fold, d, s in runs:
    pair = np.load(d / 'NEW_MODALITY_PROBABILITIES.npy')
    q = np.clip(pair, 1e-07, 1 - 1e-07)
    Z = np.c_[np.log(q / (1 - q)), np.ones(42)]
    ps = {'Warm-adapter Soft Voting': pair.mean(1), 'Warm-adapter Logit Stacking': expit(Z @ read(d / 'STACKER.json')['coefficients'])}
    for method, p in ps.items():
        rows = [dict(subject_id=k, true_label=int(y[ids.index(k)]), probability=float(p[ids.index(k)]), predicted_label=int(p[ids.index(k)] >= 0.5), threshold=0.5, seed=seed, test_fold=fold, model_id=f'{d.name}_{method}') for k in s['test']]
        save(d / (('VOTING' if method.endswith('Voting') else 'STACKING') + '_TEST_PREDICTIONS.json'), rows)
        pool.setdefault((method, seed), []).extend(rows)
        foldmetrics.append(dict(method=method, seed=seed, fold=fold, metrics=dict(zip(NAMES, map(float, metrics([r['true_label'] for r in rows], [r['probability'] for r in rows]))))))
per = []
summary = []
for (method, seed), rows in pool.items():
    assert len(rows) == 42 and len({r['subject_id'] for r in rows}) == 42
    per.append(dict(method=method, seed=seed, metrics=dict(zip(NAMES, map(float, metrics([r['true_label'] for r in rows], [r['probability'] for r in rows])))), predictions=rows))
for method in ['Warm-adapter Soft Voting', 'Warm-adapter Logit Stacking']:
    rr = [r for r in per if r['method'] == method]
    summary.append(dict(method=method, metrics={k: dict(mean=float(np.mean([r['metrics'][k] for r in rr])), sd=float(np.std([r['metrics'][k] for r in rr], ddof=1))) for k in NAMES}))
save(R / 'SUMMARY.json', summary)
save(R / 'POOLED_PER_SEED.json', per)
save(R / 'FOLD_METRICS.json', foldmetrics)
assert all((sha(R / r['path']) == r['sha256'] for r in locks))
save(R / 'AUDIT.json', dict(passed=True, adapters=30, stackers=15, all_native_reloads_matched=True, initial_probabilities_match_frozen_parent=True, current_train_validation_test_disjoint=True, all_runs_retained=True, seconds=time.time() - start))
tab = ''.join(('<tr><td>' + r['method'] + '</td>' + ''.join(('<td>%.4f ± %.4f</td>' % (r['metrics'][k]['mean'] * 100, r['metrics'][k]['sd'] * 100) for k in ['accuracy', 'recall', 'f1'])) + '</tr>' for r in summary))
(R / 'REPORT_EN.html').write_text('<!doctype html><meta charset="utf-8"><title>Warm adapter voting and stacking</title><style>body{font-family:Arial;margin:24px}td,th{padding:8px;border:1px solid #ccc}</style><h1>Warm-adapter Soft Voting and Logit Stacking</h1><p>Historical Agentic-source branch checkpoints remain frozen. Their audited cached probabilities supply a fixed parent logit. New stochastic residual adapters begin at exactly those probabilities; original branch weights are not fine-tuned. Three seeds × five subject partitions. All runs retained. No target-score selection. No new Qwen calls. </p><p>Adapters: train-only normalization, Dense16/ReLU, Dropout0.3, zero-initialized output, bounded logit correction3, residual penalty0.01, Adam0.003,25epochs. Class-stratified training Bootstrap. Fixed classification threshold0.5. Stacking fits current training in-sample updated branch probabilities, not nested OOF.</p><table><tr><th>Method</th><th>Accuracy%</th><th>Recall%</th><th>F1%</th></tr>' + tab + '</table><p><a href="SUMMARY.json">All metrics</a> | <a href="POOLED_PER_SEED.json">Predictions</a> | <a href="PLAN.json">Protocol</a> | <a href="AUDIT.json">Audit</a></p>', encoding='utf-8')
save(R / 'PROGRESS.json', dict(stage='complete_pending_host_audit', completed=15))
print(json.dumps(summary), flush=True)
