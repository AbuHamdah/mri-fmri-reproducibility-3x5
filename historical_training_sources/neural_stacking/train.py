raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import json, hashlib, time, html
from pathlib import Path
import numpy as np, tensorflow as tf
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
a = np.load(R / 'inputs/SUBJECTS.npz')
ids = a['subject_ids'].tolist()
y = a['labels'].astype('float32')
locks = []
started = time.time()
for seed in [42, 43, 44]:
    for fold in range(1, 6):
        d = R / 'runs' / f'seed{seed}_fold{fold}'
        s = read(d / 'SPLIT.json')
        tr = np.array([ids.index(k) for k in s['train']])
        va = np.array([ids.index(k) for k in s['validation']])
        pair = np.load(d / 'PAIR.npy').astype('float32')
        q = np.clip(pair, 1e-07, 1 - 1e-07)
        X = np.log(q) - np.log1p(-q)
        p = np.clip(pair.mean(1), 1e-07, 1 - 1e-07)
        prior = (np.log(p) - np.log1p(-p))[:, None]
        mean = X[tr].mean(0)
        std = np.maximum(X[tr].std(0), 0.01)
        tf.keras.backend.clear_session()
        runseed = seed * 1000 + fold
        tf.keras.utils.set_random_seed(runseed)
        rng = np.random.default_rng(runseed)
        boot = rng.choice(tr, size=len(tr), replace=True)
        assert len(set(y[boot])) == 2
        x = tf.keras.Input((2,), name='branch_logits')
        b = tf.keras.Input((1,), name='soft_voting_prior_logit')
        v = tf.keras.layers.Normalization(mean=mean, variance=std ** 2)(x)
        v = tf.keras.layers.Dense(8, activation='tanh')(v)
        v = tf.keras.layers.Dropout(0.4)(v)
        v = tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(v)
        res = tf.keras.layers.Rescaling(3.0)(tf.keras.layers.Activation('tanh')(v))
        out = tf.keras.layers.Activation('sigmoid')(tf.keras.layers.Add()([b, res]))
        m = tf.keras.Model([x, b], out)
        m.add_loss(0.01 * tf.reduce_mean(tf.square(res)))
        m.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy')
        initial = m([X, prior], training=False).numpy().ravel()
        assert np.max(abs(initial - pair.mean(1))) < 1e-06
        m.save(d / 'INITIAL_MODEL.keras')
        t = time.time()
        h = m.fit([X[boot], prior[boot]], y[boot], epochs=30, batch_size=8, shuffle=True, verbose=0, validation_data=([X[va], prior[va]], y[va]))
        m.save(d / 'MODEL.keras')
        save(d / 'TRAINING_HISTORY.json', {k: list(map(float, v)) for k, v in h.history.items()})
        save(d / 'TRAINING_EVIDENCE.json', dict(seed=runseed, bootstrap_ids=[ids[i] for i in boot], epochs=30, training_seconds=time.time() - t, iterations=int(m.optimizer.iterations.numpy()), initial_model_sha256=sha(d / 'INITIAL_MODEL.keras'), final_model_sha256=sha(d / 'MODEL.keras')))
        locks.append(dict(run=d.name, sha256=sha(d / 'MODEL.keras')))
        save(R / 'PROGRESS.json', dict(stage='training', completed=len(locks), total=15))
        print('Trained', len(locks), 'of15', flush=True)
save(R / 'ALL_MODELS_FROZEN_BEFORE_TEST.json', dict(records=locks, test_metrics_generated=False))
pool = {}
replays = []
for record in locks:
    d = R / 'runs' / record['run']
    s = read(d / 'SPLIT.json')
    assert sha(d / 'MODEL.keras') == record['sha256']
    seed = s['seed']
    fold = s['fold']
    pair = np.load(d / 'PAIR.npy').astype('float32')
    q = np.clip(pair, 1e-07, 1 - 1e-07)
    X = np.log(q) - np.log1p(-q)
    p = np.clip(pair.mean(1), 1e-07, 1 - 1e-07)
    prior = (np.log(p) - np.log1p(-p))[:, None]
    m = tf.keras.models.load_model(d / 'MODEL.keras', compile=False)
    pp = m([X, prior], training=False).numpy().ravel()
    m2 = tf.keras.models.load_model(d / 'MODEL.keras', compile=False)
    delta = float(np.max(abs(pp - m2([X, prior], training=False).numpy().ravel())))
    assert delta < 1e-06
    replays.append(dict(run=d.name, max_reload_difference=delta))
    for method, prob in [('Warm-adapter Soft Voting', pair.mean(1)), ('Stochastic Neural Stacking', pp)]:
        rows = [dict(subject_id=k, true_label=int(y[ids.index(k)]), probability=float(prob[ids.index(k)]), predicted_label=int(prob[ids.index(k)] >= 0.5), threshold=0.5, seed=seed, test_fold=fold, model_id=d.name + '_' + method) for k in s['test']]
        save(d / (('VOTING' if method.endswith('Voting') else 'STACKING') + '_TEST_PREDICTIONS.json'), rows)
        pool.setdefault((method, seed), []).extend(rows)
per = []
summary = []
for (method, seed), rows in pool.items():
    assert len(rows) == 42 and len({v['subject_id'] for v in rows}) == 42
    per.append(dict(method=method, seed=seed, metrics=dict(zip(NAMES, map(float, metrics([v['true_label'] for v in rows], [v['probability'] for v in rows])))), predictions=rows))
for method in ['Warm-adapter Soft Voting', 'Stochastic Neural Stacking']:
    rr = [v for v in per if v['method'] == method]
    summary.append(dict(method=method, metrics={k: dict(mean=float(np.mean([v['metrics'][k] for v in rr])), sd=float(np.std([v['metrics'][k] for v in rr], ddof=1))) for k in NAMES}))
save(R / 'SUMMARY.json', summary)
save(R / 'POOLED_PER_SEED.json', per)
save(R / 'NATIVE_REPLAY_AUDIT.json', dict(passed=True, records=replays))
save(R / 'COMPLETE.json', dict(training_models=15, test_prediction_rows=252, actual_seconds=time.time() - started, all_runs_reported=True))
tab = ''.join(('<tr><td>' + r['method'] + '</td>' + ''.join(('<td>%.4f ± %.4f</td>' % (r['metrics'][k]['mean'] * 100, r['metrics'][k]['sd'] * 100) for k in ['accuracy', 'recall', 'f1'])) + '</tr>' for r in summary))
(R / 'REPORT_EN.html').write_text('<!doctype html><meta charset="utf-8"><title>Stochastic Neural Stacking</title><style>body{font-family:Arial;margin:24px}td,th{padding:8px;border:1px solid #ccc}</style><h1>Stochastic Neural Stacking on fixed warm-adapter branches</h1><p>One fixed protocol, three seeds × five current subject partitions. Training-only ordinary child Bootstrap, random hidden-layer initialization, active Dropout0.4 and shuffled batches. Dense8/tanh with zero-initialized bounded residual3 over the equal Soft Voting logit. Adam0.005,30epochs, residual penalty0.01, threshold0.5. Validation logged without checkpoint selection. All runs retained.</p><p>Meta-model training uses in-sample current-training branch outputs; not nested OOF.  No new Qwen calls. This is a neural stacker, not the previous logistic stacker. Original results preserved.</p><table><tr><th>Method</th><th>Accuracy%</th><th>Recall%</th><th>F1%</th></tr>' + tab + '</table><p><a href="SUMMARY.json">All metrics</a> | <a href="POOLED_PER_SEED.json">Predictions</a> | <a href="PLAN.json">Protocol</a></p>', encoding='utf-8')
print(json.dumps(summary), flush=True)
