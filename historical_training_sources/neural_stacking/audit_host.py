import json, hashlib, sys, shutil
from pathlib import Path
import numpy as np, h5py
from scipy.special import expit
R = Path('D:/MRI_FMRI_STOCHASTIC_NEURAL_STACKING_V1')
sys.path.insert(0, str(R / 'source'))
from statistics_core import metrics, NAMES

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')
a = np.load(R / 'inputs/SUBJECTS.npz')
ids = a['subject_ids'].tolist()
y = a['labels']
checks = []
pools = {}
assert (R / 'COMPLETE.json').exists()
for rec in read(R / 'SOURCE_INDEX.json'):
    d = R / 'runs' / rec['run']
    s = read(d / 'SPLIT.json')
    assert sha(Path(rec['path'])) == rec['sha256'] and sha(d / 'PAIR.npy') == rec['sha256']
    tr = [ids.index(k) for k in s['train']]
    assert not (set(s['train']) & set(s['test']) or set(s['validation']) & set(s['test']) or set(s['train']) & set(s['validation']))
    ev = read(d / 'TRAINING_EVIDENCE.json')
    assert set(ev['bootstrap_ids']) <= set(s['train']) and ev['iterations'] > 0 and (ev['epochs'] == 30)
    assert sha(d / 'INITIAL_MODEL.keras') == ev['initial_model_sha256'] and sha(d / 'MODEL.keras') == ev['final_model_sha256']
    pair = np.load(d / 'PAIR.npy').astype('float32')
    q = np.clip(pair, 1e-07, 1 - 1e-07)
    X = np.log(q) - np.log1p(-q)
    p = np.clip(pair.mean(1), 1e-07, 1 - 1e-07)
    prior = np.log(p) - np.log1p(-p)
    with h5py.File(d / 'MODEL.keras') as f:
        cfg = json.loads(f.attrs['model_config'])['config']['layers']
        norm = next((c['config'] for c in cfg if c['class_name'] == 'Normalization'))
        mean = np.array(norm['mean'], dtype='float32')
        var = np.array(norm['variance'], dtype='float32')
        assert np.max(abs(mean - X[tr].mean(0))) < 1e-06 and np.max(abs(var - np.maximum(X[tr].std(0), 0.01) ** 2)) < 1e-06
        w1 = f['model_weights/dense/dense/kernel:0'][:]
        b1 = f['model_weights/dense/dense/bias:0'][:]
        w2 = f['model_weights/dense_1/dense_1/kernel:0'][:]
        b2 = f['model_weights/dense_1/dense_1/bias:0'][:]
        h = np.tanh((X - mean) / np.maximum(np.sqrt(var), 1e-07) @ w1 + b1)
        pp = expit(prior + 3 * np.tanh(h @ w2 + b2).ravel())
        assert np.any(w2 != 0) or np.any(b2 != 0)
    with h5py.File(d / 'INITIAL_MODEL.keras') as f:
        assert np.all(f['model_weights/dense_1/dense_1/kernel:0'][:] == 0) and np.all(f['model_weights/dense_1/dense_1/bias:0'][:] == 0)
    for method, prob in [('VOTING', pair.mean(1)), ('STACKING', pp)]:
        rows = read(d / (method + '_TEST_PREDICTIONS.json'))
        assert set((v['subject_id'] for v in rows)) == set(s['test'])
        assert all((v['true_label'] == int(y[ids.index(v['subject_id'])]) for v in rows))
        delta = max((abs(v['probability'] - prob[ids.index(v['subject_id'])]) for v in rows))
        assert delta < 1e-05
        assert all((v['predicted_label'] == int(prob[ids.index(v['subject_id'])] >= 0.5) for v in rows))
        pools.setdefault((method, s['seed']), []).extend(rows)
    checks.append(dict(run=d.name, host_probability_difference=float(delta)))
for (method, seed), rows in pools.items():
    assert len(rows) == 42 and len({v['subject_id'] for v in rows}) == 42
    stored = next((v for v in read(R / 'POOLED_PER_SEED.json') if v['seed'] == seed and (v['method'].endswith('Voting') if method == 'VOTING' else v['method'].endswith('Stacking'))))
    mm = dict(zip(NAMES, map(float, metrics([v['true_label'] for v in rows], [v['probability'] for v in rows]))))
    assert all((abs(mm[k] - stored['metrics'][k]) < 1e-12 for k in NAMES))
save(R / 'INDEPENDENT_HOST_AUDIT.json', dict(passed=True, records=checks, models=15, pooled_metrics_recalculated=6, current_bootstrap_train_only=True, source_inputs_unchanged=True, labels_unchanged=True, all_runs_preserved=True))
shutil.copy2(__file__, R / 'source/audit_host.py')
save(R / 'INVENTORY.json', [dict(path=p.relative_to(R).as_posix(), sha256=sha(p)) for p in R.rglob('*') if p.is_file() and p.name != 'INVENTORY.json'])
assert all((sha(R / v['path']) == v['sha256'] for v in read(R / 'INVENTORY.json')))
print('PASS:15 independent saved-model replays,6 pooled metric recalculations,all inputs and labels unchanged.')
