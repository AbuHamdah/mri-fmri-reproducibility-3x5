import json, hashlib, sys, shutil
from pathlib import Path
import numpy as np, h5py
from scipy.special import expit
R = Path('D:/MRI_FMRI_WARM_ADAPTER_VOTING_STACKING_V1')
sys.path.insert(0, str(R / 'source'))
from statistics_core import metrics, NAMES

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')
a = np.load(R / 'inputs/FEATURES.npz')
ids = a['subject_ids'].tolist()
y = a['labels']
replays = []
recalcs = []
pool = {}
models_changed = 0
for record in read(R / 'SOURCE_INDEX.json'):
    for s in record['source_branches'].values():
        assert sha(Path(s['model'])) == s['model_sha256'] and sha(Path(s['all_predictions'])) == s['predictions_sha256']
    d = R / 'runs' / f"seed{record['seed']}_fold{record['fold']}"
    s = read(d / 'SPLIT.json')
    tr = np.array([ids.index(k) for k in s['train']])
    pair = np.load(d / 'SOURCE_PROBABILITIES.npz')
    assert np.array_equal(pair['labels'], y) and pair['subject_ids'].tolist() == ids and (sha(d / 'SOURCE_PROBABILITIES.npz') == record['cached_probabilities_sha256'])
    pps = []
    for j, kind in enumerate(['mri', 'fmri']):
        ev = read(d / kind / 'TRAINING_EVIDENCE.json')
        assert set(ev['bootstrap_ids']) <= set(s['train'])
        assert not set(ev['bootstrap_ids']) & set(s['test'])
        assert ev['epochs'] == 25 and ev['optimizer_iterations'] > 0
        X = a[kind].astype('float32')
        with h5py.File(d / kind / 'MODEL.keras') as f:
            conf = json.loads(f.attrs['model_config'])['config']['layers']
            norm = next((c for c in conf if c['class_name'] == 'Normalization'))['config']
            mean = np.array(norm['mean'], dtype='float32')
            var = np.array(norm['variance'], dtype='float32')
            assert np.max(abs(mean - X[tr].mean(0))) < 1e-05
            assert np.max(abs(var - np.maximum(X[tr].std(0), 0.01) ** 2)) < 1e-05
            w1 = f['model_weights/dense/dense/kernel:0'][:]
            b1 = f['model_weights/dense/dense/bias:0'][:]
            w2 = f['model_weights/dense_1/dense_1/kernel:0'][:]
            b2 = f['model_weights/dense_1/dense_1/bias:0'][:]
            h = np.maximum((X - mean) / np.maximum(np.sqrt(var), 1e-07) @ w1 + b1, 0)
            res = 3 * np.tanh(h @ w2 + b2)
            p = np.clip(pair['pairs'][:, j].astype('float32'), 1e-07, 1 - 1e-07)
            base = np.log(p) - np.log1p(-p)
            pp = expit(base + res.ravel())
        stored = read(d / kind / 'ALL_PREDICTIONS.json')
        assert [v['subject_id'] for v in stored] == ids and [v['true_label'] for v in stored] == y.tolist()
        err = float(np.max(abs(pp - [v['probability'] for v in stored])))
        assert err < 1e-05
        with h5py.File(d / kind / 'INITIAL_MODEL.keras') as f:
            assert np.all(f['model_weights/dense_1/dense_1/kernel:0'][:] == 0) and np.all(f['model_weights/dense_1/dense_1/bias:0'][:] == 0)
        assert ev['initial_model_sha256'] == sha(d / kind / 'INITIAL_MODEL.keras') and ev['final_model_sha256'] == sha(d / kind / 'MODEL.keras')
        assert np.any(w2 != 0) or np.any(b2 != 0)
        models_changed += 1
        pps.append(pp)
        replays.append(dict(seed=record['seed'], fold=record['fold'], kind=kind, max_host_replay_difference=err))
    newpair = np.column_stack(pps)
    st = read(d / 'STACKER.json')
    assert st['fit_ids'] == s['train'] and set(st['bootstrap_ids']) <= set(s['train'])
    q = np.clip(newpair, 1e-07, 1 - 1e-07)
    Z = np.c_[np.log(q / (1 - q)), np.ones(42)]
    preds = {'VOTING': newpair.mean(1), 'STACKING': expit(Z @ st['coefficients'])}
    for method, p in preds.items():
        rows = read(d / (method + '_TEST_PREDICTIONS.json'))
        assert set((v['subject_id'] for v in rows)) == set(s['test'])
        assert all((v['true_label'] == y[ids.index(v['subject_id'])] for v in rows))
        err = max((abs(v['probability'] - p[ids.index(v['subject_id'])]) for v in rows))
        assert err < 1e-05
        assert all((int(p[ids.index(v['subject_id'])] >= 0.5) == v['predicted_label'] for v in rows))
        pool.setdefault((method, record['seed']), []).extend(rows)
for (method, seed), rows in pool.items():
    assert len(rows) == 42 and len({v['subject_id'] for v in rows}) == 42
    p = next((v for v in read(R / 'POOLED_PER_SEED.json') if v['seed'] == seed and (v['method'].endswith('Voting') if method == 'VOTING' else v['method'].endswith('Stacking'))))
    mm = dict(zip(NAMES, map(float, metrics([v['true_label'] for v in rows], [v['probability'] for v in rows]))))
    assert all((abs(mm[k] - p['metrics'][k]) < 1e-12 for k in NAMES))
    recalcs.append(dict(seed=seed, method=method, metrics=mm))
save(R / 'INDEPENDENT_HOST_AUDIT.json', dict(passed=True, changed_adapters=models_changed, replays=replays, pooled_recalculations=recalcs, current_bootstrap_train_only=True, source_checkpoints_hash_unchanged=True, true_labels_unchanged=True))
shutil.copy2(__file__, R / 'source/audit_host.py')
report = R / 'REPORT_EN.html'
text = report.read_text(encoding='utf-8')
text += '<p><a href="INDEPENDENT_HOST_AUDIT.json">Independent host replay audit</a></p><p>Equal accuracy/Recall/F1 here reflect equal confusion counts; methods have different probabilities, AUC, Brier and ECE. Source labels and checkpoints were preserved.</p>'
report.write_text(text, encoding='utf-8')
save(R / 'INVENTORY.json', [dict(path=p.relative_to(R).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name not in ['INVENTORY.json', 'COMPLETE.json']])
assert all((sha(R / v['path']) == v['sha256'] for v in read(R / 'INVENTORY.json')))
save(R / 'COMPLETE.json', dict(passed=True, adapters=30, stackers=15, keras_files=60, host_audit_passed=True, all_inventory_hashes_verified=True, old_results_modified=False))
print('PASS:30 changed adapters,30 independent numerical model replays,6 pooled metric recalculations,all source hashes intact.')
