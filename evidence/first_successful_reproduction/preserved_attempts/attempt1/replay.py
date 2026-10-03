import json, time, hashlib, traceback, gc, sys
from pathlib import Path
import numpy as np, tensorflow as tf
from statistics_core import metrics, NAMES
P = Path(__file__).resolve().parents[1]
OUT = P / 'reproduction_run'
OUT.mkdir(exist_ok=True)
execution_log = (OUT / 'EXECUTION.log').open('a', encoding='utf-8')

class Tee:

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        execution_log.write(text)
        execution_log.flush()

    def flush(self):
        self.stream.flush()
        execution_log.flush()
sys.stdout = Tee(sys.stdout)
sys.stderr = Tee(sys.stderr)

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
inputs = read(P / 'INPUTS.json')
ids = sorted({r['subject_id'] for r in inputs})
labels = {s: next((r['label'] for r in inputs if r['subject_id'] == s)) for s in ids}
y = np.array([labels[s] for s in ids])
index = {s: i for i, s in enumerate(ids)}
bykind = {k: {s: sorted([r['path'] for r in inputs if r['subject_id'] == s and r['kind'] == k]) for s in ids} for k in ['mri', 'fmri']}
records = []
pools = {}
selected = {}
started = time.time()

def loadmodel(path):
    return tf.keras.models.load_model(path, compile=False)

def one(m, path):
    x = np.load(P / path)
    shape = tuple(m.input_shape[1:])
    x = x[None] if x.shape == shape else x
    return float(m(x, training=False).numpy().ravel()[0])

def add(key, seed, fold, d, rows, prob, threshold=0.5, model=''):
    pp = np.asarray(prob).ravel()
    expected = np.array([r['probability'] for r in rows])
    delta = float(np.max(abs(pp - expected)))
    assert delta <= 0.0001, (key, seed, fold, delta)
    thresholds = np.broadcast_to(threshold, pp.shape)
    assert np.array_equal(pp >= thresholds, expected >= thresholds), (key, seed, fold, 'classification mismatch')
    rr = [dict(subject_id=r['subject_id'], true_label=int(labels[r['subject_id']]), probability=float(p), threshold=float(t), prediction=int(p >= t), seed=seed, test_fold=fold, model=model) for r, p, t in zip(rows, pp, thresholds)]
    assert all((r['true_label'] == labels[r['subject_id']] for r in rows))
    assert len({r['subject_id'] for r in rr}) == len(rr)
    pools.setdefault((key, seed), []).extend(rr)
    rec = dict(result=key, seed=seed, fold=fold, model=model, model_sha256=sha(P / model) if model else None, test_children=len(rr), maximum_probability_difference=delta)
    records.append(rec)
    save(OUT / 'progress.json', dict(stage='replaying_checkpoints', completed=len(records), elapsed_seconds=time.time() - started))
    print('Replay', len(records), key, seed, fold, 'difference', delta, flush=True)

def descriptors(paths):
    xx = []
    for path in paths:
        a = np.load(P / path).astype('float64')[..., 0]
        fm = a.mean((1, 2))
        fs = a.std((1, 2))
        xx.append([a.mean(), a.std(), fm.std(), fs.mean(), np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=2)).mean(), a[:, 24:40, 24:40].mean() - a.mean(), np.mean(a != 0)])
    xx = np.array(xx)
    return np.r_[xx.mean(0), xx.std(0)]
try:
    cache = np.load(P / 'fusion/agent_controlled/cache/FEATURES.npz')
    assert cache['subject_ids'].tolist() == ids
    assert np.array_equal(cache['labels'], y)
    feature_audit = []
    blocks = {}
    for kind in ['mri', 'fmri']:
        m = loadmodel(P / 'extractors' / f'{kind}_features128.keras')
        out = []
        for s in ids:
            vals = []
            for path in bykind[kind][s]:
                x = np.load(P / path).astype('float32')
                x = x[None] if x.shape == tuple(m.input_shape[1:]) else x
                vals.append(m(x, training=False).numpy()[0])
            out.append(np.mean(vals, axis=0))
        blocks[kind] = np.array(out, dtype='float32')
        delta = float(np.max(abs(blocks[kind] - cache[kind])))
        assert delta <= 1e-05, (kind, 'feature delta', delta)
        feature_audit.append(dict(kind=kind, children=42, processed_inputs=sum((len(v) for v in bykind[kind].values())), maximum_feature_difference=delta))
        del m
        gc.collect()
        print('Regenerated features:', kind, delta, flush=True)
    save(OUT / 'FEATURE_REGENERATION_AUDIT.json', dict(passed=True, records=feature_audit, raw_nifti_reprocessed=False))
    X = np.c_[blocks['mri'], blocks['fmri']]
    for seed in [42, 43, 44]:
        for fold in range(1, 6):
            d = P / 'selected/mri_baseline/adopted_models' / f'seed{seed}_fold{fold}'
            c = read(P / 'selected/mri_baseline/calibration' / f'seed{seed}_fold{fold}.json')
            m = loadmodel(d / 'model.keras')
            rows = c['predictions']
            prob = [one(m, bykind['mri'][r['subject_id']][0]) for r in rows]
            add('mri_baseline', seed, fold, d, rows, prob, c['selected_threshold'], str((d / 'model.keras').relative_to(P)))
            del m
            tf.keras.backend.clear_session()
    for series in ['mri_baseline', 'mri_agentic']:
        for seed in [42, 43, 44]:
            for fold in range(1, 6):
                d = P / 'selected' / series / 'branches' / series / f'seed{seed}_fold{fold}'
                m = loadmodel(d / 'model.keras')
                full = read(d / 'ALL_SUBJECT_PREDICTIONS.json')
                by = {r['subject_id']: r for r in full}
                prob = np.array([one(m, bykind['mri'][s][0]) for s in ids])
                expected = np.array([by[s]['probability'] for s in ids])
                assert np.max(abs(prob - expected)) < 0.0001
                selected[series, seed, fold] = prob
                if series == 'mri_agentic':
                    s = read(d / 'SPLIT.json')
                    rows = [by[k] for k in s['test']]
                    add(series, seed, fold, d, rows, prob[[index[k] for k in s['test']]], model=str((d / 'model.keras').relative_to(P)))
                del m
                tf.keras.backend.clear_session()
    for series in ['fmri_baseline', 'fmri_agentic']:
        root = P / 'selected' / series
        ca = read(root / 'cache' / series / 'SOURCE_RUN_PROBABILITIES.json')
        assert ca['subject_ids'] == ids
        m = loadmodel(P / 'parent_models' / f'{series}.keras')
        raw = []
        for s in ids:
            raw.append(m(np.stack([np.load(P / path) for path in bykind['fmri'][s]]), training=False).numpy().ravel())
        raw = np.array(raw)
        delta = float(np.max(abs(raw - np.array(ca['three_run_probabilities']))))
        assert delta <= 0.0001, (series, 'parent replay', delta)
        del m
        tf.keras.backend.clear_session()
        q = np.clip(raw.astype('float32').mean(1).astype('float64'), 1e-12, 1 - 1e-12)
        offset = np.log(q / (1 - q))
        if series == 'fmri_baseline':
            xx = np.c_[raw, offset]
        else:
            f = np.array([descriptors(bykind['fmri'][s]) for s in ids])
            expected = np.array([r['features'] for r in read(root / 'INPUT_FEATURES.json')['records']])
            assert np.max(abs(f - expected)) < 1e-12
            xx = np.c_[raw, f, offset]
        for seed in [42, 43, 44]:
            for fold in range(1, 6):
                d = root / 'branches' / series / f'seed{seed}_fold{fold}'
                m = loadmodel(d / 'model.keras')
                prob = m(xx, training=False).numpy().ravel()
                by = {r['subject_id']: r for r in read(d / 'ALL_SUBJECT_PREDICTIONS.json')}
                assert np.max(abs(prob - np.array([by[s]['probability'] for s in ids]))) < 0.0001
                selected[series, seed, fold] = prob
                s = read(d / 'SPLIT.json')
                rows = [by[k] for k in s['test']]
                add(series, seed, fold, d, rows, prob[[index[k] for k in s['test']]], model=str((d / 'model.keras').relative_to(P)))
                del m
                tf.keras.backend.clear_session()
    for series_dir in sorted((P / 'comparisons/branches').iterdir()):
        series = series_dir.name
        for seed in [42, 43, 44]:
            for fold in range(1, 6):
                d = series_dir / f'seed{seed}_fold{fold}'
                rows = read(d / 'test_predictions.json')
                m = loadmodel(d / 'model.keras')
                prob = [float(np.mean([one(m, path) for path in row['input_paths']])) for row in rows]
                add(series, seed, fold, d, rows, prob, model=str((d / 'model.keras').relative_to(P)))
                del m
                tf.keras.backend.clear_session()
                gc.collect()
    for mode_dir in sorted((P / 'fusion/traditional').iterdir()):
        mode = mode_dir.name
        for method_dir in sorted(mode_dir.iterdir()):
            method = method_dir.name
            for seed in [42, 43, 44]:
                for fold in range(1, 6):
                    d = method_dir / f'seed{seed}_fold{fold}'
                    protocol = read(d / 'protocol.json')
                    pair = np.c_[selected[f'mri_{mode}', seed, fold], selected[f'fmri_{mode}', seed, fold]]
                    m = loadmodel(d / 'model.keras')
                    prob = m(pair, training=False).numpy().ravel()
                    rows = read(d / 'test_predictions.json')
                    add(mode + ' / ' + method, seed, fold, d, rows, prob[[index[r['subject_id']] for r in rows]], protocol['fixed_threshold'], str((d / 'model.keras').relative_to(P)))
                    del m
                    tf.keras.backend.clear_session()
    for seed in [42, 43, 44]:
        for fold in range(1, 6):
            d = P / 'fusion/warm_adapters/runs' / f'seed{seed}_fold{fold}'
            s = read(d / 'SPLIT.json')
            src = np.load(d / 'SOURCE_PROBABILITIES.npz')
            assert src['subject_ids'].tolist() == ids
            pair = src['pairs'].astype('float32')
            assert np.max(abs(pair[:, 0] - selected['mri_agentic', seed, fold])) < 0.0001 and np.max(abs(pair[:, 1] - selected['fmri_agentic', seed, fold])) < 0.0001
            new = []
            for j, kind in enumerate(['mri', 'fmri']):
                m = loadmodel(d / kind / 'MODEL.keras')
                q = np.clip(pair[:, j], 1e-07, 1 - 1e-07)
                base = (np.log(q) - np.log1p(-q))[:, None]
                new.append(m([blocks[kind], base], training=False).numpy().ravel())
                del m
                tf.keras.backend.clear_session()
            newpair = np.column_stack(new)
            stored = np.load(d / 'NEW_MODALITY_PROBABILITIES.npy')
            assert np.max(abs(newpair - stored)) < 0.0001
            rows = read(d / 'VOTING_TEST_PREDICTIONS.json')
            add('Warm-adapter Soft Voting', seed, fold, d, rows, newpair.mean(1)[[index[r['subject_id']] for r in rows]], model=str((d / 'mri/MODEL.keras').relative_to(P)))
            nd = P / 'fusion/neural_stacking/runs' / d.name
            q = np.clip(newpair.astype('float32'), 1e-07, 1 - 1e-07)
            z = np.log(q) - np.log1p(-q)
            q = np.clip(newpair.astype('float32').mean(1), 1e-07, 1 - 1e-07)
            base = (np.log(q) - np.log1p(-q))[:, None]
            m = loadmodel(nd / 'MODEL.keras')
            pp = m([z, base], training=False).numpy().ravel()
            rows = read(nd / 'STACKING_TEST_PREDICTIONS.json')
            add('Stochastic Neural Stacking', seed, fold, nd, rows, pp[[index[r['subject_id']] for r in rows]], model=str((nd / 'MODEL.keras').relative_to(P)))
            del m
            tf.keras.backend.clear_session()
            fd = P / 'fusion/agent_controlled/runs' / d.name
            prior = np.load(fd / 'INPUT_PROBABILITIES.npz')
            assert prior['subject_ids'].tolist() == ids
            pair = prior['pairs'].astype('float32')
            assert np.max(abs(pair[:, 0] - selected['mri_agentic', seed, fold])) < 0.0001 and np.max(abs(pair[:, 1] - selected['fmri_agentic', seed, fold])) < 0.0001
            q = np.clip(pair.mean(1), 1e-07, 1 - 1e-07)
            base = (np.log(q) - np.log1p(-q))[:, None]
            m = loadmodel(fd / 'model.keras')
            pp = m([X, base], training=False).numpy().ravel()
            rows = read(fd / 'TEST_PREDICTIONS.json')
            add('Agent-controlled Feature Fusion', seed, fold, fd, rows, pp[[index[r['subject_id']] for r in rows]], model=str((fd / 'model.keras').relative_to(P)))
            del m
            tf.keras.backend.clear_session()
    result = read(P / 'RESULTS.json')
    wanted = {r['name']: r for r in result['adopted'] + result['source_comparisons'] + result['previous_fusion_comparisons']}
    summaries = []
    per = []
    for (key, seed), rr in pools.items():
        assert len(rr) == 42 and len({r['subject_id'] for r in rr}) == 42, (key, seed, len(rr))
        mm = dict(zip(NAMES, map(float, metrics([r['true_label'] for r in rr], [r['probability'] for r in rr], np.array([r['threshold'] for r in rr])))))
        per.append(dict(result=key, seed=seed, metrics=mm, predictions=rr))
    for key, original in wanted.items():
        rr = [r for r in per if r['result'] == key]
        assert len(rr) == 3
        mm = {k: dict(mean=float(np.mean([r['metrics'][k] for r in rr])), sd=float(np.std([r['metrics'][k] for r in rr], ddof=1))) for k in NAMES}
        diff = {k: {field: abs(mm[k][field] - original['metrics'][k][field]) for field in ['mean', 'sd']} for k in NAMES}
        assert all((diff[k][field] < 1e-12 for k in ['accuracy', 'recall', 'f1', 'precision', 'specificity', 'kappa'] for field in ['mean', 'sd']))
        assert all((diff[k][field] < 0.0001 for k in ['auc', 'brier', 'ece'] for field in ['mean', 'sd'])), (key, diff)
        summaries.append(dict(result=key, reproduced_metrics=mm, difference_from_report=diff))
    assert len(summaries) == 33
    save(OUT / 'REPRODUCED_RESULTS.json', summaries)
    save(OUT / 'POOLED_REGENERATED_PREDICTIONS.json', per)
    save(OUT / 'MODEL_RESULT_INDEX.json', records)
    save(OUT / 'REPRODUCTION_AUDIT.json', dict(passed=True, result_rows=33, repeated_seed_metrics=99, result_model_records=len(records), checkpoint_probability_tolerance=0.0001, discrete_metrics_match_exactly=True, continuous_metrics_match_within_tolerance=True, input_scope='Copied processed arrays; copied extractors regenerate features; parent models replayed; complete downstream saved-checkpoint inference. Raw NIfTI preprocessing and training not repeated.', elapsed_seconds=time.time() - started))
    print('ALL33 REQUESTED RESULTS REPRODUCED', flush=True)
except Exception:
    save(OUT / 'FAILURE.json', dict(traceback=traceback.format_exc(), completed_records=len(records), elapsed_seconds=time.time() - started))
    raise
