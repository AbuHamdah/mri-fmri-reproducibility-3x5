raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import os, json, time, hashlib, traceback, gc
from pathlib import Path
import numpy as np
import tensorflow as tf
from statistics_core import metrics, NAMES
P = Path('/package')
O = Path('/out')
plan = json.loads((O / 'PLAN.json').read_text())
lock = O / 'RUN.lock'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, indent=2), encoding='utf8')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def score(y, p, t=0.5):
    return dict(zip(NAMES, metrics(y, p, t).tolist()))

def bce(y, p):
    p = np.clip(p, 1e-07, 1 - 1e-07)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))

def choose_threshold(y, p):
    u = np.unique(p)
    c = np.unique(np.r_[0, 0.5, 1, (u[:-1] + u[1:]) / 2])
    scores = []
    for t in c:
        m = score(y, p, t)
        scores.append({'threshold': float(t), 'f1': m['f1'], 'balanced_accuracy': (m['recall'] + m['specificity']) / 2})
    best = max(scores, key=lambda r: (r['f1'], r['balanced_accuracy'], -abs(r['threshold'] - 0.5), r['threshold']))
    return (best['threshold'], scores)

def summarize_predictions(rows, split, seed, fold, series):
    by = {r['subject_id']: r for r in rows}
    v = [by[s] for s in split['validation']]
    t = [by[s] for s in split['test']]
    y = np.array([r['true_label'] for r in v])
    p = np.array([r['probability'] for r in v])
    threshold, scores = choose_threshold(y, p)
    for r in rows:
        r.update(seed=seed, test_fold=fold, model_id=f'{series}_seed{seed}_fold{fold}', selected_threshold=threshold, predicted_label=int(r['probability'] >= threshold))
    yy = np.array([r['true_label'] for r in t])
    pp = np.array([r['probability'] for r in t])
    return ({'threshold': threshold, 'threshold_candidates_validation': scores, 'fixed_metrics': score(yy, pp), 'calibrated_metrics': score(yy, pp, threshold), 'test_n': len(t), 'fixed_correct': int(((pp >= 0.5) == yy).sum()), 'calibrated_correct': int(((pp >= threshold) == yy).sum())}, t, v)

def encoder_fingerprint(enc):
    h = hashlib.sha256()
    for w in enc.get_weights():
        h.update(str(w.dtype).encode())
        h.update(str(w.shape).encode())
        h.update(w.tobytes())
    configs = []
    for layer in enc.layers:
        c = layer.get_config()
        c.pop('trainable', None)
        configs.append({'class': layer.__class__.__name__, 'config': c})
    h.update(json.dumps(configs, sort_keys=True).encode())
    return h.hexdigest()

def verify_native(loaded, enc, head, features, te, kind, inputs):
    boundary = 'dropout1' if kind == 'mri' else 'features128'
    le = tf.keras.Model(loaded.inputs, loaded.get_layer(boundary).output)
    a = enc.get_weights()
    b = le.get_weights()
    assert len(a) == len(b) and all((np.array_equal(x, y) for x, y in zip(a, b)))
    lh = loaded.layers[-1]
    assert all((np.array_equal(x, y) for x, y in zip(head.get_weights(), lh.get_weights())))
    expected = np.array([float(head(features[i:i + 1], training=False).numpy().ravel()[0]) for i in te])
    from_loaded = np.array([float(lh(features[i:i + 1], training=False).numpy().ravel()[0]) for i in te])
    cache_delta = float(abs(expected - from_loaded).max())
    check = te if kind == 'mri' else te[:1]
    native = np.array([float(loaded(np.load(P / inputs[i]['path'])[None], training=False).numpy().ravel()[0]) for i in check])
    wanted = np.array([float(head(features[i:i + 1], training=False).numpy().ravel()[0]) for i in check])
    raw_delta = float(abs(native - wanted).max())
    delta = max(cache_delta, raw_delta)
    assert delta <= 0.0001
    return (delta, {'all_test_cached_head_samples_checked': len(te), 'native_raw_test_samples_checked': len(check), 'frozen_encoder_weights_exact': True, 'selected_head_weights_exact': True, 'cached_loaded_head_delta': cache_delta, 'raw_native_spot_delta': raw_delta, 'scope': 'MRI all raw test inputs; fMRI all cached test-head outputs plus frozen-encoder byte equality and one raw test input per run.'})
allinputs = read(P / 'INPUTS.json')
subject_ids = sorted({r['subject_id'] for r in allinputs})
labels = {s: next((r['label'] for r in allinputs if r['subject_id'] == s)) for s in subject_ids}
sy = np.array([labels[s] for s in subject_ids])
sid = {s: i for i, s in enumerate(subject_ids)}
completed = len(list((O / 'branches').rglob('RESULT_CURRENT.json')))
total = 210
with lock.open('x') as f:
    f.write(str(os.getpid()))
try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for spec in plan['series']:
        series, kind = (spec['series'], spec['kind'])
        done = list((O / 'branches' / series).glob('*/RESULT_CURRENT.json'))
        if len(done) == 15:
            for rf in done:
                assert sha(rf.parent / 'model.keras') == read(rf)['model_sha256']
            continue
        source = P / spec['source_model']
        sourcehash = sha(source)
        cache = O / 'cache' / series
        cache.mkdir(parents=True, exist_ok=True)
        inputs = sorted([r for r in allinputs if r['kind'] == kind], key=lambda r: (r['subject_id'], r['path']))
        idx = {s: np.array([i for i, r in enumerate(inputs) if r['subject_id'] == s]) for s in subject_ids}
        ry = np.array([r['label'] for r in inputs], dtype=np.float32)

        def pooled(x):
            return np.array([np.mean(x[idx[s]], axis=0) for s in subject_ids])
        write(O / 'progress.json', {'stage': 'FEATURE_EXTRACTION', 'series': series, 'completed_branch_runs': completed, 'total_branch_runs': total})
        base = tf.keras.models.load_model(source, compile=False)
        dense_name = 'features128' if kind == 'mri' else 'dense64'
        enc = tf.keras.Model(base.inputs, base.get_layer(dense_name).input)
        enc.trainable = False
        initial = base.get_layer(dense_name).get_weights() + base.get_layer('prediction').get_weights()

        def make_head():
            head = tf.keras.Sequential([tf.keras.layers.Dense.from_config(base.get_layer(dense_name).get_config()), tf.keras.layers.Dense.from_config(base.get_layer('prediction').get_config())])
            head(features[:1])
            head.trainable = True
            head.set_weights(initial)
            return head
        fingerprint = encoder_fingerprint(enc)
        reused = None
        if not (cache / 'features.npy').exists():
            for audit_file in (O / 'cache').glob('*/CACHE_AUDIT.json'):
                ad = read(audit_file)
                if ad.get('encoder_fingerprint') == fingerprint and [r['path'] for r in ad.get('inputs', [])] == [r['path'] for r in inputs]:
                    assert all((sha(P / r['path']) == r['sha256'] for r in ad['inputs']))
                    features = np.load(audit_file.parent / 'features.npy')
                    np.save(cache / 'features.npy', features)
                    write(cache / 'CACHE_AUDIT.json', dict(ad, source_sha256=sourcehash, reused_from=str(audit_file.parent), reuse_reason='Identical encoder weights, configs and input fingerprints'))
                    reused = audit_file.parent
                    break
        if (cache / 'features.npy').exists():
            assert read(cache / 'CACHE_AUDIT.json')['source_sha256'] == sourcehash
            features = np.load(cache / 'features.npy')
        else:
            extractor = tf.keras.Model(base.inputs, [enc.output, base.output])
            ff = []
            orig = []
            for i, r in enumerate(inputs):
                f, p = extractor(np.load(P / r['path'])[None], training=False)
                ff.append(f.numpy())
                orig.append(float(p.numpy().ravel()[0]))
                if (i + 1) % 10 == 0:
                    write(O / 'progress.json', {'stage': 'FEATURE_EXTRACTION', 'series': series, 'inputs_completed': i + 1, 'inputs_total': len(inputs), 'completed_branch_runs': completed, 'total_branch_runs': total})
            features = np.concatenate(ff)
            np.save(cache / 'features.npy', features)
            head = make_head()
            delta = float(np.max(abs(head(features, training=False).numpy().ravel() - orig)))
            assert delta <= 0.0001
            write(cache / 'CACHE_AUDIT.json', {'encoder_fingerprint': fingerprint, 'source_sha256': sourcehash, 'cache_parity_max_difference': delta, 'inputs': [dict(r, sha256=sha(P / r['path'])) for r in inputs]})
        assert len(features) == len(inputs)
        ad = read(cache / 'CACHE_AUDIT.json')
        ad['encoder_fingerprint'] = fingerprint
        write(cache / 'CACHE_AUDIT.json', ad)
        for seed in plan['seeds']:
            for fold in plan['folds']:
                previous = O / 'branches' / series / f'seed{seed}_fold{fold}' / 'RESULT_CURRENT.json'
                if previous.exists():
                    assert sha(previous.parent / 'model.keras') == read(previous)['model_sha256']
                    continue
                tick = time.time()
                d = O / 'branches' / series / f'seed{seed}_fold{fold}'
                d.mkdir(parents=True, exist_ok=True)
                s = read(P / 'branches' / series / f'seed{seed}_fold{fold}' / 'SPLIT.json')
                train, val, test = (s['train'], s['validation'], s['test'])
                assert not set(train) & set(val) and (not set(train) & set(test)) and (not set(val) & set(test))
                assert set(train + val + test) == set(subject_ids)
                write(O / 'progress.json', {'stage': 'BRANCH_TRAINING', 'series': series, 'seed': seed, 'fold': fold, 'completed_branch_runs': completed, 'total_branch_runs': total})
                vi = np.array([sid[z] for z in val])
                tr = np.concatenate([idx[z] for z in train])
                tf.keras.utils.set_random_seed(seed)
                if (d / 'model.keras').exists() and (d / 'history.json').exists():
                    full = tf.keras.models.load_model(d / 'model.keras', compile=False)
                    head = full.layers[-1]
                    history = read(d / 'history.json')
                    selected = min(history, key=lambda r: r['validation_bce'])['epoch']
                    seconds = 0
                    te = np.concatenate([idx[z] for z in test])
                    reload_delta, verification = verify_native(full, enc, head, features, te, kind, inputs)
                    protocol = {'epochs': 50, 'learning_rate': 0.0001, 'batch_size': 8, 'selected_epoch': selected, 'selection': plan['checkpoint_selection'], 'trainable_layers': [dense_name, 'prediction'], 'source_model': spec['source_model'], 'source_sha256': sourcehash, 'actual_fitting_ids': train, 'validation_ids': val, 'test_ids': test, 'recovered_saved_selected_model_after_parity_failure': True, 'max_cached_reloaded_native_test_difference': reload_delta, 'parity_tolerance': 0.0001, 'verification': verification, 'inference_batch_size': 1, 'checkpoint_validation_batching': 'All cached run samples', 'inference_batch_size': 1, 'checkpoint_validation_batching': 'All cached run samples, as in original training'}
                    write(d / 'protocol.json', protocol)
                else:
                    head = make_head()
                    np.savez(d / 'initial_head_weights.npz', **{f'weight{k}': w for k, w in enumerate(initial)})
                    opt = tf.keras.optimizers.Adam(plan['learning_rate'])
                    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 17]))
                    best = bce(sy[vi], pooled(head(features, training=False).numpy().ravel())[vi])
                    bestw = head.get_weights()
                    selected = 0
                    updates = 0
                    history = [{'epoch': 0, 'validation_bce': best}]
                    for epoch in range(1, plan['epochs'] + 1):
                        order = rng.permutation(tr)
                        losses = []
                        for start in range(0, len(order), plan['batch_size']):
                            batch = order[start:start + plan['batch_size']]
                            with tf.GradientTape() as tape:
                                prob = head(features[batch], training=True)
                                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(ry[batch, None], prob))
                            grads = tape.gradient(loss, head.trainable_variables)
                            assert all((np.isfinite(g.numpy()).all() for g in grads))
                            opt.apply_gradients(zip(grads, head.trainable_variables))
                            updates += 1
                            losses.append(float(loss))
                        vp = pooled(head(features, training=False).numpy().ravel())[vi]
                        v = bce(sy[vi], vp)
                        history.append({'epoch': epoch, 'training_bce': float(np.mean(losses)), 'validation_bce': v})
                        if v < best:
                            best = v
                            bestw = head.get_weights()
                            selected = epoch
                    head.set_weights(bestw)
                    write(d / 'history.json', history)
                    full = tf.keras.Model(enc.inputs, head(enc.output), name=series)
                    full.save_weights(d / 'selected.weights.h5')
                    full.save(d / 'model.keras', save_format='keras_v3')
                    loaded = tf.keras.models.load_model(d / 'model.keras', compile=False)
                    te = np.concatenate([idx[z] for z in test])
                    reload_delta, verification = verify_native(loaded, enc, head, features, te, kind, inputs)
                    protocol = {'epochs': plan['epochs'], 'learning_rate': plan['learning_rate'], 'batch_size': plan['batch_size'], 'selected_epoch': selected, 'selection': plan['checkpoint_selection'], 'gradient_updates': updates, 'trainable_layers': [dense_name, 'prediction'], 'source_model': spec['source_model'], 'source_sha256': sourcehash, 'source_historical_controller': spec['historical_controller'], 'new_controller_calls': 0, 'new_hpo_search': False, 'fresh_from_scratch_training': False, 'initialization': 'inherited', 'frozen_preceding_encoder': True, 'dropout_active': False, 'actual_fitting_ids': train, 'validation_ids': val, 'test_ids': test, 'true_labels_unchanged': True, 'max_cached_reloaded_native_test_difference': reload_delta, 'parity_tolerance': 0.0001, 'verification': verification, 'inference_batch_size': 1, 'checkpoint_validation_batching': 'All cached run samples', 'selected_weight_l2_change': float(np.sqrt(sum((np.sum((a - b) ** 2) for a, b in zip(head.get_weights(), initial)))))}
                    write(d / 'protocol.json', protocol)
                    del loaded
                    seconds = time.time() - tick
                runp = np.array([float(head(features[i:i + 1], training=False).numpy().ravel()[0]) for i in range(len(features))])
                subjectp = pooled(runp)
                embedding = pooled(head.layers[0](features).numpy())
                np.save(d / 'subject_embeddings.npy', embedding)
                rows = [{'subject_id': z, 'true_label': int(labels[z]), 'probability': float(subjectp[sid[z]]), 'run_probabilities': runp[idx[z]].tolist(), 'input_paths': [inputs[i]['path'] for i in idx[z]]} for z in subject_ids]
                result, trows, vrows = summarize_predictions(rows, s, seed, fold, series)
                result.update(series=series, seed=seed, fold=fold, selected_epoch=selected, model_sha256=sha(d / 'model.keras'), reload_max_difference=reload_delta, seconds=seconds, source_model=spec['source_model'])
                write(d / 'ALL_SUBJECT_PREDICTIONS.json', rows)
                write(d / 'test_predictions.json', trows)
                write(d / 'validation_predictions.json', vrows)
                write(d / 'SPLIT.json', s)
                write(d / 'RESULT_CURRENT.json', result)
                if series != 'mri_baseline':
                    completed += 1
                print(json.dumps({'series': series, 'seed': seed, 'fold': fold, 'completed': completed, 'fixed_accuracy': result['fixed_metrics']['accuracy'], 'calibrated_accuracy': result['calibrated_metrics']['accuracy']}), flush=True)
                del full, head
                gc.collect()
        assert sha(source) == sourcehash
        del base, enc, features
        tf.keras.backend.clear_session()
        gc.collect()
    write(O / 'BRANCH_COMPLETE.json', {'completed': completed, 'total': total})
    write(O / 'progress.json', {'stage': 'BRANCHES_COMPLETE', 'completed_branch_runs': completed, 'total_branch_runs': total})
    print('BRANCHES COMPLETE', flush=True)
except Exception as e:
    write(O / 'FAILED.json', {'error': repr(e), 'traceback': traceback.format_exc()})
    raise
finally:
    lock.unlink(missing_ok=True)
