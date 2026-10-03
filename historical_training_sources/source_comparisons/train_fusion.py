raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import os, json, time, gc, traceback
from pathlib import Path
import numpy as np
import tensorflow as tf
from evaluation_common import P, O, plan, read, write, sha, score, bce, subject_ids, labels, sy, sid, summarize_predictions
lock = O / 'FUSION.lock'
with lock.open('x') as f:
    f.write(str(os.getpid()))
try:
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    count = 0
    for mode in ['baseline', 'agentic']:
        while any((len(list((O / 'branches' / f'{kind}_{mode}').glob('*/RESULT_CURRENT.json'))) != 15 for kind in ['mri', 'fmri'])):
            write(O / 'fusion_progress.json', {'stage': 'WAITING_FOR_MODE_BRANCHES', 'mode': mode, 'completed_fusion_runs': count, 'total_fusion_runs': 180})
            time.sleep(5)
        for seed in plan['seeds']:
            for fold in plan['folds']:
                tick = time.time()
                m = O / 'branches' / f'mri_{mode}' / f'seed{seed}_fold{fold}'
                f = O / 'branches' / f'fmri_{mode}' / f'seed{seed}_fold{fold}'
                s = read(m / 'SPLIT.json')
                fs = read(f / 'SPLIT.json')
                for k in ['train', 'validation', 'test']:
                    assert s[k] == fs[k]
                mr = {r['subject_id']: r for r in read(m / 'ALL_SUBJECT_PREDICTIONS.json')}
                fr = {r['subject_id']: r for r in read(f / 'ALL_SUBJECT_PREDICTIONS.json')}
                mp = np.array([mr[z]['probability'] for z in subject_ids])
                fp = np.array([fr[z]['probability'] for z in subject_ids])
                xx = np.column_stack([mp, fp])
                em = np.load(m / 'subject_embeddings.npy')
                ef = np.load(f / 'subject_embeddings.npy')
                assert len(em) == 42 and len(ef) == 42
                features = np.concatenate([em, ef], axis=1)
                tr = np.array([sid[z] for z in s['train']])
                vi = np.array([sid[z] for z in s['validation']])
                alphas = np.linspace(0, 1, 101)
                vl = [bce(sy[vi], a * mp[vi] + (1 - a) * fp[vi]) for a in alphas]
                chosen = min(range(len(alphas)), key=lambda i: (vl[i], abs(alphas[i] - 0.5), -alphas[i]))
                alpha = float(alphas[chosen])
                for method in plan['fusion_methods']:
                    d = O / 'fusion' / mode / method / f'seed{seed}_fold{fold}'
                    d.mkdir(parents=True, exist_ok=True)
                    write(O / 'fusion_progress.json', {'stage': 'FUSION', 'mode': mode, 'method': method, 'seed': seed, 'fold': fold, 'branch_completion_scope': 'Required mode branch sources complete', 'completed_fusion_runs': count, 'total_fusion_runs': 180})
                    tf.keras.utils.set_random_seed(seed)
                    protocol = {'mode': mode, 'method': method, 'seed': seed, 'fold': fold, 'mri_branch_model': str((m / 'model.keras').relative_to(O)), 'mri_model_sha256': sha(m / 'model.keras'), 'fmri_branch_model': str((f / 'model.keras').relative_to(O)), 'fmri_model_sha256': sha(f / 'model.keras'), 'true_labels_unchanged': True, 'threshold_selection': plan['threshold_selection'], 'input_subject_order': subject_ids, 'fitting_ids': s['train'], 'validation_ids': s['validation'], 'test_ids': s['test']}
                    if method == 'equal_soft_voting':
                        pp = 0.5 * mp + 0.5 * fp
                        protocol.update(mri_weight=0.5, fmri_weight=0.5, weight_selection='fixed before experiment')
                    elif method == 'fixed_weighted':
                        pp = 0.4 * mp + 0.6 * fp
                        protocol.update(mri_weight=0.4, fmri_weight=0.6, weight_selection='fixed 0.4/0.6 before experiment')
                    elif method == 'validation_weighted':
                        pp = alpha * mp + (1 - alpha) * fp
                        protocol.update(mri_weight=alpha, fmri_weight=1 - alpha, weight_selection='minimum validation BCE; grid 0..1 by 0.01; no test selection', validation_weight_grid=[{'mri_weight': float(a), 'validation_bce': float(v)} for a, v in zip(alphas, vl)])
                    elif method == 'hard_voting':
                        pp = ((mp >= 0.5).astype(float) + (fp >= 0.5).astype(float)) / 2
                        protocol.update(rule='Vote fraction from two branch decisions at 0.5. At fixed threshold 0.5, ties resolve positive.', score_interpretation='Vote fraction, not a fitted diagnostic probability')
                    else:
                        X = xx if method == 'stacking' else features
                        mean = X[tr].mean(axis=0)
                        std = X[tr].std(axis=0)
                        std = np.where(std < 1e-06, 1, std)
                        X = ((X - mean) / std).astype(np.float32)
                        np.save(d / 'inputs_all_subjects.npy', X)
                        np.savez(d / 'training_normalization.npz', mean=mean, std=std)
                        head = tf.keras.Sequential([tf.keras.layers.Input(shape=(X.shape[1],)), tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='zeros')]) if method == 'stacking' else tf.keras.Sequential([tf.keras.layers.Input(shape=(X.shape[1],)), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')])
                        head(X[:1])
                        init = [w.copy() for w in head.get_weights()]
                        np.savez(d / 'initial_weights.npz', **{f'weight{k}': w for k, w in enumerate(init)})
                        opt = tf.keras.optimizers.Adam(plan['fusion_head_learning_rate'])
                        rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 39]))
                        best = bce(sy[vi], head(X[vi]).numpy().ravel())
                        bestw = head.get_weights()
                        selected = 0
                        history = [{'epoch': 0, 'validation_bce': best}]
                        for epoch in range(1, plan['fusion_head_epochs'] + 1):
                            order = rng.permutation(tr)
                            losses = []
                            for start in range(0, len(order), 8):
                                ix = order[start:start + 8]
                                with tf.GradientTape() as tape:
                                    prob = head(X[ix], training=True)
                                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(sy[ix, None].astype(np.float32), prob))
                                grads = tape.gradient(loss, head.trainable_variables)
                                assert all((np.isfinite(g.numpy()).all() for g in grads))
                                opt.apply_gradients(zip(grads, head.trainable_variables))
                                losses.append(float(loss))
                            valbce = bce(sy[vi], head(X[vi]).numpy().ravel())
                            history.append({'epoch': epoch, 'training_bce': float(np.mean(losses)), 'validation_bce': valbce})
                            if valbce < best:
                                best = valbce
                                bestw = head.get_weights()
                                selected = epoch
                        head.set_weights(bestw)
                        head.save_weights(d / 'selected.weights.h5')
                        head.save(d / 'model.keras', save_format='keras_v3')
                        native = tf.keras.models.load_model(d / 'model.keras', compile=False)
                        pp = native(X).numpy().ravel()
                        delta = float(np.max(abs(pp - head(X).numpy().ravel())))
                        assert delta <= 1e-06
                        write(d / 'history.json', history)
                        protocol.update(selected_epoch=selected, epochs=plan['fusion_head_epochs'], learning_rate=plan['fusion_head_learning_rate'], normalization='Training children mean/std only', native_reload_max_difference=delta, model_sha256=sha(d / 'model.keras'), model_input='Normalized cached branch probability pair' if method == 'stacking' else 'Normalized cached trained branch embedding concatenation', nested_oof_training=False, limitation='Stacking receives in-sample branch predictions on training children, not nested OOF' if method == 'stacking' else '')
                        del head, native
                    rows = [{'subject_id': z, 'true_label': int(labels[z]), 'probability': float(pp[sid[z]])} for z in subject_ids]
                    result, test, val = summarize_predictions(rows, s, seed, fold, f'fusion_{mode}_{method}')
                    result.update(series=f'fusion_{mode}_{method}', seed=seed, fold=fold)
                    write(d / 'ALL_SUBJECT_PREDICTIONS.json', rows)
                    write(d / 'test_predictions.json', test)
                    write(d / 'validation_predictions.json', val)
                    write(d / 'SPLIT.json', s)
                    write(d / 'protocol.json', protocol)
                    write(d / 'RESULT_CURRENT.json', result)
                    count += 1
                    print(json.dumps({'mode': mode, 'method': method, 'seed': seed, 'fold': fold, 'completed_fusion': count, 'fixed_accuracy': result['fixed_metrics']['accuracy'], 'calibrated_accuracy': result['calibrated_metrics']['accuracy']}), flush=True)
                tf.keras.backend.clear_session()
                gc.collect()
    write(O / 'FUSION_COMPLETE.json', {'completed': count, 'total': 180})
    write(O / 'fusion_progress.json', {'stage': 'FUSION_COMPLETE', 'branch_completion_scope': 'Required mode branch sources complete', 'completed_fusion_runs': count, 'total_fusion_runs': 180})
except Exception as e:
    write(O / 'FUSION_FAILED.json', {'error': repr(e), 'traceback': traceback.format_exc()})
    raise
finally:
    lock.unlink(missing_ok=True)
