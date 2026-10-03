import json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.special import expit
import tensorflow as tf
from statistics_core import metrics
O = Path('/out')
P = Path('/package')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = read(O / 'PLAN.json')
features = read(O / 'INPUT_FEATURES.json')['records']
records = []
start = time.time()
for branch in plan['branches']:
    ca = read(O / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json')
    assert sha(P / ca['source_model']) == ca['source_model_sha256']
    ids = ca['subject_ids']
    assert ids == [r['subject_id'] for r in features]
    y = np.array(ca['labels'])
    source = np.array(ca['three_run_probabilities'])
    x = np.c_[source, np.array([r['features'] for r in features])]
    q = np.clip(source.astype('float32').mean(1).astype('float64'), 1e-12, 1 - 1e-12)
    offset = np.log(q / (1 - q))
    inputs = np.c_[x, offset]
    for seed in plan['seeds']:
        for fold in plan['folds']:
            d = O / 'branches' / branch / f'seed{seed}_fold{fold}'
            s = read(d / 'SPLIT.json')
            sets = [set(s[k]) for k in ['train', 'validation', 'test']]
            assert all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
            tr = np.array([ids.index(z) for z in s['train']])
            vi = np.array([ids.index(z) for z in s['validation']])
            counts = np.bincount(y[tr], minlength=2)
            assert np.all(counts > 0)
            sample_weights = len(tr) / (2 * counts[y[tr]])
            cuts = []
            for j in range(19):
                values = np.unique(x[tr, j])
                for threshold in (values[:-1] + values[1:]) / 2:
                    left = x[tr, j] < threshold
                    if left.sum() >= 3 and (~left).sum() >= 3:
                        cuts.append({'feature': j, 'threshold': float(threshold)})
            allowed = read(d / 'ALLOWED_TRAIN_SPLITS.json')
            assert cuts == allowed['cuts'] and allowed['fit_ids'] == s['train']
            mask = np.stack([x[:, c['feature']] < c['threshold'] for c in cuts], axis=1)
            M = mask[tr].astype('float64')
            cc = read(d / 'VALIDATION_CANDIDATES.json')
            assert len(cc) == 19
            for c in cc:
                z = offset.copy()
                lr = c['learning_rate']
                l2 = c['leaf_l2']
                for t in c['trees']:
                    p = expit(z[tr])
                    g = sample_weights * (y[tr] - p)
                    h = sample_weights * p * (1 - p)
                    GL = g @ M
                    HL = h @ M
                    GR = g.sum() - GL
                    HR = h.sum() - HL
                    gain = GL * GL / (HL + l2) + GR * GR / (HR + l2) - g.sum() ** 2 / (h.sum() + l2)
                    k = int(np.argmax(gain))
                    assert cuts[k]['feature'] == t['feature'] and cuts[k]['threshold'] == t['threshold']
                    left = float(np.clip(GL[k] / (HL[k] + l2), -2, 2))
                    right = float(np.clip(GR[k] / (HR[k] + l2), -2, 2))
                    assert abs(left - t['left_correction']) < 1e-12 and abs(right - t['right_correction']) < 1e-12
                    z += lr * np.where(mask[:, k], left, right)
                p = expit(z)
                m = metrics(y[vi], p[vi])
                loss = float(np.mean(np.logaddexp(0, z[vi]) - y[vi] * z[vi]))
                assert abs(c['validation_balanced_accuracy'] - (m[5] + m[6]) / 2) < 1e-12 and abs(c['validation_bce'] - loss) < 1e-12
            chosen = max(cc, key=lambda c: (c['validation_balanced_accuracy'], -c['validation_bce'], -len(c['trees']), float('inf') if c['leaf_l2'] is None else c['leaf_l2'], -c['learning_rate']))
            assert chosen == read(d / 'SELECTED_CHECKPOINT.json')
            inp = tf.keras.Input((20,), dtype='float64')
            base_layer = tf.keras.layers.Dense(1, use_bias=False, dtype='float64', name='frozen_parent_logit')
            base = base_layer(inp)
            base_layer.set_weights([np.r_[np.zeros((19, 1)), np.ones((1, 1))]])
            outputs = [base]
            for i, t in enumerate(chosen['trees']):
                select = tf.keras.layers.Dense(1, use_bias=False, dtype='float64', name=f'feature{i}')
                v = select(inp)
                w = np.zeros((20, 1))
                w[t['feature']] = 1
                select.set_weights([w])
                left = tf.cast(tf.less(v, tf.constant(t['threshold'], dtype='float64')), tf.float64)
                correction = tf.keras.layers.Dense(1, dtype='float64', name=f'correction{i}')
                outputs.append(correction(left))
                lr = chosen['learning_rate']
                correction.set_weights([np.array([[lr * (t['left_correction'] - t['right_correction'])]]), np.array([lr * t['right_correction']])])
            logit = tf.keras.layers.Add(dtype='float64')(outputs) if len(outputs) > 1 else base
            out = tf.keras.layers.Activation('sigmoid', dtype='float64')(logit)
            model = tf.keras.Model(inp, out)
            model.trainable = False
            model.save(d / 'model.keras', save_format='keras_v3')
            loaded = tf.keras.models.load_model(d / 'model.keras', compile=False)
            p = loaded(inputs, training=False).numpy().ravel()
            rows = read(d / 'ALL_SUBJECT_PREDICTIONS.json')
            error = float(np.max(abs(p - [r['probability'] for r in rows])))
            assert error <= 1e-06
            for i, r in enumerate(rows):
                assert r['subject_id'] == ids[i] and r['true_label'] == y[i] and (r['predicted_label'] == int(p[i] >= 0.5))
            records.append({'series': branch, 'seed': seed, 'fold': fold, 'model_sha256': sha(d / 'model.keras'), 'maximum_probability_difference': error, 'all42_classifications_match': True, 'selected_rounds': len(chosen['trees'])})
            print(json.dumps({'completed': len(records), 'total': 30}), flush=True)
            tf.keras.backend.clear_session()
(O / 'NATIVE_MODEL_REPLAY_AUDIT.json').write_text(json.dumps({'passed': True, 'selected_models': 30, 'all_subject_predictions': 1260, 'current_train_only_split_thresholds_verified': True, 'every_candidate_tree_fit_recomputed': True, 'validation_choice_recomputed': True, 'true_labels_unchanged': True, 'records': records, 'actual_seconds': time.time() - start, 'source_native_replay_evidence': 'D:/MRI_FMRI_FMRI_BATCH3_RUN_RELIABILITY_3X5_V1/FULL_RAW_REPLAY_AUDIT.json'}, indent=2))
