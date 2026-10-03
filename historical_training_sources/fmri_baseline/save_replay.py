import json, hashlib, time
from pathlib import Path
import numpy as np
import tensorflow as tf
O = Path('/out')
P = Path('/package')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
start = time.time()
records = []
for branch in ['fmri_baseline', 'fmri_agentic']:
    ca = read(O / 'cache' / branch / 'SOURCE_RUN_PROBABILITIES.json')
    assert sha(P / ca['source_model']) == ca['source_model_sha256']
    raw = np.array(ca['three_run_probabilities'])
    q = np.clip(raw.astype('float32').mean(1).astype('float64'), 1e-12, 1 - 1e-12)
    inputs = np.c_[raw, np.log(q / (1 - q))]
    for seed in [42, 43, 44]:
        for fold in [1, 2, 3, 4, 5]:
            d = O / 'branches' / branch / f'seed{seed}_fold{fold}'
            norm = read(d / 'NORMALIZATION.json')
            c = read(d / 'SELECTED_COEFFICIENTS.json')
            w = np.array(c['coefficients'])
            inp = tf.keras.Input((4,), dtype='float64', name='three_source_run_probabilities_and_parent_logit')
            offset = tf.keras.layers.Dense(1, use_bias=False, dtype='float64', name='frozen_parent_logit')(inp)
            offset_layer = None
            model_offset = tf.keras.Model(inp, offset)
            model_offset.layers[-1].set_weights([np.array([[0.0], [0.0], [0.0], [1.0]])])
            logit = offset
            if c['gamma'] is not None:
                norm_layer = tf.keras.layers.Dense(3, dtype='float64', name='train_only_normalization')
                z = norm_layer(inp)
                norm_layer.set_weights([np.r_[np.diag(1 / np.array(norm['std'])), np.zeros((1, 3))], -np.array(norm['mean']) / np.array(norm['std'])])
                radial = []
                for i, center in enumerate(norm['normalized_centers']):
                    subtract = tf.keras.layers.Dense(3, dtype='float64', name=f'center{i}')
                    delta = subtract(z)
                    subtract.set_weights([np.eye(3), -np.array(center)])
                    squared = tf.keras.layers.Multiply(dtype='float64')([delta, delta])
                    distance = tf.keras.layers.Dense(1, use_bias=False, activation='exponential', dtype='float64', name=f'rbf{i}')
                    radial.append(distance(squared))
                    distance.set_weights([np.full((3, 1), -c['gamma'] / 3)])
                features = tf.keras.layers.Concatenate(dtype='float64')(radial)
                residual = tf.keras.layers.Dense(1, dtype='float64', name='trained_residual_coefficients')
                r = residual(features)
                residual.set_weights([w[:-1, None], w[-1:]])
                logit = tf.keras.layers.Add(dtype='float64')([offset, r])
            prob = tf.keras.layers.Activation('sigmoid', dtype='float64')(logit)
            model = tf.keras.Model(inp, prob)
            model.trainable = False
            model.save(d / 'model.keras', save_format='keras_v3')
            loaded = tf.keras.models.load_model(d / 'model.keras', compile=False)
            p = loaded(inputs, training=False).numpy().ravel()
            rows = read(d / 'ALL_SUBJECT_PREDICTIONS.json')
            expected = np.array([r['probability'] for r in rows])
            error = float(np.max(abs(p - expected)))
            assert error <= 1e-06 and np.array_equal(p >= 0.5, expected >= 0.5), (branch, seed, fold, error)
            records.append({'branch': branch, 'seed': seed, 'fold': fold, 'children': 42, 'maximum_probability_difference': error, 'all_classifications_match': True, 'model_sha256': sha(d / 'model.keras')})
            (O / 'NATIVE_SAVE_PROGRESS.json').write_text(json.dumps({'completed': len(records), 'total': 30}))
            print(json.dumps(records[-1]), flush=True)
            tf.keras.backend.clear_session()
(O / 'NATIVE_MODEL_REPLAY_AUDIT.json').write_text(json.dumps({'passed': True, 'selected_models': 30, 'all_subject_head_predictions': 1260, 'model_records': records, 'actual_seconds': time.time() - start, 'scope': 'All selected saved Keras heads on preserved native source probabilities. Parent/input native replay is inherited and separately audited; not raw NIfTI replay or independentCV.'}, indent=2))
print('All30 native saved heads replayed')
