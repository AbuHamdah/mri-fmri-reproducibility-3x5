import json
from pathlib import Path
import numpy as np
import tensorflow as tf
P = Path('/package')
O = Path('/out')
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
series = 'mri_optuna'
d = O / 'branches' / series / 'seed43_fold2'
spec = next((s for s in json.loads((O / 'PLAN.json').read_text())['series'] if s['series'] == series))
source = tf.keras.models.load_model(P / spec['source_model'], compile=False)
model = tf.keras.models.load_model(d / 'model.keras', compile=False)
enc = tf.keras.Model(source.inputs, source.get_layer('features128').input)
renc = tf.keras.Model(model.inputs, model.get_layer('dropout1').output)
head = model.layers[-1]
inputs = sorted([r for r in json.loads((P / 'INPUTS.json').read_text()) if r['kind'] == 'mri'], key=lambda r: (r['subject_id'], r['path']))
ff = np.load(O / 'cache' / series / 'features.npy')
idx = {r['subject_id']: i for i, r in enumerate(inputs)}
test = json.loads((P / 'branches' / series / 'seed43_fold2/SPLIT.json').read_text())['test']
rows = []
for s in test:
    i = idx[s]
    x = np.load(P / inputs[i]['path'])[None]
    a = enc(x, training=False).numpy()
    b = renc(x, training=False).numpy()
    native = float(model(x, training=False).numpy().ravel()[0])
    cached = float(head(ff[i:i + 1], training=False).numpy().ravel()[0])
    rows.append({'id': s, 'prob_native': native, 'prob_cached': cached, 'prob_delta': abs(native - cached), 'source_vs_saved_encoder_max_difference': float(abs(a - b).max()), 'cached_vs_live_encoder_max_difference': float(abs(a - ff[i:i + 1]).max()), 'cache_feature_maxabs': float(abs(ff[i]).max())})
batchp = head(ff[[idx[z] for z in test]], training=False).numpy().ravel()
delta = float(np.max(abs(batchp - np.array([r['prob_native'] for r in rows]))))
result = {'batch_vs_single_probability_difference': delta, 'single_sample_cached_vs_native_max_difference': max((r['prob_delta'] for r in rows)), 'rows': rows}
(O / 'NATIVE_PARITY_FAILURE_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
