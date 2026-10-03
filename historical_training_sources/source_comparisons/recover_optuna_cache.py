import json, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
P = Path('/package')
O = Path('/out')
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
spec = next((s for s in json.loads((O / 'PLAN.json').read_text())['series'] if s['series'] == 'fmri_optuna'))
source = P / spec['source_model']
base = tf.keras.models.load_model(source, compile=False)
enc = tf.keras.Model(base.inputs, base.get_layer('dense64').input)
enc.trainable = False
inputs = sorted([r for r in json.loads((P / 'INPUTS.json').read_text()) if r['kind'] == 'fmri'], key=lambda r: (r['subject_id'], r['path']))
cache = O / 'cache/fmri_optuna'
features = np.load(cache / 'features.npy')
head = tf.keras.Sequential([tf.keras.layers.Dense.from_config(base.get_layer('dense64').get_config()), tf.keras.layers.Dense.from_config(base.get_layer('prediction').get_config())])
head(features[:1])
head.set_weights(base.get_layer('dense64').get_weights() + base.get_layer('prediction').get_weights())
batch = head(features).numpy().ravel()
single = np.array([float(head(features[i:i + 1]).numpy().ravel()[0]) for i in range(len(features))])
native = []
livefeats = []
extractor = tf.keras.Model(base.inputs, [enc.output, base.output])
for i, r in enumerate(inputs):
    f, p = extractor(np.load(P / r['path'])[None], training=False)
    livefeats.append(f.numpy())
    native.append(float(p.numpy().ravel()[0]))
    assert np.array_equal(f.numpy(), features[i:i + 1])
    delta = abs(native[-1] - single[i])
    assert delta <= 0.0001
    if (i + 1) % 10 == 0:
        (O / 'progress.json').write_text(json.dumps({'stage': 'VERIFY_ORPHAN_OPTUNA_CACHE', 'inputs_checked': i + 1, 'inputs_total': 126, 'completed_branch_runs': 195, 'total_branch_runs': 210}))
np.save(cache / 'source_native_probabilities.npy', np.array(native))
diagnostic = {'batch_vs_single_max_difference': float(abs(batch - single).max()), 'single_vs_source_native_max_difference': float(abs(single - native).max()), 'all_126_cached_features_byte_equal_to_live_source': True}
(O / 'FMRI_OPTUNA_CACHE_NUMERICAL_DIAGNOSTIC.json').write_text(json.dumps(diagnostic, indent=2))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()
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
ad = {'encoder_fingerprint': h.hexdigest(), 'source_sha256': sha(source), 'cache_parity_max_difference': diagnostic['single_vs_source_native_max_difference'], 'cache_inference_batch_size': 1, 'batching_numerical_difference': diagnostic['batch_vs_single_max_difference'], 'inputs': [dict(r, sha256=sha(P / r['path'])) for r in inputs]}
(cache / 'CACHE_AUDIT.json').write_text(json.dumps(ad, indent=2))
print(json.dumps(diagnostic))
