import json, hashlib, gc
from pathlib import Path
import tensorflow as tf
P = Path('/package')
O = Path('/out')
rows = []
for s in json.loads((O / 'PLAN.json').read_text())['series']:
    m = tf.keras.models.load_model(P / s['source_model'], compile=False)
    name = 'features128' if s['kind'] == 'mri' else 'dense64'
    enc = tf.keras.Model(m.inputs, m.get_layer(name).input)
    enc.trainable = False
    h = hashlib.sha256()
    count = 0
    for w in enc.get_weights():
        h.update(str(w.dtype).encode())
        h.update(str(w.shape).encode())
        h.update(w.tobytes())
        count += w.size
    configs = []
    for l in enc.layers:
        c = l.get_config()
        c.pop('trainable', None)
        configs.append({'class': l.__class__.__name__, 'config': c})
    h.update(json.dumps(configs, sort_keys=True).encode())
    rows.append({'series': s['series'], 'kind': s['kind'], 'encoder_fingerprint': h.hexdigest(), 'frozen_parameter_count': count})
    del m, enc
    tf.keras.backend.clear_session()
    gc.collect()
(O / 'SOURCE_ENCODER_EQUIVALENCE.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
