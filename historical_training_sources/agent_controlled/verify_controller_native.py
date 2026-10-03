import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from statistics_core import metrics, NAMES
R = Path('/experiment')
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
z = np.load(R / 'cache/FEATURES.npz')
X = z['combined']
y = z['labels']
ids = z['subject_ids'].tolist()
checks = []
for action in read(R / 'AGENT_DECISION_INDEX.json'):
    d = R / f"runs/seed{action['seed']}_fold{action['fold']}"
    a = d / 'agent_decisions' / f"epoch{action['epoch']}" / f"attempt{action['successful_attempt']}"
    request = read(a / 'REQUEST.json')
    payload = json.loads(request['messages'][1]['content'])
    split = read(d / 'SPLIT.json')
    pair = np.load(d / 'INPUT_PROBABILITIES.npz')
    p = np.clip(pair['pairs'].mean(1).astype(np.float32), 1e-07, 1 - 1e-07)
    base = (np.log(p) - np.log1p(-p))[:, None]
    model = tf.keras.models.load_model(d / f"checkpoint_before_agent_epoch{action['epoch']}.keras", compile=False)
    values = {}
    for part in ['train', 'validation']:
        ix = np.array([ids.index(s) for s in split[part]])
        pp = model([X[ix], base[ix]], training=False).numpy().ravel()
        vals = dict(zip(NAMES, map(float, metrics(y[ix], pp))))
        actual = dict(n=len(ix), balanced_accuracy=round(0.5 * (vals['recall'] + vals['specificity']), 6), accuracy=round(vals['accuracy'], 6), brier=round(vals['brier'], 6))
        key = 'training' if part == 'train' else 'validation'
        assert actual == payload[key], (d, part, actual, payload[key])
        values[part] = actual
    checks.append(dict(seed=action['seed'], fold=action['fold'], epoch=action['epoch'], agent_metrics_replayed=True, metrics=values))
(R / 'AGENT_INPUT_NATIVE_REPLAY_AUDIT.json').write_text(json.dumps(dict(passed=True, decisions=len(checks), checkpoint_train_validation_replays=2 * len(checks), records=checks), indent=2), encoding='utf-8')
print('Replayed actual train/validation evidence for', len(checks), 'controller decisions.')
