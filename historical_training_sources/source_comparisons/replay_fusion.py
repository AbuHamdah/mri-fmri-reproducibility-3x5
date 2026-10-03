"""Replay fusion heads/decision rules from stored branch outputs or cached head inputs. This is not a raw-image branch replay."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
p = argparse.ArgumentParser()
p.add_argument('--delivery', default='/out')
p.add_argument('--mode', choices=['baseline', 'agentic'], required=True)
p.add_argument('--method', choices=['equal_soft_voting', 'fixed_weighted', 'validation_weighted', 'hard_voting', 'stacking', 'feature'], required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--fold', type=int, required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
O = Path(a.delivery)
d = O / 'fusion' / a.mode / a.method / f'seed{a.seed}_fold{a.fold}'
proto = json.loads((d / 'protocol.json').read_text())
result = json.loads((d / 'RESULT_CURRENT.json').read_text())
stored = json.loads((d / 'test_predictions.json').read_text())
order = proto['input_subject_order']
if a.method in ['stacking', 'feature']:
    X = np.load(d / 'inputs_all_subjects.npy')
    model = tf.keras.models.load_model(d / 'model.keras', compile=False)
    pp = model(X, training=False).numpy().ravel()
else:
    probs = []
    for kind in ['mri', 'fmri']:
        rows = json.loads((O / 'branches' / f'{kind}_{a.mode}' / f'seed{a.seed}_fold{a.fold}' / 'ALL_SUBJECT_PREDICTIONS.json').read_text())
        by = {r['subject_id']: r['probability'] for r in rows}
        probs.append(np.array([by[s] for s in order]))
    m, f = probs
    if a.method == 'hard_voting':
        pp = ((m >= 0.5).astype(float) + (f >= 0.5).astype(float)) / 2
    else:
        pp = proto['mri_weight'] * m + proto['fmri_weight'] * f
ix = {s: i for i, s in enumerate(order)}
rows = []
for r in stored:
    prob = float(pp[ix[r['subject_id']]])
    label = int(prob >= result['threshold'])
    assert abs(prob - r['probability']) <= 1e-06 and label == r['predicted_label']
    rows.append({'subject_id': r['subject_id'], 'probability': prob, 'predicted_label': label, 'absolute_difference': abs(prob - r['probability'])})
Path(a.output).write_text(json.dumps({'mode': a.mode, 'method': a.method, 'seed': a.seed, 'fold': a.fold, 'scope': 'Replay from stored branch probability/embedding inputs; not a raw-image end-to-end branch replay', 'predictions': rows}, indent=2))
print('Fusion cached-input replay passed')
