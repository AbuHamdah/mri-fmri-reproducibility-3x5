import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
p = argparse.ArgumentParser()
p.add_argument('--package', default='/package')
p.add_argument('--delivery', default='/out')
p.add_argument('--series', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--fold', type=int, required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
P = Path(a.package)
O = Path(a.delivery)
d = O / 'branches' / a.series / f'seed{a.seed}_fold{a.fold}'
rows = json.loads((d / 'test_predictions.json').read_text())
result = json.loads((d / 'RESULT_CURRENT.json').read_text())
model = tf.keras.models.load_model(d / 'model.keras', compile=False)
out = []
for row in rows:
    probs = [float(model(np.load(P / path)[None], training=False).numpy().ravel()[0]) for path in row['input_paths']]
    prob = float(np.mean(probs))
    out.append({'subject_id': row['subject_id'], 'probability': prob, 'run_probabilities': probs, 'threshold': result['threshold'], 'predicted_label': int(prob >= result['threshold']), 'stored_probability': row['probability'], 'absolute_difference': abs(prob - row['probability'])})
assert max((r['absolute_difference'] for r in out)) <= 0.0001
assert all((r['predicted_label'] == row['predicted_label'] for r, row in zip(out, rows)))
Path(a.output).write_text(json.dumps({'series': a.series, 'seed': a.seed, 'fold': a.fold, 'native_model_replay': out}, indent=2), encoding='utf8')
print('Native model replay matched stored probabilities within 1e-4')
