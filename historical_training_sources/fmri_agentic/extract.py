import json, hashlib, time
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[1]
P = Path('D:\\MRI_FMRI_LAST_LAYER_ALL_3X5_V1')
A = Path('D:\\MRI_FMRI_FMRI_BATCH3_RUN_RELIABILITY_3X5_V1')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def descriptors(path):
    a = np.load(path).astype('float64')
    assert a.shape == (120, 64, 64, 1) and np.isfinite(a).all()
    a = a[..., 0]
    fm = a.mean((1, 2))
    fs = a.std((1, 2))
    return np.array([a.mean(), a.std(), fm.std(), fs.mean(), np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=2)).mean(), a[:, 24:40, 24:40].mean() - a.mean(), np.mean(a != 0)])

def generate():
    start = time.time()
    inputs = [r for r in read(P / 'INPUTS.json') if r['kind'] == 'fmri']
    ca = read(A / 'cache' / 'fmri_agentic' / 'SOURCE_RUN_PROBABILITIES.json')
    records = []
    for sid, label in zip(ca['subject_ids'], ca['labels']):
        items = sorted([r for r in inputs if r['subject_id'] == sid], key=lambda z: z['path'])
        assert len(items) == 3 and all((r['label'] == label for r in items))
        xx = np.stack([descriptors(P / r['path']) for r in items])
        records.append({'subject_id': sid, 'true_label': label, 'input_paths': [r['path'] for r in items], 'input_sha256': [sha(P / r['path']) for r in items], 'per_run_descriptors': xx.tolist(), 'features': np.r_[xx.mean(0), xx.std(0)].tolist()})
    return {'records': records, 'descriptor_names': ['global_mean', 'global_std', 'frame_mean_std', 'mean_frame_spatial_std', 'mean_abs_axis1_difference', 'mean_abs_axis2_difference', 'center16_mean_minus_global_mean', 'nonzero_fraction'], 'child_aggregation': 'Mean and population SD of each descriptor over three ordered runs', 'axis_caveat': 'Processed frame-axis numerical descriptors only; no verified temporal BOLD, motion or clinical QC interpretation', 'input_scope': 'Saved processed arrays, not raw NIfTI', 'actual_seconds': time.time() - start}
if __name__ == '__main__':
    assert not (O / 'INPUT_FEATURES.json').exists()
    (O / 'INPUT_FEATURES.json').write_text(json.dumps(generate(), indent=2))
    print('Extracted16 numeric child features from126 processed arrays')
