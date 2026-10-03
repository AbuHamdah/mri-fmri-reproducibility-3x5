import os, json, time, hashlib, traceback, gc
from pathlib import Path
import numpy as np
import tensorflow as tf
from statistics_core import metrics, NAMES
P = Path('/package')
O = Path('/out')
plan = json.loads((O / 'PLAN.json').read_text())
lock = O / 'RUN.lock'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, indent=2), encoding='utf8')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()

def score(y, p, t=0.5):
    return dict(zip(NAMES, metrics(y, p, t).tolist()))

def bce(y, p):
    p = np.clip(p, 1e-07, 1 - 1e-07)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))

def choose_threshold(y, p):
    u = np.unique(p)
    c = np.unique(np.r_[0, 0.5, 1, (u[:-1] + u[1:]) / 2])
    scores = []
    for t in c:
        m = score(y, p, t)
        scores.append({'threshold': float(t), 'f1': m['f1'], 'balanced_accuracy': (m['recall'] + m['specificity']) / 2})
    best = max(scores, key=lambda r: (r['f1'], r['balanced_accuracy'], -abs(r['threshold'] - 0.5), r['threshold']))
    return (best['threshold'], scores)

def summarize_predictions(rows, split, seed, fold, series):
    by = {r['subject_id']: r for r in rows}
    v = [by[s] for s in split['validation']]
    t = [by[s] for s in split['test']]
    y = np.array([r['true_label'] for r in v])
    p = np.array([r['probability'] for r in v])
    threshold, scores = choose_threshold(y, p)
    for r in rows:
        r.update(seed=seed, test_fold=fold, model_id=f'{series}_seed{seed}_fold{fold}', selected_threshold=threshold, predicted_label=int(r['probability'] >= threshold))
    yy = np.array([r['true_label'] for r in t])
    pp = np.array([r['probability'] for r in t])
    return ({'threshold': threshold, 'threshold_candidates_validation': scores, 'fixed_metrics': score(yy, pp), 'calibrated_metrics': score(yy, pp, threshold), 'test_n': len(t), 'fixed_correct': int(((pp >= 0.5) == yy).sum()), 'calibrated_correct': int(((pp >= threshold) == yy).sum())}, t, v)
allinputs = read(P / 'INPUTS.json')
subject_ids = sorted({r['subject_id'] for r in allinputs})
labels = {s: next((r['label'] for r in allinputs if r['subject_id'] == s)) for s in subject_ids}
sy = np.array([labels[s] for s in subject_ids])
sid = {s: i for i, s in enumerate(subject_ids)}
