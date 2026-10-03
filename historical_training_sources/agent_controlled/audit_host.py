import json, hashlib, shutil, html, importlib.util, re, collections
from pathlib import Path
import numpy as np
import h5py
R = Path('D:/MRI_FMRI_QWEN_CONTROLLED_FEATURE_FUSION_3X5_V26')
S = Path('D:/MRI_FMRI_BOOTSTRAP_RESIDUAL_3X5_V24')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')
assert read(R / 'NATIVE_REPLAY_AUDIT.json')['passed'] and read(R / 'AGENT_INPUT_NATIVE_REPLAY_AUDIT.json')['passed']
shutil.copy2(__file__, R / 'source/audit_host.py')
plan = read(R / 'PLAN.json')
locks = read(R / 'ALL_TRAINING_MODELS_FROZEN.json')
assert locks['plan_sha256'] == sha(R / 'PLAN.json')
z = np.load(R / 'cache/FEATURES.npz')
assert sha(R / 'cache/FEATURES.npz') == plan['features_sha256']
y = z['labels']
ids = z['subject_ids'].tolist()
decisions = read(R / 'AGENT_DECISION_INDEX.json')
counts = collections.Counter((a['action'] for a in decisions))
checks = []
successful_wait = 0
api_attempts = 0
server_time = 0
prompt_tokens = 0
generated_tokens = 0
for lock in locks['records']:
    seed, fold = (lock['seed'], lock['fold'])
    d = R / f'runs/seed{seed}_fold{fold}'
    split = read(d / 'SPLIT.json')
    sets = [set(split[k]) for k in ['train', 'validation', 'test']]
    assert set.union(*sets) == set(ids) and all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
    assert sha(d / 'model.keras') == lock['model_sha256'] and sha(d / 'TRAINING_HISTORY.json') == lock['history_sha256']
    assert split == read(S / f'runs/seed{seed}_fold{fold}/SPLIT.json')
    sample = read(d / 'BOOTSTRAP_SAMPLE.json')
    assert set(sample['sampled_child_ids']).issubset(sets[0]) and sample == read(S / f'runs/seed{seed}_fold{fold}/BOOTSTRAP_SAMPLE.json')
    e = read(d / 'TRAINING_EVIDENCE.json')
    assert e['initial_weight_sha256'] == read(d / 'REFERENCE_TRAINING_EVIDENCE.json')['initial_weight_sha256']
    history = read(d / 'TRAINING_HISTORY.json')
    assert [h['epoch'] for h in history] == list(range(1, e['epochs_trained'] + 1)) and e['epochs_trained'] <= 25
    for action in e['actions']:
        before, after = (action['before'], action['after'])
        epoch = action['epoch']
        assert before['optimizer_iterations'] == after['optimizer_iterations'] == history[epoch - 1]['optimizer_iterations']
        folder = d / 'agent_decisions' / f'epoch{epoch}'
        attempt = folder / f"attempt{action['successful_attempt']}"
        assert sha(attempt / 'REQUEST.json') == action['request_sha256'] and sha(attempt / 'RAW_RESPONSE.json') == action['response_sha256']
        raw = read(attempt / 'RAW_RESPONSE.json')
        request = read(attempt / 'REQUEST.json')
        payload = json.loads(request['messages'][1]['content'])
        assert set(payload) == {'epoch', 'max_epochs', 'learning_rate', 'dropout', 'recent_history', 'training', 'validation', 'actions'}
        assert json.loads(raw['message']['content'])['action'] == action['action'] and raw['model'] == 'qwen2.5:3b'
        successful_wait += action['latency_seconds']
        if action['action'] == 'reduce_lr':
            assert abs(after['learning_rate'] - 0.5 * before['learning_rate']) < 1e-10 and after['dropout'] == before['dropout']
        elif action['action'] == 'increase_dropout':
            assert abs(after['dropout'] - min(before['dropout'] + 0.1, 0.5)) < 1e-12 and action['training_function_retrace_requested']
        elif action['action'] == 'stop':
            assert e['epochs_trained'] == epoch
        else:
            assert before == after
        if epoch < e['epochs_trained']:
            next_h = history[epoch]
            assert abs(next_h['learning_rate'] - after['learning_rate']) < 1e-10 and abs(next_h['dropout'] - after['dropout']) < 1e-12 and (next_h['optimizer_iterations'] > after['optimizer_iterations'])
        checks.append(dict(seed=seed, fold=fold, epoch=epoch, action=action['action'], applied_setting_verified=True, subsequent_gradient_updates=epoch < e['epochs_trained']))
        for a in folder.glob('attempt*'):
            api_attempts += 1
            if (a / 'RAW_RESPONSE.json').exists():
                rr = read(a / 'RAW_RESPONSE.json')
                server_time += rr.get('total_duration', 0) / 1000000000.0
                prompt_tokens += rr.get('prompt_eval_count', 0)
                generated_tokens += rr.get('eval_count', 0)
    with h5py.File(d / 'model.keras', 'r') as model:
        config = json.loads(model.attrs['model_config'])
    rate = next((l['config']['rate'] for l in config['config']['layers'] if l['class_name'] == 'Dropout'))
    assert abs(rate - e['final_dropout']) < 1e-12
spec = importlib.util.spec_from_file_location('st', R / 'source/statistics_core.py')
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)
rows = read(R / 'TEST_PREDICTIONS.json')
assert len(rows) == 126
for p in read(R / 'POOLED_PER_SEED.json'):
    rr = sorted([r for r in rows if r['seed'] == p['seed']], key=lambda r: r['subject_id'])
    assert [r['subject_id'] for r in rr] == ids and all((r['true_label'] == int(y[ids.index(r['subject_id'])]) and r['prediction'] == int(r['probability'] >= 0.5) for r in rr))
    pp = np.array([r['probability'] for r in rr])
    a = dict(zip(st.NAMES, map(float, st.metrics(y, pp))))
    assert max((abs(a[k] - p['metrics'][k]) for k in st.NAMES)) < 1e-12
save(R / 'AGENT_COST.json', dict(actual_api_attempts=api_attempts, successful_decisions=len(decisions), successful_wait_seconds=successful_wait, server_total_seconds=server_time, prompt_tokens=prompt_tokens, generated_tokens=generated_tokens, monetary_cost_measured=False, timing_scope='Reported training seconds include inference/serialization overhead; successful agent waiting subtracted, not isolated pure gradient time.'))
tags = read(R / 'OLLAMA_TAGS.json')
q = next((v for v in tags['models'] if v['name'] == 'qwen2.5:3b'))
save(R / 'AGENT_PROVENANCE.json', dict(model=q['name'], digest=q['digest'], actual_api_attempts=api_attempts, successful_decisions=len(decisions), action_counts=dict(counts), role='Actual control of fresh fusion-head gradient training: LR, Dropout, continuation/stopping', encoders_and_original_branch_training_not_controlled=True))
save(R / 'INDEPENDENT_HOST_AUDIT.json', dict(passed=True, models=15, initializations_match_fixed_reference=15, controller_action_checks=checks, current_test_rows=126, unique_children=42, pooled_metric_recalculations=3, choices_locked_before_current_test=True))
summary = read(R / 'SUMMARY.json')
table = ''.join((f"<tr><td>{k}</td><td>{(100 if k in ['accuracy', 'recall', 'precision', 'f1', 'specificity'] else 1) * v['mean']:.4f} ± {(100 if k in ['accuracy', 'recall', 'precision', 'f1', 'specificity'] else 1) * v['sd']:.4f}</td></tr>" for k, v in summary['metrics'].items()))
action_rows = ''.join((f"<tr><td>{a['seed']}</td><td>{a['fold']}</td><td>{a['epoch']}</td><td>{a['action']}</td><td>{a['after']['learning_rate']:.8f}</td><td>{a['after']['dropout']:.2f}</td></tr>" for a in decisions))
(R / 'REPORT_EN.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Agent-controlled Feature Fusion</title><h1>Agent-controlled Feature Fusion — Qwen training controller</h1><p>Actual local Qwen2.5:3b calls control fresh gradient training of the fusion head at epochs10 and20: continue, halve learning rate, increase Dropout, or stop. Actions are applied before subsequent gradient updates. The selected model is the final trained epoch under this control; all15 runs are retained and locked before current test scoring.</p><p> The encoders and original branches were not newly agent-controlled.  Prior cohort test results were repeatedly examined. Same initial head weights and bootstrap draws as the fixed-controller reference were verified; no proof of generalization or agent superiority is inferred.</p><table border="1"><tr><th>Metric (% for classification metrics)</th><th>Mean ± seed SD</th></tr>' + table + '</table><h2>Actual applied decisions</h2><p>' + html.escape(str(dict(counts))) + '</p><table border="1"><tr><th>Seed</th><th>Partition</th><th>Epoch</th><th>Action</th><th>Resulting LR</th><th>Resulting Dropout</th></tr>' + action_rows + '</table><p><a href="AGENT_PROVENANCE.json">Qwen identity and action counts</a> | <a href="AGENT_DECISION_INDEX.json">All decisions</a> | <a href="AGENT_COST.json">Actual call costs</a> | <a href="TRAINING_INDEX.json">Training evidence</a> | <a href="ALL_TRAINING_MODELS_FROZEN.json">Pre-test model locks</a> | <a href="SUMMARY.json">Metrics</a> | <a href="POOLED_PER_SEED.json">Seed metrics</a> | <a href="TEST_PREDICTIONS.json">Child predictions</a> | <a href="INDEPENDENT_HOST_AUDIT.json">Host audit</a> | <a href="AGENT_INPUT_NATIVE_REPLAY_AUDIT.json">Agent input replay</a></p></html>', encoding='utf-8')
bad = []
for p in R.rglob('*.html'):
    for href in re.findall('href="([^"]+)"', p.read_text(encoding='utf-8')):
        if not (p.parent / href).is_file():
            bad.append(dict(file=str(p), link=href))
assert not bad
inv = [dict(path=p.relative_to(R).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in R.rglob('*') if p.is_file() and '__pycache__' not in str(p) and (p.name not in ['FINAL_INVENTORY.json', 'FINAL_DELIVERY_AUDIT.json'])]
save(R / 'FINAL_INVENTORY.json', inv)
assert all((sha(R / v['path']) == v['sha256'] for v in inv))
save(R / 'FINAL_DELIVERY_AUDIT.json', dict(passed=True, files=len(inv), all_hashes_verified=True, actual_api_attempts=api_attempts, successful_decisions=len(decisions), broken_links=bad))
print(json.dumps(summary))
print('Verified applied controller actions', dict(counts))
