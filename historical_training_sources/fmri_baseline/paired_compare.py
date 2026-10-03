import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
O = Path(__file__).resolve().parents[1]
records = json.loads((O / 'POOLED_PER_SEED.json').read_text())
by = {(r['series'], r['seed']): r for r in records}
result = []
differences = []
for seed in [42, 43, 44]:
    b = {r['subject_id']: r for r in by['fmri_baseline', seed]['predictions']}
    a = {r['subject_id']: r for r in by['fmri_agentic', seed]['predictions']}
    assert set(a) == set(b)
    better = []
    worse = []
    diff = []
    for sid in sorted(b):
        assert a[sid]['true_label'] == b[sid]['true_label']
        bc = b[sid]['predicted_label'] == b[sid]['true_label']
        ac = a[sid]['predicted_label'] == a[sid]['true_label']
        diff.append(int(ac) - int(bc))
        if ac and (not bc):
            better.append(sid)
        if bc and (not ac):
            worse.append(sid)
    result.append({'seed': seed, 'accuracy_difference_points': float(np.mean(diff) * 100), 'agentic_correct_baseline_wrong': better, 'baseline_correct_agentic_wrong': worse, 'exact_paired_mcnemar_p': float(binomtest(len(better), len(better) + len(worse), 0.5).pvalue) if better or worse else 1.0})
    differences.append(diff)
d = np.array(differences).mean(0)
rng = np.random.default_rng(42)
idx = rng.integers(0, 42, (10000, 42))
boot = d[idx].mean(1) * 100
ci = np.quantile(boot, [0.025, 0.975])
out = {'per_seed': result, 'mean_accuracy_difference_points': float(d.mean() * 100), 'paired_child_bootstrap95_difference_points': {'low': float(ci[0]), 'high': float(ci[1]), 'resamples': 10000, 'sampling_unit': 'Same child across all three seed evaluations;42 unique children, not126 independent observations'}, 'interval_includes_zero': bool(ci[0] <= 0 <= ci[1]), 'scope': ' No independent generalization or statistical LLM benefit is established.'}
(O / 'PAIRED_COMPARISON.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out))
