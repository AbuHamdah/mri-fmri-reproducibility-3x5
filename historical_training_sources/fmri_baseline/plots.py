from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = Path('/out')
rows = json.loads((O / 'POOLED_PER_SEED.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, k in zip(axes, ['accuracy', 'recall', 'f1']):
    for branch, label, color, dx in [('fmri_baseline', 'Baseline', '#3568ad', -0.18), ('fmri_agentic', 'Agentic', '#de7b24', 0.18)]:
        rr = sorted([r for r in rows if r['series'] == branch], key=lambda r: r['seed'])
        ax.bar(np.arange(3) + dx, [r['metrics'][k] * 100 for r in rr], width=0.34, label=label, color=color)
    ax.set_xticks(range(3), ['42', '43', '44'])
    ax.set_xlabel('Seed')
    ax.set_ylabel(k.capitalize() + ' (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.2)
axes[0].legend()
fig.suptitle('Multiscale RBF residual: actual pooled42-child results\n', fontsize=11)
fig.tight_layout()
fig.savefig(O / 'PER_SEED_METRICS.png', dpi=160)
plt.close(fig)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for bi, branch in enumerate(['fmri_baseline', 'fmri_agentic']):
    for fi, fold in enumerate(range(1, 6)):
        ax = axes[bi, fi]
        for seed in [42, 43, 44]:
            d = O / 'branches' / branch / f'seed{seed}_fold{fold}'
            selected = json.loads((d / 'SELECTED_COEFFICIENTS.json').read_text())
            hist = json.loads((d / 'candidates' / selected['name'] / 'OPTIMIZER_HISTORY.json').read_text())
            ax.plot([r['iteration'] for r in hist], [r['train_objective'] for r in hist], label=str(seed))
        ax.set_title(branch.replace('fmri_', '') + f' partition{fold}', fontsize=9)
        ax.set_xlabel('Optimizer iteration')
        ax.set_ylabel('Train objective')
        ax.grid(alpha=0.2)
axes[0, 0].legend(title='Seed', fontsize=8)
fig.suptitle('Selected residual optimization traces (identity controls have no fitted trace)', fontsize=11)
fig.tight_layout()
fig.savefig(O / 'OPTIMIZER_CURVES.png', dpi=140)
plt.close(fig)
