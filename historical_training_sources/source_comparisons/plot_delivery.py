from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = Path('/out')
pooled = json.loads((O / 'POOLED_PER_SEED.json').read_text())
C = O / 'curves'
C.mkdir(exist_ok=True)
manifest = []
for series in sorted({r['series'] for r in pooled}):
    rows = [r for r in pooled if r['series'] == series]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for r in rows:
        pr = r['predictions']
        y = np.array([a['true_label'] for a in pr])
        p = np.array([a['probability'] for a in pr])
        ths = np.r_[np.inf, np.unique(p)[::-1], -np.inf]
        tpr = []
        fpr = []
        for t in ths:
            h = p >= t
            tpr.append(((y == 1) & h).sum() / max((y == 1).sum(), 1))
            fpr.append(((y == 0) & h).sum() / max((y == 0).sum(), 1))
        axes[0].plot(fpr, tpr, label=f"Seed {r['seed']} AUC {r['calibrated_metrics']['auc']:.4f}")
        xp = []
        yp = []
        for i in range(10):
            mask = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)
            if mask.any():
                xp.append(float(p[mask].mean()))
                yp.append(float(y[mask].mean()))
        axes[1].plot(xp, yp, 'o-', label=f"Seed {r['seed']}")
        axes[2].scatter([r['seed']], [r['fixed_metrics']['accuracy'] * 100], marker='x', s=80, color='tab:blue', label='Fixed 0.5' if r['seed'] == 42 else None)
        axes[2].scatter([r['seed']], [r['calibrated_metrics']['accuracy'] * 100], marker='o', s=70, color='tab:orange', label='Validation threshold' if r['seed'] == 42 else None)
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='Per-seed ROC')
    axes[1].plot([0, 1], [0, 1], '--', color='grey')
    axes[1].set(xlabel='Mean score in bin', ylabel='Observed label-1 fraction', title='Per-seed reliability (10 bins)')
    axes[2].set(xlabel='Seed', ylabel='Accuracy (%)', title='Fixed vs validation-selected threshold')
    axes[2].set_xticks([42, 43, 44])
    axes[2].set_ylim(0, 101)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(series + '')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    name = series + '_evaluation.png'
    fig.savefig(C / name, dpi=150)
    plt.close(fig)
    manifest.append({'series': series, 'file': 'curves/' + name, 'scope': ''})
for branch in sorted((O / 'branches').iterdir()):
    if not branch.is_dir():
        continue
    fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True)
    for i, seed in enumerate([42, 43, 44]):
        for j, fold in enumerate(range(1, 6)):
            d = branch / f'seed{seed}_fold{fold}'
            h = json.loads((d / 'history.json').read_text())
            result = json.loads((d / 'RESULT_CURRENT.json').read_text())
            ax = axes[i, j]
            ax.plot([r['epoch'] for r in h], [r['validation_bce'] for r in h], label='Validation subject BCE')
            hh = [r for r in h if 'training_bce' in r]
            ax.plot([r['epoch'] for r in hh], [r['training_bce'] for r in hh], label='Mean minibatch BCE')
            ax.axvline(result['selected_epoch'], color='grey', linestyle='--')
            ax.set_title(f'Seed {seed}, partition {fold}')
            ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle(branch.name + '')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = branch.name + '_training.png'
    fig.savefig(C / name, dpi=120)
    plt.close(fig)
    manifest.append({'series': branch.name, 'file': 'curves/' + name, 'type': 'All 15 training and validation histories'})
for mode in ['baseline', 'agentic']:
    for method in ['stacking', 'feature']:
        fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True)
        for i, seed in enumerate([42, 43, 44]):
            for j, fold in enumerate(range(1, 6)):
                d = O / 'fusion' / mode / method / f'seed{seed}_fold{fold}'
                h = json.loads((d / 'history.json').read_text())
                proto = json.loads((d / 'protocol.json').read_text())
                ax = axes[i, j]
                ax.plot([r['epoch'] for r in h], [r['validation_bce'] for r in h], label='Validation subject BCE')
                hh = [r for r in h if 'training_bce' in r]
                ax.plot([r['epoch'] for r in hh], [r['training_bce'] for r in hh], label='Mean minibatch BCE')
                ax.axvline(proto['selected_epoch'], color='grey', linestyle='--')
                ax.set_title(f'Seed {seed}, partition {fold}')
                ax.grid(alpha=0.25)
        axes[0, 0].legend()
        fig.suptitle(f'{mode} {method}')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        name = f'fusion_{mode}_{method}_training.png'
        fig.savefig(C / name, dpi=120)
        plt.close(fig)
        manifest.append({'series': f'fusion_{mode}_{method}', 'file': 'curves/' + name, 'type': 'Fusion head training histories'})
    fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex=True)
    for i, seed in enumerate([42, 43, 44]):
        for j, fold in enumerate(range(1, 6)):
            proto = json.loads((O / 'fusion' / mode / 'validation_weighted' / f'seed{seed}_fold{fold}' / 'protocol.json').read_text())
            grid = proto['validation_weight_grid']
            ax = axes[i, j]
            ax.plot([r['mri_weight'] for r in grid], [r['validation_bce'] for r in grid])
            ax.axvline(proto['mri_weight'], color='grey', linestyle='--')
            ax.set_title(f'Seed {seed}, partition {fold}')
            ax.set_xlabel('MRI weight')
            ax.set_ylabel('Validation BCE')
            ax.grid(alpha=0.25)
    fig.suptitle(f'{mode}')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = f'fusion_{mode}_validation_weights.png'
    fig.savefig(C / name, dpi=120)
    plt.close(fig)
    manifest.append({'series': f'fusion_{mode}_validation_weighted', 'file': 'curves/' + name, 'type': 'Validation weight grids for every partition'})
(O / 'CURVES_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
report = (O / 'START_ALL_LAST_LAYER_EN.html').read_text()
section = '<h2>Evaluation and training curves</h2>'
for r in manifest:
    section += f"<p><a href='{r['file']}'>{r['series']} - {r.get('type', 'Evaluation')}</a></p>"
(O / 'START_ALL_LAST_LAYER_EN.html').write_text(report.replace('</html>', section + '</html>'))
print('Saved', len(manifest), 'figures')
