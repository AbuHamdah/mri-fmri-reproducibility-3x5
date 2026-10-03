"""Run all three seeds and five folds of the fresh supervised protocol."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from train_fresh import VERSION, prepare, read, write

SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3, 4, 5)


def preflight(root):
    partitions = {}
    reference = None
    for seed in SEEDS:
        tested = []
        for fold in FOLDS:
            ids, labels, _, split, _, _ = prepare(root, seed, fold)
            identity = dict(zip(ids, labels.tolist()))
            if reference is None:
                reference = identity
            if identity != reference:
                raise ValueError('Cohort or labels differ between folds')
            partitions[seed, fold] = split
            tested.extend(split['test'])
        if Counter(tested) != Counter({sid: 1 for sid in reference}):
            raise ValueError(f'Seed {seed}: every participant must appear in test exactly once')
    return reference, partitions


def jobs(root, output, kind, epochs, batch_size, lr):
    result = []
    for seed in SEEDS:
        for fold in FOLDS:
            run = f'seed{seed}_fold{fold}'
            for modality in (('mri', 'fmri') if kind == 'all' else (kind,)):
                target = output / modality / run
                command = [sys.executable, str(root / 'source/train_fresh.py'),
                           '--package', str(root), '--output', str(target), '--kind', modality,
                           '--seed', str(seed), '--fold', str(fold), '--epochs', str(epochs),
                           '--batch-size', str(batch_size), '--lr', str(lr)]
                result.append(dict(kind=modality, seed=seed, fold=fold, output=str(target), command=command))
            if kind == 'all':
                target = output / 'fusion' / run
                command = [sys.executable, str(root / 'source/train_fusion_fresh.py'),
                           '--package', str(root), '--output', str(target),
                           '--mri', str(output / 'mri' / run), '--fmri', str(output / 'fmri' / run),
                           '--seed', str(seed), '--fold', str(fold), '--epochs', str(epochs)]
                result.append(dict(kind='fusion', seed=seed, fold=fold, output=str(target), command=command))
    return result


def aggregate(output, kind, labels, partitions):
    from statistics_core import metrics, NAMES
    methods = {kind: (kind, 'test_predictions.json')} if kind != 'all' else {
        'mri': ('mri', 'test_predictions.json'), 'fmri': ('fmri', 'test_predictions.json'),
        'stacking': ('fusion', 'stacking_test_predictions.json'),
        'feature': ('fusion', 'feature_test_predictions.json'),
        'weighted': ('fusion', 'weighted_test_predictions.json')}
    pooled, summary = [], []
    for method, (directory, filename) in methods.items():
        values = []
        for seed in SEEDS:
            rows = []
            for fold in FOLDS:
                folder = output / directory / f'seed{seed}_fold{fold}'
                if read(folder / 'SPLIT.json') != partitions[seed, fold]:
                    raise ValueError('Output partition does not match the planned fold')
                predictions = read(folder / filename)
                if Counter(p['subject_id'] for p in predictions) != Counter(partitions[seed, fold]['test']):
                    raise ValueError('Missing, duplicate or unexpected test predictions')
                for p in predictions:
                    probability = p['probability']
                    if p['true_label'] != labels[p['subject_id']] or not np.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError('Invalid label or probability')
                    rows.append(dict(**p, seed=seed, fold=fold))
            if Counter(p['subject_id'] for p in rows) != Counter({s: 1 for s in labels}):
                raise ValueError('Incomplete pooled test predictions')
            rows.sort(key=lambda p: p['subject_id'])
            value = metrics([p['true_label'] for p in rows], [p['probability'] for p in rows])
            values.append(value)
            pooled.append(dict(method=method, seed=seed, subjects=len(rows), metrics=dict(zip(NAMES, value.tolist())), predictions=rows))
        values = np.asarray(values)
        summary.append(dict(method=method, metrics={name: dict(mean=float(values[:, i].mean()), sd=float(values[:, i].std(ddof=1))) for i, name in enumerate(NAMES)}))
    write(output / 'POOLED_PER_SEED.json', pooled)
    write(output / 'SUMMARY.json', dict(objective_version=VERSION, seeds=list(SEEDS), folds=list(FOLDS),
          aggregation='Pool the five test folds per seed, then mean and sample SD across three seeds', results=summary))


def execute(plan, output, runner=subprocess.run):
    logs = output / 'logs'
    logs.mkdir()
    for i, job in enumerate(plan):
        name = f"{job['kind']}_seed{job['seed']}_fold{job['fold']}"
        write(output / 'progress.json', dict(stage='running', completed=i, total=len(plan), current=name))
        with (logs / (name + '.log')).open('w', encoding='utf-8') as log:
            result = runner(job['command'], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(output / 'FAILED.json', dict(job=name, exit_code=result.returncode, completed=i, log='logs/' + name + '.log'))
            raise RuntimeError('Training stopped: ' + name)
        print(f'Completed {i + 1}/{len(plan)}: {name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['mri', 'fmri', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, output = args.package.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error('Output must be a new directory outside the package')
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive epochs, batch size and finite learning rate are required')
    labels, partitions = preflight(root)
    plan = jobs(root, output, args.kind, args.epochs, args.batch_size, args.lr)
    if args.check_only:
        print(json.dumps(dict(passed=True, seeds=list(SEEDS), folds=list(FOLDS), subjects=len(labels),
              runs_per_method=15, subprocess_jobs=len(plan), kind=args.kind, training_started=False)))
        return
    output.mkdir(parents=True)
    write(output / 'PLAN.json', dict(objective_version=VERSION, jobs=plan))
    try:
        execute(plan, output)
        aggregate(output, args.kind, labels, partitions)
    except Exception as error:
        if not (output / 'FAILED.json').exists():
            write(output / 'FAILED.json', dict(error=str(error)))
        raise
    write(output / 'progress.json', dict(stage='complete', completed=len(plan), total=len(plan)))
    write(output / 'COMPLETE.json', dict(objective_version=VERSION, kind=args.kind, runs_per_method=15, jobs=len(plan)))


if __name__ == '__main__':
    main()
