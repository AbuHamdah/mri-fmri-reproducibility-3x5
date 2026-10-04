"""New supervised protocol on processed arrays; outputs never replace saved results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from model_utils import label_targets, split_indices
from training_control import ValidationControl, dropout_configuration

VERSION = 'fresh_supervised_v4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare(root, seed, fold):
    inputs = read(root / 'INPUTS.json')
    labels = {}
    paths = {kind: {} for kind in ('mri', 'fmri')}
    for row in inputs:
        sid, label = row['subject_id'], row['label']
        if sid in labels and labels[sid] != label:
            raise ValueError('Conflicting labels for ' + sid)
        labels[sid] = label
        p = (root / row['path']).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError('Missing or external input: ' + str(p))
        paths[row['kind']].setdefault(sid, []).append(p)
    ids = sorted(labels)
    y = label_targets(ids, labels)
    source = root / 'selected/fmri_baseline/branches/fmri_baseline' / f'seed{seed}_fold{fold}' / 'SPLIT.json'
    split = read(source)
    split = {key: split[key] for key in ('train', 'validation', 'test')}
    indices = split_indices(ids, split)
    for kind in paths:
        if set(paths[kind]) != set(ids):
            raise ValueError('Missing modality subjects: ' + kind)
        for values in paths[kind].values():
            values.sort()
    architectures = {
        'mri': root / 'selected/mri_baseline/adopted_models' / f'seed{seed}_fold{fold}' / 'model.keras',
        'fmri': root / 'parent_models/fmri_baseline.keras',
    }
    if not all(p.is_file() for p in architectures.values()):
        raise ValueError('Missing architecture archive')
    return ids, y, paths, split, indices, architectures


def branch(root, output, kind, seed, epochs, batch_size, lr, prepared):
    import tensorflow as tf
    from fresh_model import fresh_model
    ids, y, paths, split, indices, architectures = prepared
    model = fresh_model(architectures[kind], seed)
    if model.output_shape[-1] != 1:
        raise ValueError('Expected a binary classifier architecture')
    optimizer = tf.keras.optimizers.Adam(lr)
    def arrays(sid):
        return np.concatenate([np.load(p, allow_pickle=False).reshape((-1,) + tuple(model.input_shape[1:]))
                               for p in paths[kind][sid]], axis=0)
    def predict(which):
        return np.array([float(tf.reduce_mean(model(arrays(ids[i]), training=False))) for i in which])
    def loss(prob, target):
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        return float(np.mean(-target * np.log(prob) - (1 - target) * np.log1p(-prob)))
    checkpoint = output / 'selected.weights.h5'
    val = indices['validation']
    best = loss(predict(val), y[val]); selected = 0
    model.save_weights(checkpoint)
    control = ValidationControl()
    control.observe(best, lr)
    dropout = dropout_configuration(model)
    history = [dict(epoch=0, training_bce=loss(predict(indices['train']), y[indices['train']]), validation_bce=best, learning_rate=lr, dropout=dropout)]
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        epoch_lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        order = rng.permutation(indices['train']); losses = []; sizes = []
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            with tf.GradientTape() as tape:
                probability = tf.stack([tf.reduce_mean(model(arrays(ids[i]), training=True)) for i in batch])
                target = tf.cast(y[batch], probability.dtype)
                objective = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target, probability))
                if model.losses:
                    objective += tf.add_n(model.losses)
            gradients = tape.gradient(objective, model.trainable_variables)
            if any(g is None or not np.isfinite(g.numpy()).all() for g in gradients):
                raise ValueError('Missing or nonfinite gradient')
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(float(objective)); sizes.append(len(batch))
        validation = loss(predict(val), y[val])
        improved, next_lr, stop = control.observe(validation, epoch_lr)
        history.append(dict(epoch=epoch, training_bce=loss(predict(indices['train']), y[indices['train']]), train_bce_with_regularization=float(np.average(losses, weights=sizes)), validation_bce=validation, learning_rate=epoch_lr, next_learning_rate=next_lr, dropout=dropout, early_stop=stop))
        optimizer.learning_rate.assign(next_lr)
        if improved:
            best, selected = validation, epoch
            model.save_weights(checkpoint)
        write(output / 'history.json', history)
        write(output / 'last_epoch_validation_predictions.json', [dict(subject_id=ids[i], true_label=int(y[i]), probability=float(p)) for i, p in zip(val, predict(val))])
        if stop:
            break
    model.load_weights(checkpoint)
    model.save(output / 'model.keras', save_format='keras_v3')
    test = indices['test']; probabilities = predict(test)
    restored = tf.keras.models.load_model(output / 'model.keras', compile=False)
    reloaded_probabilities = np.array([float(tf.reduce_mean(restored(arrays(ids[i]), training=False))) for i in test])
    reload_difference = float(np.max(np.abs(probabilities - reloaded_probabilities)))
    if reload_difference > 1e-6:
        raise ValueError('Saved model does not reproduce test probabilities')
    rows = [dict(subject_id=ids[i], true_label=int(y[i]), probability=float(p)) for i, p in zip(test, probabilities)]
    write(output / 'test_predictions.json', rows)
    write(output / 'protocol.json', dict(objective_version=VERSION, initialization='fresh_architecture_only',
          modality=kind, architecture=str(architectures[kind].relative_to(root)),
          architecture_archive_sha256=sha(architectures[kind]), target_source='INPUTS.json labels',
          split=split, epochs=epochs, batch_size=batch_size, lr=lr, selected_epoch=selected,
          max_reload_difference=reload_difference, actual_epochs=history[-1]['epoch'], early_stopping_patience=control.patience, lr_plateau_patience=control.lr_patience, dropout_configuration=dropout,
          controller='validation_only', model_sha256=sha(output / 'model.keras')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['mri', 'fmri'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive epochs, batch size and finite learning rate are required')
    root, output = args.package.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error('Output must be a new directory outside the existing package')
    prepared = prepare(root, args.seed, args.fold)
    if args.check_only:
        print(json.dumps(dict(passed=True, subjects=len(prepared[0]), objective_version=VERSION, kind=args.kind)))
        return
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    output.mkdir(parents=True)
    write(output / 'SPLIT.json', prepared[3])
    branch(root, output, args.kind, args.seed, args.epochs, args.batch_size, args.lr, prepared)


if __name__ == '__main__':
    main()
