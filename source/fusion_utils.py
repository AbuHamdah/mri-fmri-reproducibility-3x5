"""Fresh fusion heads with preprocessing fitted only on training subjects."""
import numpy as np
import tensorflow as tf


def build_heads(pairs, mri_features, fmri_features, train_indices, seed):
    tf.keras.utils.set_random_seed(seed)
    index = np.asarray(train_indices, dtype=np.int64)
    if index.ndim != 1 or not len(index) or len(set(index.tolist())) != len(index):
        raise ValueError('Training indices must be nonempty and unique')
    arrays = [np.asarray(v, dtype=np.float32) for v in (pairs, mri_features, fmri_features)]
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Fusion inputs must be finite matrices')
    if len({len(v) for v in arrays}) != 1 or np.any(index < 0) or np.any(index >= len(arrays[0])):
        raise ValueError('Mismatched input sizes or invalid training indices')
    pairs, mri_features, fmri_features = arrays
    layers = tf.keras.layers
    pair_input = layers.Input(shape=pairs.shape[1:])
    training = pairs[index]
    norm = layers.Normalization(mean=training.mean(0), variance=training.var(0))
    stack = tf.keras.Model(pair_input, layers.Dense(1, activation='sigmoid')(norm(pair_input)), name='stacking_head')
    left = layers.Input(shape=mri_features.shape[1:])
    right = layers.Input(shape=fmri_features.shape[1:])
    training = np.concatenate([mri_features[index], fmri_features[index]], axis=1)
    norm = layers.Normalization(mean=training.mean(0), variance=training.var(0))
    value = norm(layers.Concatenate()([left, right]))
    for units in (128, 64, 32):
        value = layers.Dropout(0.5)(layers.Dense(units, activation='relu')(value))
    feature = tf.keras.Model([left, right], layers.Dense(1, activation='sigmoid')(value), name='feature_head')
    return stack, feature


def validation_weight(pairs, labels, validation_indices):
    index = np.asarray(validation_indices, dtype=np.int64)
    if index.ndim != 1 or not len(index) or len(set(index.tolist())) != len(index):
        raise ValueError('Validation indices must be nonempty and unique')
    pairs = np.asarray(pairs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or labels.shape != (len(pairs),):
        raise ValueError('Expected two probabilities and one label per subject')
    if np.any(index < 0) or np.any(index >= len(pairs)):
        raise ValueError('Validation index out of range')
    probabilities = pairs[index]
    truth = labels[index]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Invalid validation probabilities')
    if not np.isin(truth, [0., 1.]).all():
        raise ValueError('Validation labels must be binary')
    candidates = np.linspace(0., 1., 101)
    mixture = np.clip(probabilities[:, :1] * candidates + probabilities[:, 1:] * (1 - candidates), 1e-7, 1 - 1e-7)
    loss = -(truth[:, None] * np.log(mixture) + (1 - truth[:, None]) * np.log(1 - mixture)).mean(0)
    return float(candidates[np.argmin(loss)])
