"""Validated labels and subject partitions for supervised training."""
import numpy as np


def label_targets(ids, labels):
    """Return recorded binary labels in the requested ID order."""
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Subject IDs must be nonempty and unique')
    missing = [sid for sid in ids if sid not in labels]
    if missing:
        raise ValueError(f'Missing subject labels: {missing}')
    values = [labels[sid] for sid in ids]
    if any(not isinstance(v, (bool, int, float, np.number)) for v in values):
        raise ValueError('Labels must be numeric binary values')
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(ids),) or not np.all(np.isin(result, [0., 1.])):
        raise ValueError('Labels must be finite binary values 0 or 1')
    return result


def split_indices(ids, split):
    """Validate a complete, disjoint subject partition before optimization."""
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Subject IDs must be nonempty and unique')
    lookup = {sid: i for i, sid in enumerate(ids)}
    seen = set()
    indices = {}
    for part in ('train', 'validation', 'test'):
        members = split.get(part)
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError(f'{part} must contain subject IDs')
        if len(members) != len(set(members)):
            raise ValueError(f'Duplicate subjects in {part}')
        unknown = set(members) - set(lookup)
        overlap = set(members) & seen
        if unknown or overlap:
            raise ValueError(f'Invalid {part}: unknown={sorted(unknown)}, overlap={sorted(overlap)}')
        indices[part] = np.asarray([lookup[sid] for sid in members], dtype=np.int64)
        seen.update(members)
    if seen != set(ids):
        raise ValueError('The split must cover every subject exactly once')
    return indices


if __name__ == '__main__':
    raise SystemExit('Use source/train_fresh.py to train with recorded labels.')
