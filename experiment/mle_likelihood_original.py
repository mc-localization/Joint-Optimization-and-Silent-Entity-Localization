"""Training-only empirical multinomial likelihood for absorbing-receiver counts."""
from __future__ import annotations

import numpy as np


def interval_counts(features, start, length, M, n_rx):
    """Convert raw cumulative sequences to mutually exclusive window intervals."""
    if not (1 <= length <= M and 1 <= start <= M - length + 1):
        raise ValueError('Infeasible detection window')
    raw = np.asarray(features, dtype=np.float64).reshape(-1, n_rx, M)
    if not np.isfinite(raw).all() or (raw < 0).any() or (np.diff(raw, axis=2) < -1e-8).any():
        raise ValueError('Expected finite nonnegative nondecreasing raw cumulative counts')
    left = start - 1
    previous = raw[:, :, left - 1:left] if left else np.zeros((len(raw), n_rx, 1))
    observed = raw[:, :, left:left + length]
    counts = np.diff(np.concatenate([previous, observed], axis=2), axis=2)
    return np.maximum(counts, 0).reshape(len(raw), -1)


def smooth_probabilities(counts, Q, repeats, pseudocount=.5):
    """Jeffreys-style symmetric pseudocount across absorption cells and residual."""
    counts = np.asarray(counts, dtype=np.float64)
    repeats = np.asarray(repeats, dtype=np.float64)
    if Q <= 0 or pseudocount <= 0 or counts.ndim != 2 or repeats.shape != (len(counts),) or (repeats <= 0).any():
        raise ValueError('Invalid molecule budget, template repeats, or smoothing')
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(axis=1) > Q + 1e-7).any():
        raise ValueError('Template absorption counts exceed Q')
    successes = counts * repeats[:, None]
    remainder = np.maximum(Q - counts.sum(axis=1), 0) * repeats
    all_counts = np.column_stack([successes, remainder])
    probabilities = (all_counts + pseudocount) / (Q * repeats[:, None] + pseudocount * all_counts.shape[1])
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12):
        raise ValueError('Multinomial probabilities do not sum to one')
    return probabilities


def predict(observations, probabilities, positions, Q, device, batch_size=128, template_block=256):
    """Argmax plug-in multinomial log likelihood; shared data-only constants omitted."""
    import torch
    observations = np.asarray(observations, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if batch_size <= 0 or template_block <= 0 or Q <= 0:
        raise ValueError('Invalid inference batch sizes or molecule budget')
    if observations.ndim != 2 or probabilities.shape != (len(positions), observations.shape[1] + 1) or positions.shape != (len(positions), 2) or not len(positions):
        raise ValueError('Likelihood matrix dimensions differ')
    if not np.isfinite(observations).all() or (observations < 0).any() or (observations.sum(axis=1) > Q + 1e-7).any():
        raise ValueError('Observed absorption counts exceed Q')
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12):
        raise ValueError('Invalid multinomial template probabilities')
    logp = np.log(probabilities)
    output = np.empty((len(observations), 2), dtype=np.float64)
    # Float64 avoids cancellation when Q=20000 and likelihoods are close.
    # Blocking bounds GPU memory and preserves a deterministic first-index tie break.
    for begin in range(0, len(observations), batch_size):
        block = observations[begin:begin + batch_size]
        full = np.column_stack([block, np.maximum(Q - block.sum(axis=1), 0)])
        data = torch.as_tensor(full, dtype=torch.float64, device=device)
        best = torch.full((len(block),), -torch.inf, dtype=torch.float64, device=device)
        index = torch.zeros(len(block), dtype=torch.long, device=device)
        with torch.no_grad():
            for offset in range(0, len(positions), template_block):
                template = torch.as_tensor(logp[offset:offset + template_block].T.copy(), dtype=torch.float64, device=device)
                score, local_index = torch.max(data @ template, dim=1)
                improve = score > best
                best = torch.where(improve, score, best)
                index = torch.where(improve, local_index + offset, index)
        output[begin:begin + len(block)] = positions[index.cpu().numpy()]
    return output


def build_training_templates(root, expected_rows, expected_positions, Q, M, n_rx, data_dir=None):
    """Stream training shards; validation and test data never enter templates."""
    sums, repeats, total = {}, {}, 0
    # An explicit directory keeps the active fine-grid run separate from old data.
    training = (data_dir if data_dir is not None else root / 'data') / 'data_train'
    for source in sorted(training.glob('*.npz')):
        with np.load(source, allow_pickle=False) as data:
            features = np.asarray(data['X'], dtype=np.float64)
            xy = np.round(np.column_stack([data['x'], data['y']]), 6)
        if features.shape != (len(xy), M * n_rx) or not np.isfinite(xy).all():
            raise ValueError('Training template shard dimensions differ')
        counts = interval_counts(features, 1, M, M, n_rx)
        if (counts.sum(axis=1) > Q + 1e-7).any():
            raise ValueError('Training shard absorption exceeds molecule budget')
        unique, inverse = np.unique(xy, axis=0, return_inverse=True)
        local_sums = np.zeros((len(unique), M * n_rx), dtype=np.float64)
        np.add.at(local_sums, inverse, features)
        local_repeats = np.bincount(inverse, minlength=len(unique))
        for point, cumulative, count in zip(unique, local_sums, local_repeats):
            key = tuple(point)
            if key in sums:
                sums[key] += cumulative
                repeats[key] += int(count)
            else:
                sums[key] = cumulative.copy()
                repeats[key] = int(count)
        total += len(xy)
    if total != expected_rows or len(sums) != expected_positions:
        raise ValueError('Complete formal training split was not used for templates')
    keys = sorted(sums)
    counts = np.asarray([repeats[key] for key in keys], dtype=np.int64)
    templates = np.stack([sums[key] / repeats[key] for key in keys])
    return templates, np.asarray(keys, dtype=np.float64), counts
