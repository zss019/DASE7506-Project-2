"""NumPy-only candidate selector; no reference access during inference."""
import json
import numpy as np
from common import parse_answer


def prepare(bank, arrays):
    x = np.asarray(arrays['embeddings'], dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    answers = [[parse_answer(c['text']) for c in q['candidates']] for q in bank]
    valid = np.array([[a is not None for a in row] for row in answers])
    votes = np.array([[sum(a == b for b in row) if a is not None else 0 for a in row] for row in answers])
    return x, valid, votes


def choose(scores, valid):
    scores = np.where(valid | ~valid.any(axis=1, keepdims=True), scores, -np.inf)
    return np.argmax(scores, axis=1).astype(int).tolist()


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    config = json.loads((checkpoint / 'config.json').read_text())
    x, valid, votes = prepare(bank, arrays)
    if config['method'] == 'majority':
        scores = votes
    elif config['method'] in ('votes_linear', 'plain_linear'):
        features = x if config['method'] == 'plain_linear' else np.concatenate([x, votes[..., None] / 4, valid[..., None].astype(float)], axis=-1)
        with np.load(checkpoint / 'weights.npz', allow_pickle=False) as f:
            scores = features @ f['weights']
    elif config['method'] == 'mlp':
        parts = [x]
        if config['feature_mode'] != 'candidate':
            parts.append(x - x.mean(axis=1, keepdims=True))
        if config['feature_mode'] == 'context_votes':
            parts.extend([votes[..., None] / 4, valid[..., None].astype(float)])
        features = np.concatenate(parts, axis=-1)
        with np.load(checkpoint / 'weights.npz', allow_pickle=False) as f:
            features = (features - f['mean']) / f['scale']
            hidden = np.maximum(features @ f['w1'] + f['b1'], 0)
            scores = (hidden @ f['w2'] + f['b2']).squeeze(-1)
    else:
        with np.load(checkpoint / 'weights.npz', allow_pickle=False) as f:
            scores = x @ f['weights'] + f['bias']
    return choose(scores, valid)
