"""AvgWD and EigenWD from sampled-response token embeddings."""

from pathlib import Path

import numpy as np
import ot
from scipy.spatial.distance import cdist


def _responses(embeddings):
    clouds = [np.asarray(x, dtype=np.float64) for x in embeddings]
    if len(clouds) < 2:
        raise ValueError("At least two sampled responses are required.")
    if any(x.ndim != 2 or min(x.shape) == 0 for x in clouds):
        raise ValueError("Each response must have shape (tokens, hidden_dim).")
    if any(x.shape[1] != clouds[0].shape[1] for x in clouds):
        raise ValueError("Responses must have the same hidden dimension.")
    if any(not np.isfinite(x).all() for x in clouds):
        raise ValueError("Embeddings must be finite.")
    return clouds


def project_responses(embeddings, projection=None, dim=128, seed=2023):
    """Apply one shared Gaussian projection to all response token states."""
    clouds = _responses(embeddings)
    if projection is None:
        if not isinstance(dim, (int, np.integer)) or dim < 1:
            raise ValueError("Projection dimension must be a positive integer.")
        rng = np.random.default_rng(seed)
        projection = rng.standard_normal((clouds[0].shape[1], dim)) / np.sqrt(dim)
    projection = np.asarray(projection, dtype=np.float64)
    if (projection.ndim != 2 or projection.shape[0] != clouds[0].shape[1]
            or projection.shape[1] < 1 or not np.isfinite(projection).all()):
        raise ValueError("Projection must be finite with shape (hidden_dim, dim).")
    return [x @ projection for x in clouds]


def pairwise_wasserstein(embeddings):
    """Exact W2 distances with uniform token weights and squared L2 costs."""
    clouds = _responses(embeddings)
    distances = np.zeros((len(clouds), len(clouds)), dtype=np.float64)
    for i, x in enumerate(clouds):
        a = np.full(len(x), 1.0 / len(x))
        for j in range(i + 1, len(clouds)):
            y = clouds[j]
            b = np.full(len(y), 1.0 / len(y))
            squared_w2 = float(ot.emd2(a, b, cdist(x, y, metric="sqeuclidean")))
            if not np.isfinite(squared_w2) or squared_w2 < -1e-9:
                raise ArithmeticError("Invalid transport cost returned by the solver.")
            distances[i, j] = distances[j, i] = np.sqrt(max(squared_w2, 0.0))
    return distances


def _distances(matrix):
    d = np.asarray(matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[0] != d.shape[1]:
        raise ValueError("Distance matrix must be square with K >= 2.")
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Distances must be finite and nonnegative.")
    if not np.allclose(d, d.T, atol=1e-10, rtol=1e-10):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(d), 0, atol=1e-10, rtol=0):
        raise ValueError("Distance matrix must have a zero diagonal.")
    return d


def avgwd(distance_matrix):
    """Mean Wasserstein distance over unordered response pairs."""
    d = _distances(distance_matrix)
    return float(d[np.triu_indices(len(d), k=1)].mean())


def eigenwd(distance_matrix, p=0.1, alpha=1e-6, eps=1e-6):
    """Gaussian-kernel singular-value ratio ||sigma||_p / ||sigma||_2."""
    d = _distances(distance_matrix)
    if not np.isfinite(p) or not 0 < p < 2:
        raise ValueError("p must satisfy 0 < p < 2.")
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("alpha and eps must be finite and positive.")
    positive = d[d > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-d ** 2 / (2 * (bandwidth ** 2 + eps)))
    kernel += alpha * np.eye(len(d))
    singular_values = np.linalg.svd(kernel, compute_uv=False)
    singular_values /= singular_values.max()
    return float(np.sum(singular_values ** p) ** (1 / p)
                 / np.linalg.norm(singular_values))


def score(embeddings, p=0.1, alpha=1e-6, eps=1e-6):
    """Return (AvgWD, EigenWD); inputs exclude prompt, EOS, and padding tokens."""
    distances = pairwise_wasserstein(embeddings)
    return avgwd(distances), eigenwd(distances, p=p, alpha=alpha, eps=eps)


def read_data(path):
    """Read questions and variable-length response features from the example NPZ."""
    with np.load(Path(path), allow_pickle=False) as data:
        features = data["features"]
        lengths = data["lengths"]
        questions = data["questions"].tolist()
    if features.ndim != 4 or min(features.shape) < 1 or features.shape[1] < 2:
        raise ValueError("features must have shape (questions, responses, tokens, dim).")
    if (lengths.shape != features.shape[:2] or lengths.dtype.kind not in "iu"
            or (lengths < 1).any() or (lengths > features.shape[2]).any()):
        raise ValueError("lengths must contain valid response token counts.")
    if len(questions) != len(features):
        raise ValueError("questions and features must contain the same number of items.")
    embeddings = [[x[:int(n)].copy() for x, n in zip(row, counts)]
                  for row, counts in zip(features, lengths)]
    return questions, embeddings
