"""BREM: 36 class-frequency sums and 18 local log-support differences."""

from dataclasses import dataclass

import numpy as np


_BITS = 15
_RELATIONS = (
    [(i,) for i in range(3)]
    + [(i, j) for i in range(3) for j in range(3) if i != j]
    + [(i, j + 3) for i in range(3) for j in range(3)]
)


def _samples(samples):
    x = np.asarray(samples)
    if x.ndim != 3 or x.shape[2] != 3 or x.shape[1] < 2 or len(x) == 0:
        raise ValueError("Expected a nonempty (N, T, 3) array with T >= 2.")
    if x.dtype.kind not in "iu" or np.any(x < 0) or np.any(x > 32767):
        raise ValueError("Codeword indices must be integers in [0, 32767].")
    return x


def _patterns(samples, relation):
    length = samples.shape[1] - max(relation) // 3
    packed = np.zeros((len(samples), length), dtype=np.int64)
    for index in relation:
        offset, column = divmod(index, 3)
        packed = (packed << _BITS) | samples[
            :, offset:offset + length, column
        ].astype(np.int64)
    return packed


@dataclass
class _Counts:
    keys: np.ndarray
    values: np.ndarray
    total: int
    parent_keys: np.ndarray
    parent_values: np.ndarray

    @classmethod
    def build(cls, packed, width):
        keys, values = np.unique(packed, return_counts=True)
        if width > 1:
            parents, inverse = np.unique(keys >> _BITS, return_inverse=True)
            parent_values = np.bincount(inverse, weights=values).astype(np.int64)
        else:
            parents = np.empty(0, dtype=np.int64)
            parent_values = np.empty(0, dtype=np.int64)
        return cls(keys, values, packed.size, parents, parent_values)

    @staticmethod
    def lookup(keys, values, query):
        index = np.searchsorted(keys, query)
        clipped = np.minimum(index, len(keys) - 1)
        matched = (index < len(keys)) & (keys[clipped] == query)
        return np.where(matched, values[clipped], 0)

    def probability(self, query, width, epsilon):
        denominator = self.total if width == 1 else self.lookup(
            self.parent_keys, self.parent_values, query >> _BITS
        )
        return self.lookup(self.keys, self.values, query) / (denominator + epsilon)


class BREM:
    """Estimate reference tables on P and extract 54-dimensional S+G features.

    Input: integer arrays (recordings, frames, 3), with the three spectral
    quantization indices in codec order. Adjacent-frame relations stay inside
    each recording. All calls use the same segment length.
    """

    def __init__(self, epsilon=1e-6):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive.")
        self.epsilon = float(epsilon)

    def fit(self, cover, stego):
        """Pool P-cover and source-rate P-stego counts within each class."""
        cover, stego = _samples(cover), _samples(stego)
        if cover.shape[1] != stego.shape[1]:
            raise ValueError("Cover and stego must have the same frame count.")
        self.frames_ = cover.shape[1]
        self.tables_ = {
            relation: tuple(
                _Counts.build(_patterns(x, relation), len(relation))
                for x in (cover, stego)
            )
            for relation in _RELATIONS
        }
        return self

    def transform(self, samples):
        """Return [S0(r1), S1(r1), ..., S0(r18), S1(r18), G(r1), ..., G(r18)]."""
        if not hasattr(self, "tables_"):
            raise RuntimeError("Fit the reference tables before extraction.")
        x = _samples(samples)
        if x.shape[1] != self.frames_:
            raise ValueError("Use the same frame count as the reference tables.")
        sums, log_differences = [], []
        for relation in _RELATIONS:
            query = _patterns(x, relation)
            cover, stego = (
                table.probability(query, len(relation), self.epsilon)
                for table in self.tables_[relation]
            )
            sums.append(np.stack([
                np.cumsum(cover, axis=1)[:, -1],
                np.cumsum(stego, axis=1)[:, -1],
            ], axis=1))
            log_support = (
                np.log(cover + self.epsilon) - np.log(stego + self.epsilon)
            )
            log_differences.append(
                (np.cumsum(log_support, axis=1)[:, -1] / query.shape[1])[:, None]
            )
        # Consistent column layout also fixes the scaler's reduction order.
        return np.asfortranarray(np.concatenate(sums + log_differences, axis=1))
