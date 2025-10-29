import numpy as np
from typing import Dict
N = 25

A = np.array([
    [-1, +1, +1, +1, -1],
    [+1, -1, -1, -1, +1],
    [+1, +1, +1, +1, +1],
    [+1, -1, -1, -1, +1],
    [+1, -1, -1, -1, +1],
], dtype=int)

E = np.array([
    [+1, +1, +1, +1, +1],
    [+1, -1, -1, -1, -1],
    [+1, +1, +1, +1, +1],
    [+1, -1, -1, -1, -1],
    [+1, +1, +1, +1, +1],
], dtype=int)

J = np.array([
    [+1, +1, +1, +1, +1],
    [-1, -1, -1, -1, +1],
    [-1, -1, -1, -1, +1],
    [+1, -1, -1, -1, +1],
    [-1, +1, +1, +1, -1],
], dtype=int)

S = np.array([
    [+1, +1, +1, +1, +1],
    [+1, -1, -1, -1, -1],
    [+1, +1, +1, +1, +1],
    [-1, -1, -1, -1, +1],
    [+1, +1, +1, +1, +1],
], dtype=int)


def build_patterns() -> Dict[str, np.ndarray]:
    pats = {
        "A": A.reshape(-1),
        "E": E.reshape(-1),
        "J": J.reshape(-1),
        "S": S.reshape(-1),
    }
    for k, v in pats.items():
        assert v.shape == (N,), f"Patrón {k} no es 25-long"
        assert set(np.unique(v)).issubset({-1, 1}), f"Patrón {k} debe ser ±1"
    return pats