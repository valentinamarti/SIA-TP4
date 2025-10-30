import numpy as np
from typing import Dict
N = 25

T = np.array([
    [+1, +1, +1, +1, +1],
    [-1, -1, +1, -1, -1],
    [-1, -1, +1, -1, -1],
    [-1, -1, +1, -1, -1],
    [-1, -1, +1, -1, -1],
], dtype=int)

X = np.array([
    [+1, -1, -1, -1, +1],
    [-1, +1, -1, +1, -1],
    [-1, -1, +1, -1, -1],
    [-1, +1, -1, +1, -1],
    [+1, -1, -1, -1, +1],
], dtype=int)

H = np.array([
    [+1, -1, -1, -1, +1],
    [+1, -1, -1, -1, +1],
    [+1, +1, +1, +1, +1],
    [+1, -1, -1, -1, +1],
    [+1, -1, -1, -1, +1],
], dtype=int)

C = np.array([
    [-1, +1, +1, +1, +1],
    [+1, -1, -1, -1, -1],
    [+1, -1, -1, -1, -1],
    [+1, -1, -1, -1, -1],
    [-1, +1, +1, +1, +1],
], dtype=int)


def build_patterns() -> Dict[str, np.ndarray]:
    pats = {
        "T": T.reshape(-1),
        "X": X.reshape(-1),
        "H": H.reshape(-1),
        "C": C.reshape(-1),
    }
    for k, v in pats.items():
        assert v.shape == (N,), f"Patrón {k} no es 25-long"
        assert set(np.unique(v)).issubset({-1, 1}), f"Patrón {k} debe ser ±1"
    return pats


# Calcular similitud (correlación normalizada) entre patrones
patterns = [T, X, H, C]
pattern_names = ["T", "X", "H", "C"]

# Aplanar cada letra en vector de 25 elementos
vecs = [p.flatten() for p in patterns]

# Calcular productos internos normalizados
n = 25
print("\n--- Similitud entre patrones almacenados ---")
for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        corr = np.sum(vecs[i] * vecs[j]) / n
        print(f"Similitud entre patrón {pattern_names[i]} y {pattern_names[j]}: {corr:.3f}")
print()