import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
from graphs.patterns_graphs import summary_for_run
from patterns.stored_paterns import A, E, J, N, S, build_patterns
matplotlib.use("Agg")  # para entornos sin display
import matplotlib.pyplot as plt

# -------------------- Configuración --------------------

ON, OFF = 1, -1
BASE_DIR = os.getcwd()
RUN_ROOT = os.path.join(BASE_DIR, "hopfield_runs")
SUM_ROOT = os.path.join(BASE_DIR, "hopfield_run_summaries")
os.makedirs(RUN_ROOT, exist_ok=True)
os.makedirs(SUM_ROOT, exist_ok=True)

# Estado de render
_run_label = "run"
_step_counter = 0

# -------------------- Utilidades de render --------------------

def set_run_label(label: str) -> None:
    """Inicia una nueva subcarpeta para un intento y resetea el contador de pasos."""
    global _run_label, _step_counter
    _run_label = label
    _step_counter = 0

def _draw_white_grid_with_stars(v: np.ndarray, title: str) -> str:
    """
    Dibuja una grilla 5x5:
      - Fondo y celdas completamente blancas.
      - Dibuja '*' SOLO en las celdas con valor +1.
      - En celdas -1 no se dibuja nada (queda en blanco).
    """
    global _step_counter
    grid = v.reshape(5, 5)

    fig = plt.figure(figsize=(3.2, 3.2), dpi=180)
    ax = plt.gca()
    ax.set_facecolor("white")

    # Fondo blanco uniforme y límites de celdas (sin colorear celdas)
    ax.imshow(np.ones((5, 5)), interpolation="nearest", vmin=0, vmax=1, cmap="gray")

    # Dibujar estrellas en +1
    for i in range(5):
        for j in range(5):
            if grid[i, j] == ON:
                ax.text(j, i, "*", ha="center", va="center", fontsize=16, color="black")

    # Grid fino
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.6, alpha=0.8)

    # Ocultar ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Título
    ax.set_title(title, fontsize=9)
    plt.tight_layout(pad=0.35)

    # Guardar
    outdir = os.path.join(RUN_ROOT, _run_label)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"step_{_step_counter:03d}.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    _step_counter += 1
    return outpath

def show(v: np.ndarray, title: str = "") -> None:
    _draw_white_grid_with_stars(v, title)



# -------------------- Hopfield (núcleo) --------------------

def sgn_with_tie(z: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Función signo con desempate: si h_i == 0, conserva el estado previo x_i."""
    out = np.sign(z)
    zeros = (out == 0)
    out[zeros] = prev[zeros]
    return out.astype(int)

def energy(W: np.ndarray, x: np.ndarray) -> float:
    """E(S) = -1/2 * S^T W S"""
    return -0.5 * float(x @ W @ x)

def train_hebb(patterns: Dict[str, np.ndarray]) -> np.ndarray:
    """
    W = (X @ X.T) / N, con diagonal en 0 y simetría exacta. N = número de neuronas.
    """
    X = np.stack([v for v in patterns.values()], axis=1)  # (N, p)
    W = (X @ X.T) / N
    np.fill_diagonal(W, 0.0)
    W = (W + W.T) / 2.0  # asegurar simetría exacta ante redondeos 
    return W

def is_fixed_point(W: np.ndarray, x: np.ndarray) -> bool:
    """
    Condición teórica de estabilidad (punto fijo):
      x = sgn(Wx) con empate conservando el valor previo.
    Equivalente: para todo i, x_i * (Wx)_i >= 0
    """
    h = W @ x
    return np.all(x * h >= 0)

def recall_async_until_convergence(
    W: np.ndarray,
    x0: np.ndarray,
    max_sweeps: int = 200,
    verbose: bool = True,
    rng=None
) -> Tuple[np.ndarray, int]:
    """
    Dinámica síncrona de Hopfield que itera usando la regla:
        S(t+1) = sign( W S(t) )
    Es decir, en cada sweep todos los nodos se actualizan SIMULTÁNEAMENTE
    según el producto W @ x, como en la ecuación dada en la imagen.
    Corta cuando el estado no cambia en un sweep (punto fijo).
    """
    x = x0.copy()
    if verbose:
        show(x, f"sweep=0 | Energía: {energy(W, x):.1f}")

    for s in range(1, max_sweeps + 1):
        x_new = sgn_with_tie(W @ x, x)
        if verbose:
            show(x_new, f"sweep={s} | Energía: {energy(W, x_new):.1f}")
        if np.array_equal(x_new, x):
            # No hubo cambios → punto fijo
            return x_new, s
        x = x_new

    # No convergió: devuelve el último estado
    return x, max_sweeps

def flip_noise(x: np.ndarray, p: float, rng) -> np.ndarray:
    """Invierte aleatoriamente una fracción p de las entradas (±1 -> ∓1)."""
    x = x.copy()
    k = max(1, int(round(p * x.size)))
    idx = rng.choice(x.size, size=k, replace=False)
    x[idx] *= -1
    return x

def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())



# -------------------- Orquestación --------------------

def run_a(p_noise: float = 0.24, max_sweeps: int = 50, seed: int = 7) -> Dict[str, str]:
    """Ejecuta: render de patrones, asociaciones ruidosas y un caso espurio, con resúmenes por intento."""
    rng = np.random.default_rng(seed)
    patterns = build_patterns()
    W = train_hebb(patterns)

    # Patrones almacenados
    set_run_label("stored_patterns")
    for name, v in patterns.items():
        show(v, f"Letra {name} (patrón) | Energía: {energy(W,v):.1f}")
    summary_for_run(os.path.join(RUN_ROOT, "stored_patterns"),
                    os.path.join(SUM_ROOT, "stored_patterns_summary.png"))

    # Asociación desde ruido
    for name, clean in patterns.items():
        set_run_label(f"assoc_{name}")
        noisy = flip_noise(clean, p_noise, rng)
        show(clean, f"Objetivo {name} - patrón limpio | Energía: {energy(W,clean):.1f}")
        show(noisy, f"Objetivo {name} - consulta ruidosa | Energía: {energy(W,noisy):.1f}")
        x_rec, sweeps = recall_async_until_convergence(W, noisy, max_sweeps=max_sweeps, verbose=True, rng=rng)
        dists = {n: hamming(x_rec, v) for n, v in patterns.items()}
        best = min(dists, key=dists.get)
        show(x_rec, f"Final → '{best}' | sweeps={sweeps} | Hamming {dists} | Energía: {energy(W,x_rec):.1f}")
        summary_for_run(os.path.join(RUN_ROOT, f"assoc_{name}"),
                        os.path.join(SUM_ROOT, f"assoc_{name}_summary.png"))