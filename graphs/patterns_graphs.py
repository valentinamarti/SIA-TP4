
import os
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
from patterns.stored_paterns import A,E,J,S
import matplotlib.pyplot as plt
try:
    from PIL import Image
    RESAMPLER = getattr(Image, "Resampling", Image).LANCZOS  # Pillow ≥10
except Exception:
    Image = None
    RESAMPLER = None

def _pick_rect_grid(n: int) -> Tuple[int, int]:
    """Elige (rows, cols) para formar una grilla rectangular. Evita una sola fila si n>1."""
    if n <= 0:
        return (0, 0)
    if n == 1:
        return (1, 1)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    if rows == 1:
        rows = 2
        cols = int(np.ceil(n / rows))
    return rows, cols

def summary_for_run(run_folder: str, outfile: str, tile_size=(260, 260), bg=(255, 255, 255)) -> str:
    """Crea un collage rectangular con TODOS los frames de un intento."""
    if Image is None:
        return outfile  # si PIL no está disponible, omite sin error
    pngs = [os.path.join(run_folder, f) for f in sorted(os.listdir(run_folder))
            if f.lower().endswith(".png")]
    if not pngs:
        return outfile
    imgs = [Image.open(p).convert("RGBA").resize(tile_size, RESAMPLER) for p in pngs]
    n = len(imgs)
    rows, cols = _pick_rect_grid(n)
    Wc, Hc = tile_size
    canvas = Image.new("RGBA", (cols * Wc, rows * Hc), bg)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        canvas.paste(im, (c * Wc, r * Hc), im)
    canvas.convert("RGB").save(outfile, quality=95)
    return outfile
