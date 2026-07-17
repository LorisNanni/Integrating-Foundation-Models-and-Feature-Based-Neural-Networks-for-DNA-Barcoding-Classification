import numpy as np
import pandas as pd
from typing import Sequence, Union

# -------------------------------------------------------------
# 1.  Map every IUPAC symbol to its row in the coordinate table
# -------------------------------------------------------------
BASE_INDEX = {
    "A": 0, "C": 1, "G": 2, "T": 3,
    "R": 4, "Y": 5, "K": 6, "M": 7,
    "S": 8, "W": 9, "B": 10, "D": 11,
    "H": 12, "V": 13, "N": 14,
}

# -------------------------------------------------------------
# 2.  One step of the Chaos-Game -– midpoint between
#     previous point and the vertex for the current base
# -------------------------------------------------------------
def _next_coords(
    base: str,
    prev: np.ndarray,
    base_coords: np.ndarray,
) -> np.ndarray:
    """Return the next point given the previous point + current base."""
    try:
        idx = BASE_INDEX[base.upper()]
    except KeyError:
        raise ValueError(f"Unsupported nucleotide symbol: {base!r}")

    return 0.5 * (prev + base_coords[idx])


# -------------------------------------------------------------
# 3A.  Low-level function: return a NumPy matrix  (n+1) × dim
# -------------------------------------------------------------
def chaos_game(
    seq: Union[str, Sequence[str]],
    base_coords: np.ndarray,
) -> np.ndarray:
    """
    Produce the full Chaos-Game Representation of `seq`.

    Parameters
    ----------
    seq         : DNA sequence (string or list of single-char strings)
    base_coords : array with shape (15, 3) or (15, d)
                  rows = vertices for A,C,G,T,R,Y,K,M,S,W,B,D,H,V,N

    Returns
    -------
    coords : NumPy array of shape (len(seq)+1, 3)
             first row is the origin (all zeros)
    """
    # Accept either a raw string or a pre-split list
    if isinstance(seq, str):
        seq = list(seq.upper())
    else:
        seq = [b.upper() for b in seq]

    n_points, dim = len(seq) + 1, base_coords.shape[1]
    coords = np.zeros((n_points, dim), dtype=float)

    prev = np.zeros(dim, dtype=float)
    for i, base in enumerate(seq, start=1):       # row 0 stays all-zeros
        coords[i] = _next_coords(base, prev, base_coords)
        prev = coords[i]

    return coords


# -------------------------------------------------------------
# 3B.  Convenience wrapper – return a tidy Pandas DataFrame
# -------------------------------------------------------------
def chaos_game_df(
    seq: Union[str, Sequence[str]],
    base_coords: np.ndarray,
    axes: Sequence[str] = ("i", "j", "k"),
) -> pd.DataFrame:
    """Same as `chaos_game` but with nice column labels."""
    coords = chaos_game(seq, base_coords)
    if len(axes) != coords.shape[1]:
        raise ValueError("axes names must match the coordinate dimension")
    return pd.DataFrame(coords, columns=list(axes))



