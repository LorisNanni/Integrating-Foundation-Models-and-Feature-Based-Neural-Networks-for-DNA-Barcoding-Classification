import numpy as np
from typing import Sequence, Union, Dict, Iterable, List, Callable, Tuple, Optional
from chaos_game import chaos_game
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
import math
from itertools import combinations
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull

# --- helper: which canonical bases belong to each IUPAC symbol ----
_READ_DICT = {
    "R": ("A", "G"),
    "Y": ("C", "T"),
    "K": ("G", "T"),
    "M": ("A", "C"),
    "S": ("C", "G"),
    "W": ("A", "T"),
    "B": ("C", "G", "T"),
    "D": ("A", "G", "T"),
    "H": ("A", "C", "T"),
    "V": ("A", "C", "G"),
    "N": ("A", "C", "G", "T"),
}


def CGR_TABLE(A_: np.ndarray,
              C_: np.ndarray,
              G_: np.ndarray,
              T_: np.ndarray) -> pd.DataFrame:
    """

    Parameters
    ----------
    A_, C_, G_, T_ : length-3 NumPy vectors with the (i,j,k) coordinates
                     of the four canonical nucleotides.

    Returns
    -------
    pandas.DataFrame  (15 rows × 4 columns: i, j, k, Nucleotide)
    """
    # first 4 rows – A, C, G, T
    rows = {
        "A": np.asarray(A_),
        "C": np.asarray(C_),
        "G": np.asarray(G_),
        "T": np.asarray(T_),
    }

    # add the 11 degenerate codes as means of their constituent bases
    for code, bases in _READ_DICT.items():
        rows[code] = np.vstack([rows[b] for b in bases]).mean(axis=0)

    df = pd.DataFrame.from_dict(rows, orient="index",
                                columns=["i", "j", "k"])
    df["Nucleotide"] = df.index          
    df.reset_index(drop=True, inplace=True)
    return df


_scale = 2.0 * np.sqrt(1.0 / 3.0)          # 2*sqrt(1/3)

coord1 = CGR_TABLE(
    (np.array([0, 0, 0]) - 0.5) * _scale,   # A
    (np.array([1, 0, 1]) - 0.5) * _scale,   # C
    (np.array([0, 1, 1]) - 0.5) * _scale,   # G
    (np.array([1, 1, 0]) - 0.5) * _scale,   # T
)
coord1_mat = coord1[["i", "j", "k"]].to_numpy()

def seq_to_cgr(
    dna_seq: Union[str, Sequence[str]],
    CGR_coord: np.ndarray = coord1_mat,
    axes: Sequence[str] = ("i", "j", "k")
) -> "np.recarray":
    """
    Convert a DNA sequence to its 3-D Chaos Game Representation.

    Parameters
    ----------
    dna_seq   : str  |  list/tuple of single-char bases
    CGR_coord : np.ndarray  shape (4, 3) – rows **A,C,G,T**
                (additional rows for degenerate bases are ignored here)
    axes      : names for the three output columns

    Returns
    -------
    CGR trace as *numpy record-array*
      • shape  = (len(sequence),)
      • fields = axes ('i','j','k'  by default)
    """
    # --- argument sanity checks ------------------------------------------
    if CGR_coord.shape[1] != len(axes):
        raise ValueError("Number of coord columns must equal len(axes)")

    # --- normalise / sanitise the sequence -------------------------------
    if isinstance(dna_seq, str):
        dna_seq = list(dna_seq.upper())
    else:
        dna_seq = [b.upper() for b in dna_seq]

    dna_seq = [b for b in dna_seq if b != "-"]   # strip gaps “-”

    # --- run the chaos-game ---------------------------------------------
    #trace = chaos_game(dna_seq, CGR_coord[:4, :])   # use only A,C,G,T rows
    trace = chaos_game(dna_seq, CGR_coord)

    # --- return as a nice labelled record-array --------------------------
    rec = np.core.records.fromarrays(
        trace.T,            # fields are columns
        names=",".join(axes)
    )
    return rec

# -------------------------------------------------------------
#  Oriented (signed) angle between two 3-D vectors
# -------------------------------------------------------------
def oriented_angle(a: Union[np.ndarray, Sequence[float]],
                   b: Union[np.ndarray, Sequence[float]],
                   n: Union[np.ndarray, Sequence[float]],
                   debug: bool = False,
                   tol: float = None,
                   high_precision: bool = True) -> float:
    # choose dtype (longdouble may be wider than float64 on your platform)
    dtype = np.longdouble if high_precision else np.float64

    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    n = np.asarray(n, dtype=dtype)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na * nb == 0:
        return 0.0

    # unsigned angle via dot
    x = np.dot(a, b) / (na * nb)

    x = np.clip(x, -1.0, 1.0)

    theta = float(np.arccos(x))

    # orientation (use normalized vectors to reduce scaling error)
    nn = np.linalg.norm(n)
    if nn == 0:
        triple_unit = dtype(0)
    else:
        a_u = a / na
        b_u = b / nb
        n_u = n / nn
        triple_unit = np.dot(n_u, np.cross(a_u, b_u))  # sign we care about

    # tolerance relative to machine epsilon
    if tol is None:
        tol = dtype(64.0) * np.finfo(dtype).eps

    # If too close to 0, compute scalar triple product via explicit 3×3 determinant
    if abs(triple_unit) < tol and nn != 0:
        # det([a_u b_u n_u]) == n_u · (a_u × b_u)
        ax, ay, az = a_u
        bx, by, bz = b_u
        nx, ny, nz = n_u
        triple_det = (
            nx * (ay*bz - az*by)
          - ny * (ax*bz - az*bx)
          + nz * (ax*by - ay*bx)
        )
        triple = float(triple_det)
    else:
        triple = float(triple_unit)

    # stable sign test with tolerance (treat near-zero as +)
    sign = -1.0 if triple < 0.0 else 1.0

    return sign * theta
# -------------------------------------------------------------
#  orientedangle1 – apply to each *pair* of successive rows
# -------------------------------------------------------------
def orientedangle1(points: np.ndarray,
                   v: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """
    Direct port of:
        NumericVector orientedangle1(arma::mat points, arma::rowvec v)

    For each consecutive pair of rows p1, p2 in `points`, compute
        angles[i] = oriented_angle(p1, p2, v)

    Parameters
    ----------
    points : (N, 3) array-like
    v      : (3,) array-like

    Returns
    -------
    angles : (N-1,) ndarray
    """
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(v,      dtype=float)

    n = pts.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    angles = np.empty(n - 1, dtype=float)
    for i in range(n - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        angles[i] = oriented_angle(p1, p2, ref)
    return angles



def by3roworientedangle4(points: np.ndarray,
                         v: Union[np.ndarray, Sequence[float]],
                         *,
                         debug: bool = True) -> np.ndarray:
    """
    Python translation of:
        NumericVector by3roworientedangle4(arma::mat points, arma::rowvec v)

    For each sliding window (p1, p2, p3) along the rows of `points`,
    compute a = p1 - p2, b = p3 - p2 and return oriented_angle(a, b, v).

    Parameters
    ----------
    points : (N, 3) array-like
    v      : (3,) array-like
    debug  : if True, print a, b, and ref (v) for each window

    Returns
    -------
    angles : (N-2,) ndarray
    """
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(v,      dtype=float)

    n = pts.shape[0]

    if n < 3:
        return np.array([], dtype=float)

    angles = np.empty(n - 2, dtype=float)
    for i in range(n - 2):
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2]
        a = p1 - p2
        b = p3 - p2

        angles[i] = oriented_angle(a, b, ref, False)
    return angles

def oriented_distance(a: Union[np.ndarray, Sequence[float]],
                      b: Union[np.ndarray, Sequence[float]],
                      n: Union[np.ndarray, Sequence[float]]
                     ) -> float:
    """
    Signed Euclidean distance between two points **a** and **b**
    in the direction of reference vector **n**.

    Equivalent to the Rcpp / Armadillo implementation:

        double oriented_distance(arma::rowvec a,
                                 arma::rowvec b,
                                 arma::rowvec n)

    Parameters
    ----------
    a, b : length-3 arrays (or any equal-length dimensionality)
        The two endpoints of the segment.
    n    : same length array
        Reference direction.  
        • If dot(b − a, n) < 0  → distance reported as **negative**.  
        • Otherwise             → **positive** (or zero).

    Returns
    -------
    float
        Signed distance  ‖b − a‖  with the sign determined by **n**.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = np.asarray(n, dtype=float)

    diff = b - a
    dist = np.linalg.norm(diff)
    sign = -1.0 if np.dot(diff, n) < 0.0 else 1.0
    return sign * dist

def orienteddistance1(points: Union[np.ndarray, Sequence[Sequence[float]]],
                      v: Union[np.ndarray, Sequence[float]] = (1.0, 0.0, 0.0)
                     ) -> np.ndarray:
    """
    Python equivalent of the Rcpp function

        NumericVector orienteddistance1(NumericMatrix points,
                                        NumericVector v = {1,0,0})

    For every successive **pair** of points (pᵢ, pᵢ₊₁) along the CGR
    trace, compute the *signed* distance ‖pᵢ₊₁ − pᵢ‖ with the sign
    determined by the dot-product with the reference vector **v**.

    Behavioural quirk kept from the original code
    ---------------------------------------------
    In the C++ code the loop starts at `i = 1` and writes directly to
    `distances[i]`, so element 0 of the output is left as the default
    (zero).  We reproduce that peculiarity here.

    Parameters
    ----------
    points : (N, 3) array-like
        Sequence of 3-D coordinates.
    v      : length-3 array-like, default (1,0,0)
        Reference direction used to set the sign.

    Returns
    -------
    distances : ndarray, shape (N − 1,)
        Signed step lengths; first element is **0** by design.
    """
    points = np.asarray(points, dtype=float)
    v      = np.asarray(v,      dtype=float)

    n = points.shape[0]
    if n < 2:
        raise ValueError("Need at least two points.")

    distances = np.zeros(n - 1, dtype=float)      # element 0 stays 0

    for i in range(1, n - 1):                      # i = 1 … n-2
        p1, p2 = points[i], points[i + 1]
        distances[i] = oriented_distance(p1, p2, v)

    return distances

# ----------------------------------------------------------------------
# oriented_distance  – signed Euclidean distance between two vectors
# ----------------------------------------------------------------------
def oriented_distance(a: Union[np.ndarray, Sequence[float]],
                      b: Union[np.ndarray, Sequence[float]],
                      n: Union[np.ndarray, Sequence[float]] = (1.0, 0.0, 0.0)
                     ) -> float:
    """
    Signed distance ‖b − a‖ with the sign determined by dot(b − a, n).

    Parameters
    ----------
    a, b : 1-D arrays (same dimension)
    n    : reference vector (same dimension)

    Returns
    -------
    float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = np.asarray(n, dtype=float)

    dist = np.linalg.norm(b - a)
    sign = -1.0 if np.dot(b - a, n) < 0 else 1.0
    return sign * dist


# ----------------------------------------------------------------------
# by3roworienteddistance4  – sliding-window “edge” distances
# ----------------------------------------------------------------------
def by3roworienteddistance4(points: Union[np.ndarray, Sequence[Sequence[float]]],
                            v: Union[np.ndarray, Sequence[float]] = (1.0, 0.0, 0.0)
                           ) -> np.ndarray:
    """
    Sliding 3-point window along the CGR trace.  
    For each window [p₁, p₂, p₃] compute:

        a = p₁ − p₂
        b = p₃ − p₂
        d = oriented_distance(a, b, v)

    The result is a vector of length N−2 (where N = #points).

    Parameters
    ----------
    points : (N, d) array-like  – sequence of coordinates
    v      : reference vector (default (1,0,0))

    Returns
    -------
    ndarray, shape (N-2,)
    """
    pts = np.asarray(points, dtype=float)
    v   = np.asarray(v,      dtype=float)

    N = pts.shape[0]
    if N < 3:
        raise ValueError("Need at least three points to use a 3-row window.")

    out = np.empty(N - 2, dtype=float)

    for i in range(N - 2):
        p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2]
        a = p1 - p2
        b = p3 - p2
        out[i] = oriented_distance(a, b, v)

    return out


# ------------------------------------------------------------
# common_function 
# ------------------------------------------------------------
def common_function(list_of_vectors: Iterable[np.ndarray],
                    func: Callable,
                    *args, **kwargs) -> float:
    """
    Apply *func* to every vector, then again to the collection
    of returned scalarss.

    Example
    -------
     common_function([np.arange(3), np.arange(10, 20)], max)
    19
    """
    scalars = [func(vec, *args, **kwargs) for vec in list_of_vectors]
    return func(scalars)


# ------------------------------------------------------------
# feature_histograms
# ------------------------------------------------------------
def feature_histograms(
    features            : List[np.ndarray],
    bin_count           : int,
    *,
    return_breaks       : bool = False,
    return_bin_centers  : bool = False
) -> Union[np.ndarray,
           Tuple[np.ndarray, np.ndarray]]:
    """
    Parameters
    ----------
    features           : list of 1-D arrays – one array per sequence
    bin_count          : number of histogram bins wanted
    return_breaks      : if True, also return the array of bin edges
    return_bin_centers : if True, also return the array of bin centres
                         (mutually exclusive with *return_breaks*).

    Returns
    -------
    counts                       – 2-D (n_seq × bin_count)
    or (counts, breaks)          – if *return_breaks*  is True
    or (counts, centres)         – if *return_bin_centers* is True
    """

    # --------------------------------------------------------
    # 2·1  Decide where the bin edges should be
    #      (special handling for angle ranges ±π)
    # --------------------------------------------------------
    # Are the global min / max exactly −π or +π ?
    is_full_angle_span = (
        np.isclose(common_function(features, np.max),  np.pi) or
        np.isclose(common_function(features, np.min), -np.pi)
    )

    if is_full_angle_span:
        # — remove unattainable values immediately next to ±π —
        def remove_pi(x: np.ndarray) -> np.ndarray:
            mask = (np.abs(np.pi - x) > 3e-8) & (np.abs(-np.pi - x) > 3e-8)
            return x[mask]

        cleaned   = remove_pi(np.concatenate(features))
        interior  = np.linspace(cleaned.min(), cleaned.max(), bin_count - 1)
        breaks    = np.concatenate(([-np.pi], interior, [np.pi]))
    else:
        breaks = np.linspace(
            common_function(features, np.min),
            common_function(features, np.max),
            bin_count + 1
        )

    # --------------------------------------------------------
    # 2·2  Histogram every vector with bin semantics
    # --------------------------------------------------------
    counts = []
    for vec in features:
        vec = np.asarray(vec, dtype=float)

        # findInterval with (rightmost.closed = TRUE, left.open = TRUE)
        idx  = np.searchsorted(breaks, vec, side='right')
        idx  = np.clip(idx, 1, len(breaks) - 1)   # all.inside = TRUE
        bins = idx - 1                            # to 0-based bin numbers
        counts.append(np.bincount(bins, minlength=bin_count))

    counts = np.vstack(counts)                    # shape (n_seq, bin_count)
    centres = (breaks[:-1] + breaks[1:]) / 2.0    # rollmean(k = 2)
    if return_breaks:
        return counts, breaks
    if return_bin_centers:
        return counts, centres
    return counts

# ---------------------------------------------------------------------
# Helper utilities                                                  |
# ---------------------------------------------------------------------
def _zero_variance_columns(mat: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Return a Boolean mask: True ⇢ column is (almost) constant."""
    return np.ptp(mat, axis=0) < atol          # ptp = max – min

def _global_min(list_of_vecs: Iterable[np.ndarray]) -> float:
    return min(vec.min() for vec in list_of_vecs)


def _global_max(list_of_vecs: Iterable[np.ndarray]) -> float:
    return max(vec.max() for vec in list_of_vecs)


def _hist_one_sequence(
    arr: np.ndarray,
    breaks: List[np.ndarray]
) -> np.ndarray:
    """
    Count how many points of `arr` fall in each interval of `breaks`
    (one break vector per coordinate).  Bins are *left-open, right-closed*
    just like the R `findInterval(..., rightmost.closed=TRUE, left.open=TRUE)`.
    The counts for all coordinates are concatenated into one long vector:
        [axis-1-bin-1, …, axis-1-bin-k,   axis-2-bin-1, …]
    """
    counts_per_axis = []
    for dim in range(arr.shape[1]):
        edges = breaks[dim]
        # numpy digitise gives i ∈ [1, len(edges)] with right=True → right-closed
        idx   = np.digitize(arr[:, dim], edges, right=True)
        # clamp so everything is inside (equiv. to all.inside = TRUE)
        idx   = np.clip(idx, 1, len(edges) - 1) - 1   # to 0-based bins
        counts = np.bincount(idx,
                             minlength=len(edges) - 1)  # (#breaks − 1) bins
        counts_per_axis.append(counts)
    return np.hstack(counts_per_axis)


# ---------------------------------------------------------------------
#  Main function – Python port of  coordinate_histograms()          |
# ---------------------------------------------------------------------
def coordinate_histograms(
    cgr_coords           : Sequence[np.ndarray],
    bin_count            : int,
    *,
    return_breaks        : bool = False,
    return_bin_centers   : bool = False
) -> Union[np.ndarray,
           Tuple[np.ndarray, List[np.ndarray]],
           Tuple[np.ndarray, np.ndarray]]:
    """
    Build 1-D histograms *for every coordinate axis*, for every CGR trace.

    Parameters
    ----------
    cgr_coords : list/tuple of (Nᵢ × d) numpy arrays (d ≤ 3)
                 Each array is the CGR point cloud of one sequence.
    bin_count  : how many bins per axis.
    return_breaks        : also return the list of break arrays.
    return_bin_centers   : alternatively return the matrix of bin centres
                           (shape d × bin_count).  Only one of the two
                           can be True.

    Returns
    -------
    counts_matrix                 – shape (#seq  ×  d·bin_count)
    or (counts_matrix, breaks)
    or (counts_matrix, centres)
    """
    if return_breaks and return_bin_centers:
        raise ValueError("Choose either return_breaks OR return_bin_centers.")

    # ------------------------------------------------------------
    # 1·1  Drop columns that are constant across *all* sequences
    # ------------------------------------------------------------
    all_points = np.vstack(cgr_coords)
    keep_mask  = ~_zero_variance_columns(all_points)
    cgr_coords = [pts[:, keep_mask] for pts in cgr_coords]

    # ------------------------------------------------------------
    # 1·2  Remove the origin row (0,0,0) if present
    # ------------------------------------------------------------
    def _strip_origin(arr: np.ndarray) -> np.ndarray:
        return arr[~np.all(arr == 0.0, axis=1)] if np.all(arr[0] == 0) else arr

    cgr_coords = [_strip_origin(pts) for pts in cgr_coords]

    d = cgr_coords[0].shape[1]                 # remaining dimensions

    # ------------------------------------------------------------
    # 1·3  Determine global histogram edges for every axis
    # ------------------------------------------------------------
    breaks: List[np.ndarray] = []
    for dim in range(d):
        lb = _global_min([pts[:, dim] for pts in cgr_coords])
        ub = _global_max([pts[:, dim] for pts in cgr_coords])

        edges = np.linspace(lb, ub, bin_count + 1)
        if edges.size < 2:                     # pathological case
            edges = np.array([lb, lb + 1.0])
        breaks.append(edges)

    # Bin centres like  zoo::rollmean(k = 2)
    centres = np.vstack([(b[:-1] + b[1:]) / 2.0 for b in breaks])

    # ------------------------------------------------------------
    # 1·4  Histogram every sequence
    # ------------------------------------------------------------
    counts = np.vstack([_hist_one_sequence(pts, breaks)
                        for pts in cgr_coords])   # (#seq × d·k)

    # ------------------------------------------------------------
    # 1·5  Output exactly in the style of the R version
    # ------------------------------------------------------------
    if return_breaks:
        return counts, breaks
    if return_bin_centers:
        return counts, centres
    return counts

ArrayLike = Union[np.ndarray, pd.DataFrame]

def split_coord_histograms(histograms: ArrayLike,
                           num_axes: int) -> List[ArrayLike]:
    """
    Slice a big histogram-matrix into *num_axes* equal column blocks.

    Parameters
    ----------
    histograms : np.ndarray | pandas.DataFrame   (N × M)
        Each row is a sequence; columns are concatenated
        coordinate-histogram bins (e.g. X-bins ⧺ Y-bins ⧺ Z-bins).
    num_axes   : int
        How many coordinate axes were concatenated (typically 3).

    Returns
    -------
    blocks : list of np.ndarray / DataFrame
        `blocks[i]` contains the columns that belong to axis *i*
        (all rows preserved, only the relevant slice of columns).
    """
    # --- convert to a NumPy view for the indexing math ------------
    is_df  = isinstance(histograms, pd.DataFrame)
    data   = histograms.values if is_df else np.asarray(histograms)

    n_cols = data.shape[1]
    if n_cols % num_axes:
        raise ValueError("Number of columns must be divisible by `num_axes`")

    block_size = n_cols // num_axes
    blocks: List[ArrayLike] = []

    for axis in range(num_axes):
        start, end = axis * block_size, (axis + 1) * block_size
        block_data = data[:, start:end]

        # keep the DataFrame “look & feel” if that’s what we got in
        if is_df:
            block = pd.DataFrame(block_data,
                                 index=histograms.index,
                                 columns=histograms.columns[start:end])
        else:
            block = block_data

        blocks.append(block)

    return blocks


# ---------------------------------------------------------------------
# helpers that R defined *inside* the original function
# ---------------------------------------------------------------------
def _power_mean(dist_mats: List[np.ndarray], p: float) -> np.ndarray:
    """
    Combine several distance-matrices with a (generalised) power-mean.

    • p = 1  → arithmetic mean
    • p = 0  → geometric   mean
    • p =-1  → harmonic    mean
    • p = 2  → quadratic   mean (RMS)
    """
    n = len(dist_mats)
    stack = np.stack(dist_mats, axis=0)          # shape (n, N, N)

    if p == 0:                                   # geometric mean
        with np.errstate(divide="ignore"):
            out = np.exp(np.log(stack).sum(axis=0) / n)
    else:
        out = (stack ** p).sum(axis=0) * (1.0 / n)
        out = out ** (1.0 / p)

    # numerical safety: make sure main diagonal is exactly zero
    np.fill_diagonal(out, 0.0)
    return out


def _row_zscore(mat: np.ndarray):
    """Row-wise centring & scaling ."""
    m  = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0                             # avoid /0 for constant rows
    return (mat - m) / sd

# ------------------------------------------------------------------
# helper – turn the recarray (fields i,j,k) returned by seq_to_cgr
#          into a plain ndarray so that numpy/scipy maths works
# ------------------------------------------------------------------
def _rec_to_xyz(rec: np.recarray) -> np.ndarray:
    """recarray → (N, 3) float64 ndarray."""
    return np.column_stack((rec["i"], rec["j"], rec["k"])).astype(float)

# ------------------------------------------------------------------
# utility – always give me an (N × B) 2-D array          
# ------------------------------------------------------------------
def _as_2d_rows(mat: Union[np.ndarray, Sequence[float]],
                n_rows: int) -> np.ndarray:
    """
    Coerce *mat* into (n_rows × B) 2-D float64.

    • already 2-D  → unchanged
    • 1-D length == n_rows      → reshape to (N,1)
    • 1-D anything else         → tile that single row across N rows
    """
    mat = np.asarray(mat, dtype=float)

    if mat.ndim == 2:
        return mat

    if mat.size == n_rows:                 # one scalar per sequence
        return mat.reshape(n_rows, 1)

    # any other 1-D size (typically B) – replicate for every sequence
    return np.tile(mat.reshape(1, -1), (n_rows, 1))

# ==============================================================
#  main distance function – with DEBUG PRINTS
# ==============================================================

def cgr_distance2(
    seq_cg: Union[Sequence[np.ndarray], Dict[str, np.ndarray]],
    *,
    frac: float = 1 / 15,
    cs: bool = True,
    power: float = 1.0,
    normalize: bool = False,
    seq_cg_test: Optional[Union[Sequence[np.ndarray], Dict[str, np.ndarray]]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:

    """
    Compute a fused distance matrix between CGR traces.

    Parameters
    ----------
    seq_cg : list/dict of CGR traces (numpy recarrays or ndarrays)
        The training/reference sequences.
    frac : float, default 1/15
        Fraction of average sequence length to use as histogram bin count.
    cs : bool, default True
        If True, z-score normalise the histogram rows before distance calc.
    power : float, default 1.0
        Power parameter for the final fusion of distance matrices.
    normalize : bool, default False
        If True, normalize each distance matrix to [0,1] before fusion.
    seq_cg_test : optional list/dict of CGR traces
        If provided, compute distances between `seq_cg_test` and `seq_cg` only.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[pd.DataFrame]]] : distance matrix (or matrices if test set provided, in the form (D_test, D_ref)).
    """

    is_train_test_mode = (seq_cg_test is not None)

    # 0 ─────────────────────────────────────────────────────────

    if isinstance(seq_cg, dict):
        names  = list(seq_cg.keys())
        traces = [_rec_to_xyz(t) for t in seq_cg.values()]
    else:
        names  = [f"s{i}" for i in range(len(seq_cg))]
        traces = [_rec_to_xyz(t) for t in seq_cg]
    
    N = len(traces)
    if N < 2:
        raise ValueError("Need at least two sequences.")
    

    if is_train_test_mode:
        if isinstance(seq_cg_test, dict):
            test_names  = list(seq_cg_test.keys())
            test_traces = [_rec_to_xyz(t) for t in seq_cg_test.values()]
        else:
            test_names  = [f"s{i}" for i in range(len(seq_cg_test))]
            test_traces = [_rec_to_xyz(t) for t in seq_cg_test]

        test_N = len(test_traces)
        if test_N < 2:
            raise ValueError("Need at least two sequences for the reference set.")

    # 1 ─────────────────────────────────────────────────────────
    avg_len = np.mean([t.shape[0] for t in traces])
    bins    = int(math.ceil(frac * (avg_len - 1)))  # Bins need to be computed only with training data (in order not to have data leakage)

    # 2  Feature generators
    ref_vecs = [np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1])]

    def _skip_origin(x): return x[1:]

    feat_fcns = [
        lambda pts, v: orientedangle1(_skip_origin(pts), v),
        by3roworientedangle4,
        orienteddistance1,
        by3roworienteddistance4,
    ]
        

    # 3  Shape-feature distance matrices
    shape_dists: List[np.ndarray] = []
    shape_dists_test: List[np.ndarray] = [] if is_train_test_mode else None


    for f_idx, f in enumerate(feat_fcns):
        for v_idx, v in enumerate(ref_vecs):

            # 1. Compute features for each trace
            feats = [f(t, v) for t in traces]  # list of arrays

            # 2. Convert to histogram representation
            H = feature_histograms(feats, bin_count=bins)
            H = _as_2d_rows(H, n_rows=N)

            # 3. Normalize rows (z-score)
            if cs:
                H = _row_zscore(H)

            # 4. Compute pairwise distances
            D = squareform(pdist(H, metric="euclidean"))
            if normalize and D.max() > 0:
                D = D / D.max()
            shape_dists.append(D)


            if is_train_test_mode:
                # Also compute distances between test traces and training traces

                # 1.
                test_feats = [f(t, v) for t in test_traces]

                # 2.
                H_test = feature_histograms(test_feats, bin_count=bins)
                H_test = _as_2d_rows(H_test, n_rows=test_N)

                # 3.
                if cs:
                    H_test = _row_zscore(H_test)

                # 4.
                D_test = cdist(H_test, H, metric="euclidean")
                if normalize and D_test.max() > 0:
                    D_test = D_test / D_test.max()

                shape_dists_test.append(D_test)


    # 4  Coordinate-histogram distance matrices
    coord_hist = coordinate_histograms(traces, bin_count=bins)

    coord_hist = _as_2d_rows(coord_hist, n_rows=N)
    #print("coord_hist shape:", coord_hist.shape)
    coord_blocks = split_coord_histograms(coord_hist, num_axes=3)
    #print("coord_blocks", coord_blocks)

    coord_dists: List[np.ndarray] = []
    for axis, H in enumerate(coord_blocks):
        H = H.values if isinstance(H, pd.DataFrame) else H
        H = _as_2d_rows(H, n_rows=N)
        if cs:
            H = _row_zscore(H)
        D = squareform(pdist(H, metric="euclidean"))
        if normalize and D.max() > 0:
            D = D / D.max()

        coord_dists.append(D)

    if is_train_test_mode:
        coord_hist_test = coordinate_histograms(test_traces, bin_count=bins)
        coord_hist_test = _as_2d_rows(coord_hist_test, n_rows=test_N)
        coord_blocks_test = split_coord_histograms(coord_hist_test, num_axes=3)

        coord_dists_test: List[np.ndarray] = []
        for axis, H_test in enumerate(coord_blocks_test):
            H_test = H_test.values if isinstance(H_test, pd.DataFrame) else H_test
            H_test = _as_2d_rows(H_test, n_rows=test_N)
            if cs:
                H_test = _row_zscore(H_test)
            D_test = cdist(H_test, H, metric="euclidean")
            if normalize and D_test.max() > 0:
                D_test = D_test / D_test.max()

            coord_dists_test.append(D_test)

    # 5  Fuse all 15 metrics
    D_final = _power_mean(shape_dists + coord_dists, p=power)

    if is_train_test_mode:
        D_final_test = _power_mean(shape_dists_test + coord_dists_test, p=power)
        return (pd.DataFrame(D_final, index=names, columns=names), pd.DataFrame(D_final_test, index=test_names, columns=names))
    
    return pd.DataFrame(D_final, index=names, columns=names)


Array = np.ndarray


def _as_list_with_names(
    cg_list: Union[Sequence[Array], Dict[str, Array]]
) -> Tuple[List[Array], List[str]]:
    """Accept list or dict; return ([arrays], [names])."""
    if isinstance(cg_list, dict):
        names = list(cg_list.keys())
        arrs = [np.asarray(cg_list[k], dtype=float) for k in names]
    else:
        arrs = [np.asarray(a, dtype=float) for a in cg_list]
        names = [str(i + 1) for i in range(len(arrs))]
    return arrs, names


def _estimate_bandwidth_fixed(_: Array, value: float) -> float:
    """R's estimate_bandwidth(..., method='fixed', value=bandwidth)."""
    return float(value)


def _gaussian_jitter_samples(
    data: Array, bandwidth: float, samples_per_point: int, seed: Optional[int] = None
) -> Array:
    """
    Sample 'samples_per_point' Gaussian-jittered points per original point.
    data: (N, d); returns (N*spp, d).
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape
    if n == 0:
        return np.empty((0, d))
    # Tile the data and add Gaussian noise (iid per dim)
    base = np.repeat(data, repeats=samples_per_point, axis=0)
    noise = rng.normal(loc=0.0, scale=bandwidth, size=base.shape)
    return base + noise


def _convex_hull(points: Array) -> Optional[ConvexHull]:
    """
    Safe convex hull for d up to 3. Returns None if not enough unique points.
    """
    # Deduplicate to avoid Qhull precision issues
    pts = np.unique(points, axis=0)
    d = pts.shape[1]
    # Need at least d+1 non-coplanar points
    if pts.shape[0] <= d:
        return None
    try:
        return ConvexHull(pts)
    except Exception:
        return None


def _in_hull_mask(points: Array, hull: ConvexHull) -> np.ndarray:
    """
    Check if points are inside a convex hull using half-space representation:
    For each facet: A x + b <= 0
    """
    # hull.equations has shape (n_facets, d+1): [*normal, offset]
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    # Inside if all inequalities <= 0 (with small tolerance)
    return (A @ points.T + b[:, None] <= 1e-12).all(axis=0)


def _bbox(points: Array) -> Tuple[Array, Array]:
    """Axis-aligned bounding box (min, max)."""
    return points.min(axis=0), points.max(axis=0)


def _bbox_volume(lo: Array, hi: Array) -> float:
    side = np.maximum(hi - lo, 0.0)
    return float(np.prod(side))


def _sample_uniform_in_bbox(
    lo: Array, hi: Array, n: int, seed: Optional[int] = None
) -> Array:
    rng = np.random.default_rng(seed)
    u = rng.random((n, lo.size))
    return lo + u * (hi - lo)


def volume_intersection_tanimoto(
    cg_list: Union[Sequence[Array], Dict[str, Array]],
    bandwidth: float = 0.003,
    hv_args: Optional[dict] = None,
    vi_args: Optional[dict] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    cg_list : list of (Ti, d) arrays OR dict name->array
        CGR point clouds per sequence (d<=3 recommended).
        If dict, names are taken from keys; otherwise names are '1','2',...
    bandwidth : float
        Fixed Gaussian jitter bandwidth (matches R's estimate_bandwidth(..., value=bandwidth)).
    hv_args : dict
        Extra args for 'hypervolume' build. Recognized keys:
          - 'samples_per_point' or 'samples.per.point' (default 5000)
          - 'sd_count' or 'sd.count' (kept for parity; not used in this simple approximation)
          - 'seed'
    vi_args : dict
        Extra args for intersection. Recognized keys:
          - 'num_points_max' or 'num.points.max' (default 150000)
          - 'seed'
    n_jobs : int
        Parallel jobs for hull building and pairwise intersections (-1 = all cores).
    random_state : int or None
        Base seed for reproducibility.

    Returns
    -------
    tanimoto_df : pandas.DataFrame (N x N)
        Symmetric matrix of Tanimoto similarities (1 on diagonal).
        Row/column labels are the sequence names.
    """
    hv_args = {} if hv_args is None else dict(hv_args)
    vi_args = {} if vi_args is None else dict(vi_args)

    # Map R-style keys to pythonic ones
    samples_per_point = int(
        hv_args.get("samples_per_point", hv_args.get("samples.per.point", 5000))
    )
    sd_count = hv_args.get("sd_count", hv_args.get("sd.count", 4))  # kept for parity
    hv_seed = hv_args.get("seed", random_state)

    num_points_max = int(vi_args.get("num_points_max", vi_args.get("num.points.max", 150000)))
    vi_seed = vi_args.get("seed", random_state)

    clouds, names = _as_list_with_names(cg_list)
    n = len(clouds)
    if n == 0:
        return pd.DataFrame()

    # --- Build per-sequence "hypervolume" (convex hull of Gaussian-jittered samples)
    def build_hv(idx: int):
        rng_seed = None if hv_seed is None else (hv_seed + idx)
        data = np.asarray(clouds[idx], dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Sequence {names[idx]} has ndim={data.ndim}, expected 2D array.")
        # Drop (0,0,0) first point if present
        if data.shape[0] > 0 and np.allclose(data[0], 0.0):
            data = data[1:]

        bw = _estimate_bandwidth_fixed(data, value=bandwidth)
        samples = _gaussian_jitter_samples(data, bandwidth=bw,
                                           samples_per_point=samples_per_point,
                                           seed=rng_seed)
        # If too few samples or degenerate, fallback to raw data
        pts = samples if samples.shape[0] >= data.shape[1] + 1 else data
        hull = _convex_hull(pts)
        if hull is None:
            # Degenerate: volume ~ 0
            return {
                "name": names[idx],
                "hull": None,
                "volume": 0.0,
                "bbox": (_bbox(pts) if pts.size else (np.zeros(data.shape[1]), np.zeros(data.shape[1]))),
            }
        lo, hi = _bbox(hull.points[hull.vertices])
        return {"name": names[idx], "hull": hull, "volume": float(hull.volume), "bbox": (lo, hi)}

    hypervolumes = Parallel(n_jobs=n_jobs)(
        delayed(build_hv)(i) for i in range(n)
    )

    # --- Pairwise intersections (Monte Carlo over union bbox, count points inside both hulls)
    def pair_tanimoto(i: int, j: int):
        hv1 = hypervolumes[i]; hv2 = hypervolumes[j]
        vol1 = hv1["volume"]; vol2 = hv2["volume"]
        hull1 = hv1["hull"];  hull2 = hv2["hull"]

        # Quick outs
        if vol1 == 0.0 or vol2 == 0.0 or hull1 is None or hull2 is None:
            return i, j, 0.0

        # Union bbox
        lo1, hi1 = hv1["bbox"]; lo2, hi2 = hv2["bbox"]
        lo = np.minimum(lo1, lo2); hi = np.maximum(hi1, hi2)
        bbox_vol = _bbox_volume(lo, hi)
        if bbox_vol == 0.0:
            # Identical degenerate bbox; if hulls equal, intersection==vol1==vol2
            inter = min(vol1, vol2)
            denom = vol1 + vol2 - inter
            return i, j, (inter / denom) if denom > 0 else 0.0

        rng_seed = None if vi_seed is None else (vi_seed + i * n + j)
        # Monte Carlo points in union bbox
        P = _sample_uniform_in_bbox(lo, hi, n=num_points_max, seed=rng_seed)

        inside1 = _in_hull_mask(P, hull1)
        inside2 = _in_hull_mask(P, hull2)
        both = inside1 & inside2
        inter_vol_est = bbox_vol * (both.mean())  # proportion * bbox volume

        denom = vol1 + vol2 - inter_vol_est
        tanimoto = inter_vol_est / denom if denom > 0 else 0.0
        # Clamp to [0,1] due to MC noise
        tanimoto = float(np.clip(tanimoto, 0.0, 1.0))
        return i, j, tanimoto

    pairs = list(combinations(range(n), 2))
    pair_results = Parallel(n_jobs=n_jobs)(
        delayed(pair_tanimoto)(i, j) for (i, j) in pairs
    )

    # Assemble symmetric matrix with diag=1
    T = np.eye(n, dtype=float)
    for i, j, t in pair_results:
        T[i, j] = t
        T[j, i] = t

    tanimoto_df = pd.DataFrame(T, index=names, columns=names)
    return tanimoto_df

