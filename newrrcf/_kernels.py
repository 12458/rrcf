"""
JIT-compiled numerical kernels for RRCF performance optimization.

This module contains Numba-optimized numerical operations extracted from the main
RCTree implementation. These kernels handle pure array computations and are compiled
to machine code for improved performance.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def compute_min_max_over_mask(X: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute min and max over masked array efficiently.

    Parameters:
    -----------
    X: np.ndarray (n x d)
       Data array
    S: np.ndarray (n,) bool
       Boolean mask

    Returns:
    --------
    xmin, xmax: tuple of np.ndarray (d,)
                Minimum and maximum values per dimension
    """
    # Get dimensions
    d = X.shape[1]
    xmin = np.empty(d, dtype=X.dtype)
    xmax = np.empty(d, dtype=X.dtype)

    # Initialize with infinity
    for j in range(d):
        xmin[j] = np.inf
        xmax[j] = -np.inf

    # Compute min/max only over masked elements
    for i in range(len(S)):
        if S[i]:
            for j in range(d):
                if X[i, j] < xmin[j]:
                    xmin[j] = X[i, j]
                if X[i, j] > xmax[j]:
                    xmax[j] = X[i, j]

    return xmin, xmax


@jit(nopython=True, cache=True)
def compute_cut_probabilities(xmin: np.ndarray, xmax: np.ndarray) -> np.ndarray:
    """
    Compute probabilities for cut dimension selection.

    Parameters:
    -----------
    xmin, xmax: np.ndarray (d,)
                Min and max values per dimension

    Returns:
    --------
    l: np.ndarray (d,)
       Normalized probabilities proportional to range in each dimension
    """
    l = xmax - xmin
    l_sum = np.sum(l)
    if l_sum > 0:
        l = l / l_sum
    else:
        # If all dimensions have same value, uniform probability
        l = np.ones_like(l) / len(l)
    return l


@jit(nopython=True, cache=True)
def compute_lr_bbox(bbox_l_min: np.ndarray, bbox_l_max: np.ndarray,
                    bbox_r_min: np.ndarray, bbox_r_max: np.ndarray) -> np.ndarray:
    """
    Compute combined bounding box from left and right child bboxes.

    Parameters:
    -----------
    bbox_l_min, bbox_l_max: np.ndarray (d,)
                             Left child bbox min and max
    bbox_r_min, bbox_r_max: np.ndarray (d,)
                             Right child bbox min and max

    Returns:
    --------
    bbox: np.ndarray (2, d)
          Combined bounding box
    """
    d = len(bbox_l_min)
    bbox = np.empty((2, d), dtype=bbox_l_min.dtype)

    for j in range(d):
        bbox[0, j] = min(bbox_l_min[j], bbox_r_min[j])
        bbox[1, j] = max(bbox_l_max[j], bbox_r_max[j])

    return bbox


@jit(nopython=True, cache=True)
def compute_insert_cut_dimension(bbox_hat_min: np.ndarray, bbox_hat_max: np.ndarray,
                                  r: float) -> tuple[int, float, np.ndarray]:
    """
    Compute cut dimension and value for point insertion.

    Parameters:
    -----------
    bbox_hat_min, bbox_hat_max: np.ndarray (d,)
                                 Extended bounding box including new point
    r: float
       Random value for cut selection

    Returns:
    --------
    cut_dimension: int
                   Dimension to cut
    cut: float
         Value of cut
    span_sum: np.ndarray (d,)
              Cumulative sum of spans (for computing cut value)
    """
    b_span = bbox_hat_max - bbox_hat_min
    span_sum = np.cumsum(b_span)

    cut_dimension = -1
    for j in range(len(span_sum)):
        if span_sum[j] >= r:
            cut_dimension = j
            break

    if cut_dimension == -1:
        # Fallback: use last dimension
        cut_dimension = len(span_sum) - 1

    cut = bbox_hat_min[cut_dimension] + span_sum[cut_dimension] - r

    return cut_dimension, cut, span_sum


@jit(nopython=True, cache=True)
def expand_bbox_for_point(bbox_min: np.ndarray, bbox_max: np.ndarray,
                          point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand bounding box to include a new point.

    Parameters:
    -----------
    bbox_min, bbox_max: np.ndarray (d,)
                        Original bbox bounds
    point: np.ndarray (d,)
           Point to include

    Returns:
    --------
    bbox_hat_min, bbox_hat_max: np.ndarray (d,)
                                 Extended bbox bounds
    """
    d = len(point)
    bbox_hat_min = np.empty(d, dtype=bbox_min.dtype)
    bbox_hat_max = np.empty(d, dtype=bbox_max.dtype)

    for j in range(d):
        bbox_hat_min[j] = min(bbox_min[j], point[j])
        bbox_hat_max[j] = max(bbox_max[j], point[j])

    return bbox_hat_min, bbox_hat_max


@jit(nopython=True, cache=True)
def check_bbox_tighten(bbox_min: np.ndarray, bbox_max: np.ndarray,
                       child_bbox_min: np.ndarray, child_bbox_max: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Check if bbox needs tightening and return updated bounds.

    Parameters:
    -----------
    bbox_min, bbox_max: np.ndarray (d,)
                        Current bbox bounds
    child_bbox_min, child_bbox_max: np.ndarray (d,)
                                     Child bbox to check against

    Returns:
    --------
    needs_update: bool
                  Whether any update is needed
    new_min, new_max: np.ndarray (d,)
                      Updated bbox bounds (may be same as input)
    """
    d = len(bbox_min)
    needs_update = False
    new_min = bbox_min.copy()
    new_max = bbox_max.copy()

    for j in range(d):
        if child_bbox_min[j] < bbox_min[j]:
            new_min[j] = child_bbox_min[j]
            needs_update = True
        if child_bbox_max[j] > bbox_max[j]:
            new_max[j] = child_bbox_max[j]
            needs_update = True

    return needs_update, new_min, new_max


@jit(nopython=True, cache=True)
def check_bbox_contains_point(bbox_min: np.ndarray, bbox_max: np.ndarray,
                              point: np.ndarray) -> bool:
    """
    Check if a point is on the boundary of a bounding box.

    Parameters:
    -----------
    bbox_min, bbox_max: np.ndarray (d,)
                        Bbox bounds
    point: np.ndarray (d,)
           Point to check

    Returns:
    --------
    on_boundary: bool
                 True if point defines any boundary of bbox
    """
    for j in range(len(point)):
        if bbox_min[j] == point[j] or bbox_max[j] == point[j]:
            return True
    return False


@jit(nopython=True, cache=True)
def update_bbox_elementwise(mins: np.ndarray, maxes: np.ndarray,
                            point: np.ndarray) -> None:
    """
    Update min/max arrays with a point's coordinates (in-place).

    Parameters:
    -----------
    mins, maxes: np.ndarray (d,)
                 Min and max arrays to update
    point: np.ndarray (d,)
           Point to compare against

    Returns:
    --------
    None (updates in-place)
    """
    for j in range(len(point)):
        if point[j] < mins[j]:
            mins[j] = point[j]
        if point[j] > maxes[j]:
            maxes[j] = point[j]
