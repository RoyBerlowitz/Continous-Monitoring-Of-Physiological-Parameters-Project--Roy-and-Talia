import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

def contingency_coefficient(attr: pd.Series, target: pd.Series) -> float:
    """
    Compute the Class-Attribute Contingency Coefficient (CACC).
    """
    contingency_table = pd.crosstab(attr, target)
    chi2 = ((contingency_table - contingency_table.mean()) ** 2 / contingency_table.mean()).sum().sum()
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0
    return np.sqrt(chi2 / (chi2 + n))

def best_cut_point(values: np.ndarray, target: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find the best cut point that maximizes CACC.
    Returns (best_cut, best_cacc) or None if no improvement.
    """
    sorted_idx = np.argsort(values)
    values_sorted = values[sorted_idx]
    target_sorted = target[sorted_idx]

    unique_vals = np.unique(values_sorted)
    if len(unique_vals) <= 1:
        return None

    best_cut, best_cacc = None, -1
    for i in range(1, len(values_sorted)):
        if values_sorted[i] == values_sorted[i - 1]:
            continue
        cut = (values_sorted[i] + values_sorted[i - 1]) / 2
        left = pd.Series(np.where(values <= cut, f"<= {cut:.4f}", f"> {cut:.4f}"))
        cacc = contingency_coefficient(left, pd.Series(target))
        if cacc > best_cacc:
            best_cut, best_cacc = cut, cacc

    return (best_cut, best_cacc) if best_cut is not None else None

def CACC_discretization(values: np.ndarray, target: np.ndarray, min_gain: float = 1e-6) -> List[float]:
    """
    Recursively discretize a continuous attribute using CACC.
    Returns sorted list of cut points.
    """
    cut_points = []

    def recursive_cut(sub_values, sub_target):
        best = best_cut_point(sub_values, sub_target)
        if not best:
            return
        cut, cacc = best
        # Compute CACC before split
        before_cacc = contingency_coefficient(pd.Series(np.repeat("all", len(sub_values))), pd.Series(sub_target))
        if cacc - before_cacc > min_gain:
            cut_points.append(cut)
            left_mask = sub_values <= cut
            recursive_cut(sub_values[left_mask], sub_target[left_mask])
            recursive_cut(sub_values[~left_mask], sub_target[~left_mask])

    recursive_cut(values, target)
    return sorted(cut_points)