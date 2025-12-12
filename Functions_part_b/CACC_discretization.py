import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import chi2_contingency

def contingency_coefficient(attr: pd.Series, target: pd.Series) -> float:
    # This function computes the class-attribute Contingency Coefficient (CACC).
    # first we build the contigency value between each attribute to each label
    contingency_table = pd.crosstab(attr, target)
    # we calculate chi_squared of the contigency value
    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table)
    except ValueError:
        # Happens if table has rows/columns with zero variance
        chi2 = 0.0
    # then we compute the total sum of contigencies.
    n = contingency_table.sum().sum()
    # if the sum is 0, we return zero, because otherwise we will return 1 while contigency is zero
    if n == 0:
        return 0.0
    # we normalize the chi_squared to be between zero and one
    return np.sqrt(chi2 / (chi2 + n))

def best_cut_point(values: np.ndarray, target: np.ndarray) -> Optional[Tuple[float, float]]:
    # Find the best cut point that maximizes CACC.
    # Returns (best_cut, best_cacc) or None if no improvement.
    # Firstly, we are sorting the values and get their new indexed.
    sorted_idx = np.argsort(values)
    # now we re-organize the values and and the target by the new sorting
    values_sorted = values[sorted_idx]
    target_sorted = target[sorted_idx]
    # we try to find how many unique values there are. if there is only one - we stop/
    unique_vals = np.unique(values_sorted)
    if len(unique_vals) <= 1:
        return None
    # we initialize the best cut location to None and the best CACC score to -1, so we will always have score at the end (as it will surprass it).
    best_cut, best_cacc = None, -1
    # we go over the sorted values list until we identify a change in the value
    for i in range(1, len(values_sorted)):
        if values_sorted[i] == values_sorted[i - 1]:
            continue
        # we define the mean of the different values as the cut point.
        cut = (values_sorted[i] + values_sorted[i - 1]) / 2
        # we define the values before the cut as a "left" group
        left = pd.Series(np.where(values <= cut, f"<= {cut:.4f}", f"> {cut:.4f}"))
        # we calculate the CACC score of the discertiaztion
        cacc = contingency_coefficient(left, pd.Series(target))
        # if the CACC score is better, we define this cut as the best so far
        if cacc > best_cacc:
            best_cut, best_cacc = cut, cacc
    # we return the best cut and cacc if it not none
    return (best_cut, best_cacc) if best_cut is not None else None

def CACC_discretization(column: pd.Series, target_series: pd.Series, column_name: str, min_gain: float = 1e-6) -> List[float]:

    # Here, we recursively discretize a continuous attribute using CACC.
    # we return sorted list of cut points.
    values = column.values
    target = target_series.values
    print(f"started discretization for {column_name}")
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
    print(f"completed discretization for {column_name}")

    return sorted(cut_points)


def discretize_colum(df: pd.DataFrame, column: str, cut_points: List[float]) -> pd.Series:
    # 1. Define the bins (including boundaries)
    bins = [-np.inf] + cut_points + [np.inf]

    # 2. Define the labels (0, 1, 2, 3...) for the new discrete categories
    # There will be len(cut_points) + 1 intervals
    labels = np.arange(len(cut_points) + 1).astype(int)

    # 3. Apply the discretization
    # pd.cut handles the assignment of values to their corresponding interval index (label)
    discretized_series = pd.cut(
        df[column],
        bins=bins,
        labels=labels,
        include_lowest=True  # Ensures the lowest value is included
    ).astype(int)

    return discretized_series

