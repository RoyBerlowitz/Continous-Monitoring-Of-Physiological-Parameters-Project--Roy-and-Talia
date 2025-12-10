import os
import pickle

def load_cache_or_compute(cache_path, compute_fn, force_recompute=False, save=True):
    """
    Generic cache loader:
      - If cache exists and not forcing recompute → load it.
      - Otherwise compute and optionally save result.
    """
    if (not force_recompute) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Compute the result
    result = compute_fn()

    # Save if requested
    if save:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result
