from pathlib import Path
import pickle
import os

def load_cache(cache_path, compute_fn, force_recompute=False, save=True):
    """
    Generic cache loader:
      - If cache exists and not forcing recompute → load it.
      - Otherwise compute and optionally save result.
    """

    os.makedirs('pkls', exist_ok=True)

    # Full path to cache file
    cache_path = Path(__file__).resolve().parent.parent / 'pkls' / cache_path
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

def load_pickle(cache_path):
    cache_path = Path(__file__).resolve().parent.parent / 'pkls' / cache_path
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        raise Exception("Cache file not found.")

def save_pickle_to_test(data, cache_path):
    cache_path = Path(__file__).resolve().parent.parent.parent / 'o02_test' / 'pkls' / cache_path
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)