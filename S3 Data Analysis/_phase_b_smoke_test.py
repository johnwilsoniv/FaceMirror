"""Standalone smoke test for Phase B feature modules.

Avoids importing the full paralysis_utils (which pulls imblearn / seaborn).
Inlines the small number of helpers the synkinesis modules actually need by
extracting them with ast, then exercises each per-type module on synthetic
data and prints a feature inventory.

Run with any Python that has pandas + numpy.
"""
import ast
import importlib
import importlib.util
import logging
import os
import sys
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("phase_b_smoke")


def _load_paralysis_utils_subset():
    """Build a stub `paralysis_utils` module containing only the helpers used
    by the synkinesis feature modules — bypasses heavy ML imports."""
    src_path = os.path.join(ROOT, "paralysis_utils.py")
    with open(src_path) as f:
        tree = ast.parse(f.read())

    wanted = {
        "calculate_ratio",
        "calculate_percent_diff",
        "_get_au_value_series",
        "_extract_base_au_features",
        "_extract_coupling_features",
    }
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = types.ModuleType("paralysis_utils")
    module.__dict__.update({"pd": pd, "np": np, "logger": logging.getLogger("paralysis_utils")})
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<paralysis_utils-subset>", "exec"), module.__dict__)
    sys.modules["paralysis_utils"] = module
    return module


def _build_sample_frame():
    """Synthesize one row covering every (action, side, AU) the modules touch."""
    actions = ["BL", "RE", "ET", "ES", "BS", "SS", "SO", "SE", "PL", "FR", "BK", "WN", "BC", "LT"]
    aus = ["AU01_r", "AU02_r", "AU06_r", "AU10_r", "AU12_r", "AU14_r",
           "AU15_r", "AU17_r", "AU25_r", "AU45_r"]
    rng = np.random.default_rng(42)
    data = {}
    for action in actions:
        for side in ("Left", "Right"):
            for au in aus:
                base = float(rng.uniform(0.0, 2.0))
                data[f"{action}_{side} {au}"] = [base]
                data[f"{action}_{side} {au} (Normalized)"] = [base * float(rng.uniform(0.4, 0.8))]
    return pd.DataFrame(data)


def main():
    _load_paralysis_utils_subset()

    # config_paths is heavy on first import; provide a stub if it fails.
    try:
        importlib.import_module("config_paths")
    except Exception:
        cp = types.ModuleType("config_paths")
        cp.get_models_dir = lambda: os.path.join(ROOT, "models")
        cp.get_output_base_dir = lambda: os.path.join(ROOT, "_smoke_output")
        sys.modules["config_paths"] = cp

    importlib.import_module("synkinesis_config")
    importlib.import_module("synkinesis_features_base")

    df = _build_sample_frame()
    summary = []
    sample_features_by_type = {}
    for type_key in [
        "ocular_oral",
        "oral_ocular",
        "snarl_smile",
        "mentalis",
        "hypertonicity",
        "brow_cocked",
    ]:
        module = importlib.import_module(f"{type_key}_features")
        for side in ("Left", "Right"):
            features = module.extract_features(df, side)
            assert isinstance(features, pd.DataFrame), f"{type_key} {side}: not a DataFrame"
            assert features.shape[0] == 1, f"{type_key} {side}: row count != 1"
            assert features.notna().all().all(), f"{type_key} {side}: NaNs in features"
            assert np.isfinite(features.values).all(), f"{type_key} {side}: non-finite values"
        summary.append((type_key, features.shape[1]))
        sample_features_by_type[type_key] = sorted(features.columns.tolist())

    print("Per-type feature counts (Right side):")
    for type_key, n_features in summary:
        print(f"  {type_key:15s}  {n_features:4d} features")
    print()
    print("Sample feature names per type (first 6):")
    for type_key, cols in sample_features_by_type.items():
        print(f"  {type_key}:")
        for c in cols[:6]:
            print(f"    {c}")
    print()
    print("PHASE B SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
