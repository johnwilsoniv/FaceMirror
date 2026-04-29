"""Shared fixtures and helpers for the SplitFace pipeline test framework.

Most tests want one or more of:
  - The list of canary patients (parametrize over them)
  - Per-stage metric thresholds (split by normal vs paralyzed)
  - Paths to canonical input data + golden references
  - A loaded saved Jan 1 model for inference parity tests

This module centralizes those so tests stay focused on assertions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths — single source of truth so tests don't pin paths individually.
# ---------------------------------------------------------------------------

S3_ROOT = Path(__file__).resolve().parent.parent  # "S3 Data Analysis"
TESTS_ROOT = Path(__file__).resolve().parent
GOLDEN_ROOT = TESTS_ROOT / "golden"
CANARY_DATA_ROOT = TESTS_ROOT / "canary_data"

# External data (lives outside the repo on each engineer's machine)
SPLITFACE_BASE = Path("/Users/johnwilsoniv/Documents/SplitFace")
S1O_VIDEOS = SPLITFACE_BASE / "S1O Processed Files" / "Face Mirror 1.0 Output"
S2O_PYFACEAU = SPLITFACE_BASE / "S2O Coded Files"
S2O_CPP = SPLITFACE_BASE / "S2O Coded Files OF"
S3O_RESULTS = SPLITFACE_BASE / "S3O Results"
PYFACEAU_COMBINED_CSV = S3O_RESULTS / "combined_results.csv"
CPP_COMBINED_CSV = S3O_RESULTS / "combined_results_OF_v2.csv"

# Saved Jan 1 manuscript model (load with joblib, NOT pickle)
JAN1_MODEL_DIR = (
    S3_ROOT
    / "dist"
    / "Paralysis Analyzer.app"
    / "Contents"
    / "Resources"
    / "models"
)

# Optional rich C++ reference (richer than AU-only S2O CSVs)
GOLD_CPP_REFERENCE = S3_ROOT.parent / "gold_cpp_reference"

# Reproducibility env vars expected during retrain tests
DETERMINISM_ENV = {"PYTHONHASHSEED": "42", "OMP_NUM_THREADS": "1"}


# ---------------------------------------------------------------------------
# Canary patient registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Canary:
    """One canary patient + everything tests need to identify and locate it."""

    id: str
    severity: str  # "Normal" | "Partial" | "Complete"
    labels: dict[str, str]
    tier0: bool
    in_gold_cpp_reference: bool
    role: str

    @property
    def is_paralyzed(self) -> bool:
        return self.severity != "Normal"

    @property
    def threshold_bucket(self) -> str:
        """Which threshold band this patient is judged against."""
        return "paralyzed" if self.is_paralyzed else "normal"

    def video(self, side: str) -> Path:
        side = side.lower()
        return S1O_VIDEOS / f"{self.id}_{side}_mirrored.mp4"

    def pyfaceau_csv(self, side: str) -> Path:
        side = side.lower()
        return S2O_PYFACEAU / f"{self.id}_{side}_mirrored_coded.csv"

    def cpp_csv(self, side: str) -> Path:
        """C++ AU CSV (also has 2D/3D landmarks + head pose)."""
        side = side.lower()
        return S2O_CPP / f"{self.id}_{side}_mirrored.csv"

    def gold_cpp_dir(self, side: str) -> Path | None:
        """Rich C++ reference dir (if available)."""
        if not self.in_gold_cpp_reference:
            return None
        side = side.lower()
        return GOLD_CPP_REFERENCE / f"{self.id}_{side}_mirrored"


def _load_canaries() -> list[Canary]:
    yaml_path = TESTS_ROOT / "canary_patients.yaml"
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    out: list[Canary] = []
    for entry in data["canaries"]:
        out.append(
            Canary(
                id=entry["id"],
                severity=entry["severity"],
                labels=dict(entry["labels"]),
                tier0=bool(entry.get("tier0", False)),
                in_gold_cpp_reference=bool(entry.get("in_gold_cpp_reference", False)),
                role=entry["role"],
            )
        )
    return out


CANARIES: list[Canary] = _load_canaries()
CANARIES_BY_ID: dict[str, Canary] = {c.id: c for c in CANARIES}
TIER0_CANARIES: list[Canary] = [c for c in CANARIES if c.tier0]


# ---------------------------------------------------------------------------
# Metric bands
# ---------------------------------------------------------------------------


def load_metric_bands() -> dict[str, Any]:
    """Load the per-stage threshold bands. Returns nested dict:

      {stage_name: {normal: {...metrics...}, paralyzed: {...metrics...}}}

    Stages may also have a top-level 'shared' section for metrics that don't
    split by severity.
    """
    p = GOLDEN_ROOT / "metric_bands.yaml"
    if not p.exists():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def canaries() -> list[Canary]:
    return CANARIES


@pytest.fixture(scope="session")
def metric_bands() -> dict[str, Any]:
    bands = load_metric_bands()
    if not bands:
        pytest.skip(
            "metric_bands.yaml not populated; run "
            "`python tests/update_goldens.py --stage=all --reason='initial baseline'` "
            "first."
        )
    return bands


@pytest.fixture(scope="session")
def jan1_model():
    """Load the saved Jan 1 Mid Face manuscript model (joblib, not pickle)."""
    import joblib  # heavy import, defer until needed

    sys.path.insert(0, str(S3_ROOT))  # so OrdinalBinaryClassifier imports resolve
    model_path = JAN1_MODEL_DIR / "mid_face_model.pkl"
    scaler_path = JAN1_MODEL_DIR / "mid_face_scaler.pkl"
    feats_path = JAN1_MODEL_DIR / "mid_face_features.list"
    if not model_path.exists():
        pytest.skip(f"Jan 1 model not present at {model_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with feats_path.open() as f:
        features = [line.strip() for line in f if line.strip()]
    return {"model": model, "scaler": scaler, "features": features}


def _set_determinism_env() -> None:
    for k, v in DETERMINISM_ENV.items():
        os.environ.setdefault(k, v)


@pytest.fixture(scope="session", autouse=True)
def _ensure_determinism_env():
    """Auto-applied: ensures PYTHONHASHSEED + OMP_NUM_THREADS are set before
    any test runs. Tests that fork subprocesses should still pass these
    explicitly, but in-process numpy/sklearn calls benefit from this."""
    _set_determinism_env()
    yield


# ---------------------------------------------------------------------------
# Helpful parametrize decorators
# ---------------------------------------------------------------------------


def parametrize_canaries(tier: int = 1):
    """Parametrize a test over the canary set for the given tier.

    Tier 0 = the small fast-gate set (1 normal + 1 paralyzed).
    Tier 1+ = the full 10-canary set.
    """
    selected = TIER0_CANARIES if tier == 0 else CANARIES
    return pytest.mark.parametrize(
        "canary",
        selected,
        ids=[c.id for c in selected],
    )


def parametrize_canaries_sides(tier: int = 1):
    """Parametrize over (canary, side) pairs."""
    selected = TIER0_CANARIES if tier == 0 else CANARIES
    params: list[tuple[Canary, str]] = []
    ids: list[str] = []
    for c in selected:
        for side in ("left", "right"):
            params.append((c, side))
            ids.append(f"{c.id}_{side}")
    return pytest.mark.parametrize(("canary", "side"), params, ids=ids)
