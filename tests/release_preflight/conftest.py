"""Shared fixtures for release-preflight tests.

These tests run BEFORE PyInstaller / pip install steps to catch issues we'd
otherwise only discover mid-build. See tests/release_preflight/README.md for
the full test layer breakdown.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def stage_dirs() -> dict[str, Path]:
    """Path to each shippable stage's source directory."""
    return {
        "S1": REPO_ROOT / "S1_FaceMirror",
        "S2": REPO_ROOT / "S2 Action Coder",
        "S3": REPO_ROOT / "S3 Data Analysis",
    }


@pytest.fixture(scope="session")
def stage_specs(stage_dirs) -> dict[str, Path]:
    """Path to each stage's PyInstaller .spec file (ARM macOS)."""
    return {
        "S1": stage_dirs["S1"] / "Face_Mirror.spec",
        "S2": stage_dirs["S2"] / "Action_Coder.spec",
        "S3": stage_dirs["S3"] / "Paralysis_Analyzer.spec",
    }


@pytest.fixture(scope="session")
def stage_requirements(stage_dirs) -> dict[str, Path]:
    """Path to each stage's requirements.txt (macOS variant)."""
    return {
        "S1": stage_dirs["S1"] / "requirements.txt",
        "S2": stage_dirs["S2"] / "requirements.txt",
        "S3": stage_dirs["S3"] / "requirements.txt",
    }


@pytest.fixture(scope="session")
def python_executable() -> str:
    """Python in the venv we build with."""
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable
