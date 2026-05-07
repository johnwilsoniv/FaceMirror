"""Layer 1 — Pre-build environment validation.

These tests must pass BEFORE running PyInstaller for any stage. They catch
the issues we hit during v1.1.1 Mac builds:

- Missing bin/ffmpeg in worktree (ffmpeg/ffprobe not in git)
- requirements.txt entries that don't resolve on PyPI (PyQt5-multimedia)
- Stale references in spec files (paths that no longer exist on disk)
- Missing weights/ or models/ trees the spec expects to bundle

Run: pytest tests/release_preflight/test_build_env.py -v
Total runtime: <30 sec (no actual installs or builds happen).
"""
from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path
import pytest


# -----------------------------------------------------------------------------
# Required local binaries / data trees per stage
# (Hand-maintained; less brittle than parsing the .spec dynamically.)
# -----------------------------------------------------------------------------

REQUIRED_LOCAL_RESOURCES = {
    "S1": [
        ("bin/ffmpeg", "executable"),
        ("bin/ffprobe", "executable"),
        ("weights", "directory"),
        ("weights/1k3d68.onnx", "file"),
        ("weights/Alignment_RetinaFace.pth", "file"),
        ("weights/In-the-wild_aligned_PDM_68.txt", "file"),
        ("weights/AU_predictors", "directory"),
        ("weights/clnf", "directory"),
    ],
    "S2": [
        ("bin/ffmpeg", "executable"),
        ("bin/ffprobe", "executable"),
    ],
    "S3": [
        ("models", "directory"),
        ("models/upper_face_model.pkl", "file"),
        ("models/mid_face_model.pkl", "file"),
        ("models/lower_face_model.pkl", "file"),
        ("models/upper_face_scaler.pkl", "file"),
        ("models/mid_face_scaler.pkl", "file"),
        ("models/lower_face_scaler.pkl", "file"),
        ("models/upper_face_features.list", "file"),
        ("models/mid_face_features.list", "file"),
        ("models/lower_face_features.list", "file"),
        ("paper_combined_results.csv", "file"),
        ("FPRS FP Key.csv", "file"),
    ],
}


@pytest.mark.parametrize("stage,resource_path,kind",
    [(s, p, k) for s, lst in REQUIRED_LOCAL_RESOURCES.items() for p, k in lst],
    ids=lambda v: str(v))
def test_required_local_resource_present(stage, resource_path, kind, stage_dirs):
    """Each stage's spec expects certain files/dirs to be on disk at build
    time. Catch missing ffmpeg/weights/models BEFORE PyInstaller runs.

    Note: ffmpeg + ffprobe are gitignored (binary, large) so they must be
    placed manually before building. This test is the prompt for that.
    """
    full_path = stage_dirs[stage] / resource_path
    assert full_path.exists(), (
        f"{stage}: required {kind} missing at {full_path}.\n"
        f"  PyInstaller will fail or silently produce a broken build.\n"
        f"  If this is an external binary (ffmpeg/ffprobe), copy it manually "
        f"into the stage's bin/ directory before building."
    )
    if kind == "executable":
        assert os.access(full_path, os.X_OK), f"{stage}: {full_path} not executable (chmod +x)"
    elif kind == "directory":
        assert full_path.is_dir(), f"{stage}: {full_path} exists but is not a directory"
    elif kind == "file":
        assert full_path.is_file(), f"{stage}: {full_path} exists but is not a regular file"


# -----------------------------------------------------------------------------
# requirements.txt resolvability
# -----------------------------------------------------------------------------

def _parse_requirements(req_path: Path) -> list[str]:
    """Extract package specs from a requirements.txt, ignoring comments and
    blank lines."""
    out = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        # Skip -r references and editable installs (covered by their own files)
        if line.startswith(("-r", "-e", "-c", "--")):
            continue
        out.append(line)
    return out


@pytest.mark.parametrize("stage", ["S1", "S2", "S3"])
def test_requirements_file_exists(stage, stage_requirements):
    """Each stage must have a requirements.txt the build pipeline can read."""
    req_path = stage_requirements[stage]
    assert req_path.exists(), f"{stage}: missing {req_path}"


@pytest.mark.parametrize("stage", ["S1", "S2", "S3"])
def test_requirements_have_no_typos(stage, stage_requirements, python_executable):
    """Run pip's resolver in dry-run mode to catch packages that don't exist
    on PyPI (e.g. PyQt5-multimedia is not a real package; PyQt5 already has
    QtMultimedia built in).

    Uses --dry-run + --report=- to avoid actually installing anything.
    Catches: typos, deprecated package names, PyPI-removed versions.
    """
    req_path = stage_requirements[stage]
    if not req_path.exists():
        pytest.skip(f"{stage}: no requirements.txt")
    # pip 23+ supports --dry-run; falls back gracefully on older pips
    result = subprocess.run(
        [python_executable, "-m", "pip", "install", "--dry-run", "--quiet",
         "-r", str(req_path)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{stage}: pip dry-run rejected requirements.txt:\n"
            f"  STDERR: {result.stderr[-2000:]}\n"
            f"  STDOUT: {result.stdout[-1000:]}"
        )


# -----------------------------------------------------------------------------
# .spec file static checks
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("stage", ["S1", "S2", "S3"])
def test_spec_file_exists(stage, stage_specs):
    """Each stage must ship a PyInstaller .spec at the expected name."""
    spec = stage_specs[stage]
    assert spec.exists(), f"{stage}: missing PyInstaller spec at {spec}"


@pytest.mark.parametrize("stage", ["S1", "S2", "S3"])
def test_spec_app_version_set(stage, stage_specs):
    """Catches the bug class where we forget to bump app_version when tagging
    a release.
    """
    spec = stage_specs[stage]
    text = spec.read_text()
    m = re.search(r"app_version\s*=\s*['\"]([^'\"]+)['\"]", text)
    assert m, f"{stage}: spec doesn't define app_version"
    version = m.group(1)
    assert re.match(r"^\d+\.\d+\.\d+", version), (
        f"{stage}: app_version='{version}' isn't semver-shaped"
    )


@pytest.mark.parametrize("stage", ["S1", "S2", "S3"])
def test_spec_datas_paths_exist_on_disk(stage, stage_specs, stage_dirs):
    """Regex-extract simple `datas += [('path', '...')]` literals from the
    spec and verify each source path resolves on disk.

    This is intentionally a lightweight static check (no exec). Will catch:
    - the bin/ffmpeg case (path literal points at missing file)
    - typos in resource paths
    Won't catch: dynamically-computed paths, conditional bundling. Those need
    the full PyInstaller analysis pass to validate.
    """
    spec = stage_specs[stage]
    text = spec.read_text()
    # Match: datas += [('source_path', 'dest')] OR datas += [('source_path', 'dest', ...)]
    pattern = re.compile(r"datas\s*\+?=\s*\[?\s*\(\s*['\"]([^'\"]+)['\"]\s*,")
    paths = set(pattern.findall(text))
    missing = []
    for p in paths:
        # Skip computed paths, glob patterns, absolute paths (probably system tools)
        if any(c in p for c in ("*", "{", "$")) or p.startswith("/"):
            continue
        full = stage_dirs[stage] / p
        if not full.exists():
            missing.append(p)
    assert not missing, (
        f"{stage}: {len(missing)} datas paths in spec don't exist on disk:\n  "
        + "\n  ".join(sorted(missing))
        + f"\n(checked relative to {stage_dirs[stage]})"
    )
