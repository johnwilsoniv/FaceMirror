"""Layer 2 — Post-install dependency smoke.

Runs AFTER `pip install` in the build venv, BEFORE PyInstaller. Catches
broken installs that pip thinks succeeded but actually didn't:

- PyAV's libavcodec dylib mismatch (libavcodec.61.19.101 missing while
  libavformat.61.7.100 is present — a partial install we silently shipped
  on Windows v1.1.1)
- torch / opencv version drift between pip resolution and what PyInstaller
  later finds on disk
- Missing native deps (libomp on Mac, MSVC runtime on Windows)

For each of a curated list of "load-bearing" packages, exercise an API
call that touches the bundled native libs. If pip "succeeded" but the
package can't actually run, fail the test.

Total runtime: ~10 sec.
"""
from __future__ import annotations
import importlib
import sys
import pytest


# (import_name, optional minimal API call to exercise native deps)
# The API call is what would catch a partial-install where import succeeds
# but later usage segfaults / dlopens fail.
NATIVE_DEPS = [
    ("numpy",        "import numpy as np; np.zeros(3)"),
    ("pandas",       "import pandas as pd; pd.DataFrame({'a':[1]})"),
    ("torch",        "import torch; torch.zeros(3)"),
    ("torchvision",  "import torchvision"),
    ("torchaudio",   "import torchaudio"),
    ("cv2",          "import cv2; cv2.__version__"),
    ("av",           "import av; av.open"),  # touches libavcodec/libavformat dlopen
    ("onnxruntime",  "import onnxruntime; onnxruntime.InferenceSession"),
    ("xgboost",      "import xgboost as xgb; xgb.DMatrix"),
    ("sklearn",      "import sklearn; sklearn.set_config"),
    ("imblearn",     "from imblearn.over_sampling import SMOTE"),
    ("pyfaceau",     "import pyfaceau; pyfaceau.__version__"),
    ("pyclnf",       "import pyclnf; pyclnf.__version__"),
    ("pymtcnn",      "import pymtcnn; pymtcnn.__version__"),
    ("pyfhog",       "import pyfhog; pyfhog.__version__"),
    ("PyQt5",        "from PyQt5 import QtCore"),
    ("PyQt5.QtMultimedia", "from PyQt5 import QtMultimedia"),  # gotcha: PyQt5-multimedia isn't a thing
    ("librosa",      "import librosa"),
    ("soundfile",    "import soundfile"),
    ("faster_whisper", "import faster_whisper"),
    ("ffmpeg",       "import ffmpeg"),  # ffmpeg-python wrapper
]


@pytest.mark.parametrize("pkg,api_call", NATIVE_DEPS,
                         ids=[name for name, _ in NATIVE_DEPS])
def test_dep_imports_and_minimal_api(pkg, api_call, python_executable):
    """For each load-bearing package, attempt `import` + a minimal API
    touch in a subprocess (so a segfault in one package doesn't crash the
    pytest session).
    """
    import subprocess
    result = subprocess.run(
        [python_executable, "-c", api_call],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        msg_lines = [
            f"{pkg}: import or minimal API call failed (returncode={result.returncode})",
            f"  Cmd: {api_call!r}",
            f"  STDERR (last 1KB):",
        ]
        msg_lines.extend("    " + line for line in result.stderr[-1024:].splitlines())
        pytest.fail("\n".join(msg_lines))


@pytest.mark.parametrize("pkg,api_call", NATIVE_DEPS,
                         ids=[name for name, _ in NATIVE_DEPS])
def test_dep_importable_via_pythonpath(pkg, api_call):
    """Same idea but in-process — a faster check for the import alone.
    Subprocess test above is the authoritative one (catches segfaults);
    this one catches silly typos in package names without subprocess overhead.
    """
    # Allow dotted names; for in-process import only the first segment matters
    top = pkg.split(".")[0]
    try:
        importlib.import_module(top)
    except ImportError as e:
        pytest.fail(f"{pkg}: cannot import in-process — {e}")
