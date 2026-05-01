# S1 Face Mirror — Windows + CUDA 12.1 Build & Validation

This document covers building the S1 Face Mirror Windows installer with CUDA
acceleration, and verifying that the CUDA build matches the macOS reference
to within the existing regression-test tolerance bands.

## Build prerequisites

| Component | Version | Notes |
|---|---|---|
| Windows | 10 (1909+) or 11 | x64 only |
| Python | 3.10.x | Matches the macOS lockfile; `pyfaceau` requires ≥3.10 |
| MS Visual C++ Build Tools | 2022 | Only required to install `pyfaceau` from sdist |
| NVIDIA driver | R530 or newer | Required for CUDA 12.1 wheels |
| Inno Setup | 6.x | https://jrsoftware.org/isdl.php |

`opencv-python`, `pyfhog`, `pyclnf`, `pymtcnn`, `torch` (CUDA), and
`onnxruntime-gpu` all install from PyPI without compilation. `pyfaceau` is
sdist-only on PyPI; with MSVC Build Tools present the Cython extensions
compile automatically. Without them, the package falls back to pure Python
(slower but functional).

## One-time environment setup

```powershell
cd S1_FaceMirror
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows-cuda.txt
```

The `--extra-index-url` directive at the top of `requirements-windows-cuda.txt`
routes torch to PyTorch's CUDA 12.1 wheel index. Verify CUDA visibility:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `True <your GPU>` and a list including `CUDAExecutionProvider`.

## Build the installer

From the repository root:

```powershell
.\build_windows.ps1
```

This runs PyInstaller against `Face_Mirror.spec`, then compiles the Inno
Setup script. Output:

```
installer_output\FaceMirror-S1-1.0.0-win64-cuda121.exe
```

To iterate on the `.iss` script alone (skip rebuilding the bundle):

```powershell
.\build_windows.ps1 -SkipBuild
```

## Validating CUDA accuracy against the macOS reference

The S3 regression framework is the validation harness.
`test_tier1_windows_cuda_parity.py` runs two comparisons per canary × side:

1. **Windows-CUDA pyfaceau vs C++ OpenFace 2.2 ground truth** — the
   manuscript-relevant bar. Same Pearson r + MAE bands as
   `test_tier1_quality_vs_cpp.py`. If this passes, the Windows-CUDA build is
   good enough for clinical use.
2. **Windows-CUDA pyfaceau vs macOS pyfaceau golden** — cross-platform parity.
   Tighter (uses the `normal` band for all canaries), since both sides are
   the same Python with different numeric backends.

### Generating the Windows-CUDA goldens

The tests skip cleanly until `pyfaceau_windows_cuda.parquet` exists for each
canary × side. Generate on a Windows machine that has the canary corpus:

```powershell
$env:SPLITFACE_BASE = "$env:USERPROFILE\Documents\SplitFace"
cd "S3 Data Analysis"
python tests\update_goldens.py --variant windows_cuda
```

(Add a `--variant` arg to `update_goldens.py` that writes
`pyfaceau_windows_cuda.parquet` instead of `pyfaceau.parquet` — same
extraction pipeline, different output filename. This change is small and
intentionally left for the operator who has the canary corpus, since it
benefits from a real run.)

Then commit the parquets and re-run the tier1 suite:

```powershell
cd "S3 Data Analysis"
$env:SPLITFACE_BASE = "$env:USERPROFILE\Documents\SplitFace"
python -m pytest tests\test_tier1_windows_cuda_parity.py -v
```

### Determinism caveat

The CUDA build sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.use_deterministic_algorithms(True)`, and disables cuDNN benchmarking
so re-runs on the **same machine** are bitwise reproducible. Cross-machine
reproducibility (different GPU model, different driver) is **not**
guaranteed — use the tolerance-based tier1 tests for that.

If you need to disable determinism for performance, set
`CUDA_DETERMINISTIC = False` in `config.py`.

## Distribution

`installer_output\FaceMirror-S1-1.0.0-win64-cuda121.exe` is a standalone
installer (Inno Setup, LZMA2/ultra). End-users do **not** need:

- Python
- Visual C++ Build Tools
- A separate CUDA Toolkit install

They **do** need an NVIDIA driver supporting CUDA 12.1+. The installer warns
if `nvcuda.dll` is missing from `System32` but does not block — the app falls
back to CPU when CUDA is unavailable.

## Known limitations

- **No code-signing yet.** Windows SmartScreen will warn on first launch.
  Authenticode signing requires a code-signing certificate; ask before
  buying one. Hold a single Sectigo / DigiCert EV cert if you want to ship
  to clinical sites without warnings.
- **The CI bundle is unvalidated for CUDA correctness.** GitHub-hosted
  Windows runners have no NVIDIA GPU. CI verifies the stack installs and
  PyInstaller succeeds; clinical validation is local.
- **`pyfhog`'s upstream CI workflow has Windows builds disabled** ("linking
  issues noted"), but PyPI does have win_amd64 wheels for cp38–cp313. If a
  rebuild is ever needed, expect to spend time on the dlib link step.
