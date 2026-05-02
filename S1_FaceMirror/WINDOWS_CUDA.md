# S1 Face Mirror — Windows + CUDA 12.8 Build & Validation

This document covers building the S1 Face Mirror Windows installer with CUDA
acceleration, and verifying that the CUDA build matches the macOS reference
to within the existing regression-test tolerance bands.

## Build prerequisites

| Component | Version | Notes |
|---|---|---|
| Windows | 10 (1909+) or 11 | x64 only |
| Python | 3.10.x | Matches the macOS lockfile; `pyfaceau` requires ≥3.10 |
| MS Visual C++ Build Tools | 2022 | **Required** — `pyfaceau` is sdist-only and its Cython extensions need MSVC at install time |
| NVIDIA driver | R570 or newer | Required for CUDA 12.8 wheels (Blackwell needs R570+) |
| Inno Setup | 6.x | https://jrsoftware.org/isdl.php — installer-build only, not runtime |

Install MSVC Build Tools 2022 with the C++ workload via winget:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override `
  "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools `
   --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
   --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
```

Wheel availability on PyPI for the FaceMirror dependency stack:
- `torch` (cu128 wheels), `onnxruntime-gpu`, `pyclnf`, `pymtcnn`, `opencv-python` — wheels available, no compile.
- `pyfhog` — version `0.1.4` is **sdist-only** and its Windows source build hits a dlib symbol-resolution error (`DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0`). The requirements file pins `<0.1.4` so pip picks `0.1.3` which has a `cp310-win_amd64` wheel.
- `pyfaceau` — sdist-only, builds from source via MSVC. The "falls back to pure Python without MSVC" pattern in the upstream README does NOT apply: setuptools errors out at the find-compiler step before any fallback can run, so MSVC is genuinely required.

## One-time environment setup

```powershell
cd S1_FaceMirror
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows-cuda.txt

# IMPORTANT: pyfaceau pulls in CPU `onnxruntime` which shadows the
# `onnxruntime-gpu` wheel (same Python module name; whichever is installed
# last wins). Remove the CPU one and force-reinstall the GPU one:
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu
```

The `--extra-index-url` directive at the top of `requirements-windows-cuda.txt`
routes torch to PyTorch's CUDA 12.8 wheel index. Verify CUDA visibility:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); cap=torch.cuda.get_device_capability(0); print(f'compute capability sm_{cap[0]}{cap[1]}')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `True <your GPU>` with a non-trivial compute capability
(`sm_75`/`sm_86`/`sm_89`/`sm_120`...) and a providers list **starting with**
`CUDAExecutionProvider` (not just CPU). If `CUDAExecutionProvider` is missing,
the onnxruntime-shadow workaround above wasn't applied.

## Build the installer

From the repository root:

```powershell
.\build_windows.ps1
```

This runs PyInstaller against `Face_Mirror.spec`, then compiles the Inno
Setup script. Output:

```
installer_output\FaceMirror-S1-1.0.0-win64-cuda128.exe
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

### Why this matters: the bundled-weights gotcha

`pyfaceau` ships a stripped-down weights bundle on PyPI: only **13 of the 17
OpenFace 2.2 AUs** have SVR `.dat` files in `~/.pyfaceau/weights/`. Run
`pyfaceau.processor.OpenFaceProcessor()` with no `weights_dir` argument and
the AU output is missing AU05/AU09/AU14/AU20.

S1 ships its own `S1_FaceMirror/weights/` bundle that includes the full
17-AU OpenFace 2.2 set. Both `S1_FaceMirror/openface_integration.py`'s
`OpenFace3Processor` (used by the running app) and the
`stage_windows_cuda_aus` golden-generation stage point at this bundle
explicitly. **Don't change either to use the auto-downloaded weights** —
column schema parity with the macOS goldens depends on it.

### Generating the Windows-CUDA goldens

The tests skip cleanly until `pyfaceau_windows_cuda.parquet` exists for each
canary × side. The `windows_cuda_aus` stage in `update_goldens.py` runs
pyfaceau **live** on each canary video using the Windows-CUDA stack
(`onnxruntime-gpu` `CUDAExecutionProvider` + `pyclnf` `use_gpu=True`) and
writes the per-frame AU output as a parquet golden:

```powershell
$env:SPLITFACE_BASE = "$env:USERPROFILE\Documents\SplitFace"
cd "S3 Data Analysis"
python tests\update_goldens.py --stage windows_cuda_aus --reason "fresh CUDA install"
```

The stage takes ~20-60 seconds per video × 20 videos (10 canaries × 2 sides),
so plan for 5-15 minutes total depending on video length. First-run cold
start adds another ~30s while CUDA JITs cuDNN/cuBLAS kernels for the GPU's
compute capability (especially noticeable on Blackwell — sm_120 wasn't in
older cu121 wheels).

If `SPLITFACE_BASE` points at a cloud-synced location (iCloud Drive,
OneDrive, etc.), check that the canary mp4s are fully materialized before
running — OpenCV's ffmpeg backend has a 60-second read timeout that's
shorter than typical on-demand-sync download times for 16+ MB videos. The
safest pattern is to copy the canary subdirs to a non-synced local path
first (e.g. `~\Documents\SplitFace\`) and point `SPLITFACE_BASE` there.

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

`installer_output\FaceMirror-S1-1.0.0-win64-cuda128.exe` is a standalone
installer (Inno Setup, LZMA2/ultra). End-users do **not** need:

- Python
- Visual C++ Build Tools
- A separate CUDA Toolkit install

They **do** need an NVIDIA driver supporting CUDA 12.8+. The installer warns
if `nvcuda.dll` is missing from `System32` but does not block — the app falls
back to CPU when CUDA is unavailable.

## Known correlation drift (under investigation)

A first end-to-end run of `windows_cuda_aus` on a Blackwell sm_120 box
(2026-05-01) reproduced the **17-AU schema** correctly but Pearson
correlation against the macOS pyfaceau goldens splits cleanly into two
buckets per canary (numbers from IMG_0422_left, n=1113 frames):

| Bucket | AUs | Pearson r |
|---|---|---|
| Healthy | AU01, AU02, AU05, AU06, AU12, AU45 | 0.84 - 0.96 |
| Drifted | AU04, AU07, AU09, AU14, AU15, AU17, AU20, AU23, AU25, AU26 | -0.45 to 0.75 |

The drifted bucket includes some AUs with **negative** correlation (AU04,
AU09, AU14, AU23) — definitively beyond numerical-backend drift. Likely
suspects (none confirmed):

- `pyclnf` GPU-vs-CPU/Metal landmark refinement disagreement (CPU pyclnf is
  separately broken on Windows: every frame returns `success=False`, so we
  cannot directly bisect)
- ONNX Runtime backend differences (CoreML on Apple Silicon vs CUDA on
  Blackwell) propagating through to AU SVR feature inputs
- `pyfaceau` version drift between the macOS S2O CSV generation date and
  current PyPI 1.3.11

The schema and 6/13 healthy AUs already let the manuscript-relevant
`test_windows_cuda_vs_cpp_ground_truth` pass for those AUs. The drifted
bucket needs a follow-up investigation that can A/B between Apple-Silicon
Mac CPU and Windows CUDA on the same canary corpus.

## Known limitations

- **No code-signing yet.** Windows SmartScreen will warn on first launch.
  Authenticode signing requires a code-signing certificate; ask before
  buying one. Hold a single Sectigo / DigiCert EV cert if you want to ship
  to clinical sites without warnings.
- **The CI bundle is unvalidated for CUDA correctness.** GitHub-hosted
  Windows runners have no NVIDIA GPU. CI verifies the stack installs and
  PyInstaller succeeds; clinical validation is local.
- **`pyfhog`'s upstream CI workflow has Windows builds disabled** ("linking
  issues noted"). PyPI has cp310-win_amd64 wheels for `pyfhog` versions
  `0.1.0`-`0.1.3` only; `0.1.4` is sdist-only and currently fails to link on
  Windows with `DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0` /
  `USER_ERROR__inconsistent_build_configuration` unresolved external symbols.
  The requirements file pins `<0.1.4` to use the wheel; lift the pin once
  upstream publishes a fixed wheel for `0.1.5+`. Tier1 tolerance bands cover
  the slight version drift vs the macOS-reference git SHA (which uses dlib
  built natively for macOS where the symbol resolution rules differ).
- **iCloud Drive on Windows is NOT a viable canary corpus location.** The
  Windows iCloud client's "files-on-demand" mode hangs OpenCV's ffmpeg
  reader on first access (60s timeout < typical iCloud download time for a
  16 MB video). Copy the canary subdirs to a local non-cloud path before
  running `windows_cuda_aus`. The `fetch_canaries.ps1` helper script in the
  repo root pulls them from a Mac on the same LAN over SMB.
