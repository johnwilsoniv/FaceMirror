# Release Preflight Tests

Catch build-and-release issues **before** they show up mid-PyInstaller or,
worse, in a shipped `.dmg` someone is downloading.

Built in response to the v1.1.1 Mac release effort, where we ate the cost
of these issues in real time:

| What broke | When discovered | Layer that would have caught it |
|---|---|---|
| `bin/ffmpeg` not in git → worktree didn't bundle it | S1 build emitted warning at minute ~10 | **Layer 1** — required local resources |
| `PyQt5-multimedia` not on PyPI | `pip install -r requirements.txt` | **Layer 1** — pip dry-run |
| PyAV `libavcodec.61.19.101` missing dylib | S2 PyInstaller analysis crash | **Layer 2** — native API call |
| Stale `torch/_C/_acc/__init__.pyi` cached path | S1 BUNDLE step (minute ~10 of build) | **Layer 3** (post-build .app validation, not yet implemented) |
| Missing `tokenizers` / `hypothesis` build deps | PyInstaller analysis crash | **Layer 2** — import smoke |
| `OnlineAUCorrection` state leak after 40 videos | only when running 111-patient batch | **Layer 4** — long-batch state-leak test (not yet implemented) |

## Layer 1 — Pre-build environment (`test_build_env.py`)

Runs **before** PyInstaller. Verifies:

- Required local binaries (`ffmpeg`, `ffprobe`) and weight/model trees are
  on disk at the paths each `.spec` expects
- Each stage's `requirements.txt` actually resolves on PyPI
  (catches typos, removed packages, etc.) via `pip install --dry-run`
- Each `.spec` references resource paths that exist on disk
- Each `.spec` defines a semver-shaped `app_version`

Runtime: ~30 sec.

## Layer 2 — Post-install dependency smoke (`test_install_health.py`)

Runs **after** `pip install`, **before** PyInstaller. For each
load-bearing package, exercises a minimal API call in a subprocess so a
segfault in one package doesn't take the whole test session down. Catches:

- Partial installs where `import pkg` succeeds but the bundled native
  library is missing or version-mismatched (PyAV libavcodec)
- Missing transitive native deps (libomp on Mac, MSVC runtime on Windows)

Runtime: ~10 sec.

## Layer 3 — Post-build `.app` smoke (planned, not yet implemented)

Runs **after** PyInstaller succeeds. Verifies:

- `Contents/MacOS/<binary>` exists and is executable
- `Contents/Info.plist` parses + has correct `CFBundleVersion`
- `Contents/Frameworks/` has the expected major dylibs (`libtorch`, `libav*`)
- The app launches with `--help` (or equivalent non-interactive flag) and
  exits 0 within 30s
- `otool -L` on the bundled `.so` files resolves all dylibs (no
  `Library not loaded` errors)

## Layer 4 — Long-batch state-leak regression (planned)

Runs the production pipeline through a synthetic 50+ video batch in one
process to catch state-leak bugs of the
`OnlineAUCorrection`-saturation-after-40-videos class. Asserts that the
50th video's AU outputs match the 1st video's outputs to within FP
tolerance. Would have caught the bug that broke 65 of 111 patients in the
Windows v1.1.1 batch.

## Run

```bash
# All layers (Layer 3+4 will be added — currently 1+2 only)
pytest tests/release_preflight/ -v

# Just one layer
pytest tests/release_preflight/test_build_env.py -v
pytest tests/release_preflight/test_install_health.py -v

# Single test
pytest tests/release_preflight/test_build_env.py::test_required_local_resource_present -v
```

## When to run

- **Manually before any release tag push**: `make preflight-release` (target
  to be added) gates `gh release create`.
- **Automatically in CI** on any commit that touches a `.spec`, a
  `requirements*.txt`, or anything under `S1_FaceMirror/`,
  `S2 Action Coder/`, `S3 Data Analysis/`. Workflow:
  `.github/workflows/release-preflight.yml` (to be added).
- **Locally before kicking off a long PyInstaller run**:
  `pytest tests/release_preflight/test_build_env.py -v`
  takes 30 sec and saves you from waiting 15 minutes only to find the spec
  references a missing file.
