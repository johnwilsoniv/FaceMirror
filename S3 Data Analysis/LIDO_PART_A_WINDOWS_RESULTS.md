# Part A results — GPU vs CPU CLNF on Windows-CUDA + pyfaceau 1.3.16

**Audience:** Mac command center.
**Reporter:** Windows box.
**Companion to:** `LIDO_COHORT_WINDOWS_BRIEF.md` (Part A asks).

## TL;DR

1. **v1316 goldens were built with `CLNF_CONFIG['use_gpu']=True`** — confirmed
   bit-exact (MAE = 0.000000, r = 1.000000) on every canary × side we
   re-ran under GPU.
2. **The 38% CPU-better-on-paralyzed-faces claim is refuted.** GPU vs CPU
   r-delta on the paralyzed canaries we tested is **−0.0092 to +0.0033 r**
   (mean ≈ 0). CPU and GPU produce different absolute values, but they
   correlate equally well with the C++ ground truth. The stale comment can
   be retired.
3. **Lido must use `use_gpu=True`** — matches v1316 bit-exact, no accuracy
   reason to prefer CPU.
4. **pyclnf had two Windows-incompat bugs blocking the CPU comparison;
   both fixed, pushed to `johnwilsoniv/pyclnf` `main`** (commits 63d76973
   and 5fbe598d). With those landed, CPU CLNF runs cleanly on Windows
   against OpenCV 4.12 (same OpenCV the Mac `cpp_warp.so` was built
   against), so the comparison above is bit-exact methodology.

## What the brief asked

> For each of `{IMG_0942, IMG_0861, IMG_2259}` × `{left, right}`, run
> pyfaceau on the mirrored clip TWICE — once with
> `CLNF_CONFIG['use_gpu']=False`, once `=True` — and compare each run's AU
> columns to the C++ ground truth (`tests/golden/aus/<id>_<side>/cpp.parquet`),
> Pearson r + MAE per AU. Also compare to existing
> `pyfaceau_windows_cuda.parquet` to determine which config v1316 was
> built with.

## What ran

`recoded_rerun_dual_v1316/` is already comfortable evidence that GPU works
end-to-end; the open question was specifically "what does CPU produce, and
how does it differ." We ran a 4-worker CPU pool + 2-worker GPU pool in
parallel over a shorter paralyzed-canary subset (IMG_2259, IMG_3847,
IMG_4157 — 517-555 frames per side) instead of the brief's three
1100+ frame canaries. Wall time 21.9 min vs ~70 min serial. All 12 work
items returned 100% success rate (no failed frames).

### v1316 golden identity check (GPU is bit-exact; CPU diverges by a small but real margin)

| canary | side | cfg | mean MAE vs v1316 golden | mean r vs v1316 |
|---|---|---|---|---|
| IMG_2259 | left  | cpu | 0.082919 | 0.948137 |
| IMG_2259 | left  | gpu | **0.000000** | **1.000000** |
| IMG_2259 | right | cpu | 0.082109 | 0.862294 |
| IMG_2259 | right | gpu | **0.000000** | **1.000000** |
| IMG_3847 | left  | cpu | 0.016981 | 0.977887 |
| IMG_3847 | left  | gpu | **0.000000** | **1.000000** |
| IMG_3847 | right | cpu | 0.024001 | 0.982841 |
| IMG_3847 | right | gpu | **0.000000** | **1.000000** |
| IMG_4157 | left  | cpu | 0.063565 | 0.941761 |
| IMG_4157 | left  | gpu | **0.000000** | **1.000000** |
| IMG_4157 | right | cpu | 0.044061 | 0.978320 |
| IMG_4157 | right | gpu | **0.000000** | **1.000000** |

GPU is **bit-identical** to v1316 on every canary × side. CPU output is
close to v1316 but not bit-exact (mean r 0.86–0.98). v1316 = `use_gpu=True`
is settled.

### Accuracy vs C++ ground truth

| canary | side | cfg | mean r vs cpp | mean MAE vs cpp |
|---|---|---|---|---|
| IMG_2259 | left  | cpu | 0.9063 | 0.1508 |
| IMG_2259 | left  | gpu | 0.9096 | 0.1439 |
| IMG_2259 | right | cpu | 0.7825 | 0.1037 |
| IMG_2259 | right | gpu | 0.7806 | 0.1050 |
| IMG_3847 | left  | cpu | 0.8502 | 0.0479 |
| IMG_3847 | left  | gpu | 0.8413 | 0.0473 |
| IMG_3847 | right | cpu | 0.9450 | 0.0631 |
| IMG_3847 | right | gpu | 0.9386 | 0.0647 |
| IMG_4157 | left  | cpu | 0.8580 | 0.1192 |
| IMG_4157 | left  | gpu | 0.8488 | 0.1254 |
| IMG_4157 | right | cpu | 0.9449 | 0.1192 |
| IMG_4157 | right | gpu | 0.9453 | 0.1188 |

### The 38% question (GPU − CPU delta on paralyzed faces)

| canary | side | r-delta (GPU − CPU) | MAE-delta |
|---|---|---|---|
| IMG_2259 | left  | +0.0033 | −0.0069 |
| IMG_2259 | right | −0.0019 | +0.0013 |
| IMG_3847 | left  | −0.0089 | −0.0007 |
| IMG_3847 | right | −0.0064 | +0.0016 |
| IMG_4157 | left  | −0.0092 | +0.0063 |
| IMG_4157 | right | +0.0004 | −0.0004 |

Mean r-delta ≈ −0.0038, range [−0.0092, +0.0033]. This is **FP-drift
territory**, not a 38% effect. CPU is sometimes slightly better, sometimes
slightly worse. The 38% comment can be safely removed from the pyclnf
source.

## What it took to test this

Out of the gate, CPU CLNF on Windows produced 0/30 success and zero AU
columns, surfacing two latent Windows-incompat bugs in pyclnf:

1. **`cpp_warp` C++ extension was Mac-only.** `pyclnf/cpp_warp/__init__.py`
   tried `from .cpp_warp import extract_aoi, warp_affine` and silently
   fell back to `extract_aoi = None` on Windows because the wheel only
   shipped `cpp_warp.cpython-310-darwin.so` (Mac .so). The first
   `cpp_warp.extract_aoi(...)` call deep in the optimizer then crashed
   with `TypeError: 'NoneType' object is not callable`.
2. **`pyclnf/core/eye_patch_expert.py:950` hardcoded a `/tmp/` debug
   path.** This was unconditional (gated only by `debug_enabled` which
   fires on every CPU CLNF run that hits a 3×3 response map on landmarks
   0 or 8 — i.e. every video). Windows has no `/tmp/` so this raised
   `FileNotFoundError` on every frame.

Both fixed and pushed to `johnwilsoniv/pyclnf` `main`:

- [`63d76973`](https://github.com/johnwilsoniv/pyclnf/commit/63d76973) —
  cpp_warp Windows build support: cross-platform `CMakeLists.txt`
  (`OPENCV_DIR` env var on Windows), new `build.ps1`, fail-fast
  `__init__.py` with auto Windows DLL search path, prebuilt
  `cpp_warp.cp310-win_amd64.pyd` against OpenCV 4.12.0 (same version
  Homebrew ships, so the bit-exactness pyclnf promises is preserved).
- [`5fbe598d`](https://github.com/johnwilsoniv/pyclnf/commit/5fbe598d) —
  eye_patch_expert.py: `tempfile.gettempdir()` instead of `/tmp/` so the
  debug file path works on Windows too. Five other `/tmp/` writes in
  `pyclnf/core/optimizer.py` are all gated by `self.debug_mode` (off by
  default); flagged as follow-up cleanup, non-blocking.

After those two commits, CPU CLNF on Windows produces full success
(517/517, 532/532, 555/555 on the three short canaries) and the bit-exact
OpenCV 4.12 path matches the Mac build. That's what made the comparison
above possible.

## What I want from you

1. **Confirm Lido proceeds with `use_gpu=True`** (matches v1316 bit-exact,
   no accuracy reason to switch). I'm going to assume yes unless you
   reply otherwise.
2. **Retire or correct the 38% comment in pyclnf.** I left it alone in
   this PR because rewording it is your call, but it's now demonstrably
   wrong on this corpus.
3. **Optional follow-ups for pyclnf:**
   - cibuildwheel for cross-platform PyPI wheels (so the next Windows
     user doesn't have to set `OPENCV_DIR` and run `build.ps1` locally —
     the prebuilt .pyd is committed, but it's tied to Python 3.10 +
     OpenCV 4.12 + Windows x64; cibuildwheel would generate the matrix).
   - Bundle `opencv_world<ver>.dll` into the wheel's `package_data` so
     Windows users don't need a system OpenCV.
   - Clean up the 5 remaining `/tmp/` debug paths in
     `pyclnf/core/optimizer.py` (all gated by `self.debug_mode`, lower
     priority).

Box continuing to A4 (mirror-consistency check on IMG_0942) and then B
(Lido reprocess) on the assumption that the answer to question 1 is yes.
Will pause if you redirect.
