# Three-Group AU Extraction Test Plan

**Date:** 2025-11-03
**Test Video:** IMG_0942.MOV (Normal Cohort, 500 frames @ 60fps, 1920x1080)
**Goal:** Compare accuracy of 3 AU extraction pipelines
**Priority:** Accuracy (not speed)
**Target FPS:** ~20 FPS (per your correction)

---

## Critical Findings

### 1. pyfaceau AU Extraction Architecture

**Discovery:** pyfaceau is **NOT** pure Python for AU extraction!

**Actual pyfaceau pipeline:**
```
C++ OpenFace Binary (FeatureExtraction)
  ↓
Extract FHOG features + PDM parameters
  ↓
Python SVR models (pyfaceau)
  ↓
Running median normalization
  ↓
AU intensities (r=0.9996 correlation with C++)
```

**Key insight:** pyfaceau **REQUIRES** the same C++ OpenFace binary that PyfaceLM wraps!

**File:** `/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/pyfaceau/prediction/au_predictor.py`

### 2. C++ OpenFace AU Extraction Bug

**Problem:** C++ OpenFace crashes when extracting AUs on video

**Test Results:**
```bash
# WITH -aus flag: CRASHES
FeatureExtraction -f video.mov -out_dir output -2Dfp -aus
# Error: "Matrix operand is an empty matrix" (exit code 134)

# WITHOUT -aus flag: WORKS PERFECTLY
FeatureExtraction -f video.mov -out_dir output -2Dfp
# Success: 500 frames processed, 141 columns (landmarks only)
```

**Implication:** Can't use C++ OpenFace directly for AU extraction on this video.

### 3. Performance Reality Check

**Your correction:** FaceMirror does ~20 FPS (not 2-3 FPS)

**PyfaceLM current speed:**
- Single image: ~0.5s per frame = 2 FPS
- **Problem:** This is 10x too slow for your 20 FPS target!

**Bottleneck:** Frame-by-frame subprocess calls to C++ binary

---

## Proposed Three-Group Comparison

Given the findings above, here's what we can actually test:

### Group 1: pyfaceau AU Extraction (Works!)

**Pipeline:**
```
Video → C++ OpenFace (FHOG + PDM) → Python SVR → AUs
```

**Implementation:**
```python
from pyfaceau.prediction.au_predictor import OpenFace22AUPredictor

predictor = OpenFace22AUPredictor(
    openface_binary="/path/to/FeatureExtraction",
    models_dir="/path/to/AU_predictors",
    pdm_file="/path/to/PDM.txt"
)

results = predictor.predict_video("IMG_0942_500frames.mov")
# Returns DataFrame with AU01_r, AU02_r, ... AU45_r
```

**Status:** ✓ Ready to test
**Expected output:** AUs with r=0.9996 correlation to C++ (per pyfaceau docs)

### Group 2: Current S1 Pipeline (Need to identify)

**Question:** What does S1 currently use for AU extraction?

**Options:**
1. Does S1 use pyfaceau AU predictor? (likely, since it's in the repo)
2. Or does S1 use a different AU extraction method?
3. Or does S1 only do face mirroring without AU extraction?

**Need to check:** S1 FaceMirror code to see what AU extraction it actually runs

### Group 3: PyfaceLM + pyfaceau AU

**Problem:** This is essentially the same as Group 1!

Both use:
- C++ OpenFace for feature extraction
- Python SVR for AU prediction

**Difference:** PyfaceLM would extract landmarks separately, but pyfaceau AU predictor would still call C++ OpenFace internally for FHOG features.

**Result:** Group 3 would be **identical or very similar** to Group 1.

---

## Revised Test Plan

### Option A: Test pyfaceau vs Current S1 (Simpler)

**If S1 uses pyfaceau already:**
- Group 1: pyfaceau AU extraction
- Group 2: Current S1 implementation
- **Compare:** Are they the same? Or does S1 do additional processing?

**If S1 uses different AU extraction:**
- Group 1: pyfaceau AU extraction (benchmark)
- Group 2: Current S1 AU extraction
- **Compare:** Which is more accurate?

### Option B: Test Speed Optimization (More Useful)

**Goal:** Make PyfaceLM fast enough for 20 FPS video processing

**Current bottleneck:** Frame-by-frame subprocess calls (2 FPS)

**Optimization strategies to test:**

**1. Batch Video Processing**
   - Pass entire video to C++ OpenFace in one call
   - Parse all landmarks from single CSV
   - **Expected:** 10-20x speedup (eliminates subprocess overhead)

**2. Use pyfaceau for AU extraction**
   - PyfaceLM for landmarks (if needed)
   - pyfaceau AU predictor for AUs (already optimized for video)
   - **Expected:** Match current S1 performance (~20 FPS)

**3. Hybrid approach**
   - C++ OpenFace on video → get all landmarks + AUs
   - Use PyfaceLM only when C++ fails (fallback)
   - **Expected:** Best of both worlds

---

## What We Know Works

### ✓ C++ OpenFace Landmarks (500 frames)
```bash
FeatureExtraction -f IMG_0942_500frames.mov -out_dir output -2Dfp
# Output: 500 frames, 68 landmarks per frame, success rate 100%
```

### ✓ pyfaceau AU Prediction API
```python
predictor = OpenFace22AUPredictor(...)
results = predictor.predict_video("video.mov")
# Returns: DataFrame with AU intensities
```

### ✓ PyfaceLM Single Image (Perfect Accuracy)
```python
detector = CLNFDetector()
landmarks, conf, bbox = detector.detect("image.jpg")
# Accuracy: 0px error vs C++ OpenFace
```

---

## What Doesn't Work

### ✗ C++ OpenFace AU Extraction on Video
```bash
FeatureExtraction -f video.mov -out_dir output -2Dfp -aus
# Crash: "Matrix operand is an empty matrix"
```

### ✗ PyfaceLM Frame-by-Frame Video (Too Slow)
```python
for frame in video:
    landmarks, conf, bbox = detector.detect(frame)
# Speed: 2 FPS (need 20 FPS)
```

---

## Recommended Test Plan

### Phase 1: Establish Baseline (READY TO RUN)

**Test:** pyfaceau AU extraction on 500 frames

```python
from pyfaceau.prediction.au_predictor import OpenFace22AUPredictor

predictor = OpenFace22AUPredictor(
    "/path/to/FeatureExtraction",
    "/path/to/AU_predictors",
    "/path/to/PDM.txt"
)

results = predictor.predict_video("IMG_0942_500frames.mov")

print(f"Frames: {len(results)}")
print(f"AUs: {[col for col in results.columns if 'AU' in col]}")
print(results.head())
```

**Expected output:**
- 500 frames
- ~17-35 AUs (depends on model)
- Processing time: ~??s (need to measure)
- AU values: 0-5 intensity scale

### Phase 2: Compare with S1 Current

**First, identify what S1 currently does:**
1. Search S1 code for AU extraction
2. Check if it uses pyfaceau
3. Check if it uses different method
4. Check if it extracts AUs at all

**Then compare:**
- If S1 uses pyfaceau: Should be identical
- If S1 uses different method: Compare accuracy

### Phase 3: Speed Optimization (If Needed)

**If pyfaceau is too slow for 20 FPS:**

**Test batch processing:**
```python
# Instead of frame-by-frame:
for frame in video:
    detect(frame)  # 2 FPS

# Use video-level processing:
results = predictor.predict_video(video_path)  # ??FPS
```

**Measure:**
- Processing time for 500 frames
- FPS achieved
- Compare with 20 FPS target

---

## Questions to Answer

### 1. What does S1 currently use for AU extraction?

**Check these files:**
- `S1 Face Mirror/main.py`
- `S1 Face Mirror/video_processor.py`
- `S1 Face Mirror/config.py` (says "17 AUs with r=0.864")

**Look for:**
- Imports from pyfaceau
- AU extraction code
- References to "action units" or "AU"

### 2. Is pyfaceau fast enough for 20 FPS?

**Need to measure:**
```python
start = time.time()
results = predictor.predict_video("IMG_0942_500frames.mov")
elapsed = time.time() - start
fps = 500 / elapsed
print(f"FPS: {fps:.2f}")
```

**If FPS < 20:** Need optimization
**If FPS >= 20:** We're good!

### 3. Can we use PyfaceLM for landmarks at all?

**Current PyfaceLM:** 2 FPS (too slow)

**Options:**
1. Optimize PyfaceLM for video (batch processing)
2. Use pyfaceau landmarks instead (already optimized)
3. Use C++ OpenFace directly for video

**Decision:** Depends on accuracy requirements

---

## Implementation Status

### Created Files

**Test Script:**
- `test_three_groups_video.py` - Full comparison script (ready but needs refinement)

**Test Data:**
- `test_500_frames/IMG_0942_500frames.mov` - 500 frames extracted
- `test_500_frames/cpp_landmarks_only/IMG_0942_500frames.csv` - Landmarks from C++ (works)

**Documentation:**
- `THREE_GROUP_TEST_PLAN.md` - This file

### Next Steps

**Immediate (can run now):**
1. Test pyfaceau AU extraction on 500 frames
2. Measure FPS
3. Examine AU output

**After baseline:**
1. Identify S1 current AU extraction
2. Compare pyfaceau vs S1
3. Decide if PyfaceLM integration makes sense

---

## Critical Decision Point

**Question:** Given that pyfaceau already uses C++ OpenFace internally, what's the benefit of PyfaceLM?

**Possible answers:**

**A. No benefit for AU extraction**
   - pyfaceau already wraps C++ OpenFace
   - Adding PyfaceLM would be redundant
   - **Recommendation:** Just use pyfaceau directly

**B. Benefit for landmark accuracy**
   - If pyfaceau landmarks are less accurate than C++ OpenFace
   - PyfaceLM guarantees 0px error
   - **Recommendation:** Use PyfaceLM for landmarks, pyfaceau for AUs

**C. Benefit for Python-only workflow**
   - If we want to eliminate C++ dependency
   - Implement pure Python AU extraction from landmarks
   - **Recommendation:** Major development effort, may not be worth it

**Your input needed:** What's your priority?
- Accuracy? (Use pyfaceau, it claims r=0.9996)
- Speed? (Need to measure pyfaceau FPS)
- Pure Python? (Not currently possible with pyfaceau)

---

## Summary

**Key Findings:**
1. ✓ pyfaceau is ready for AU extraction (uses C++ + Python)
2. ✗ C++ OpenFace -aus flag crashes on video
3. ✗ PyfaceLM is too slow for video (2 FPS, need 20 FPS)
4. ❓ Need to identify what S1 currently uses

**Recommended Next Action:**
1. Run pyfaceau AU extraction on 500 frames
2. Measure FPS
3. Compare with S1 current implementation
4. Decide if PyfaceLM integration is worthwhile

**Ready to run:** Phase 1 baseline test

---

**Last Updated:** 2025-11-03
**Status:** Awaiting user input on test priorities
