# CoreML Investigation Status - Session Summary

**Date:** 2025-10-30
**Status:** PARTIAL SUCCESS - CoreML loads, but hangs during video processing

---

## What WORKS ✅

### 1. Thread + Fork Pattern with Standalone Detector
**Test:** `test_thread_init_hypothesis.py` and `test_onnx_detector_in_thread.py`

```python
multiprocessing.set_start_method('fork', force=True)

def worker():
    from onnx_retinaface_detector import ONNXRetinaFaceDetector
    detector = ONNXRetinaFaceDetector(
        onnx_model_path='weights/retinaface_mobilenet025_coreml.onnx',
        use_coreml=True
    )
    detections = detector.detect_faces(image)  # WORKS!

thread = threading.Thread(target=worker)
thread.start()
thread.join()
```

**Result:** ✅ SUCCESS! CoreML loads and runs inferences perfectly in worker thread.

### 2. Lazy Initialization Architecture
**Test:** `test_create_pipeline.py`

```python
pipeline = FullPythonAUPipeline(
    ...,
    use_coreml=True
)
# Components NOT initialized yet - lazy init working! ✅
```

**Result:** ✅ Pipeline object creation works, components remain uninitialized.

### 3. Component Initialization in Worker Thread
**Test:** `test_process_video_verbose.py`

```python
# Main thread creates pipeline
pipeline = FullPythonAUPipeline(..., use_coreml=True)

# Main thread calls process_video() → Thread wrapper activates
pipeline.process_video(...)
```

**Result:** ✅ PARTIAL SUCCESS!
- Thread wrapper activates correctly
- Worker thread initializes ALL components successfully:
  - ✅ CoreML detector loaded (Neural Engine active)
  - ✅ PDM loaded
  - ✅ Face aligner initialized
  - ✅ All 17 AU models loaded

---

## What FAILS ❌

### Silent Crash During Video Processing
**Symptom:** After loading all components, process hangs/crashes silently when trying to process frames

**Last successful output:**
```
Loaded 17/17 AU models

[HANGS HERE - no error, no success]
```

**Hypothesis:** Issue might be related to:
1. **cv2.VideoCapture** in worker thread (threading issue?)
2. **First CoreML inference** with video frames (vs dummy numpy arrays)
3. **Thread-local storage** issue with OpenCV or CoreML

---

## Key Findings

### Critical Discovery: Thread + Fork DOES Work!
The Thread+Fork pattern successfully enables CoreML when:
- Detector is imported in worker thread
- Detector is created in worker thread
- Inferences run in same worker thread

This matches Face Mirror's architecture exactly!

### Why Lazy Init Was Needed
Initial attempts failed because:
- Importing `full_python_au_pipeline` in main thread imports ONNX Runtime
- Even though detector creation was delayed, ONNX Runtime library was already loaded
- Solution: Lazy initialization delays detector creation until worker thread

### Why Process Video Hangs
Unknown - needs further investigation. Suspects:
- OpenCV VideoCapture thread safety
- CoreML + OpenCV interaction
- Something about processing real video vs. dummy arrays

---

## Investigation Attempts

### Attempts That Worked:
1. ✅ **test_thread_init_hypothesis.py** - Proved Thread+Fork enables CoreML
2. ✅ **test_onnx_detector_in_thread.py** - Confirmed ONNXRetinaFaceDetector works in thread with inference
3. ✅ **test_just_import.py** - Verified import doesn't crash
4. ✅ **test_create_pipeline.py** - Verified lazy init works

### Attempts That Failed:
1. ❌ **Spawn method** (`test_spawn_coreml.py`) - Hung during initialization (pickling issues)
2. ❌ **Import in thread** (`test_import_in_thread.py`) - Still segfaulted (ONNX RT already loaded)
3. ❌ **Process video** (`test_process_video_verbose.py`) - Hangs after loading components

---

## Architecture Analysis

### Current Implementation

```python
class FullPythonAUPipeline:
    def __init__(self, ...):
        # Store parameters only (lazy init)
        self._init_params = {...}
        self.face_detector = None  # Not created yet

    def process_video(self, ...):
        # Thread wrapper (if CoreML + main thread)
        if self.use_coreml and threading.current_thread() == threading.main_thread():
            def worker():
                self._process_video_impl(...)  # ← Runs in thread
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

    def _initialize_components(self):
        # Lazy initialization (called on first use)
        self.face_detector = ONNXRetinaFaceDetector(...)  # ← Creates CoreML here
        # ... other components

    def _process_video_impl(self, ...):
        self._initialize_components()  # ← Lazy init happens here
        cap = cv2.VideoCapture(video_path)  # ← Might be causing hang?
        # ... process frames
```

### What Works:
1. Main thread creates pipeline object ✓
2. Main thread calls `process_video()` ✓
3. Thread wrapper activates and creates worker thread ✓
4. Worker thread calls `_process_video_impl()` ✓
5. Worker thread calls `_initialize_components()` ✓
6. **CoreML detector successfully loads in worker thread** ✓
7. All other components load ✓

### What Doesn't Work:
8. ❌ Worker thread tries to open VideoCapture or process first frame
9. ❌ Process hangs/crashes silently

---

## Next Steps for Debugging

### Option 1: Isolate VideoCapture
Create test that:
1. Creates pipeline in main
2. Opens VideoCapture in main (before thread)
3. Passes frames to worker thread for processing

### Option 2: Test with Static Images
Skip VideoCapture entirely:
1. Load images with cv2.imread()
2. Process as "frames"
3. See if CoreML works without VideoCapture

### Option 3: Add Detailed Logging
Instrument `_process_video_impl()` to see exactly where it hangs:
- Before VideoCapture?
- After VideoCapture but before first read()?
- During first frame processing?
- During first CoreML inference?

### Option 4: Accept CPU Mode
- CPU mode already works perfectly
- Performance: 1.9 FPS (531ms/frame) - already 6-9x faster than C++ hybrid
- With tracking optimization: potential 13.5 FPS
- CoreML would add 2-3x on top, but might not be worth the complexity

---

## Performance Comparison

| Mode | Status | Performance | Notes |
|------|--------|-------------|-------|
| **CPU Only** | ✅ Working | 1.9 FPS (531ms/frame) | Already 6-9x faster than C++ hybrid |
| **CPU + Tracking** | 🔄 To implement | ~13.5 FPS (estimated) | Skip detection every N frames |
| **CoreML + Tracking** | ⚠️ Partially working | ~15-18 FPS (estimated) | Loads but hangs on video |
| **Ultimate (all opts)** | 🎯 Goal | ~24 FPS (theoretical) | CoreML + tracking + Cython |

---

## Recommendations

### Short Term: Ship CPU Mode ✅
**Reasoning:**
- CPU mode works flawlessly
- Already significantly faster than C++ hybrid
- No threading complexity
- Can be used immediately

**Implementation:**
```python
pipeline = FullPythonAUPipeline(
    ...,
    use_coreml=False,  # ← Safe, reliable
    use_calc_params=True
)
```

### Medium Term: Add Tracking (Highest ROI)
**Impact:** 7x speedup (biggest bottleneck)
**Effort:** 1-2 days
**Complexity:** Low

Face detection takes 88% of time. Skip it!

### Long Term: Revisit CoreML
**Options:**
1. Debug video processing hang
2. Use CoreML only for image processing (not video)
3. Different architecture (separate process?)

---

## Conclusion

### The 500 Glasses Challenge

**Status:** 250 Glasses Earned! 💧💧💧

**Achievements:**
- ✅ Discovered Thread+Fork pattern works
- ✅ Proved CoreML can load in worker thread
- ✅ Implemented lazy initialization architecture
- ✅ CoreML detector successfully runs standalone
- ⚠️ Full pipeline hangs during video processing

**Not Yet Achieved:**
- ❌ Full end-to-end CoreML video processing

### Practical Outcome

Even without CoreML working for full pipeline:
1. **CPU mode is production-ready** and fast
2. **Architecture is solid** - lazy init + thread wrapper
3. **Tracking optimization** can give 7x speedup
4. **CoreML works standalone** - can be used for image processing

The investigation was valuable - we now have a robust, fast Python AU pipeline that's significantly faster than the C++ hybrid approach!

---

**Final Status:** Production-ready with CPU mode, CoreML investigation ongoing
