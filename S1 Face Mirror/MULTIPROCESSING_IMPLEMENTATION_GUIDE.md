# Multiprocessing Implementation Guide

**Date:** 2025-10-17
**Status:** Ready to implement
**Estimated Time:** 4-6 hours for basic implementation

---

## 🎯 Goal

Replace ThreadPoolExecutor with multiprocessing.Pool to achieve true parallel execution with CoreML.

**Expected Performance:**
- Current (threads): 9.5-11 fps (CoreML serializes across threads)
- Target (processes): 40-50+ fps (each process gets isolated CoreML session)

---

## 📊 Why Multiprocessing?

### Test Results Summary:
```
CPU-only (threads):
- Per-frame: 555ms (too slow)
- 6 threads @ 600% CPU: 11 fps
- ❌ CPU is 5x slower than CoreML Neural Engine

CoreML (threads):
- Per-frame: 105ms (fast)
- 6 threads @ serialized: 9.5 fps
- ❌ CoreML serializes inference (no true parallelism)

CoreML (processes) - TARGET:
- Per-frame: 105ms (fast)
- 6 processes @ 75% efficiency: ~43 fps
- ✅ Each process gets isolated CoreML session
```

---

## 🏗️ Architecture

### Current (Threading):
```python
Main Thread
├── Read batch → memory
├── ThreadPoolExecutor (6 workers)
│   ├── Thread 1: process_frame() → CoreML (SERIALIZED)
│   ├── Thread 2: process_frame() → CoreML (WAITING)
│   └── ... (all threads share ONE CoreML session)
└── Collect results
```

### New (Multiprocessing):
```python
Main Process
├── Read batch → memory
├── multiprocessing.Pool (6 workers)
│   ├── Process 1: init models → process_frame() → CoreML session 1
│   ├── Process 2: init models → process_frame() → CoreML session 2
│   ├── ... (each process has ISOLATED CoreML session)
│   └── Process 6: init models → process_frame() → CoreML session 6
└── Collect results
```

---

## 🔧 Implementation Steps

### **Step 1: Create Global Worker Function** (30 min)

Workers must be at module level (pickle-able). Cannot use class methods directly with multiprocessing.Pool.

**Add to `openface_integration.py` (AFTER imports, BEFORE class):**

```python
# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================
# These functions must be at module level to be pickle-able by multiprocessing
# ============================================================================

# Global processor instance (one per worker process)
_worker_processor = None

def _init_worker_process(weights_dir, device, confidence_threshold, nms_threshold, calculate_landmarks):
    """
    Initialize OpenFace3Processor in each worker process.

    This runs once per worker process at startup, giving each process
    its own isolated CoreML session.
    """
    global _worker_processor
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        _worker_processor = OpenFace3Processor(
            device=device,
            weights_dir=weights_dir,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            calculate_landmarks=calculate_landmarks,
            num_threads=1,  # Each process is single-threaded
            debug_mode=False
        )
    print(f"  Worker process {mp.current_process().pid} initialized")


def _process_frame_multiprocessing(frame_data):
    """
    Process a single frame in a worker process.

    Args:
        frame_data: tuple of (frame_index, frame, fps)

    Returns:
        tuple of (frame_index, csv_row_dict)
    """
    global _worker_processor
    if _worker_processor is None:
        raise RuntimeError("Worker process not initialized")

    frame_index, frame, fps = frame_data

    # Use the processor's existing _process_frame_worker logic
    return _worker_processor._process_frame_worker(frame_data)
```

---

### **Step 2: Modify `process_video()` Method** (2-3 hours)

**Location:** `openface_integration.py`, line ~427 (the `process_video` method)

**Changes needed:**

#### 2A. Add multiprocessing parameter (line ~77 in `__init__`):
```python
def __init__(self, device=None, weights_dir=None, confidence_threshold=0.5,
             nms_threshold=0.4, calculate_landmarks=False, num_threads=6,
             debug_mode=False, use_multiprocessing=True):  # NEW PARAMETER
    """
    Args:
        ...
        use_multiprocessing: Use multiprocessing.Pool instead of threads (default: True)
    """
    self.use_multiprocessing = use_multiprocessing
    # ... rest of init
```

#### 2B. Replace ThreadPoolExecutor section (line ~552-605):

**OLD CODE (ThreadPoolExecutor):**
```python
if True:  # ENABLED: Parallel processing with pre-loaded frames
    # Multi-threaded CPU processing
    from concurrent.futures import as_completed

    with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
        # Submit all frames in current batch
        futures = {executor.submit(self._process_frame_worker, frame_data): frame_data[0]
                  for frame_data in current_batch}

        # Collect results as they complete
        for future in as_completed(futures):
            idx, csv_row = future.result()
            batch_results[idx] = csv_row
            # ... progress updates
```

**NEW CODE (multiprocessing.Pool):**
```python
if self.use_multiprocessing:
    # ============================================================================
    # MULTIPROCESSING: Each process gets isolated CoreML session
    # ============================================================================
    # Process pool is created once per video (expensive to create/destroy)
    # Each worker process initializes its own OpenFace3Processor instance
    # ============================================================================

    # Create pool if not exists (at start of video processing)
    if not hasattr(self, '_mp_pool') or self._mp_pool is None:
        print(f"  Initializing multiprocessing pool with {self.num_threads} workers...")

        # Determine weights directory
        if self.weights_dir is None:
            script_dir = Path(__file__).parent
            weights_dir = script_dir / 'weights'
        else:
            weights_dir = self.weights_dir

        # Create pool with initializer
        self._mp_pool = Pool(
            processes=self.num_threads,
            initializer=_init_worker_process,
            initargs=(
                str(weights_dir),
                self.device,
                self.confidence_threshold,
                self.nms_threshold,
                self.calculate_landmarks
            )
        )
        print(f"  ✓ Worker processes initialized (each has isolated CoreML)")

    # Submit batch to pool
    async_results = []
    for frame_data in current_batch:
        result = self._mp_pool.apply_async(_process_frame_multiprocessing, (frame_data,))
        async_results.append((frame_data[0], result))  # (frame_idx, AsyncResult)

    # Collect results as they complete
    frames_done = 0
    for frame_idx, result in async_results:
        idx, csv_row = result.get(timeout=60)  # Wait up to 60s per frame
        batch_results[idx] = csv_row
        frames_done += 1

        # UPDATE PROGRESS BAR
        if pbar is not None:
            pbar.update(1)

        # Send progress updates every 10 frames
        if frames_done % 10 == 0:
            tqdm_rate = pbar.format_dict.get('rate', 0) or 0 if pbar is not None else 0
            if progress_callback and tqdm_rate > 0:
                try:
                    progress_callback(idx + 1, total_frames, tqdm_rate)
                except Exception:
                    pass

else:
    # FALLBACK: Use ThreadPoolExecutor (old behavior)
    # ... keep existing threading code as fallback
```

#### 2C. Add pool cleanup at end of `process_video()` (line ~729):

**Add BEFORE `return success_count`:**
```python
        # Cleanup multiprocessing pool
        if self.use_multiprocessing and hasattr(self, '_mp_pool') and self._mp_pool is not None:
            print("  Closing multiprocessing pool...")
            self._mp_pool.close()
            self._mp_pool.join()
            self._mp_pool = None
            print("  ✓ Worker processes terminated")

        return success_count
```

---

### **Step 3: Store weights_dir in __init__** (5 min)

**Location:** `openface_integration.py`, line ~112

**Change:**
```python
# Determine weights directory
if weights_dir is None:
    script_dir = Path(__file__).parent
    weights_dir = script_dir / 'weights'
else:
    weights_dir = Path(weights_dir)

self.weights_dir = weights_dir  # STORE FOR LATER USE

print("Initializing OpenFace 3.0 models...")
```

---

### **Step 4: Test with Small Video** (30 min)

```bash
cd "S1 Face Mirror"
python3 -c "
from openface_integration import OpenFace3Processor
from pathlib import Path

processor = OpenFace3Processor(use_multiprocessing=True, num_threads=6)

# Test with short video
video_path = Path('path/to/test/video_right_mirrored.mp4')
output_csv = Path('test_output.csv')

processor.process_video(video_path, output_csv)
"
```

**Expected output:**
```
Initializing multiprocessing pool with 6 workers...
  Worker process 12345 initialized
  Worker process 12346 initialized
  ...
  ✓ Worker processes initialized (each has isolated CoreML)

Extracting AUs: 100%|██████████| 854/854 [00:18<00:00, 47.2frame/s]

  ✓ Processed 854 frames successfully
  Output: test_output.csv
```

**Success criteria:**
- FPS > 40 (target: 40-50+ fps)
- No errors or crashes
- Output CSV matches threading version

---

## 🐛 Debugging Common Issues

### Issue 1: "Can't pickle local object"
**Cause:** Using lambdas or local functions
**Fix:** Ensure `_init_worker_process` and `_process_frame_multiprocessing` are at module level

### Issue 2: Workers hang or timeout
**Cause:** CoreML not initializing properly in workers
**Fix:** Check that each worker prints initialization message

### Issue 3: Memory explosion
**Cause:** Pool not getting cleaned up between videos
**Fix:** Ensure `_mp_pool.close()` and `_mp_pool.join()` are called

### Issue 4: FPS still low (~10 fps)
**Cause:** Processes still serializing somehow
**Fix:** Check Activity Monitor - should see multiple Python processes at 100% CPU each

### Issue 5: Results out of order
**Cause:** Async results collected in wrong order
**Fix:** Use `batch_results[idx]` dict to store by frame index, then sort before writing

---

## 📊 Performance Expectations

### Conservative Estimate (75% efficiency):
```
Single process: 9.5 fps (105ms/frame)
6 processes: 9.5 × 6 × 0.75 = 42.75 fps
```

### Optimistic Estimate (85% efficiency):
```
Single process: 9.5 fps (105ms/frame)
6 processes: 9.5 × 6 × 0.85 = 48.45 fps
```

### Realistic Target:
**40-50 fps** (4-5x improvement over baseline)

---

## 🚧 Known Limitations

1. **Memory Usage:** Each process loads full model into memory (6x memory vs threading)
   - Baseline: ~2 GB (1 model)
   - Multiprocessing: ~12 GB (6 models)
   - Still acceptable for 16-32 GB systems

2. **Startup Time:** Pool initialization takes ~10-15 seconds (models warm up in each process)
   - Acceptable for long videos (>100 frames)
   - May not be worth it for very short videos (<30 frames)

3. **Complexity:** More complex to debug than threading
   - Use `print()` statements in worker functions
   - Check logs for each worker process

---

## 🎯 Success Checklist

- [ ] Worker functions at module level (`_init_worker_process`, `_process_frame_multiprocessing`)
- [ ] `use_multiprocessing` parameter added to `__init__`
- [ ] `self.weights_dir` stored in `__init__`
- [ ] ThreadPoolExecutor replaced with `multiprocessing.Pool`
- [ ] Pool cleanup added at end of `process_video()`
- [ ] Tested with small video (verify FPS > 40)
- [ ] Tested with large video (verify no memory issues)
- [ ] AU output matches baseline (spot-check CSV files)

---

## 🚀 Next Steps After Implementation

1. **Validate Performance:**
   - Run on IMG_0435 (854 frames)
   - Target: >40 fps
   - Compare to baseline: 9.5 fps

2. **Test with AU45:**
   - Enable `calculate_landmarks=True`
   - Target: >20 fps with AU45
   - If <20 fps → STAR optimization needed (Future Option A)

3. **Production Testing:**
   - Test on multiple videos
   - Test different resolutions (720p, 1080p, 4K)
   - Verify CSV output quality

4. **Git Commit:**
```bash
git add -A
git commit -m "Implement multiprocessing for AU extraction (40-50+ fps)

- Replace ThreadPoolExecutor with multiprocessing.Pool
- Each process gets isolated CoreML session (no serialization)
- Performance: 9.5 fps → 40-50 fps (4-5x improvement)
- Memory usage: ~12 GB for 6 worker processes

Test Results:
- IMG_0435 (854 frames): XX.X fps
- CPU utilization: 600% (6 cores @ 100% each)
- No CoreML serialization bottleneck

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 📝 Implementation Notes

- This guide assumes `openface_integration.py` is the file to modify
- The threading code is kept as fallback (`use_multiprocessing=False`)
- Pool is created once per video (not per batch) for efficiency
- Progress updates work the same way (tqdm + GUI callback)

---

**Ready to implement! Estimated time: 4-6 hours for complete implementation and testing.**
