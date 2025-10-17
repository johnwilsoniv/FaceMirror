# AU Extraction Future Paths

**Date:** 2025-10-17
**Status:** ❌ CPU-only test FAILED - Proceeding with multiprocessing

---

## 🧪 CPU-Only Test Results (COMPLETED)

**Test Configuration:**
- ONNX Runtime: CPU Execution Provider only
- Threads: 6 worker threads
- Video: IMG_0435 (854 frames @ 59 fps)

**Results:**
- ✅ **CPU Utilization:** 600-650% (parallelization working correctly)
- ❌ **FPS:** 11 fps (expected 50-80+ fps)
- ❌ **Per-frame time:** ~555ms per frame (vs CoreML's 105ms)

**Analysis:**
```
CoreML Neural Engine:
- Per-frame: ~105ms (serialized)
- Total: 9.5 fps with 6 threads (serialization bottleneck)

CPU-only:
- Per-frame: ~555ms (5x slower than CoreML!)
- Total: 11 fps with 6 threads (good parallelization but too slow)
```

**Conclusion:**
- ❌ **Hypothesis was WRONG:** CPU-only is 5x slower per-frame than CoreML Neural Engine
- ✅ Threading/parallelization works correctly (600% CPU proves it)
- 🎯 **Solution:** Need multiprocessing with CoreML for best of both worlds

**Next Step:** Path 3 - Multiprocessing.Pool implementation

---

## 🎯 Current Situation

**Problem:** AU extraction running at 9.5-11 fps (unacceptable performance)
**Root Cause:** CoreML Execution Provider serializes inference across threads/processes
**Solution:** Implement multiprocessing with CoreML (each process gets isolated CoreML session)

---

## 🔀 Decision Tree: Next Steps

### **Path 1: CPU-Only Test Results ≥85 fps** ❌ DID NOT ACHIEVE
**What it means:** CPU-only ONNX provides sufficient parallelization
**Effort:** 0 hours (already done)
**Risk:** None
**Result:** Got 11 fps (CPU too slow compared to Neural Engine)

**Next Steps:**
1. ✅ Confirm CPU-only test achieves ≥85 fps
2. Test with AU45 enabled (STAR landmark detection)
3. If AU45 still ≥60 fps → **DONE!**
4. If AU45 drops to <60 fps → See Path 3

**Expected Performance:**
- Without AU45: 83-143 fps (6x parallelization, 70-85% efficiency)
- With AU45: 9-15 fps (STAR bottleneck remains)

**Pros:**
- ✅ Minimal code changes
- ✅ No multiprocessing complexity
- ✅ Stable, maintainable
- ✅ Works within existing architecture

**Cons:**
- ⚠️ Per-frame slower than CoreML (30-50ms vs 15-30ms)
- ⚠️ May not help AU45 performance (STAR still slow)

---

### **Path 2: CPU-Only Test Results 60-84 fps** ⚠️ MARGINAL
**What it means:** Close to target but not quite sufficient
**Effort:** 1-3 days
**Risk:** Medium

**Options (in order of preference):**

#### **Option 2A: INT8 Quantization** (1 day)
Convert ONNX models to INT8 for 1.5-2x speedup

**Steps:**
1. Install ONNX quantization tools
2. Quantize MTL model (static calibration)
3. Test accuracy vs speed tradeoff
4. If accuracy acceptable → Deploy

**Expected Performance:**
- Per-frame: 15-30ms → 8-20ms
- With 6 threads: 60-84 fps → 90-168 fps

**Pros:**
- ✅ Simple implementation
- ✅ No architecture changes
- ✅ Compatible with CPU-only

**Cons:**
- ⚠️ Slight accuracy loss (~1-2%)
- ⚠️ Need to validate AU output quality

#### **Option 2B: Reduce Batch Size** (1 hour)
Test if smaller batches improve parallelization

**Steps:**
1. Test BATCH_SIZE = 50 (vs current 100)
2. Check if memory pressure was limiting performance
3. If improvement → Adjust default

**Expected Performance:**
- Marginal improvement (5-10%)
- Unlikely to close gap to 85 fps

**Pros:**
- ✅ Trivial to test
- ✅ No risk

**Cons:**
- ❌ Unlikely to solve problem alone

#### **Option 2C: Move to Path 3** (2-3 days)
If marginal performance insufficient, implement multiprocessing

---

### **Path 3: Multiprocessing with CoreML** 🚀 ACTIVE PATH
**What it means:** Use process-based parallelism + CoreML for best performance
**Effort:** 2-3 days
**Risk:** Medium-High

**Status:** ✅ Implementing now (CPU-only test proved CoreML is essential)

**Architecture:**
```
Main Process
├── Video Reading (sequential)
├── Frame Distribution Queue
├── Worker Pool (6 processes)
│   ├── Worker 1: RetinaFace + MTL + STAR (ONNX)
│   ├── Worker 2: RetinaFace + MTL + STAR (ONNX)
│   ├── ...
│   └── Worker 6: RetinaFace + MTL + STAR (ONNX)
├── Results Collection Queue
└── CSV Writing (sequential)
```

**Expected Performance:**
```
With CoreML per process:
- Single-process: ~9.5 fps (105ms per frame, CoreML Neural Engine)
- 6 processes with 75% efficiency: ~43 fps (conservative)
- 6 processes with 85% efficiency: ~48 fps (optimistic)

Reality Check:
- CPU-only: 11 fps @ 555ms/frame (too slow)
- CoreML threads: 9.5 fps @ 105ms/frame (serialized)
- CoreML processes: Target 40-50+ fps (no serialization)

**Target: 40+ fps (4-5x improvement over baseline)**
```

**Implementation Steps:**

**Day 1: Basic Multiprocessing (4-6 hours)**
1. Replace ThreadPoolExecutor with multiprocessing.Pool
2. Implement shared memory for frames (avoid pickling overhead)
3. Parallel model warmup (load ONNX in each process)
4. Test with small video

**Day 2: Production Hardening (4-6 hours)**
1. Queue-based IPC for GUI progress callbacks
2. Graceful shutdown handling (Ctrl+C, errors)
3. Memory cleanup between batches
4. Test with large videos (10+ minutes)

**Day 3: Integration & Testing (2-4 hours)**
1. Integrate with native_dialogs progress window
2. Test on multiple videos (1080p, 4K, portrait, landscape)
3. Verify AU output matches single-thread results
4. Performance benchmarking and validation

**Pros:**
- ✅ Guaranteed performance improvement (100+ fps)
- ✅ True process isolation (no GIL, no serialization)
- ✅ Each worker has dedicated ONNX session
- ✅ Scales better with AU45 enabled

**Cons:**
- ⚠️ Higher memory usage (6x model copies)
- ⚠️ More complex IPC for GUI updates
- ⚠️ Debugging more difficult
- ⚠️ Startup overhead (model loading)

**Risks:**
- Shared memory implementation may be tricky on macOS
- GUI progress updates require Queue-based IPC
- Memory usage may be prohibitive for 4K videos

---

### **Path 4: CPU-Only Comparable to Baseline (~10 fps)** 🚨 BROKEN
**What it means:** Something is wrong with threading
**Effort:** 2-4 hours debugging
**Risk:** Unknown

**Likely Issues:**
1. NUM_THREADS not set correctly (check main.py:52)
2. Exceptions silently failing in worker threads
3. ThreadPoolExecutor not actually parallelizing
4. cv2.VideoCapture still causing deadlocks

**Debugging Steps:**
1. Add per-thread timing logs in openface_integration.py
2. Check Activity Monitor for CPU utilization (should be 400-600%)
3. Verify frames are pre-loaded correctly (read_batch_preload)
4. Check for exceptions in worker threads

**If debugging fails:** Move to Path 3 (multiprocessing)

---

## 🔮 Long-Term Optimization Paths

### **Future Option A: STAR ONNX Optimization**
**Goal:** Reduce STAR landmark detection from 400-500ms to <100ms
**Effort:** 1-2 weeks
**Risk:** High

**Approaches:**
1. INT8 quantization for STAR model
2. Model pruning (reduce 98 landmarks to only eyes)
3. Alternative models (but user rejected MediaPipe)
4. Custom CoreML conversion with aggressive optimization

**Expected Impact:**
- AU45 performance: 9-15 fps → 30-60 fps

---

### **Future Option B: GPU Acceleration**
**Goal:** Use Metal/MPS for true GPU parallelization
**Effort:** 2-3 weeks
**Risk:** Very High

**Approach:**
1. Convert ONNX models to PyTorch with MPS backend
2. Batch inference across 6 frames simultaneously
3. GPU memory management for large batches

**Expected Impact:**
- Without AU45: 150-300 fps
- With AU45: 60-120 fps

**Challenges:**
- MPS not as mature as CoreML
- May have compatibility issues with OpenFace 3.0
- GPU memory constraints with large batches

---

### **Future Option C: Hybrid Architecture**
**Goal:** CoreML for detection, CPU for landmark/AU
**Effort:** 1-2 days
**Risk:** Medium

**Approach:**
1. Keep RetinaFace + MTL on CoreML (fast, Neural Engine)
2. Run STAR on CPU-only in parallel
3. Split workload to avoid serialization

**Expected Impact:**
- Marginal improvement over CPU-only (10-20%)

---

## 📊 Performance Matrix (Updated with Test Results)

| Path | Effort | Risk | Actual FPS | Expected MP FPS | Complexity |
|------|--------|------|------------|-----------------|------------|
| **Baseline: CoreML threads** | 0h | None | 9.5 fps | N/A | Current |
| **Path 1: CPU-Only** | 0h | None | ❌ 11 fps | N/A | ❌ Failed |
| **Path 2A: INT8 Quant** | 1d | Medium | N/A | 15-20 fps | Low |
| **Path 2B: Batch Size** | 1h | None | N/A | 10-12 fps | None |
| **Path 3: MP + CoreML** | 2-3d | Medium | Testing | 40-50 fps | High |
| **Future A: STAR Opt** | 1-2w | High | N/A | TBD | Very High |
| **Future B: GPU/MPS** | 2-3w | Very High | N/A | TBD | Very High |

**Note:** All estimates now based on actual measured CoreML performance (105ms/frame = 9.5 fps serialized)

---

## 🎯 Recommended Strategy (UPDATED)

### **Phase 1: Test CPU-Only** ✅ COMPLETE
Result: 11 fps (CPU too slow vs CoreML Neural Engine)
- ❌ Did not achieve 85 fps target
- ✅ Proved parallelization works (600% CPU)
- ❌ Proved CPU is 5x slower than CoreML per-frame
- 🎯 Conclusion: Must use multiprocessing + CoreML

### **Phase 2: Implement Multiprocessing** 🚀 CURRENT
Target: 40-50+ fps (4-5x improvement over baseline)

**Tonight's Work (4-6 hours):**
1. Basic multiprocessing.Pool implementation
2. Revert to CoreML execution provider
3. Process-level model initialization
4. Test with small video

**Tomorrow:**
1. Production hardening (error handling, cleanup)
2. GUI integration (Queue-based IPC)
3. Full testing and validation

### **Phase 3: AU45 Validation (after multiprocessing)**
Once base performance meets target (40+ fps):
1. Test with AU45 enabled
2. If AU45 ≥20 fps → **Acceptable**
3. If AU45 <20 fps → Consider STAR optimization (Future Option A)

---

## 🚀 Implementation Plan: Multiprocessing with CoreML

### **Step 1: Revert to CoreML** (5 minutes)
Undo CPU-only test changes in `onnx_mtl_detector.py`:
- Remove forced `CPUExecutionProvider`
- Restore original CoreML configuration
- Each process will get isolated CoreML session

### **Step 2: Replace ThreadPoolExecutor with multiprocessing.Pool** (2 hours)
Key changes in `openface_integration.py`:
- Import `multiprocessing` instead of `threading`
- Create process pool with 6 workers
- Each worker loads models independently (isolated CoreML)
- Use `multiprocessing.Queue` for results collection

### **Step 3: Handle GUI Progress** (1 hour)
- Use `multiprocessing.Queue` for IPC from workers to main
- Main process polls queue and updates GUI callback
- Workers send progress tuples: `(frame_idx, fps_estimate)`

### **Step 4: Test and Validate** (1 hour)
- Run on test video
- Verify FPS improvement (target: 40-50+ fps)
- Check AU output matches baseline
- Monitor memory usage

---

**Ready to implement! Starting multiprocessing implementation now.** 🚀
