# AU Extraction Future Paths

**Date:** 2025-10-17
**Status:** CPU-only test ready, awaiting results

---

## 🎯 Current Situation

**Problem:** AU extraction running at 9.5 fps (pathetic performance)
**Root Cause:** CoreML Execution Provider serializes inference across threads
**Current Test:** CPU-only ONNX to validate parallelization hypothesis

---

## 🔀 Decision Tree: Next Steps

### **Path 1: CPU-Only Test Results ≥85 fps** ✅ IDEAL
**What it means:** CPU-only ONNX provides sufficient parallelization
**Effort:** 0 hours (already done)
**Risk:** None

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

### **Path 3: CPU-Only Test Results <60 fps** ❌ INSUFFICIENT
**What it means:** CPU-only not sufficient, need process-based parallelism
**Effort:** 2-3 days
**Risk:** Medium-High

**Required:** Implement multiprocessing.Pool solution

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
- Single-process: 10-15 fps
- 6 processes with 75% efficiency: 100-150 fps
- **Guaranteed to exceed 85 fps target**

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

## 📊 Performance Matrix

| Path | Effort | Risk | Without AU45 | With AU45 | Complexity |
|------|--------|------|--------------|-----------|------------|
| **Path 1: CPU-Only** | 0h | None | 83-143 fps | 9-15 fps | Low |
| **Path 2A: INT8 Quant** | 1d | Medium | 90-168 fps | 12-20 fps | Low |
| **Path 2B: Batch Size** | 1h | None | 65-90 fps | 10-12 fps | None |
| **Path 3: Multiprocessing** | 2-3d | Medium | 100-150 fps | 60-90 fps | High |
| **Future A: STAR Opt** | 1-2w | High | 83-143 fps | 30-60 fps | Very High |
| **Future B: GPU/MPS** | 2-3w | Very High | 150-300 fps | 60-120 fps | Very High |

---

## 🎯 Recommended Strategy

### **Phase 1: Test CPU-Only (NOW)**
Run the test and measure FPS:
- ≥85 fps → **Stop here, success**
- 60-84 fps → Phase 2A (INT8 quantization)
- <60 fps → Phase 3 (multiprocessing)

### **Phase 2: Quick Wins (if marginal)**
1. Try INT8 quantization (1 day)
2. If insufficient → Phase 3

### **Phase 3: Multiprocessing (if necessary)**
1. Basic implementation (1 day)
2. Production hardening (1 day)
3. Testing and integration (0.5 day)

### **Phase 4: AU45 Validation (after primary path)**
Once base performance meets target:
1. Test with AU45 enabled
2. If AU45 ≥60 fps → **Done**
3. If AU45 <60 fps → Consider STAR optimization (Future Option A)

---

## 🛌 Before You Sleep: What's Ready

✅ **CPU-only test is implemented and ready to run**
✅ **Documentation complete (this file + CPU_ONLY_TEST_RESULTS.md)**
✅ **Multiprocessing solution documented (ready to implement if needed)**
✅ **All paths have clear effort estimates and risk assessments**

---

## 🌅 When You Wake Up: Test Checklist

1. Run Face Mirror with test video (IMG_0437.MOV)
2. Check initialization message (confirm CPU-only active)
3. Record total processing time for 971 frames
4. Calculate FPS: `971 / process_time`
5. Check Activity Monitor CPU usage (should be 400-600%)
6. Report results and choose path based on FPS achieved

---

**Sleep well! The optimization strategy is documented and ready to execute.** 🚀
