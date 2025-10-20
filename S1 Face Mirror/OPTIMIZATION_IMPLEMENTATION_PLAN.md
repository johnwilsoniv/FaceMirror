# Face Mirror Optimization Implementation Plan

**Created:** 2025-10-17
**Baseline:** face_mirror_performance_20251017_170249.txt
**Total Processing Time:** 344.6s for 1111 frames (~3.2 FPS)
**Target:** 145-165s (~6.7-7.6 FPS) - **2.1-2.4x speedup**

---

## Performance Baseline (Actual Profiling Data)

| Component | Time | % of Total | Avg/Call | Calls | Status |
|-----------|------|------------|----------|-------|--------|
| **STAR_coreml** | **185.4s** | **56.3%** | **166.9ms** | 1111 | **PRIMARY BOTTLENECK** |
| MTL_coreml | 128.9s | 39.1% | 58.0ms | 2223 | Secondary Bottleneck |
| RetinaFace_coreml | 15.0s | 4.6% | 102.3ms | 147 | Minor |
| RetinaFace_postprocess | 6.0s | 1.8% | 40.8ms | 147 | Quick Win Opportunity |
| Other (pre/post) | 9.3s | 2.7% | - | - | Negligible |

**CoreML Status:**
- STAR: 82.6% ANE coverage, 18 partitions
- MTL: 69.0% ANE coverage, 28 partitions
- RetinaFace: 81.3% ANE coverage, 3 partitions

---

## Optimization Phases (Prioritized by Impact)

### ❌ PHASE 1: RetinaFace Postprocessing Optimization (FAILED & REVERTED)
**Impact:** Target was 8-10x speedup - Actual: Made performance WORSE
**Effort:** 2 hours (implementation + testing + revert)
**Risk:** Low (reverted with no permanent changes)
**Status:** ❌ FAILED & REVERTED (2025-10-17)

**Current Performance (Baseline):**
- 6.0s total (1.8% of processing)
- 40.8ms average per call
- **py_cpu_nms is optimal** for this workload

**Attempted Optimizations:**

| Attempt | Approach | Result | Time |
|---------|----------|--------|------|
| Baseline | py_cpu_nms | ✅ BEST | 40.8ms |
| Attempt 1 | MPS + torchvision.ops.nms | ❌ WORSE | 118.6ms (3x slower) |
| Attempt 2 | CPU + torchvision.ops.nms | ❌ WORSE | 76.0ms (1.9x slower) |
| **Final** | **Reverted to py_cpu_nms** | **✅ RESTORED** | **40.8ms** |

**Why It Failed:**
1. **MPS overhead:** Device transfers cost more than computation benefits
2. **torchvision.ops.nms overhead:** Tensor operations slower than optimized NumPy
3. **Original is already optimal:** py_cpu_nms was specifically optimized for this use case

**Key Learnings:**
- MPS/GPU not always faster (see `PHASE1_LESSONS_LEARNED.md`)
- Device transfers have real costs (~10-20ms each)
- Small operations (< 5000 elements) don't benefit from GPU
- **Don't optimize what's already optimal**

**Decision:** SKIP Phase 1 - Focus on Phase 2 (STAR) where optimization will actually help

**Implementation Details:**
See `PHASE1_LESSONS_LEARNED.md` for detailed analysis.

**Files Modified:**
- `onnx_retinaface_detector.py`

**Reference:** STAR_OPTIMIZATION_GUIDE.md lines 515-648

---

### ⬜ PHASE 2: STAR Model Optimization (MAIN EFFORT)
**Impact:** 3-4x speedup, ~120s saved per session
**Effort:** 3-4 weeks
**Risk:** Medium (requires model retraining)

**Current Performance:**
- 185.4s total (56.3% of processing) - **LARGEST BOTTLENECK**
- 166.9ms average per call
- 82.6% ANE coverage, 18 partitions

**Target Performance:**
- 55-66s total (17-20% of processing)
- 50-60ms average per call
- 95%+ ANE coverage, <5 partitions

#### Week 1-2: Model Architecture Modifications
- [ ] **Day 1-2: Analysis & Setup**
  - [ ] Load STAR PyTorch checkpoint from OpenFace 3.0
  - [ ] Document current architecture (layers, operations, shapes)
  - [ ] Identify all LayerNorm/InstanceNorm layers
  - [ ] Identify all non-ReLU activations (GELU, SiLU, etc.)
  - [ ] Profile memory layout (NCHW vs NHWC)

- [ ] **Day 3-4: Replace Incompatible Operations**
  - [ ] Replace LayerNorm → BatchNorm2d (retain dimensionality)
  - [ ] Replace GELU/SiLU → ReLU6
  - [ ] Optimize attention mechanism (use Conv2d instead of Linear)
  - [ ] Remove dynamic reshapes (use fixed shapes)
  - [ ] Fuse Conv + BN + Activation layers

- [ ] **Day 5: Fine-tuning & Validation**
  - [ ] Fine-tune modified model on OpenFace training data (1-2 epochs)
  - [ ] Validate landmark accuracy (target: <2% degradation)
  - [ ] Save modified PyTorch checkpoint

#### Week 2-3: CoreML Conversion & Optimization
- [ ] **Day 6-7: Initial Conversion**
  - [ ] Convert model to channels-last memory format (NHWC)
  - [ ] Trace with torch.jit.trace
  - [ ] Convert to CoreML with MLProgram format, FP16, CPU_AND_NE
  - [ ] Use EnumeratedShapes: `(1, 3, 224, 224)` and `(1, 3, 256, 256)`
  - [ ] Save initial CoreML model

- [ ] **Day 8-9: Analysis & Iteration**
  - [ ] Use Xcode Performance Report to analyze compute unit assignment
  - [ ] Count partitions and ANE coverage (target: <5 partitions, >95% ANE)
  - [ ] Identify remaining CPU/GPU operations
  - [ ] Iterate on model modifications to eliminate incompatibilities
  - [ ] Re-convert and re-analyze

- [ ] **Day 10: Validation & Benchmarking**
  - [ ] Benchmark inference time on M1 (target: 50-60ms)
  - [ ] Validate landmark accuracy on test set
  - [ ] Profile memory usage
  - [ ] Test on M1 Pro to verify performance gains

#### Week 3-4: Integration & Testing
- [ ] **Day 11-13: Integration**
  - [ ] Create new ONNX model from optimized CoreML
  - [ ] Update `onnx_star_detector.py` to use new model
  - [ ] Run integration tests with full pipeline
  - [ ] Validate end-to-end processing

- [ ] **Day 14-15: Final Validation**
  - [ ] Run full pipeline profiling with new model
  - [ ] Compare accuracy on validation dataset
  - [ ] Document performance improvements
  - [ ] Update profiling report

**Expected Outcome:**
- Time: 185.4s → 55-66s (~120s saved)
- Percentage: 56.3% → 17-20% of total time
- Pipeline FPS: 3.2 → 4.8-5.0 FPS (**1.5-1.6x overall speedup**)

**Files Created:**
- `convert_star_to_coreml.py` (new conversion script)
- `star_landmark_98_optimized.mlpackage` (new CoreML model)
- `star_landmark_98_optimized.onnx` (new ONNX model)

**Files Modified:**
- `onnx_star_detector.py` (update model path)

**Reference:** STAR_OPTIMIZATION_GUIDE.md lines 155-313, 652-729

---

### ⬜ PHASE 3: MTL Model Optimization
**Impact:** 1.7-1.9x speedup, ~55-65s saved per session
**Effort:** 2-3 weeks
**Risk:** Medium (can reuse STAR methodology)

**Current Performance:**
- 128.9s total (39.1% of processing)
- 58.0ms average per call
- 69.0% ANE coverage, 28 partitions (worst fragmentation)

**Target Performance:**
- 65-75s total (20-23% of processing)
- 30-35ms average per call
- 90%+ ANE coverage, <8 partitions

#### Week 1: Model Architecture Modifications
- [ ] **Day 1-2: Analysis**
  - [ ] Load MTL EfficientNet-B0 checkpoint
  - [ ] Analyze architecture (similar to STAR analysis)
  - [ ] Identify ANE-incompatible operations

- [ ] **Day 3-4: Apply STAR Methodology**
  - [ ] Replace incompatible operations (LayerNorm → BatchNorm, etc.)
  - [ ] Optimize activation functions (→ ReLU6)
  - [ ] Simplify graph where possible
  - [ ] Fine-tune if needed

- [ ] **Day 5: Validation**
  - [ ] Validate AU extraction accuracy
  - [ ] Compare with baseline outputs

#### Week 2: CoreML Conversion
- [ ] **Day 6-8: Conversion**
  - [ ] Convert to CoreML with optimal settings
  - [ ] Analyze with Xcode Performance Report
  - [ ] Iterate on conversion parameters
  - [ ] Target: <8 partitions, 90%+ ANE

- [ ] **Day 9-10: Benchmarking**
  - [ ] Benchmark inference time (target: 30-35ms)
  - [ ] Validate accuracy on test dataset
  - [ ] Profile memory usage

#### Week 3: Integration
- [ ] **Day 11-12: Integration**
  - [ ] Create new ONNX model
  - [ ] Update `onnx_mtl_detector.py`
  - [ ] Run integration tests

- [ ] **Day 13-15: Final Testing**
  - [ ] Full pipeline profiling
  - [ ] Validate AU extraction quality
  - [ ] Document improvements

**Expected Outcome:**
- Time: 128.9s → 65-75s (55-65s saved)
- Percentage: 39.1% → 20-23% of total time
- Pipeline FPS: 5.0 → 6.7-7.6 FPS (**additional 1.3-1.5x speedup**)

**Files Created:**
- `convert_mtl_to_coreml.py` (new conversion script)
- `mtl_efficientnet_b0_optimized.mlpackage` (new CoreML model)
- `mtl_efficientnet_b0_optimized.onnx` (new ONNX model)

**Files Modified:**
- `onnx_mtl_detector.py` (update model path)

**Reference:** Reuse STAR_OPTIMIZATION_GUIDE.md methodology

---

## Cumulative Performance Projections

| Phase | Pipeline Time | Pipeline FPS | Speedup | Time Saved |
|-------|--------------|--------------|---------|------------|
| Baseline | 344.6s | 3.2 FPS | 1.0x | - |
| After Phase 1 | 339.1s | 3.3 FPS | 1.02x | 5.5s |
| **After Phase 2** | **219-230s** | **4.8-5.0 FPS** | **1.5-1.6x** | **~125s** |
| **After Phase 3** | **145-165s** | **6.7-7.6 FPS** | **2.1-2.4x** | **~180-200s** |

---

## Validation Checklist

### Accuracy Validation
- [ ] STAR landmark localization error: <2% increase from baseline
- [ ] MTL AU extraction accuracy: <1% degradation
- [ ] RetinaFace detection recall: no degradation
- [ ] End-to-end pipeline outputs match original (visual inspection)

### Performance Validation
- [ ] STAR inference: 50-60ms average on M1
- [ ] MTL inference: 30-35ms average on M1
- [ ] RetinaFace postprocess: <5ms average
- [ ] Pipeline FPS: 6.7-7.6 FPS sustained
- [ ] Memory usage: <2GB per process
- [ ] Thermal stability: no throttling during 5-minute continuous processing

### Quality Validation
- [ ] No visual artifacts in landmark detection
- [ ] Stable tracking (no jitter frame-to-frame)
- [ ] Consistent performance across face sizes and orientations
- [ ] AU intensity values within expected ranges

---

## Profiling & Monitoring Tools

### During Implementation
- [ ] Use `performance_profiler.py` for timing measurements
- [ ] Monitor with `sudo powermetrics --samplers ane_power -i 1000`
- [ ] Analyze with Xcode Performance Report (CoreML models)
- [ ] Optional: `sudo asitop` for real-time ANE monitoring

### Profiling Commands
```bash
# Monitor Neural Engine usage
sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 1000

# Real-time ANE activity (requires asitop)
sudo asitop

# Export profiling data
python -c "from performance_profiler import get_profiler; get_profiler().export_json('results.json')"
```

---

## Resources & References

### Documentation
1. **CoreML Tools:** https://apple.github.io/coremltools/
2. **Apple ML Research:** https://machinelearning.apple.com/research/neural-engine-transformers
3. **WWDC 2024:** https://developer.apple.com/videos/play/wwdc2024/10159/

### Project Files
- `STAR_OPTIMIZATION_GUIDE.md` - Primary reference for optimization techniques
- `COREML_PERFORMANCE_ANALYSIS.md` - Current model analysis
- `performance_profiler.py` - Timing infrastructure

### Model Weights
- STAR PyTorch: `weights/Landmark_98.pkl`
- MTL PyTorch: `weights/EfficientNet_B0_MTL.pkl` (or similar)
- RetinaFace PyTorch: `weights/Alignment_RetinaFace.pth`

---

## Notes & Learnings

### Key Insights from Profiling
- STAR (56.3%) is the dominant bottleneck, not MTL (39.1%)
- Actual timings match STAR_OPTIMIZATION_GUIDE.md baseline within 5%
- RetinaFace runs every ~7.6 frames (adaptive detection)
- 28 MTL partitions cause significant overhead despite lower per-call time

### Optimization Priorities (Confirmed)
1. **STAR first** - Largest absolute impact (120s saved)
2. **RetinaFace postprocess** - Quick win to validate approach
3. **MTL last** - Reuse STAR methodology for efficiency

### Risk Mitigation
- Test each optimization in isolation before combining
- Keep original models for rollback
- Validate accuracy at each step
- Use profiler to confirm gains before proceeding

---

## Timeline Summary

| Phase | Duration | Start Date | Completion | Status |
|-------|----------|------------|------------|--------|
| Phase 1: RetinaFace | 2 hours | 2025-10-17 | 2025-10-17 | ❌ Failed & Reverted |
| Phase 2: STAR | 3-4 weeks | TBD | TBD | ⬜ Not Started (PRIORITY) |
| Phase 3: MTL | 2-3 weeks | TBD | TBD | ⬜ Not Started |
| **Total** | **6-8 weeks** | 2025-10-17 | | **Ready for Phase 2** |

---

## Success Criteria

### Must Have (Required for Success)
- ✅ STAR inference: <60ms average
- ✅ MTL inference: <40ms average
- ✅ Pipeline FPS: >6 FPS sustained
- ✅ Landmark accuracy: <2% degradation
- ✅ AU accuracy: <1% degradation

### Should Have (Target Goals)
- ⭐ STAR inference: <55ms average
- ⭐ MTL inference: <35ms average
- ⭐ Pipeline FPS: >7 FPS sustained
- ⭐ ANE coverage: >95% for STAR, >90% for MTL
- ⭐ Partitions: <5 for STAR, <8 for MTL

### Nice to Have (Stretch Goals)
- 🎯 STAR inference: <50ms average
- 🎯 Pipeline FPS: >8 FPS sustained
- 🎯 ANE coverage: >98% for both models
- 🎯 Memory usage: <1.5GB per process

---

**Last Updated:** 2025-10-17
**Status:** Phase 1 - Failed & Reverted | Ready for Phase 2
**Next Step:** Begin Phase 2 (STAR Model Optimization) - Expected 3-4x speedup, ~120s saved
**Key Lesson:** Focus optimization efforts where they'll actually help (model inference, not postprocessing)
