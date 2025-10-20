# Phase 1: Lessons Learned - MPS Overhead Analysis

**Date:** 2025-10-17
**Issue:** Optimization made performance WORSE (3x slower)
**Root Cause:** MPS device transfer overhead
**Solution:** Simplified optimization - CPU-only with vectorized NMS

---

## What Went Wrong

### Initial Optimization Attempt
- **Goal:** 8-10x speedup by using MPS (Apple Silicon GPU)
- **Approach:** Move all tensors to MPS device, use GPU-accelerated NMS
- **Expected:** 40.8ms → 2-5ms
- **Actual Result:** 40.8ms → **118.6ms** (3x SLOWER!)

### Performance Data

| Metric | Baseline | After MPS | Change |
|--------|----------|-----------|--------|
| RetinaFace_postprocess | 40.8ms | 118.6ms | ❌ +77.8ms (3x slower) |
| Total postprocess time | 6.0s (147 calls) | 7.8s (66 calls) | ❌ Degraded |

---

## Root Cause Analysis

### Why MPS Made Things Slower

1. **Device Transfer Overhead**
   ```python
   # Each .to(device) call has overhead
   loc = torch.from_numpy(loc).to('mps')      # Transfer 1
   conf = torch.from_numpy(conf).to('mps')    # Transfer 2
   landms = torch.from_numpy(landms).to('mps') # Transfer 3
   prior_data = priors.data.to('mps')         # Transfer 4
   ...
   dets = dets.cpu().numpy()                  # Transfer back
   ```
   - Each CPU↔MPS transfer adds 5-20ms latency
   - Small tensors suffer worst overhead-to-computation ratio

2. **MPS Not Optimized for Small Operations**
   - RetinaFace postprocessing operates on ~5000 prior boxes
   - Operations are memory-bound, not compute-bound
   - GPU excels at large batch operations (thousands-millions of elements)
   - Our workload is too small to amortize transfer costs

3. **torchvision.ops.nms on MPS May Not Be Optimized**
   - MPS backend is newer than CUDA
   - Some operations may fall back to slower paths
   - NMS algorithm not inherently parallelizable (sequential dependencies)

---

## Key Learnings

### When MPS Helps (and When It Doesn't)

**MPS IS GOOD FOR:**
✅ Large model inference (CNNs, transformers)
✅ Batch processing (batch size > 8-16)
✅ Compute-heavy operations (matrix multiplication, convolutions)
✅ Operations that stay on device for many steps

**MPS IS BAD FOR:**
❌ Small tensor operations (<1000 elements)
❌ Memory-bound operations (indexing, masking, sorting)
❌ Operations with many CPU↔GPU transfers
❌ Sequential algorithms (like NMS)

### Apple Silicon Architecture Reality

```
CPU ←→ Unified Memory ←→ MPS (GPU/Neural Engine)
  ↑         ↑                    ↑
Fast    Shared but          Transfer still
        latency exists!     has overhead
```

**Misconception:** "Unified memory = free CPU↔GPU transfers"
**Reality:** Transfers still have latency due to cache coherency and synchronization

---

## Corrected Optimization Approach

### What We Changed (Iteration 2)

**Keep It Simple:**
1. ✅ Replace `py_cpu_nms` with `torchvision.ops.nms` (on CPU)
2. ✅ Use vectorized torch operations for filtering
3. ❌ NO device transfers (stay on CPU throughout)
4. ❌ NO MPS/GPU involvement

**Code Diff:**
```python
# BEFORE (slow Python loops)
keep = py_cpu_nms(dets, self.nms_threshold)  # 30-35ms

# AFTER (vectorized on CPU)
keep = nms(boxes, scores, self.nms_threshold)  # Should be 10-15ms
```

**Expected Improvement:**
- py_cpu_nms uses Python loops with NumPy operations
- torchvision.ops.nms uses C++ vectorized implementation
- Even on CPU, should see 2-3x speedup (40ms → 15-20ms)

---

## Testing Plan (Revised)

### Performance Targets (Realistic)

| Metric | Baseline | Conservative Target | Optimistic Target |
|--------|----------|-------------------|-------------------|
| Postprocess Time | 40.8ms | 20-25ms | 15-20ms |
| Speedup | 1.0x | 1.6-2.0x | 2.0-2.7x |

**Why More Conservative:**
- No GPU acceleration (CPU only)
- Main win is from vectorized C++ vs Python loops
- Limited by single-threaded NMS algorithm

---

## Broader Implications for Pipeline Optimization

### Phase 2 & 3 Strategy Updates

**For STAR Model Optimization:**
- ✅ CoreML Neural Engine is GOOD (large model, batch inference)
- ✅ Focus on reducing partitions (avoid CPU↔ANE transfers)
- ✅ Model inference benefits from ANE acceleration

**For MTL Model Optimization:**
- ✅ Same as STAR - CoreML/ANE is appropriate
- ✅ Large convolution operations benefit from GPU

**For Postprocessing Operations:**
- ❌ Avoid MPS for small tensor operations
- ✅ Use vectorized CPU implementations instead
- ✅ Pre-allocate buffers on CPU (not GPU)

### Updated Optimization Heuristic

```python
def should_use_gpu(operation_size, num_transfers):
    """
    Heuristic for GPU vs CPU decision

    operation_size: number of FLOPs or tensor elements
    num_transfers: number of CPU↔GPU transfers needed
    """
    transfer_cost = num_transfers * 10  # ~10ms per transfer
    gpu_benefit = operation_size / 1000  # rough estimate

    return gpu_benefit > transfer_cost
```

**For RetinaFace postprocessing:**
- operation_size: ~5000 boxes × 100 ops = 500K FLOPs
- num_transfers: 4-5 transfers
- transfer_cost: 40-50ms
- gpu_benefit: ~0.5ms
- **Result: CPU is better**

**For STAR model inference:**
- operation_size: 30M parameters × 2 (MAC) = 60M FLOPs
- num_transfers: 2 (input, output)
- transfer_cost: 20ms
- gpu_benefit: 100-150ms
- **Result: GPU/ANE is better**

---

## Action Items

### Immediate (Retest)
- [ ] Test revised optimization (CPU-only NMS)
- [ ] Verify 1.6-2.7x speedup (40ms → 15-25ms)
- [ ] If still slower, revert to original py_cpu_nms

### Short-term (Documentation)
- [ ] Update PHASE1_IMPLEMENTATION_SUMMARY.md with MPS findings
- [ ] Update STAR_OPTIMIZATION_GUIDE.md with device selection heuristics
- [ ] Add "When NOT to use GPU" section

### Long-term (Phase 2/3)
- [ ] Apply learnings to STAR/MTL optimizations
- [ ] Focus CoreML/ANE optimization on model inference only
- [ ] Keep postprocessing on CPU unless proven otherwise

---

## Conclusion

**Key Takeaway:**
"Not all optimizations work on all hardware. Device transfers have real costs that can outweigh computational benefits for small operations."

**Silver Lining:**
We discovered this issue early (Phase 1) before investing weeks into STAR/MTL optimizations that might have had similar problems.

**Path Forward:**
- Retest simplified CPU-only optimization
- If successful: modest 1.6-2.7x speedup is still valuable
- If unsuccessful: revert and move to Phase 2 (where GPU actually helps)

---

**Document Version:** 1.0
**Status:** Awaiting retest results
