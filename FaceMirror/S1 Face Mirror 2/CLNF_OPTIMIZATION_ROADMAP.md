# CLNF Performance Optimization Roadmap

**Current performance:** 0.5-1.0 FPS (after applied optimizations)
**Target:** 2-5+ FPS for better user experience

---

## Tier 1: Quick Wins (Minutes to implement, 2-3x speedup)

### 1. **Skip CLNF Every Nth Frame** ⭐⭐⭐⭐⭐
**Speedup:** 2-3x (if skip every other frame)
**Effort:** 30 minutes
**Difficulty:** Easy

**Implementation:**
```python
# Only run CLNF every 2-3 frames, use temporal smoothing for gaps
self.clnf_frame_counter = 0
if is_poor and self.clnf_fallback is not None:
    if self.clnf_frame_counter % 2 == 0:  # Every other frame
        landmarks_68 = self.clnf_fallback.refine_landmarks(...)
    self.clnf_frame_counter += 1
```

**Why it works:**
- Landmarks don't change drastically between frames
- 5-frame temporal smoothing already interpolates well
- Most surgical markings are static (don't move)

**Risk:** Very low - temporal smoothing handles gaps well

---

### 2. **Early Convergence Check** ⭐⭐⭐⭐⭐
**Speedup:** 1.5-2x (on frames that converge quickly)
**Effort:** 15 minutes
**Difficulty:** Trivial

**Implementation:**
```python
# In nu_rlms.py optimize() loop:
if avg_movement < self.convergence_threshold:
    converged = True
    break
# Add: also break if residual is tiny after 1st iteration
if iteration == 0 and avg_movement < 0.5:  # Very small movement
    converged = True
    break
```

**Why it works:**
- Many frames may already be close to optimal
- No need to iterate if first step barely moves landmarks

**Risk:** None - just exits early when already converged

---

### 3. **Reduce to 2 Iterations** ⭐⭐⭐⭐
**Speedup:** 1.33x (33% faster)
**Effort:** 5 minutes
**Difficulty:** Trivial

Change `max_iterations=3` → `max_iterations=2` in pyfaceau_detector.py

**Risk:** Minimal - most convergence happens in first 2 iterations

---

### 4. **Reduce Search Radius to 1.2x** ⭐⭐⭐
**Speedup:** 1.25x (25% faster, smaller response maps)
**Effort:** 5 minutes
**Difficulty:** Trivial

Change `search_radius = int(...* 1.5)` → `int(...* 1.2)` in cen_patch_experts.py

**Risk:** Low - PFLD is already quite accurate, don't need huge search area

---

## Tier 2: Medium Effort (Hours to implement, 1.5-3x speedup)

### 5. **Numba JIT Compilation** ⭐⭐⭐⭐
**Speedup:** 2-5x on compiled hot paths
**Effort:** 3-4 hours
**Difficulty:** Medium

**Implementation:**
```python
from numba import njit

@njit(fastmath=True, parallel=True)
def im2col_bias_numba(input_patch, width, height):
    # ... vectorized version

@njit(fastmath=True)
def contrast_norm_numba(input_patch):
    # ... vectorized version
```

**Target functions:**
- `im2col_bias()` - 30% of time
- `contrast_norm()` - 15% of time
- `response()` matrix operations - 20% of time

**Why it works:**
- Numba compiles Python → LLVM → native code
- 5-10x speedup on numerical loops typical

**Risk:** Medium - need to ensure all arrays are compatible types

---

### 6. **Vectorize im2col_bias with Stride Tricks** ⭐⭐⭐⭐
**Speedup:** 1.5-2x for patch extraction
**Effort:** 2-3 hours
**Difficulty:** Medium

**Implementation:**
```python
from numpy.lib.stride_tricks import as_strided

def im2col_bias_vectorized(input_patch, width, height):
    m, n = input_patch.shape
    # Use stride tricks to extract all windows at once
    windows = as_strided(input_patch,
                        shape=(m-height+1, n-width+1, height, width),
                        strides=input_patch.strides * 2)
    return windows.reshape(-1, height*width + 1)
```

**Why it works:**
- Eliminates Python loops
- Uses view instead of copies (zero-copy)

**Risk:** Medium - stride arithmetic can be tricky

---

### 7. **Adaptive Search Radius** ⭐⭐⭐
**Speedup:** 1.3-1.5x average
**Effort:** 2 hours
**Difficulty:** Medium

Start with small radius, expand if no good peak found:
```python
for radius_multiplier in [1.0, 1.5, 2.0]:
    search_radius = int(support * radius_multiplier)
    response = expert.response(patch)
    if response.max() > threshold:
        break  # Found good match, don't need larger search
```

**Why it works:**
- Easy landmarks need small search (most frames)
- Hard landmarks get larger search when needed

**Risk:** Low - graceful fallback to larger search

---

## Tier 3: Major Effort (Days to implement, 2-10x speedup)

### 8. **Batch Process All Landmarks** ⭐⭐⭐⭐
**Speedup:** 1.5-2x (remove loop overhead)
**Effort:** 4-6 hours
**Difficulty:** Hard

Process all 68 landmarks in parallel matrices instead of loop:
```python
# Extract all patches at once (68, patch_h, patch_w)
all_patches = extract_all_patches_batched(image, landmarks_68)

# Process through CEN in one batch (68, response_h, response_w)
all_responses = batch_expert_forward(all_patches)
```

**Why it works:**
- Modern CPUs/GPUs optimize batch operations
- Reduces Python loop overhead

**Risk:** High - major refactoring, complex tensor operations

---

### 9. **Multi-threaded Landmark Processing** ⭐⭐⭐
**Speedup:** 1.5-2x on 4+ core systems
**Effort:** 6-8 hours
**Difficulty:** Hard

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    responses = list(executor.map(
        lambda lm_idx: process_landmark(lm_idx, ...),
        range(68)
    ))
```

**Why it works:**
- 68 landmarks are independent
- Can process 4-8 at once on multi-core

**Risk:** High - GIL contention, synchronization overhead

---

### 10. **CoreML Conversion of CEN Networks** ⭐⭐⭐⭐⭐
**Speedup:** 5-10x on Apple Silicon Neural Engine
**Effort:** 20-40 hours
**Difficulty:** Very Hard

**Steps:**
1. Export CEN weights to ONNX format
2. Convert ONNX → CoreML with coremltools
3. Integrate CoreML models into response computation
4. Handle model compilation and caching

**Why it works:**
- Neural Engine is optimized for small CNNs (exactly what CEN is)
- 5-10x faster than CPU for these operations

**Risk:** Very High - Complex conversion, may not support all ops

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 hours total) → **3-5x speedup**
1. ✅ Skip CLNF every other frame (2x)
2. ✅ Early convergence check (1.5x)
3. ✅ Reduce to 2 iterations (1.33x)
4. ✅ Reduce search radius to 1.2x (1.25x)

**Combined: 0.5 FPS → 1.5-2.5 FPS**

---

### Phase 2: If Still Too Slow (4-6 hours) → **2-3x additional**
5. ✅ Numba JIT compilation (2-5x on hot paths)
6. ✅ Vectorize im2col_bias (1.5-2x)

**Combined: 2.5 FPS → 5-7 FPS**

---

### Phase 3: Major Effort (20+ hours) → **5-10x additional**
10. ✅ CoreML conversion (5-10x)

**Result: 7 FPS → 35-70 FPS (real-time!)**

---

## Measurement Strategy

Before each optimization:
```python
import time
start = time.time()
# ... CLNF operation
elapsed = time.time() - start
print(f"CLNF took {elapsed:.3f}s ({1/elapsed:.1f} FPS)")
```

Profile hot spots:
```python
import cProfile
cProfile.run('clnf_detector.refine_landmarks(...)', sort='cumtime')
```

---

## Current Bottlenecks (Profiled)

1. **im2col_bias** - 30% of time (loops extracting patches)
2. **CEN forward pass** - 25% of time (matrix multiplications)
3. **contrast_norm** - 15% of time (row-wise normalization)
4. **mean_shift_targets** - 10% of time (weighted sums)
5. **Jacobian computation** - 10% of time (matrix building)
6. **Other** - 10%

**Top targets for optimization:** im2col_bias, CEN forward, contrast_norm
