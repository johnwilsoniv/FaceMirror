# CLNF Implementation Roadmap

## Executive Summary

Based on literature review and your testing showing CLNF consistently outperforms the current approach, implementing CLNF is **highly feasible** using existing resources.

## Available Resources

### 1. Official Implementation (Already Installed!)
**OpenFace 2.2** - Located at `~/repo/fea_tool/external_libs/openFace/OpenFace/`
- ✅ Complete CLNF implementation in C++
- ✅ Pre-trained models (PDM + LNF patch experts)
- ✅ Optimized for real-time performance
- ✅ Already tested and working

### 2. Training Code
**CCNF Repository**: https://github.com/TadasBaltrusaitis/CCNF
- Matlab implementation for training patch experts
- Includes training scripts for LNF
- Can retrain on custom data

### 3. Framework Code
**CLM-framework**: https://github.com/TadasBaltrusaitis/CLM-framework
- Matlab implementation of full pipeline
- Good for understanding algorithm flow
- Useful for prototyping

### 4. Python Port
**OpenFace_Python**: https://github.com/liulianjushi/OpenFace_Python
- Community Python wrapper
- Quality/completeness unknown
- Worth investigating as reference

### 5. Literature
- ✅ Original CLNF paper (2013) - we have it
- ✅ OpenFace 2.0 paper (2018)
- ✅ CCNF paper (2014)
- All mathematical details documented in papers

## Recommended Implementation Strategy

### Phase 1: Leverage Existing C++ Implementation (FASTEST PATH)

**Option 1A: Python Wrapper Around OpenFace C++**

```python
# Conceptual API
class OpenFaceCLNF:
    def __init__(self, model_path):
        # Load OpenFace models via ctypes/pybind11
        self.clnf = load_openface_clnf(model_path)

    def detect_landmarks(self, frame, bbox):
        # Initialize from RetinaFace bbox
        # Run CLNF optimization
        # Return 68 landmarks
        return landmarks_68
```

**Advantages:**
- ✅ Use proven, optimized code
- ✅ Pre-trained models work out of the box
- ✅ Fast (OpenFace runs 2-10 FPS on CPU)
- ✅ Can implement in 1-2 weeks

**Implementation Steps:**
1. Create Python bindings to OpenFace C++ (ctypes or pybind11)
2. Expose CLNF initialization and fitting functions
3. Feed RetinaFace bbox as initialization
4. Return landmarks to Python

**Challenges:**
- Need to create Python bindings
- Handle C++/Python data passing
- OpenCV version compatibility (already solved)

### Phase 1B: Direct C++ Integration

**Modify OpenFace to accept external bbox:**

```cpp
// In OpenFace C++ code
bool CLNF::DetectLandmarks(const cv::Mat& image, const cv::Rect& external_bbox) {
    // Initialize from external_bbox instead of running face detector
    // Run existing CLNF optimization
    // Return results
}
```

Then call from Python using subprocess or shared library.

**Advantages:**
- ✅ Minimal code changes
- ✅ Keep all optimizations
- ✅ Straightforward integration

**Implementation Steps:**
1. Modify OpenFace to accept bbox parameter
2. Disable internal face detection
3. Export landmarks to standardized format (CSV/JSON)
4. Call from Python, parse results

### Phase 2: Train Custom Patch Experts (OPTIONAL)

Only if pre-trained models don't perform well on facial paralysis data.

**Use CCNF Training Code:**

```matlab
% Train LNF patch experts on custom data
% From CCNF repository
train_LNF_patch_expert(training_images, landmarks, scale, orientation)
```

**Steps:**
1. Prepare training data (facial paralysis videos with landmarks)
2. Extract patches at multiple scales/orientations
3. Train using CCNF Matlab code
4. Export trained models
5. Load into OpenFace C++

**Training Requirements:**
- ~1000+ labeled images (you likely have this)
- Multiple scales: 17px, 23px, 30px interocular distance
- Multiple orientations: -20°, 0°, +20° yaw
- Positive samples (on landmark) + negative samples (off landmark)

### Phase 3: Pure Python Implementation (IF NEEDED)

Only if C++ integration proves difficult.

**Port key components to Python:**

```python
class LNFPatchExpert:
    """Local Neural Field patch expert"""
    def __init__(self, weights_file):
        self.alpha, self.beta, self.gamma, self.theta = load_weights(weights_file)

    def compute_response_map(self, patch):
        # Implement LNF forward pass
        # Vertex features (neural network)
        # Edge features (smoothness + sparsity)
        return response_map

class NURLMS:
    """Non-Uniform Regularised Landmark Mean-Shift"""
    def __init__(self, pdm, patch_experts):
        self.pdm = pdm
        self.patch_experts = patch_experts
        self.W = compute_reliability_weights(patch_experts)

    def fit(self, frame, initial_bbox):
        # Iterative optimization
        for iteration in range(max_iters):
            # Evaluate patch experts
            response_maps = [pe.compute_response_map(frame, landmark)
                           for pe in self.patch_experts]

            # Compute mean-shift vectors
            v = compute_mean_shifts(response_maps)

            # Update parameters
            delta_p = -(J.T @ W @ J + r*Lambda_inv)^-1 @ (r*Lambda_inv @ p - J.T @ W @ v)
            p += delta_p

            if converged(delta_p):
                break

        return landmarks_from_params(p)
```

**Advantages:**
- ✅ Full control over implementation
- ✅ Easy to modify/experiment
- ✅ Native Python integration

**Challenges:**
- ❌ Time-consuming (2-3 months)
- ❌ Slow without optimization
- ❌ Need to port trained models
- ❌ Need to implement PDM, LNF, NU-RLMS

## Recommended Phased Approach

### Week 1-2: Quick Proof of Concept
**Goal:** Get CLNF working with RetinaFace bbox

**Approach:** Direct C++ integration (Phase 1B)

1. **Modify OpenFace** (`FaceModelParameters.cpp`):
   ```cpp
   // Add flag to skip face detection
   bool use_external_bbox = true;
   cv::Rect external_bbox;
   ```

2. **Create Python wrapper script**:
   ```python
   def run_clnf(video_path, bbox_sequence):
       # Write bbox to temp file
       # Call OpenFace with --external-bbox flag
       # Parse landmark output
       return landmarks
   ```

3. **Integration test**:
   - Run on 5-10 test videos
   - Compare with current pipeline
   - Measure accuracy improvement
   - Profile performance

**Deliverable:** Working prototype demonstrating CLNF with RetinaFace initialization

### Week 3-4: Optimize Integration
**Goal:** Clean Python API and performance optimization

1. **Create pybind11 bindings** (if subprocess too slow)
2. **Batch processing support**
3. **Multi-scale handling**
4. **Error handling and fallbacks**

**Deliverable:** Production-ready CLNF module

### Month 2: Custom Training (If Needed)
**Goal:** Patch experts specialized for facial paralysis

1. **Prepare training data**:
   - Extract frames from paralysis videos
   - Label landmarks (or use CLNF predictions)
   - Balance normal/paralysis samples

2. **Set up CCNF training**:
   - Install Matlab (or use Octave)
   - Clone CCNF repository
   - Configure training parameters

3. **Train patch experts**:
   - Train 9 sets (3 scales × 3 orientations)
   - Validate on held-out data
   - Compare with pre-trained models

4. **Deploy custom models**:
   - Export to OpenFace format
   - Test in production
   - A/B test against pre-trained

**Deliverable:** Custom CLNF models optimized for facial paralysis

## Integration Points with Current Pipeline

### Replacing PFLD + SVR CLNF:

**Current:**
```python
RetinaFace → bbox + 5 landmarks → PFLD → 68 landmarks → SVR CLNF → refined 68
```

**New:**
```python
RetinaFace → bbox → CLNF → 68 landmarks
```

### Hybrid Approach (Best of Both Worlds):

```python
class HybridLandmarkDetector:
    def __init__(self):
        self.retinaface = RetinaFace()
        self.pfld = PFLD()
        self.clnf = OpenFaceCLNF()
        self.use_clnf = True  # Flag to switch

    def detect(self, frame):
        # Face detection
        bbox, landmarks_5 = self.retinaface(frame)

        if self.use_clnf:
            # CLNF from bbox
            landmarks_68 = self.clnf.detect_landmarks(frame, bbox)
        else:
            # PFLD + SVR CLNF
            landmarks_68 = self.pfld.detect_landmarks(frame, bbox, landmarks_5)
            landmarks_68 = self.svr_clnf.refine(frame, landmarks_68)

        return landmarks_68
```

## Technical Considerations

### 1. Initialization from RetinaFace Bbox

**OpenFace expects:**
- Face detection bbox: [x, y, width, height]
- Optional: confidence score

**RetinaFace provides:**
- Bbox: [x1, y1, x2, y2]
- 5 landmarks (optional, can discard)

**Conversion:**
```python
def retinaface_to_openface_bbox(rf_bbox):
    x1, y1, x2, y2 = rf_bbox
    return {
        'x': x1,
        'y': y1,
        'width': x2 - x1,
        'height': y2 - y1,
        'confidence': 1.0
    }
```

### 2. PDM Compatibility

**OpenFace PDM:** 68 points (dlib format)
**Your current pipeline:** 68 points (same format)

✅ **Compatible** - No conversion needed

### 3. Multi-Scale Processing

**OpenFace uses 3 scales typically:**
- Coarse: 17px interocular
- Medium: 23px interocular
- Fine: 30px interocular

**RetinaFace detects at single scale:**
- May need to resize for optimal CLNF performance
- Or train patch experts at different scales

### 4. Performance Optimization

**Target: Real-time mirroring (~30 FPS)**

**Current bottlenecks:**
- CLNF: ~100-300ms per frame (Matlab)
- OpenFace C++: ~50-100ms per frame

**Optimization strategies:**
- Use OpenFace C++ (already optimized)
- GPU acceleration (OpenFace supports)
- Reduce iterations (trade accuracy for speed)
- Use larger area of interest (faster)

## Risk Assessment

### Low Risk:
✅ **Phase 1A/1B** - Use existing OpenFace implementation
- Well-tested code
- Pre-trained models
- Clear integration path

### Medium Risk:
⚠️ **Phase 2** - Custom patch expert training
- Requires Matlab/Octave
- Need sufficient training data
- May not improve over pre-trained

### High Risk:
❌ **Phase 3** - Pure Python rewrite
- Time-consuming
- Likely slower than C++
- Maintenance burden

## Success Criteria

### Minimum Viable Product (Week 2):
- [ ] CLNF accepts RetinaFace bbox as input
- [ ] Produces 68 landmarks
- [ ] Runs on test videos
- [ ] Measurable accuracy improvement

### Production Ready (Week 4):
- [ ] Clean Python API
- [ ] Performance: <100ms per frame
- [ ] Error handling
- [ ] Fallback to PFLD on failure
- [ ] Integration tests

### Optimized (Month 2):
- [ ] Custom patch experts (if needed)
- [ ] <50ms per frame
- [ ] 95%+ detection rate
- [ ] Better than OpenFace baseline on your data

## Resources Needed

### Immediate (Phase 1):
- Access to OpenFace C++ code (✅ already have)
- Python binding tools (ctypes/pybind11) (✅ pip install)
- Test videos with ground truth (✅ already have)

### If Custom Training (Phase 2):
- Matlab license (or Octave - free)
- CCNF training code (✅ GitHub)
- 1000+ labeled training images
- GPU for training (optional, speeds up)

## Bottom Line

**Feasibility: HIGH ✅**

You have everything needed to implement CLNF:
1. ✅ Working C++ implementation (OpenFace)
2. ✅ Pre-trained models
3. ✅ Training code (if needed)
4. ✅ Complete documentation
5. ✅ Test data
6. ✅ Proof that CLNF works better (your testing)

**Recommended Path:**
Start with **Phase 1B** (direct C++ integration) - should have working prototype in 1-2 weeks.

**Expected Improvement:**
Based on your testing showing CLNF consistently outperforms current approach, expect:
- Better landmark accuracy on challenging cases
- More robust to occlusion/asymmetry
- Stronger shape constraints (no impossible faces)

## Next Steps

1. **Verify OpenFace models location**:
   ```bash
   ls ~/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/
   ```

2. **Test OpenFace with external bbox**:
   - Modify to accept bbox parameter
   - Test on single frame
   - Compare output with PFLD

3. **Create Python wrapper**:
   - Simple subprocess wrapper first
   - Optimize with pybind11 if needed

4. **Benchmark**:
   - Accuracy: CLNF vs current pipeline
   - Speed: Time per frame
   - Robustness: Failure rate

Would you like me to start with Phase 1B implementation?
