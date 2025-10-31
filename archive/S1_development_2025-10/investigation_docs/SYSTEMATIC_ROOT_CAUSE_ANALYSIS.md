# Systematic Root Cause Analysis: Why C++ Gets Upright Faces

## Goal
Understand WHY C++ produces consistently upright faces (rotation ~0°) while Python produces tilted faces (~-5° with eyes, ~+30° without eyes).

**Constraints for this analysis:**
- No code changes or tests during exploration
- Focus on rotation angle and stability ONLY
- No correlation metrics, no AU testing yet
- Pure investigation of the algorithmic difference

---

## The Mystery Stated Clearly

### Observed Behavior

**C++ Output:**
- Rotation: ~0° (upright faces)
- Stability: ~0° std (perfectly consistent)
- Expression invariant: Eye closure doesn't affect rotation
- Visual: Faces are always upright

**Python Output (WITH eyes, 24 points):**
- Rotation: ~-5° (slight CCW tilt)
- Stability: 4.51° std (varies across frames)
- Expression sensitive: 6.4° jump on eye closure
- Visual: Faces slightly tilted, varies with expression

**Python Output (WITHOUT eyes, 16 points):**
- Rotation: ~+30° (significant CCW tilt)
- Stability: 1.47° std (very consistent!)
- Expression invariant: No jump on eye closure
- Visual: Faces consistently tilted ~30° CCW

### The Core Question

Both implementations:
- ✓ Use the same Kabsch algorithm
- ✓ Use the same PDM file (In-the-wild_aligned_PDM_68.txt)
- ✓ Use the same rigid point indices (in source code)
- ✓ Use the same matrix operations (we verified step-by-step)

**So why do they produce different rotation angles?**

---

## Known Facts About the System

### 1. PDM Mean Shape is Rotated

We confirmed the PDM mean shape is rotated ~45° CCW:
- Nose vector: 64° from vertical
- Eye axis: -130° from horizontal
- This is intentional (average orientation in training data)

**Question:** If both Python and C++ use this rotated reference, why don't both produce the same rotation?

### 2. Kabsch Algorithm Implementation

Both use identical Kabsch algorithm:
1. Mean normalize source and destination
2. Compute RMS scale
3. Normalize by scale
4. SVD of `src.T @ dst`
5. Check determinant for reflection
6. Compute rotation matrix: `R = Vt.T @ corr @ U.T`
7. Return `scale * R`

**Question:** If the algorithm is identical, where does the difference arise?

### 3. Rigid Points

OpenFace C++ source shows 24 rigid indices including 8 eye landmarks.

**Observation:** Python WITH eyes uses these → -5° tilt
**Observation:** Python WITHOUT eyes (16 points) → +30° tilt
**Question:** Does C++ actually use all 24 at runtime despite what the source code shows?

---

## Possible Root Causes (Systematic List)

### Category A: Hidden Preprocessing

**A1: Coordinate System Transform**
- Could C++ apply a coordinate transform to landmarks before AlignFace?
- Could there be an image coordinate to model coordinate conversion?
- Status: We checked - no transforms found before AlignFace call

**A2: Landmark Rotation**
- Could C++ pre-rotate landmarks to "undo" the PDM rotation?
- Could there be a pose-based rotation applied?
- Status: We checked - landmarks come directly from CLNF output

**A3: Reference Shape Modification**
- Could C++ rotate the PDM mean shape before using it?
- Could there be a "de-rotation" applied to the reference?
- Status: NEED TO INVESTIGATE - could there be a rotation applied to mean_shape that we missed?

### Category B: SVD Implementation Differences

**B1: numpy.linalg.svd vs cv::SVD**
- Different underlying libraries (LAPACK vs Eigen/OpenBLAS)
- Different numerical precision (float32 vs float64)
- Different algorithms (divide-and-conquer vs Jacobi)
- Status: RESEARCH NEEDED - are there documented differences?

**B2: Sign Ambiguity**
- SVD decomposition is not unique (U and V can have column signs flipped)
- Could numpy and OpenCV make different sign choices?
- Could this cause rotation direction differences?
- Status: RESEARCH NEEDED - how do libraries handle sign ambiguity?

**B3: Singular Value Ordering**
- Could singular values be ordered differently?
- Could this affect the rotation matrix construction?
- Status: RESEARCH NEEDED

### Category C: Matrix Operations

**C1: Operation Order**
- Python: `src.T @ dst` (using @ operator)
- C++: `src.t() * dst` (using OpenCV operator)
- Could these produce different results?
- Status: Should be identical, but VERIFY

**C2: Floating Point Precision**
- OpenCV uses float32 throughout
- numpy defaults to float64
- Could precision differences accumulate?
- Status: Could cause small differences, but 5-30° seems too large

**C3: Matrix Storage (Row vs Column Major)**
- numpy: Row-major (C-style)
- OpenCV: Could be different
- Could this affect matrix multiplications?
- Status: RESEARCH NEEDED

### Category D: Hidden Post-Processing

**D1: Rotation Correction After Kabsch**
- Could C++ apply a correction rotation after Kabsch?
- Could there be a "level face" operation?
- Status: We checked AlignFace function - no correction found

**D2: Warp Matrix Construction**
- Could C++ build the warp matrix differently?
- Could there be a rotation embedded in the translation?
- Status: We verified - matches our implementation

### Category E: Runtime vs Source Code Discrepancy

**E1: Different Rigid Points at Runtime**
- Source code shows 24 rigid indices
- Could runtime use different indices?
- Could there be a flag or config that changes them?
- Status: NEED TO INVESTIGATE - check if there's runtime configuration

**E2: Conditional Logic**
- Could there be if/else logic that changes behavior?
- Could face size or quality trigger different paths?
- Status: Review AlignFace for any conditionals

**E3: PDM File Loading Differences**
- Could C++ load the PDM differently?
- Could there be a different PDM file being used?
- Could the mean_shape be processed differently?
- Status: We verified the PDM file path - same file

### Category F: The Reference Frame Interpretation

**F1: PDM Coordinate Space**
- The PDM is rotated 45° - this is intentional
- Could there be an intended "reference frame" correction?
- Could the rotation be relative to something other than the PDM?
- Status: NEED TO UNDERSTAND - what is the PDM coordinate space supposed to represent?

**F2: Canonical Pose Definition**
- What does "aligned to canonical pose" actually mean?
- Could "canonical" mean "upright" not "aligned to PDM mean"?
- Could there be a standard face orientation that overrides PDM?
- Status: RESEARCH NEEDED - what is OpenFace's canonical pose definition?

---

## Research Questions (No Implementation Yet)

### 1. numpy SVD vs OpenCV SVD

**Research:**
- Google: "numpy linalg svd vs opencv svd differences"
- Look for: Known precision differences
- Look for: Sign ambiguity handling
- Look for: Singular value ordering

**Questions:**
- Do they produce identical results for the same input?
- Are there documented cases of different outputs?
- Could 5-30° rotation difference come from SVD?

### 2. OpenFace PDM Coordinate System

**Research:**
- Read OpenFace papers/documentation
- What coordinate system is the PDM trained in?
- What does the 45° rotation mean?
- Is there an expected "de-rotation" step?

**Questions:**
- Is the PDM supposed to be used as-is?
- Should the mean_shape be rotated to canonical pose first?
- Is there a transform from PDM space to image space?

### 3. OpenFace Canonical Pose

**Research:**
- What is OpenFace's definition of "canonical pose"?
- Is it "upright face" or "aligned to mean shape"?
- Are there examples showing expected output?

**Questions:**
- Should aligned faces always be upright?
- Or should they match the PDM orientation?
- Is there a standard we're missing?

### 4. Kabsch Algorithm Variants

**Research:**
- Are there different Kabsch implementations?
- Could there be a "Kabsch with fixed orientation" variant?
- Are there papers on expression-invariant alignment?

**Questions:**
- Could C++ be using a modified Kabsch?
- Could there be additional constraints we're missing?
- Is standard Kabsch sufficient for face alignment?

---

## Specific Code Locations to Re-Examine

### 1. PDM Mean Shape Loading

**C++ Code:**
```
lib/local/LandmarkDetector/src/PDM.cpp
- Read() function (lines ~80-150)
- How is mean_shape parsed from file?
- Is any rotation applied after loading?
```

**Check for:**
- Post-load transformations
- Coordinate system conversions
- Any mention of "canonical" or "reference"

### 2. AlignFace Function Call Sites

**C++ Code:**
```
lib/local/FaceAnalyser/src/FaceAnalyser.cpp: line 264
- How is AlignFace called?
- What are the params_global values?
- Is rigid=true or rigid=false?
```

**Check for:**
- Parameter values at call site
- Any preprocessing before call
- Any post-processing after call

### 3. Rigid Points Extraction

**C++ Code:**
```
lib/local/FaceAnalyser/src/Face_utils.cpp: lines 45-107
- extract_rigid_points() function
- Are all 24 indices used?
- Is there any conditional logic?
```

**Check for:**
- Runtime configuration
- Conditional extraction
- Any filtering of points

---

## Hypotheses to Test (Later, Not Now)

### Hypothesis 1: C++ Uses Different Rigid Points at Runtime
**Test:** Add debug output to C++ to print actual rigid points used
**Expected:** Might find different indices than source code shows

### Hypothesis 2: SVD Sign Ambiguity Causes Rotation Flip
**Test:** Force same SVD signs in Python as C++
**Expected:** Rotation angles might match

### Hypothesis 3: PDM Mean Shape Has Hidden Rotation
**Test:** Load PDM in both C++ and Python, print mean_shape values
**Expected:** Might find C++ rotates mean_shape before use

### Hypothesis 4: There's a Reference Frame Transform
**Test:** Look for coordinate system transforms in PDM or CLNF
**Expected:** Might find image→PDM or PDM→canonical transforms

### Hypothesis 5: OpenCV SVD Behaves Differently Than numpy
**Test:** Use same test matrix in both, compare outputs
**Expected:** Might find systematic difference

---

## What We Need From You

Since you'll be doing visual inspection, here's what would help:

### 1. Visual Inspection Questions

When looking at aligned faces:
- Are C++ faces perfectly upright (eyes horizontal)?
- How upright is "upright" - exactly 0° or approximately?
- Do all C++ frames have identical orientation?
- Is there ANY variation in C++ rotation across frames?

### 2. Specific Frames to Compare

Key frames for comparison:
- Frame 1: Baseline
- Frame 493: Eyes open
- Frame 617: Eyes closed (expression sensitivity test)
- Frame 863: Eyes open (consistency check)

For each, compare:
- C++ rotation angle (visual estimate)
- Python with eyes rotation
- Python without eyes rotation

### 3. Expression Sensitivity

Compare frames 493 (eyes open) vs 617 (eyes closed):
- Does C++ show ANY rotation change?
- Does Python with eyes show rotation change?
- Does Python without eyes show rotation change?

---

## Next Steps (Investigation Only)

1. **Research numpy vs OpenCV SVD**
   - Find documentation on differences
   - Look for known issues or precision differences
   - Check if anyone has reported Kabsch differences

2. **Re-examine PDM mean shape usage**
   - Check if C++ applies any rotation to mean_shape
   - Look for coordinate system transforms
   - Verify PDM coordinate space interpretation

3. **Verify rigid points at runtime**
   - Could add debug output to C++ (if needed)
   - Or infer from behavior: does C++ act like it's using different points?

4. **Understand OpenFace canonical pose**
   - What is the intended output orientation?
   - Is "aligned" supposed to mean "upright"?

5. **Check for hidden rotation corrections**
   - Are there any rotations applied outside AlignFace?
   - Could there be calibration or normalization steps?

---

## Summary

**We need to understand WHY C++ gets upright faces, not just match the output.**

The most likely candidates:
1. **SVD implementation differences** (numpy vs OpenCV)
2. **Hidden rotation of PDM mean shape** (that we missed)
3. **Runtime rigid point differences** (despite source code)
4. **Coordinate system transform** (that we haven't found)

**We will NOT:**
- Make code changes yet
- Run more experiments yet
- Test AU prediction yet

**We WILL:**
- Research SVD differences
- Re-examine C++ code for hidden rotations
- Understand PDM coordinate system
- Use your visual inspection feedback

Once we understand the ROOT CAUSE, we'll know the right fix.
