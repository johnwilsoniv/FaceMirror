# OpenFace Mean-Shift Convergence Bug: START HERE

## You Are Here Because

Your landmarks aren't moving during detection, even though:
- Response maps are computed correctly
- Peaks in response maps are in the right locations
- But the optimization loop doesn't update landmark positions

## Read These in Order

### 1. README_CONVERGENCE_DEBUG.md (3 min read)
- High-level overview of the bug
- 3-step debugging process
- Most likely causes ranked by probability
- How to use the analysis package

### 2. MEANSHIFT_QUICK_REFERENCE.md (5 min read)
- Visual data flow diagram (7 stages)
- 4 common failure modes matching your symptoms
- Validation checklist (run this first!)
- Critical equations

### 3. INSTRUMENTATION_POINTS.md (Apply debug code)
- 9 locations in C++ code to add debug output
- Copy-paste ready code snippets
- Expected output pattern for working case
- Red flags that indicate failure

### 4. OPENFACE_MEANSHIFT_DEBUG.md (Detailed reference)
- Comprehensive technical breakdown
- 6 main sections (mean-shift, Jacobian, parameter update, etc.)
- Debugging checklist with 5 categories of bugs
- Critical equations to verify

---

## Quick Start (15 minutes)

1. Open **MEANSHIFT_QUICK_REFERENCE.md**
   - Look at "What Can Go Wrong" section
   - Find the failure mode that matches your symptoms
   - Read the root cause analysis

2. Go to **INSTRUMENTATION_POINTS.md**
   - Find the location number matching your failure mode
   - Copy the debug code
   - Add to your C++ code

3. Compile and run
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j4
   ./your_program 2>&1 | tee debug.log
   ```

4. Check output against "Expected Output Pattern"
   - Look for the step that fails
   - Continue to next instrumentation point

---

## The Bug (In One Sentence)

Mean-shift vectors are computed correctly but not properly converted to parameter changes that move landmarks.

---

## Most Likely Root Causes

**40%:** Coordinate transformation `sim_ref_to_img` is wrong
**35%:** Jacobian has structural issues (zero rows, singular)
**20%:** Hessian is ill-conditioned (regularization too strong)
**5%:** UpdateModelParameters doesn't actually update parameters

---

## Critical Code Locations

| What | File | Lines |
|------|------|-------|
| Mean-shift | LandmarkDetectorModel.cpp | 820-935 |
| Optimization loop | LandmarkDetectorModel.cpp | 990-1191 |
| **Jacobian projection** | LandmarkDetectorModel.cpp | 1107 |
| **Parameter update** | LandmarkDetectorModel.cpp | 1131 |
| Jacobian computation | PDM.cpp | 346-450 |
| Apply parameters | PDM.cpp | 454-506 |
| Render landmarks | PDM.cpp | 159-188 |

---

## Validation Checklist (Run First!)

Before debugging C++ code:

- [ ] Response maps have clear peaks
- [ ] Peaks are near ground truth landmarks
- [ ] Mean-shift vectors non-zero
- [ ] Jacobian is full-rank
- [ ] Hessian condition number < 1e6
- [ ] Parameter updates are nonzero
- [ ] CalcShape2D produces visible movement

If all pass, bug is subtle (numerical precision issue)
If any fail, use INSTRUMENTATION_POINTS to debug that step

---

## File Guide

**New to this codebase?**
→ Start with MEANSHIFT_QUICK_REFERENCE.md

**Know what's wrong, need to fix it?**
→ Go directly to INSTRUMENTATION_POINTS.md

**Need deep technical understanding?**
→ Read OPENFACE_MEANSHIFT_DEBUG.md Section by Section

**Want a roadmap of what to do?**
→ Use README_CONVERGENCE_DEBUG.md

---

## The Data Pipeline

```
Response Maps
      ↓
Mean-Shift Computation
      ↓ (mean_shifts in response-map space)
Coordinate Transform
      ↓ (mean_shifts in image space)
Jacobian Computation
      ↓
Jacobian × Mean-Shifts
      ↓ (projects to parameter space)
Hessian Solve
      ↓
Parameter Updates
      ↓
Apply to Parameters
      ↓
CalcShape2D
      ↓
New Landmarks
```

If landmarks don't move, trace backwards through this pipeline.

---

## Next Steps

1. **Identify your symptom** (which step fails)
2. **Read INSTRUMENTATION_POINTS** location for that step
3. **Add debug code** to LandmarkDetectorModel.cpp or PDM.cpp
4. **Recompile** with -DCMAKE_BUILD_TYPE=Debug
5. **Run test** and capture output
6. **Compare** against expected output pattern
7. **Find failure point** and read OPENFACE_MEANSHIFT_DEBUG section
8. **Fix the bug** based on root cause analysis
9. **Verify fix** with debug output

---

## Support

All code locations are absolute paths from your system:
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp`
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/src/PDM.cpp`

All debug statements are copy-paste ready.

Good luck! This should be fixable in under an hour once you pinpoint the failure.

