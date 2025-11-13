# OpenFace Mean-Shift Convergence Bug: Complete Document Index

## Quick Navigation

Start with your learning style:

**Visual/Practical:** → `00_START_HERE.md` then `INSTRUMENTATION_POINTS.md`
**Strategic/Process:** → `README_CONVERGENCE_DEBUG.md`
**Technical/Deep:** → `OPENFACE_MEANSHIFT_DEBUG.md`
**Quick Reference:** → `MEANSHIFT_QUICK_REFERENCE.md`

---

## Document Details

### 1. 00_START_HERE.md (Entry Point)
**Best for:** First time reading this analysis
**Content:**
- What you're debugging (high-level)
- Read order recommendation
- 15-minute quick start
- Data flow diagram
- Validation checklist
**Key sections:** None (entry point)
**Length:** ~2 KB

### 2. README_CONVERGENCE_DEBUG.md (Strategic Overview)
**Best for:** Understanding the full picture
**Content:**
- Package overview (5 documents)
- The bug in 1 sentence
- 3-step debugging process with timings
- 4 most likely root causes ranked by probability
- Quick verification checklist (pre-debugging)
- Key file locations table
- How to use the package
- Example debugging session
- Symptom-to-solution mapping
**Key sections:**
- "The Bug in One Sentence"
- "3-Step Debugging Process"
- "Most Likely Root Cause"
- "How to Use This Package"
**Length:** ~4 KB

### 3. MEANSHIFT_QUICK_REFERENCE.md (Technical Summary)
**Best for:** Quick lookup during debugging
**Content:**
- Complete data flow (7 stages)
- Critical equations (3 types)
- 4 common failure modes with:
  * Root causes
  * Debug code snippets
  * Expected vs actual behavior
- Validation checklist (9 items)
- 3 hypotheses ranked by likelihood
- Key file locations lookup table
**Key sections:**
- "What Can Go Wrong" (4 failure modes)
- "Critical Equations"
- "Validation Checklist"
- "Hypothesis: Why It Might Not Work"
**Length:** ~7 KB

### 4. INSTRUMENTATION_POINTS.md (Practical Guide)
**Best for:** Adding debug code to C++
**Content:**
- 9 strategic instrumentation locations:
  1. After mean-shift computation (Line 1077)
  2. Coordinate transform check (Lines 1080-1083)
  3. Jacobian verification (After line 1062)
  4. Jacobian-mean-shift projection (After line 1107)
  5. Hessian and solve (Lines 1115-1128)
  6. Parameter update effect (After line 1131)
  7. Iteration loop monitoring (Line 1036)
  8. Weight matrix check (After line 1028)
  9. Per-landmark mean-shift debug (Lines 927-931)

- For each location:
  * Purpose
  * Copy-paste ready code
  * Expected output
  * How to detect failure

- Expected output pattern (working case)
- Red flags to watch for
**Key sections:**
- Location-by-location instrumentation
- "Expected Output Pattern"
- "Red Flags"
**Length:** ~8 KB

### 5. OPENFACE_MEANSHIFT_DEBUG.md (Complete Reference)
**Best for:** Understanding the algorithm in depth
**Content:**

**SECTION 1: Mean-Shift Computation Pipeline**
- Function: `NonVectorisedMeanShift_precalc_kde()` (Lines 820-935)
- KDE kernel computation
- Weighted average calculation
- Output format
- KDE kernel equation: `v = exp(a*(vx+vy))`

**SECTION 2: Coordinate Space Transformations**
- Response-map space to image space
- `sim_ref_to_img` transformation
- Critical for Jacobian projection

**SECTION 3: NU_RLMS Optimization Loop**
- 6-step process (lines 990-1191)
- Step 1: Jacobian computation
- Step 2: Mean-shift computation
- **Step 3: Jacobian-mean-shift projection (CRITICAL)**
- Step 4: Regularization
- Step 5: Hessian build and Cholesky solve
- Step 6: Parameter application

**SECTION 4: Jacobian Computation Details**
- Matrix structure: (2n) x (6+m)
- Scaling, rotation, translation derivatives
- Small-angle approximation for rotation
- Weight matrix application
- Jacobian equation: `J[i,j] = d(landmark_2D[i])/d(param[j])`

**SECTION 5: Parameter Update Application**
- Scale/translation: Direct addition
- Rotation: Matrix multiplication (small-angle perturbation)
- Local parameters: Direct addition
- `UpdateModelParameters()` function

**SECTION 6: Shape Rendering from Parameters**
- 3D shape from local parameters
- Rigid transformation (rotation + scale + translation)
- 2D projection (weak-perspective)
- `CalcShape2D()` function

**DEBUGGING CHECKLIST: 5 Bug Categories**
- Issue 1: Jacobian-mean-shift mismatch
- Issue 2: Coordinate space transformation error
- Issue 3: Regularization overpower
- Issue 4: Convergence termination
- Issue 5: Parameter clamping

**CRITICAL EQUATIONS**
- Equation 1: Gauss-Newton solution
- Equation 2: Weak-perspective projection
- Equation 3: Mean-shift calculation

**Key sections:**
- "POTENTIAL BUGS - DEBUGGING CHECKLIST"
- "CRITICAL EQUATIONS TO VERIFY"
- "RECOMMENDED DEBUGGING STEPS"
- "KEY FILES TO INSPECT"
**Length:** ~14 KB

---

## How to Use These Documents

### Scenario 1: "I don't know where to start"
1. Read `00_START_HERE.md` (2 min)
2. Read `README_CONVERGENCE_DEBUG.md` (3 min)
3. Decide which strategy to follow

### Scenario 2: "Response maps look good but landmarks don't move"
1. Read `MEANSHIFT_QUICK_REFERENCE.md` (5 min)
2. Go to `INSTRUMENTATION_POINTS.md`
3. Find location matching your symptom
4. Add debug code and run

### Scenario 3: "I need to understand the math"
1. Read `OPENFACE_MEANSHIFT_DEBUG.md` Section 1-6
2. Review critical equations
3. Compare with code at indicated line numbers

### Scenario 4: "I found the bug location, need quick answer"
1. Open `MEANSHIFT_QUICK_REFERENCE.md`
2. Search for your symptom in "What Can Go Wrong"
3. Read root cause analysis
4. Jump to corresponding section in `OPENFACE_MEANSHIFT_DEBUG.md`

### Scenario 5: "I'm instrumenting the code"
1. Open `INSTRUMENTATION_POINTS.md`
2. Find your location (1-9)
3. Copy code snippet
4. Paste into LandmarkDetectorModel.cpp or PDM.cpp
5. Recompile
6. Run and compare output

---

## File Locations in Your System

All documents:
```
/Users/johnwilsoniv/Documents/SplitFace Open3/
  ├─ 00_START_HERE.md
  ├─ README_CONVERGENCE_DEBUG.md
  ├─ MEANSHIFT_QUICK_REFERENCE.md
  ├─ INSTRUMENTATION_POINTS.md
  ├─ OPENFACE_MEANSHIFT_DEBUG.md
  └─ DOCUMENT_INDEX.md (this file)
```

Source code to modify:
```
/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/
  └─ lib/local/LandmarkDetector/src/
     ├─ LandmarkDetectorModel.cpp (mean-shift, Jacobian projection)
     └─ PDM.cpp (Jacobian, parameter update, shape rendering)
```

---

## Critical Code Locations Reference

| Component | File | Lines | Document Section |
|-----------|------|-------|------------------|
| Mean-shift computation | LandmarkDetectorModel.cpp | 820-935 | OPENFACE_MEANSHIFT_DEBUG S1 |
| NU_RLMS loop | LandmarkDetectorModel.cpp | 990-1191 | OPENFACE_MEANSHIFT_DEBUG S3 |
| **Jacobian projection** | LandmarkDetectorModel.cpp | 1107 | INSTRUMENTATION_POINTS L4 |
| **Parameter update** | LandmarkDetectorModel.cpp | 1131 | INSTRUMENTATION_POINTS L6 |
| Jacobian computation | PDM.cpp | 346-450 | OPENFACE_MEANSHIFT_DEBUG S4 |
| UpdateModelParameters | PDM.cpp | 454-506 | OPENFACE_MEANSHIFT_DEBUG S5 |
| CalcShape2D | PDM.cpp | 159-188 | OPENFACE_MEANSHIFT_DEBUG S6 |

---

## Most Useful Sections by Problem Type

**"Mean-shift vectors are zero"** → OPENFACE_MEANSHIFT_DEBUG S1 + INSTRUMENTATION_POINTS L1
**"Parameters not updating"** → OPENFACE_MEANSHIFT_DEBUG S3-S4 + INSTRUMENTATION_POINTS L3-L5
**"Landmarks not moving despite updates"** → OPENFACE_MEANSHIFT_DEBUG S5-S6 + INSTRUMENTATION_POINTS L6
**"Coordinate space issues"** → OPENFACE_MEANSHIFT_DEBUG S2 + INSTRUMENTATION_POINTS L2
**"Jacobian singular"** → OPENFACE_MEANSHIFT_DEBUG S4 + INSTRUMENTATION_POINTS L3
**"Need equations"** → OPENFACE_MEANSHIFT_DEBUG "CRITICAL EQUATIONS" + MEANSHIFT_QUICK_REFERENCE

---

## Reading Time Estimates

**Just the essentials:** 15 minutes
- 00_START_HERE.md (2 min)
- README_CONVERGENCE_DEBUG.md (3 min)
- INSTRUMENTATION_POINTS.md relevant section (10 min)

**Good understanding:** 30 minutes
- Add MEANSHIFT_QUICK_REFERENCE.md (5 min)
- Add quick scan of OPENFACE_MEANSHIFT_DEBUG.md (5 min)

**Complete mastery:** 1-2 hours
- Read all documents in order
- Read source code at indicated lines
- Run instrumentation and compare

---

## The Bug at Each Level

**One sentence:**
Landmarks don't move despite correct response maps.

**One paragraph:**
Mean-shift vectors are computed correctly from response maps, but they are not being properly converted to parameter changes that affect landmark positions. The bug is somewhere in the Jacobian projection or parameter update pipeline.

**Technical:**
The NU_RLMS optimization loop computes mean-shift vectors (equation in OPENFACE_MEANSHIFT_DEBUG S1), projects them through the Jacobian (equation in S3), solves a Gauss-Newton system (equation in S5), and applies the resulting parameters (equation in S6). One of these steps is failing.

**Full detail:**
See OPENFACE_MEANSHIFT_DEBUG Sections 1-6.

---

## Summary

You have everything needed to debug this convergence bug:
1. Complete technical breakdown (OPENFACE_MEANSHIFT_DEBUG)
2. Quick reference guide (MEANSHIFT_QUICK_REFERENCE)
3. Copy-paste debug code (INSTRUMENTATION_POINTS)
4. Strategic overview (README_CONVERGENCE_DEBUG)
5. Entry point guide (00_START_HERE)

Expected time to fix: 1 hour after identifying the failure point.

Good luck!

