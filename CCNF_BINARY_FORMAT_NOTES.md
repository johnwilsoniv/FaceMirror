# CCNF Binary Format - Technical Notes

## Overview

The CCNF patch expert files (.txt extension but binary format) have a more complex structure than initially understood.

## Actual File Format (from C++ source analysis)

### File Header
```
Offset  | Type    | Description
--------|---------|-------------
0x00    | float64 | patchScaling (interocular distance multiplier)
0x08    | int32   | numberViews (different face orientations)
```

### For Each View (orientation)
```
1. Center Matrix (binary cv::Mat)
   - Dimensions: 3×1
   - Type: CV_64FC1 (double)
   - Contains: [pitch, yaw, roll] in degrees

2. Visibility Matrix (binary cv::Mat)
   - Dimensions: numLandmarks×1  (e.g., 68×1)
   - Type: CV_32SC1 (int)
   - Contains: 0 or 1 for each landmark (visible/not visible)
```

### For Each Landmark (in each view)
```
3. Window Sizes (array)
   - Read separately from config
   - Used for sigma computation

4. Sigma Components (matrices)
   - Also from config
   - For edge features computation

5. CCNF Patch Expert:
   - Type marker: int32 (value = 5)
   - Width: int32
   - Height: int32
   - Num neurons: int32

   If num_neurons == 0:
     - Empty marker: int32
     - (landmark not visible at this orientation)

   If num_neurons > 0:
     For each neuron:
       - Type: int32 (value = 2)
       - Neuron type: int32
       - Norm weights: float64
       - Bias: float64
       - Alpha: float64
       - Weight matrix (binary cv::Mat):
         * Rows: int32
         * Cols: int32
         * Type: int32 (OpenCV type code)
         * Data: bytes (rows×cols×elemSize)

     Beta values: [float64, ...]
     Patch confidence: float64
```

## Loading Code Reference

**C++ Function:** `Patch_experts::Read_CCNF_patch_experts()`
**Location:** `Patch_experts.cpp:555`

Key code excerpts:
```cpp
// Line 563: Read patch scaling
patchesFile.read ((char*)&patchScaling, 8);

// Line 566: Read number of views
patchesFile.read ((char*)&numberViews, 4);

// Line 575-581: Read centers for each view
for(size_t i = 0; i < centers.size(); i++) {
    cv::Mat center;
    LandmarkDetector::ReadMatBin(patchesFile, center);
    center.copyTo(centers[i]);
    centers[i] = centers[i] * M_PI / 180.0;  // Convert to radians
}

// Line 584-587: Read visibility for each view
for(size_t i = 0; i < visibility.size(); i++) {
    LandmarkDetector::ReadMatBin(patchesFile, visibility[i]);
}

// Line 590+: Read patches for each view and landmark
```

## Complexity Factors

1. **Multi-view structure**: File contains patches for multiple face orientations
2. **View-dependent visibility**: Each landmark may be visible in some views but not others
3. **Variable neuron counts**: Different landmarks have different numbers of neurons
4. **External config dependencies**: Window sizes and sigma components come from separate config
5. **Nested binary structures**: Matrices within structures within arrays

## Current Implementation Status

### ✓ Completed
- PDM text parser (fully working!)
- Understanding of CCNF binary format
- Neuron reading logic
- Patch expert reading logic (partial)

### ⏸ Needs More Work
- File header parsing (scaling + views)
- View metadata parsing (centers + visibility)
- Integration with window_sizes/sigma_components config
- Complete multi-view patch parsing

## Alternative Approaches

### Option 1: Complete Python Binary Parser
**Effort:** High (2-3 days)
**Pros:** Pure Python, no C++ dependencies
**Cons:** Complex multi-level parsing, potential bugs

### Option 2: Minimal C++ Export Utility
**Effort:** Medium (1 day)
**Pros:** Leverage existing C++ reader, one-time use
**Cons:** Requires C++ compilation once

**Recommended:** Option 2 - Create simple C++ program that:
```cpp
// Read using OpenFace code
CCNF_patch_expert patches;
patches.Read(/* ... */);

// Export to simple format
for (each patch) {
    np.save("patch_{i}_neurons.npy", neurons);
    np.save("patch_{i}_metadata.npy", metadata);
}
```

### Option 3: Start with Python Core, Use Pre-exported Models
**Effort:** Low (immediate start)
**Pros:** Can begin implementing CLNF algorithm now
**Cons:** Need to export models once via Option 2

**Recommended Path Forward:** Option 3 then Option 2
1. Begin implementing pure Python CLNF core (PDM, NU-RLMS optimizer)
2. Create minimal C++ export utility for CCNF patches (one-time use)
3. Never need C++ again after export!

## Files Examined

- `ccnf_patches_0.25_general.txt` (887 KB)
- `ccnf_patches_0.35_general.txt` (887 KB)
- `ccnf_patches_0.5_general.txt` (887 KB)

First bytes (hex):
```
0000: 0000 0000 0000 d03f    # patchScaling = 1.0
0008: 0700 0000               # numberViews = 7
000C: 0300 0000               # center matrix rows = 3
0010: 0100 0000               # center matrix cols = 1
0014: 0600 0000               # center matrix type = CV_64FC1
...
```

## Next Steps

1. ✅ PDM is already working - can start using for shape model
2. ⏸ CCNF patches - decide between Options 1, 2, or 3
3. 🎯 Focus on Python CLNF core implementation (PDM + optimizer)
4. 📦 Export CCNF models later (Option 2 when needed)

## Key Insight

**We don't need to parse CCNF immediately!**

The PDM is sufficient to start implementing:
- Point Distribution Model transforms
- Parameter optimization (NU-RLMS)
- Basic landmark fitting

CCNF patch experts can be exported via minimal C++ utility when needed for actual response map computation.

## References

- C++ Source: `Patch_experts.cpp` (lines 555-656)
- Binary matrix format: `LandmarkDetectorUtils.cpp:900` (ReadMatBin)
- CCNF patch expert class: `CCNF_patch_expert.cpp:288`
