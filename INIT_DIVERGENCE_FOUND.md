# Initialization Divergence Found

## Test Setup
- Image: `calibration_frames/patient1_frame1.jpg`
- BBox: (296, 778, 405, 407)
- Tracked Landmarks: [36, 48, 30, 8]

## Python Initialization (WS=11)
```
[PY][INIT] Initial parameters:
[PY][INIT]   scale: 2.854022
[PY][INIT]   rotation: (0.000000, 0.000000, 0.000000)
[PY][INIT]   translation: (501.379280, 961.117186)
[PY][INIT] Initial tracked landmarks:
[PY][INIT]   Landmark_36: (369.9593, 854.1730)
[PY][INIT]   Landmark_48: (418.8516, 1042.7785)
[PY][INIT]   Landmark_30: (504.8805, 937.1019)
[PY][INIT]   Landmark_8: (501.6611, 1181.0518)
```

## C++ Initialization (WS=11)
```
[CPP][INIT] Initial parameters:
[CPP][INIT]   scale: 2.81584
[CPP][INIT]   rotation: (0, 0, 0)
[CPP][INIT]   translation: (505.072, 962.331)
[CPP][INIT] Initial tracked landmarks:
[CPP][INIT]   Landmark_36: (375.41, 856.818)
[CPP][INIT]   Landmark_48: (423.648, 1042.9)
[CPP][INIT]   Landmark_30: (508.526, 938.637)
[CPP][INIT]   Landmark_8: (505.35, 1179.32)
```

## Comparison

### Parameters
| Parameter | Python | C++ | Δ |
|-----------|--------|-----|---|
| scale | 2.854022 | 2.81584 | **0.038182** |
| rotation_x | 0.000000 | 0.000000 | 0.000000 |
| rotation_y | 0.000000 | 0.000000 | 0.000000 |
| rotation_z | 0.000000 | 0.000000 | 0.000000 |
| translation_x | 501.379280 | 505.072 | **-3.693** |
| translation_y | 961.117186 | 962.331 | **-1.214** |

### Landmark Positions
| LM | Python X | Python Y | C++ X | C++ Y | ΔX | ΔY | Distance |
|----|----------|----------|-------|-------|----|----|---------|
| 36 | 369.96 | 854.17 | 375.41 | 856.82 | **-5.45** | **-2.65** | **6.06 px** |
| 48 | 418.85 | 1042.78 | 423.65 | 1042.90 | **-4.80** | **-0.12** | **4.80 px** |
| 30 | 504.88 | 937.10 | 508.53 | 938.64 | **-3.65** | **-1.54** | **3.96 px** |
| 8 | 501.66 | 1181.05 | 505.35 | 1179.32 | **-3.69** | **+1.73** | **4.08 px** |

**Average initialization error: 4.73 pixels**

## Root Cause

The divergence happens at **bounding box preprocessing** BEFORE PDM initialization.

### Investigation Results

1. **Python PDM `init_params()` implementation is CORRECT**
   - When given the SAME bbox as C++, Python produces identical parameters
   - Test: bbox (301.938, 782.149, 400.586, 400.585)
   - Python scaling: 2.8158397
   - C++ scaling: 2.81584
   - **PERFECT MATCH**

2. **C++ applies bbox preprocessing**
   - Input bbox: (296, 778, 405, 407)
   - C++ preprocessed bbox: (301.938, 782.149, 400.586, 400.585)
   - Bbox is made more square and shifted
   - This preprocessing happens in MTCNN detector pipeline

3. **Python currently uses raw bbox**
   - No bbox preprocessing applied
   - Results in 4-6px initialization offset per landmark

## Resolution Status

**Python init_params() is VERIFIED CORRECT** ✅

The 4-6px initialization difference is due to bbox preprocessing, NOT a bug in the initialization code.

## Next Steps

1. ✅ Verified Python init_params() matches C++ when given same bbox
2. ⏭️ Determine if bbox preprocessing is necessary (might not affect final accuracy)
3. ⏭️ If needed: implement exact C++ bbox preprocessing pipeline
4. ⏭️ Alternative: Use detector-provided bboxes which already include preprocessing
