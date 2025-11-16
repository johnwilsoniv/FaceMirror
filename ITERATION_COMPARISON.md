# Iteration-by-Iteration Comparison: Python vs C++ CLNF

## Test Setup
- Image: `calibration_frames/patient1_frame1.jpg`
- BBox: (301.938, 782.149, 400.586, 400.585) - **EXACT** C++ MTCNN bbox
- Tracked Landmarks: [36, 48, 30, 8]
- Window Sizes: [11, 9, 7]

## CRITICAL FINDING: Divergence at ITER1!

### Initialization (ITER0) - PERFECT MATCH ✅

**Python:**
```
scale: 2.815840
translation: (505.071760, 962.331374)
Landmark_36: (375.4100, 856.8179)
Landmark_48: (423.6482, 1042.9002)
Landmark_30: (508.5262, 938.6374)
Landmark_8: (505.3498, 1179.3236)
```

**C++:**
```
scale: 2.81584
translation: (505.072, 962.331)
Landmark_36: (375.41, 856.818)
Landmark_48: (423.648, 1042.9)
Landmark_30: (508.526, 938.637)
Landmark_8: (505.35, 1179.32)
```

**Difference:** < 0.001 pixels - PERFECT MATCH!

## Window Size = 11 Iterations

### ITER1 - **DIVERGENCE STARTS HERE!** 🔴

| Landmark | Python X | Python Y | C++ X | C++ Y | ΔX | ΔY | Distance |
|----------|----------|----------|-------|-------|----|----|---------||
| 36 | 368.70 | 855.13 | 364.63 | 858.48 | **+4.07** | **-3.35** | **5.29px** |
| 48 | 417.96 | 1045.52 | 414.79 | 1043.00 | **+3.17** | **+2.52** | **4.05px** |
| 30 | 501.84 | 939.57 | 491.85 | 942.61 | **+9.99** | **-3.04** | **10.44px** |
| 8 | 498.02 | 1174.66 | 498.56 | 1176.44 | **-0.54** | **-1.78** | **1.86px** |

**Average Divergence at ITER1: 5.41 pixels**

### ITER2 - Divergence Increases

| Landmark | Python X | Python Y | C++ X | C++ Y | ΔX | ΔY | Distance |
|----------|----------|----------|-------|-------|----|----|---------||
| 36 | 365.33 | 856.10 | 361.50 | 858.58 | **+3.83** | **-2.48** | **4.56px** |
| 48 | 416.67 | 1046.44 | 412.54 | 1043.00 | **+4.13** | **+3.44** | **5.37px** |
| 30 | 493.53 | 939.42 | 487.01 | 943.64 | **+6.52** | **-4.22** | **7.79px** |
| 8 | 495.93 | 1172.62 | 497.29 | 1175.76 | **-1.36** | **-3.14** | **3.42px** |

### ITER3 - Divergence Stabilizes

| Landmark | Python X | Python Y | C++ X | C++ Y | ΔX | ΔY | Distance |
|----------|----------|----------|-------|-------|----|----|---------||
| 36 | 363.04 | 855.37 | 360.63 | 858.55 | **+2.41** | **-3.18** | **3.99px** |
| 48 | 413.71 | 1046.84 | 412.01 | 1042.98 | **+1.70** | **+3.86** | **4.22px** |
| 30 | 492.06 | 941.21 | 485.69 | 943.93 | **+6.37** | **-2.72** | **6.93px** |
| 8 | 491.36 | 1171.03 | 497.12 | 1175.55 | **-5.76** | **-4.52** | **7.33px** |

### ITER4 - Final WS=11

| Landmark | Python X | Python Y | C++ X | C++ Y | ΔX | ΔY | Distance |
|----------|----------|----------|-------|-------|----|----|---------||
| 36 | 361.89 | 856.90 | 360.37 | 858.45 | **+1.52** | **-1.55** | **2.17px** |
| 48 | 412.92 | 1045.88 | 411.89 | 1042.95 | **+1.03** | **+2.93** | **3.10px** |
| 30 | 488.70 | 940.29 | 485.32 | 944.07 | **+3.38** | **-3.78** | **5.08px** |
| 8 | 489.55 | 1170.80 | 497.16 | 1175.48 | **-7.61** | **-4.68** | **8.93px** |

**Average Divergence after WS=11: 4.82 pixels**

## Root Cause Analysis

### Where the Divergence Occurs

Since both implementations:
1. ✅ Start from IDENTICAL initialization (ITER0)
2. 🔴 Diverge immediately at ITER1

The problem MUST be in one of these components executed during ITER1:

1. **Response Map Computation**
   - Patch expert evaluation
   - Warping to reference coordinates
   - Sigma component weighting

2. **Mean-Shift Calculation**
   - KDE computation
   - Gaussian kernel weights
   - Peak finding

3. **Jacobian Computation**
   - Similarity transform derivatives
   - Rotation matrix derivatives

4. **Parameter Update**
   - Hessian matrix construction
   - Linear system solve
   - Manifold-aware rotation update

### Next Steps

1. **Add more granular debug output** to compare:
   - Mean-shift vectors for each landmark
   - Jacobian matrix values
   - Delta parameters before/after update

2. **Compare intermediate values** for a single landmark (e.g., landmark 36):
   - Response map at ITER0
   - Mean-shift value
   - Jacobian row
   - Parameter delta

3. **Identify the exact divergence point** by binary search through the pipeline
