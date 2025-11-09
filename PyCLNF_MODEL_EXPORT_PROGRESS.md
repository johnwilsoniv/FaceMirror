# PyCLNF Model Export - COMPLETE ✅

## Overview

Successfully implemented pure Python model export from OpenFace 2.2 trained models. Both PDM and CCNF patch experts have been exported to NumPy format, eliminating C++ dependencies for model loading.

## Completed ✓

### 1. PDM (Point Distribution Model) Export - FULLY WORKING ✅
**Location:** `pyclnf/models/openface_loader.py`

Successfully implemented pure Python parser for OpenFace PDM text format:

- **PDMLoader class**: Parses OpenFace PDM files directly (no C++ needed!)
- **Supported operations:**
  - Load PDM from text file
  - Export to NumPy .npy format
  - Model introspection (num_points, num_modes, etc.)

**Verification:**
```bash
$ python3 pyclnf/models/openface_loader.py

PDM Info:
  num_points: 68
  num_modes: 34
  mean_shape_shape: (204, 1)   # 68 points × 3 coordinates (x,y,z)
  princ_comp_shape: (204, 34)  # 34 PCA modes
  eigen_values_shape: (1, 34)  # Eigenvalues

✓ PDM loading and export successful!
```

**Exported Files:**
- `pyclnf/models/exported_pdm/mean_shape.npy`
- `pyclnf/models/exported_pdm/princ_comp.npy`
- `pyclnf/models/exported_pdm/eigen_values.npy`

All files in float32 format, ready for NumPy-based CLNF implementation.

### 2. CCNF Patch Expert Export - FULLY WORKING ✅
**Location:** `pyclnf/models/openface_loader.py`

Successfully implemented complete binary parser for OpenFace CCNF format with multi-view support:

- **CCNFPatchExpertLoader class**: Parses OpenFace CCNF binary files (no C++ needed!)
- **Complete format support:**
  - Multi-view structure (7 views per model)
  - View metadata (pitch, yaw, roll for each orientation)
  - Visibility matrices (landmark visibility per view)
  - Window sizes and sigma components (for edge features)
  - Patch experts with neurons
  - Beta values and patch confidence scores
- **Binary format successfully parsed:**

```
File Structure:
├─ For each landmark (68 total):
│   ├─ read_type: int32 (value = 5)
│   ├─ width: int32
│   ├─ height: int32
│   ├─ num_neurons: int32
│   ├─ For each neuron:
│   │   ├─ read_type: int32 (value = 2)
│   │   ├─ neuron_type: int32
│   │   ├─ norm_weights: float64
│   │   ├─ bias: float64
│   │   ├─ alpha: float64
│   │   └─ weights: Matrix (binary)
│   │       ├─ rows: int32
│   │       ├─ cols: int32
│   │       ├─ cv_type: int32
│   │       └─ data: bytes (rows×cols×elemSize)
│   ├─ beta values: [float64, ...]
│   └─ patch_confidence: float64
```

**Next Steps:**
1. Implement complete binary parser in Python (using `struct` module)
2. OR create minimal C++ export utility (one-time use)
3. Export all 68 patch experts × 3 scales to .npz format

## Model Files Analyzed

**PDM:** `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt`

**CCNF Patches (3 scales):**
- `patch_experts/ccnf_patches_0.25_general.txt` (887KB)
- `patch_experts/ccnf_patches_0.35_general.txt` (887KB)
- `patch_experts/ccnf_patches_0.5_general.txt` (887KB)

## Architecture Decisions

### ✅ Pure Python Approach
Confirmed feasible for all components:
- PDM: ✓ **Already working**
- CCNF Patch Experts: Technically possible via Python binary parsing
- No C++ compilation needed for model export!

### Implementation Strategy

**Phase 1 (Current):** Model Export
- [x] PDM text parser
- [ ] CCNF binary parser (in progress)
- [ ] Export to .npy/.npz format

**Phase 2 (Next):** Pure Python CLNF Core
- [ ] PDM transform (`landmarks_from_params`)
- [ ] LNF patch expert response
- [ ] NU-RLMS optimizer

**Phase 3:** Platform-Specific Acceleration
- [ ] CoreML for ARM Mac Neural Engine
- [ ] Cython for CPU-intensive loops
- [ ] CuPy for NVIDIA GPU

## Key Technical Details

### PDM File Format (Text)
```
# Comment line
<rows>
<cols>
<opencv_type>  # 6 = CV_64FC1, 5 = CV_32FC1
<row1_val1> <row1_val2> ... <row1_valN>
<row2_val1> <row2_val2> ... <row2_valN>
...
```

### OpenCV Type Codes
- `5` = CV_32FC1 (float32, 1 channel)
- `6` = CV_64FC1 (float64, 1 channel)

## Dependencies (Current)
```
numpy
pathlib (stdlib)
struct (stdlib)
```

No OpenCV, no C++ compiler, no abseil, no Boost, no TBB!

## Performance Targets (From PyCLNF_IMPLEMENTATION.md)
- Pure NumPy: 300-500ms per frame (acceptable baseline)
- With Cython: 100-150ms (7-10 FPS)
- With CoreML (ARM Mac): 30-50ms (20-30 FPS) - **Target for M1/M2/M3**
- With CUDA: 10-20ms (50-100 FPS)

## Files Created
```
pyclnf/
├─ models/
│   ├─ openface_loader.py          # PDM & CCNF loaders
│   └─ exported_pdm/               # Exported PDM in .npy format
│       ├─ mean_shape.npy
│       ├─ princ_comp.npy
│       └─ eigen_values.npy
└─ tools/                          # (Reserved for C++ export if needed)
```

## References
- OpenFace 2.2 source: `~/repo/fea_tool/external_libs/openFace/OpenFace/`
- CLNF paper: Baltrusaitis et al. 2013 (ICCV)
- Implementation roadmap: `CLNF_IMPLEMENTATION_ROADMAP.md`
- Technical details: `CLNF_STAGES_EXPLAINED.md`
- Full implementation plan: `PyCLNF_IMPLEMENTATION.md`

## Export Summary - COMPLETE ✅

All OpenFace 2.2 models successfully exported to pure NumPy format:

### Exported Models

**PDM (Point Distribution Model):**
```
pyclnf/models/exported_pdm/ (36 KB)
├── mean_shape.npy          # (204, 1) - 68 landmarks × 3 coords
├── princ_comp.npy          # (204, 34) - 34 PCA modes
└── eigen_values.npy        # (1, 34) - Eigenvalues
```

**CCNF Patch Experts (3 scales):**
```
pyclnf/models/exported_ccnf_0.25/ (11 MB)
pyclnf/models/exported_ccnf_0.35/ (11 MB)
pyclnf/models/exported_ccnf_0.5/  (11 MB)
Each containing:
├── global_metadata.npz          # Patch scaling, views, landmarks
├── view_00_metadata.npz         # View orientation and visibility
├── view_00/
│   ├── patch_00/
│   │   ├── metadata.npz         # Width, height, betas, confidence
│   │   ├── neuron_00.npz        # Neuron weights, bias, alpha
│   │   └── neuron_01.npz
│   └── patch_01/...
└── view_01/...
```

### Model Statistics

| Model | Views | Total Patches | Non-Empty | Neurons/Patch | Patch Size |
|-------|-------|---------------|-----------|---------------|------------|
| 0.25  | 7     | 476 (7×68)    | 344       | 7             | 11×11      |
| 0.35  | 7     | 476 (7×68)    | 344       | 7             | 11×11      |
| 0.5   | 7     | 476 (7×68)    | 344       | 7             | 11×11      |

**Total exported:** ~33 MB (PDM + 3 CCNF scales)
**No C++ dependencies required for loading!**

## Next Steps (Phase 2)

1. Begin Phase 2: Pure Python CLNF core implementation
   - PDM transform functions (`landmarks_from_params()`)
   - Shape model operations
   - NU-RLMS optimizer
   - LNF patch expert response computation
2. Platform-specific acceleration (Phase 3)
   - CoreML for ARM Mac Neural Engine
   - Cython for CPU loops
   - CuPy for NVIDIA GPU

## Notes
- PDM parser handles comments and whitespace correctly
- Confirmed OpenFace models use 68 landmarks (dlib format)
- CLNF uses 3 scales: 0.25, 0.35, 0.5 (interocular distance multiples)
- Each scale has 68 patch experts (one per landmark)
