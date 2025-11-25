# OpenFace C++ Setup for Landmark Comparison

This document describes how to set up the OpenFace C++ binary to generate comparable landmark data for validation against the Python pyclnf implementation.

## Overview

The `verify_landmark_accuracy.py` script compares:
- **C++ OpenFace**: FeatureExtraction binary with CLNF landmark detection
- **Python pyclnf**: Pure Python port of OpenFace's CLNF algorithm

## OpenFace Installation

### Prerequisites

```bash
# macOS
brew install cmake boost opencv dlib

# Ubuntu/Debian
sudo apt-get install cmake libboost-all-dev libopencv-dev
```

### Build OpenFace

```bash
# Clone OpenFace
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace

# Download models (required)
./download_models.sh

# Build
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j$(nproc)
```

### Verify Installation

```bash
# Check binary exists
./bin/FeatureExtraction -h
```

## Required Modifications

### 1. No Source Code Modifications Required

The vanilla OpenFace FeatureExtraction binary works for comparison. The Python pyclnf implementation was designed to match OpenFace's default behavior.

### 2. Command Line Flags

For comparable output, use these flags:

```bash
FeatureExtraction -f <video_path> -out_dir <output_dir> -2Dfp
```

| Flag | Description |
|------|-------------|
| `-f` | Input video file |
| `-out_dir` | Output directory for CSV results |
| `-2Dfp` | Export 2D facial landmarks (68 points) |

### 3. Output Format

OpenFace outputs a CSV with columns:
- `frame`: Frame number
- `timestamp`: Time in seconds
- `confidence`: Detection confidence
- `success`: 1 if successful, 0 otherwise
- `x_0` to `x_67`: X coordinates for 68 landmarks
- `y_0` to `y_67`: Y coordinates for 68 landmarks

**Note**: Column names may have leading spaces (` x_0` vs `x_0`). The verification script handles both formats.

## Parameter Alignment

### CLNF Parameters

| Parameter | C++ OpenFace | Python pyclnf | Notes |
|-----------|--------------|---------------|-------|
| Regularization | 35 | 20 | Lower in Python for better convergence |
| Max Iterations | 10 | 10 | Per window size |
| Convergence Threshold | 0.005 | 0.01-0.1 | Python uses higher threshold |
| Window Sizes | [11, 9, 7] | [11, 9, 7] | Same multi-scale approach |
| Sigma (KDE) | 1.5 | 1.75 | Slightly different kernel width |

### Face Detection

| Aspect | C++ OpenFace | Python pyclnf |
|--------|--------------|---------------|
| Detector | Haar Cascade + HOG | PyMTCNN (CoreML/ONNX) |
| Format | Haar returns (x,y,w,h) | MTCNN returns (x,y,w,h) |

**Note**: Different face detectors may produce slightly different bounding boxes, affecting initial landmark positions.

## Known Differences

### 1. Initialization Variance
Face detector differences can cause ~1-2px variance in final landmarks due to different initial bounding boxes.

### 2. Floating Point Precision
Minor numerical differences between C++ double and Python float64 operations.

### 3. Eye Landmark Asymmetry
Python pyclnf shows systematic differences in left vs right eye landmarks:
- Left eye: ~1.42px mean error
- Right eye: ~0.84px mean error

This is under active investigation.

## Running the Comparison

### Basic Usage

```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3"

PYTHONPATH="pyclnf:pymtcnn:pyfaceau:." python3 verify_landmark_accuracy.py \
    "/path/to/video.mov"
```

### With Options

```bash
# Process every 5th frame
python3 verify_landmark_accuracy.py video.mov --skip 5

# Limit to first 100 frames
python3 verify_landmark_accuracy.py video.mov --max-frames 100
```

### Expected Output

```
======================================================================
FULL VIDEO LANDMARK ACCURACY VERIFICATION
======================================================================

Video: Shorty.mov
Size: 1080x1920, Frames: 182, FPS: 59.9
Running C++ FeatureExtraction on video...
  C++ processed 180 frames

Initializing Python pipeline...
Processing frames (skip=1)...
  Frame 50/182 (12.3 fps)
  Frame 100/182 (11.8 fps)
  Frame 150/182 (11.5 fps)

======================================================================
OVERALL LANDMARK ACCURACY (68 points)
======================================================================
Mean error:   0.76px
Median error: 0.73px
Max error:    1.82px

ACCURACY ASSESSMENT
Overall accuracy: EXCELLENT (0.76px mean error)
Eye accuracy:     GOOD (1.13px mean error)
```

## Accuracy Thresholds

| Rating | Mean Error | Description |
|--------|------------|-------------|
| EXCELLENT | < 2.0px | Production ready |
| GOOD | 2.0 - 5.0px | Acceptable for most uses |
| MODERATE | 5.0 - 10.0px | May need investigation |
| POOR | > 10.0px | Significant issues |

## Troubleshooting

### OpenFace Binary Not Found

Update the path in `verify_landmark_accuracy.py`:

```python
cmd = [
    '/path/to/OpenFace/build/bin/FeatureExtraction',  # Update this
    '-f', video_path,
    '-out_dir', out_dir,
    '-2Dfp'
]
```

### No Faces Detected

- Ensure video has clear, frontal faces
- Check lighting conditions
- Try different detection thresholds

### Large Discrepancies

If mean error > 5px:
1. Verify both use same video file
2. Check frame synchronization
3. Ensure OpenFace models are downloaded
4. Compare bounding boxes from face detectors

## Advanced: Modifying OpenFace

If you need to modify OpenFace for debugging:

### Export Additional Data

Edit `FaceAnalyser.cpp` to output intermediate values:
- Response maps
- PDM parameters
- Jacobian matrices

### Match Python Exactly

To match Python pyclnf exactly, you would need to:
1. Use same face detector (MTCNN)
2. Set regularization=20
3. Set convergence_threshold=0.01

However, this defeats the purpose of comparison testing.

## Files Reference

| File | Purpose |
|------|---------|
| `verify_landmark_accuracy.py` | Main comparison script |
| `pyclnf/pyclnf/clnf.py` | Python CLNF implementation |
| `pyclnf/pyclnf/core/pdm.py` | Point Distribution Model |
| `pyclnf/pyclnf/core/patch_expert.py` | Patch expert inference |
| `PYCLNF_ACCURACY_REPORT.md` | Detailed accuracy analysis |

## Contact

For issues with the Python implementation, check the pyclnf repository.
For OpenFace issues, see: https://github.com/TadasBaltrusaitis/OpenFace
