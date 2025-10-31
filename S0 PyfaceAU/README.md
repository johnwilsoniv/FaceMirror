# PyfaceAU - Pure Python OpenFace 2.2 AU Extraction

**A complete Python implementation of OpenFace 2.2's Facial Action Unit (AU) extraction pipeline.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

PyfaceAU is a pure Python reimplementation of the [OpenFace 2.2](https://github.com/TadasBaltrusaitis/OpenFace) Facial Action Unit extraction pipeline. It achieves **r > 0.83 correlation** with the original C++ implementation while requiring **zero compilation** and running on any platform.

### Key Features

- **🐍 100% Python** - No C++ compilation required
- **📦 Easy Installation** - `pip install` and go
- **🎯 High Accuracy** - r=0.83 overall, r=0.94 for static AUs
- **⚡ Optimized** - 4.6 FPS with CoreML acceleration on Mac
- **🔧 Modular** - Use individual components independently
- **📊 17 Action Units** - Full AU extraction (AU01, AU02, AU04, etc.)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourname/pyfaceau.git
cd pyfaceau

# Install dependencies
pip install -r requirements.txt

# Install PyFHOG (required for HOG features)
pip install pyfhog
```

### Basic Usage

```python
from pyfaceau import FullPythonAUPipeline

# Initialize pipeline
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
    au_models_dir='path/to/AU_predictors',
    triangulation_file='weights/tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,  # macOS only
    verbose=False
)

# Process video
results = pipeline.process_video(
    video_path='input.mp4',
    output_csv='results.csv'
)

# Access AU predictions
print(results[['frame', 'AU01_r', 'AU06_r', 'AU12_r']])
```

### Example Output

```csv
frame,success,AU01_r,AU02_r,AU04_r,AU06_r,AU12_r,...
0,True,0.60,0.90,0.00,1.23,2.45,...
1,True,0.55,0.85,0.00,1.20,2.50,...
```

---

## Architecture

PyfaceAU replicates the complete OpenFace 2.2 AU extraction pipeline:

```
Video Input
    ↓
Face Detection (RetinaFace ONNX)
    ↓
Landmark Detection (PFLD 68-point)
    ↓
3D Pose Estimation (CalcParams - 99.45% accuracy)
    ↓
Face Alignment (OpenFace22 aligner)
    ↓
HOG Feature Extraction (PyFHOG - r=1.0)
    ↓
Geometric Features (PDM reconstruction)
    ↓
Running Median Tracking (Cython-optimized)
    ↓
AU Prediction (17 SVR models)
    ↓
Output: 17 AU intensities
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed component descriptions.

---

## Performance

### Accuracy (vs OpenFace C++ 2.2)

| Metric | Correlation (r) |
|--------|-----------------|
| **Overall** | **0.83** |
| Static AUs (6) | 0.94 |
| Dynamic AUs (11) | 0.77 |
| Best AU (AU12) | 0.99 |

See [docs/CPP_VS_PYTHON.md](docs/CPP_VS_PYTHON.md) for detailed comparison.

### Speed

| Configuration | FPS | Per Frame |
|---------------|-----|-----------|
| CPU Mode | 1.9 | 531ms |
| CoreML + Tracking | 4.6 | 217ms |
| C++ OpenFace 2.2 | 32.9 | 30ms |

**Note:** PyfaceAU prioritizes ease-of-use and portability over raw speed.

---

## Supported Action Units

PyfaceAU extracts 17 Facial Action Units:

**Dynamic AUs (11):**
- AU01 - Inner Brow Raiser
- AU02 - Outer Brow Raiser
- AU05 - Upper Lid Raiser
- AU09 - Nose Wrinkler
- AU15 - Lip Corner Depressor
- AU17 - Chin Raiser
- AU20 - Lip Stretcher
- AU23 - Lip Tightener
- AU25 - Lips Part
- AU26 - Jaw Drop
- AU45 - Blink

**Static AUs (6):**
- AU04 - Brow Lowerer
- AU06 - Cheek Raiser
- AU07 - Lid Tightener
- AU10 - Upper Lip Raiser
- AU12 - Lip Corner Puller
- AU14 - Dimpler

---

## Requirements

### Python Dependencies

```
python >= 3.10
numpy >= 1.20.0
opencv-python >= 4.5.0
pandas >= 1.3.0
scipy >= 1.7.0
onnxruntime >= 1.10.0
pyfhog >= 0.1.0
```

### Model Files

Download OpenFace 2.2 AU predictor models:
- Available from: [OpenFace repository](https://github.com/TadasBaltrusaitis/OpenFace)
- Place in: `AU_predictors/` directory
- Required: 17 `.dat` files (AU_1_dynamic_intensity_comb.dat, etc.)

---

## Project Structure

```
S0 PyfaceAU/
├── pyfaceau/                  # Core library
│   ├── pipeline.py            # Full AU extraction pipeline
│   ├── detectors/             # Face and landmark detection
│   ├── alignment/             # Face alignment and pose estimation
│   ├── features/              # HOG and geometric features
│   ├── prediction/            # AU prediction and running median
│   └── utils/                 # Utilities and Cython extensions
├── weights/                   # Model weights
├── tests/                     # Test suite
├── examples/                  # Usage examples
└── docs/                      # Documentation
```

---

## Advanced Usage

### Process Single Frame

```python
from pyfaceau import FullPythonAUPipeline
import cv2

pipeline = FullPythonAUPipeline(...)

# Read frame
frame = cv2.imread('image.jpg')

# Process (requires landmarks and pose from CSV or detector)
aligned = pipeline.aligner.align_face(frame, landmarks, tx, ty, rz)
hog_features = pipeline.extract_hog(aligned)
aus = pipeline.predict_aus(hog_features, geom_features)
```

### Use Individual Components

```python
# Face detection only
from pyfaceau.detectors import ONNXRetinaFaceDetector
detector = ONNXRetinaFaceDetector('weights/retinaface_mobilenet025_coreml.onnx')
faces = detector.detect_faces(frame)

# Landmark detection only
from pyfaceau.detectors import CunjianPFLDDetector
landmarker = CunjianPFLDDetector('weights/pfld_cunjian.onnx')
landmarks, conf = landmarker.detect_landmarks(frame, bbox)

# Face alignment only
from pyfaceau.alignment import OpenFace22FaceAligner
aligner = OpenFace22FaceAligner('weights/In-the-wild_aligned_PDM_68.txt')
aligned = aligner.align_face(frame, landmarks, tx, ty, rz)
```

---

## Citation

If you use PyfaceAU in your research, please cite:

```bibtex
@software{pyfaceau2025,
  title={PyfaceAU: Pure Python OpenFace 2.2 AU Extraction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourname/pyfaceau}
}
```

Also cite the original OpenFace:

```bibtex
@inproceedings{baltrusaitis2018openface,
  title={OpenFace 2.0: Facial behavior analysis toolkit},
  author={Baltru{\v{s}}aitis, Tadas and Zadeh, Amir and Lim, Yao Chong and Morency, Louis-Philippe},
  booktitle={2018 13th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2018)},
  pages={59--66},
  year={2018},
  organization={IEEE}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **OpenFace** - Original C++ implementation by Tadas Baltrusaitis
- **PyFHOG** - HOG feature extraction library
- **RetinaFace** - Face detection model
- **PFLD** - Landmark detection by Cunjian Chen

---

## Support

- **Issues:** https://github.com/yourname/pyfaceau/issues
- **Documentation:** [docs/](docs/)
- **Examples:** [examples/](examples/)

---

**Built with ❤️ for the facial behavior research community**
