# PyFace - Organization Plan for Publication

**Goal:** Rename and organize the Full Python AU Pipeline for publication as "PyFace"

---

## Recommended Structure

```
pyface/
├── pyface/                      # Main package
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline class (renamed from full_python_au_pipeline.py)
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── retinaface.py        # Face detection (from onnx_retinaface_detector.py)
│   │   ├── pfld.py              # Landmark detection (from cunjian_pfld_detector.py)
│   │   └── face_aligner.py      # Face alignment (from openface22_face_aligner.py)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pdm.py               # PDM shape model (from pdm_parser.py)
│   │   ├── calc_params.py       # Pose estimation (from calc_params.py)
│   │   └── au_predictor.py      # AU SVR models (from openface22_au_predictor.py)
│   ├── features/
│   │   ├── __init__.py
│   │   └── hog.py               # HOG extraction (from fhog_extractor.py)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── running_median.py    # Running median tracker
│   │   └── triangulation.py     # Triangulation parser
│   └── config.py                # Configuration
│
├── weights/                     # Model weights
│   ├── retinaface_mobilenet025_coreml.onnx
│   ├── pfld_cunjian.onnx
│   ├── In-the-wild_aligned_PDM_68.txt
│   ├── tris_68_full.txt
│   └── AU_predictors/           # 17 SVR models
│
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── coreml_acceleration.py
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_detectors.py
│   └── test_au_prediction.py
│
├── docs/
│   ├── README.md
│   ├── API.md
│   ├── PERFORMANCE.md
│   └── COREML.md
│
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Key Files to Rename/Organize

### Core Pipeline
- `full_python_au_pipeline.py` → `pyface/pipeline.py`
  - Rename class: `FullPythonAUPipeline` → `PyFace` or `AUPipeline`

### Detectors
- `onnx_retinaface_detector.py` → `pyface/detectors/retinaface.py`
- `cunjian_pfld_detector.py` → `pyface/detectors/pfld.py`
- `openface22_face_aligner.py` → `pyface/detectors/face_aligner.py`

### Models
- `pdm_parser.py` → `pyface/models/pdm.py`
- `calc_params.py` → `pyface/models/calc_params.py`
- `openface22_au_predictor.py` → `pyface/models/au_predictor.py`
- `openface22_hog_parser.py` → `pyface/models/hog_parser.py`

### Features
- `fhog_extractor.py` → `pyface/features/hog.py`

### Utils
- `running_median_tracker.py` → `pyface/utils/running_median.py`
- `triangulation_parser.py` → `pyface/utils/triangulation.py`

### Delete (Test/Debug Files)
All the investigation/debug files from our session:
- `test_*.py` (all test files)
- `debug_*.py`
- `analyze_*.py`
- `compare_*.py`
- `diagnose_*.py`
- All markdown documentation files from debugging
- `calc_params_tool.cpp` (C++ test tool)

---

## Main API Design

### Simple API
```python
import pyface

# Basic usage
pipeline = pyface.AUPipeline(use_coreml=True)
results = pipeline.process_video('video.mp4')

# Results DataFrame
print(results[['frame', 'timestamp', 'AU01_r', 'AU06_r', 'AU12_r']])
```

### Advanced API
```python
import pyface

# Custom configuration
pipeline = pyface.AUPipeline(
    detector='retinaface',      # or 'mtcnn', custom
    landmark_model='pfld',       # or 'fan', '98pt'
    use_coreml=True,            # Apple Silicon acceleration
    use_calc_params=True,       # Accurate pose estimation
    num_aus=17,                 # Number of AUs to extract
    verbose=True
)

# Process with options
results = pipeline.process_video(
    'video.mp4',
    output_csv='results.csv',
    batch_size=32,
    skip_frames=0,              # Optional frame skipping
    progress_callback=my_callback
)
```

---

## Documentation Structure

### README.md
```markdown
# PyFace - Pure Python Action Unit Extraction

Fast, accurate facial action unit (AU) extraction using OpenFace 2.2 models with optional CoreML acceleration.

## Features
- 17 Action Units (FACS-based)
- CoreML acceleration on Apple Silicon (3-4x speedup)
- Pure Python implementation
- No C++ dependencies
- Easy to use API

## Installation
pip install pyface

## Quick Start
import pyface
pipeline = pyface.AUPipeline(use_coreml=True)
results = pipeline.process_video('video.mp4')

## Performance
- CPU: ~530ms/frame (1.9 FPS)
- CoreML (M1/M2/M3): ~160ms/frame (6.2 FPS)
- 17 AU intensities per frame
```

### API.md
- Complete API reference
- All classes and methods
- Parameters and return types
- Examples for each method

### PERFORMANCE.md
- Benchmark results
- Comparison with C++ OpenFace
- Optimization tips
- Hardware recommendations

### COREML.md
- CoreML acceleration guide
- macOS requirements
- Performance expectations
- Troubleshooting

---

## Package Metadata (setup.py)

```python
from setuptools import setup, find_packages

setup(
    name='pyface',
    version='1.0.0',
    description='Pure Python Action Unit Extraction with OpenFace 2.2',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/pyface',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
        'pandas>=1.3.0',
        'torch>=1.9.0',
        'onnxruntime>=1.11.0',  # CPU
        'onnxruntime-coreml>=1.13.0; platform_system=="Darwin"',  # macOS
        'scipy>=1.7.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'coreml': ['onnxruntime-coreml>=1.13.0'],
        'dev': ['pytest', 'black', 'flake8'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='face action-units facs openface computer-vision',
    include_package_data=True,
    package_data={
        'pyface': [
            'weights/*.onnx',
            'weights/*.txt',
            'weights/AU_predictors/*.dat',
        ],
    },
)
```

---

## Migration Steps

### Phase 1: Reorganize (1 hour)
1. Create `pyface/` directory structure
2. Move and rename core files
3. Update imports
4. Delete test/debug files

### Phase 2: Clean API (30 min)
1. Rename `FullPythonAUPipeline` → `AUPipeline`
2. Simplify `__init__` parameters
3. Add convenience methods
4. Create `pyface/__init__.py` with clean exports

### Phase 3: Documentation (1 hour)
1. Write README.md with examples
2. Create API.md reference
3. Document performance characteristics
4. Add usage examples

### Phase 4: Package (30 min)
1. Create setup.py
2. Add requirements.txt
3. Test installation
4. Create distribution

### Phase 5: Publish (optional)
1. GitHub repository
2. PyPI upload
3. Documentation site
4. Example notebooks

---

## Comparison: S1 vs PyFace

### S1 (OpenFace 3.0 ONNX)
**Pros:**
- Very fast (~35ms/frame, 28 FPS)
- Modern neural network architecture
- Integrated multitask model

**Cons:**
- Requires specific ONNX models
- Less interpretable (black box NN)
- Harder to customize AU extraction

### PyFace (OpenFace 2.2 Python)
**Pros:**
- Pure Python (easy to modify)
- Based on established OpenFace 2.2
- SVR models are interpretable
- Good accuracy (proven FACS research)
- No C++ compilation needed

**Cons:**
- Slower than S1 (~160ms/frame vs ~35ms)
- More complex pipeline
- Older architecture (SVR vs NN)

### Recommendation
**Use S1 when:**
- Speed is critical
- Standard AU extraction is sufficient
- Running production systems

**Use PyFace when:**
- Need pure Python implementation
- Want to customize AU extraction
- Research/academic use
- No C++ toolchain available
- Need interpretable models

---

## Next Steps

1. **Decision:** Should we create PyFace as a separate package or integrate into S1?

2. **Naming:**
   - `pyface` (simple, clean)
   - `pyface-au` (more specific)
   - `openface-python` (clear heritage)
   - Your preference?

3. **Scope:**
   - Just AU extraction?
   - Include face detection/landmarks?
   - Video processing utilities?

4. **Target Audience:**
   - Researchers?
   - Developers?
   - Both?

Let me know your thoughts and I can start the reorganization!
