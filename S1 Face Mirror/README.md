# S1 Face Mirror - Facial Video Processing with PyFaceAU

Processes facial videos to extract Action Units (AUs) and create mirrored comparison videos for facial asymmetry analysis. Built on PyFaceAU, a pure Python implementation of OpenFace 2.2.

## Quick Start

### Run Application

```bash
# GUI mode (default)
python main.py

# Batch mode (process entire directory)
python main.py --batch --mode 1 --input /path/to/videos --output /path/to/output
```

### Basic Workflow

1. Launch application (GUI appears automatically)
2. Select processing mode:
   - Mode 1: Face mirroring + AU extraction
   - Mode 2: Face mirroring only
   - Mode 3: AU extraction only
3. Choose input directory containing videos
4. Processing begins automatically
5. Results saved to `~/Documents/SplitFace/S1O Processed Files/`

## Features

### Core Capabilities

- **PyFaceAU AU Extraction**: 17 Action Units with r=0.864 correlation to OpenFace 2.2
- **Face Mirroring**: Left/right split-face video generation for asymmetry analysis
- **Patient-Based Processing**: Automatically groups and processes videos by patient ID
- **Batch Processing**: Process entire directories with progress tracking
- **Existing Output Detection**: Skip already-processed videos
- **High Performance**: 22.5 FPS face mirroring, 60.8 FPS AU extraction (Apple Silicon)

### Advanced Features

- **CLNF Landmark Refinement**: Improves eyebrow landmark accuracy by 7.4%
- **Multi-threaded Processing**: Parallel frame processing (configurable workers)
- **Memory Management**: Automatic garbage collection and memory checkpointing
- **Video Rotation Detection**: Auto-corrects video orientation
- **Progress Window**: Real-time status, FPS metrics, and ETA
- **Performance Profiling**: Optional detailed timing analysis

## Processing Modes

### Mode 1: Full Pipeline (Face Mirroring + AU Extraction)
Processes videos through complete pipeline:
1. Extract frames from video
2. Detect faces and landmarks (PyFaceAU)
3. Apply CLNF refinement
4. Create left/right mirrored videos
5. Extract 17 Action Units per side
6. Save CSVs and mirrored videos

**Output:**
- `~/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/`
  - `[patient_id]_source.mp4` (mirrored comparison video)
- `~/Documents/SplitFace/S1O Processed Files/Combined Data/`
  - `[patient_id]_left_mirrored.csv` (AU intensities)
  - `[patient_id]_right_mirrored.csv` (AU intensities)

### Mode 2: Face Mirroring Only
Creates mirrored videos without AU extraction (faster):
- Useful for visual inspection
- ~22.5 FPS processing speed
- Only generates mirrored video files

**Output:**
- `~/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/`
  - `[patient_id]_source.mp4`

### Mode 3: AU Extraction Only
Extracts AUs without creating mirrored videos:
- Useful when videos already exist
- ~60.8 FPS processing speed
- Generates CSV files only

**Output:**
- `~/Documents/SplitFace/S1O Processed Files/Combined Data/`
  - `[patient_id]_left_mirrored.csv`
  - `[patient_id]_right_mirrored.csv`

## Output Format

### CSV Structure

Each CSV contains per-frame AU intensities:

```csv
frame,success,confidence,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r
0,True,0.95,0.60,0.90,0.00,0.12,1.23,0.45,0.00,0.33,2.45,0.10,0.05,0.20,0.15,0.08,1.10,0.40,0.00
1,True,0.96,0.55,0.85,0.00,0.15,1.20,0.42,0.00,0.30,2.50,0.12,0.06,0.18,0.12,0.10,1.15,0.38,0.00
```

**Columns:**
- `frame`: Frame number
- `success`: Detection success flag
- `confidence`: Face detection confidence
- `AU##_r`: Action Unit intensity (0.0-5.0 scale)

### Supported Action Units

**Upper Face:**
- AU01 (Inner Brow Raiser)
- AU02 (Outer Brow Raiser)
- AU04 (Brow Lowerer)
- AU05 (Upper Lid Raiser)
- AU07 (Lid Tightener)
- AU45 (Blink)

**Mid Face:**
- AU06 (Cheek Raiser)
- AU09 (Nose Wrinkler)
- AU10 (Upper Lip Raiser)

**Lower Face:**
- AU12 (Lip Corner Puller)
- AU14 (Dimpler)
- AU15 (Lip Corner Depressor)
- AU17 (Chin Raiser)
- AU20 (Lip Stretcher)
- AU23 (Lip Tightener)
- AU25 (Lips Part)
- AU26 (Jaw Drop)

## Requirements

### System Requirements

**For End Users (Bundled App):**
- macOS 11.0+ (Big Sur or later)
- Apple Silicon (M1, M2, M3, M4) recommended
- 8 GB RAM (16 GB recommended)
- 10 GB free storage

**For Developers:**
- Python 3.10 or 3.11
- macOS (Apple Silicon recommended)
- 16+ GB RAM

### Dependencies

Core libraries:
- `pyfaceau` - PyFaceAU for AU extraction
- `opencv-python` - Video processing
- `numpy` - Array operations
- `pandas` - CSV output
- `torch` - GPU acceleration (if available)
- `onnxruntime` - ONNX model inference
- `psutil` - Memory monitoring
- `tqdm` - Progress bars

See main project README for installation instructions.

## Configuration

All settings are in `config.py`. Key parameters:

### Performance Settings

```python
NUM_THREADS = 6                    # Parallel processing workers (4-6 recommended)
BATCH_SIZE = 100                   # Frames per batch (100 for 16GB, 50 for 8GB)
MEMORY_CHECKPOINT_INTERVAL = 10    # Deep cleanup interval (every N videos)
PROGRESS_UPDATE_INTERVAL = 50      # Console update interval (frames)
```

### AU Detection Settings

```python
ENABLE_AU45_CALCULATION = True     # Enable blink detection
                                   # True: ~2-3 FPS (accurate blinks)
                                   # False: ~14-28 FPS (5-7x faster)

CONFIDENCE_THRESHOLD = 0.5         # Face detection threshold (0.0-1.0)
NMS_THRESHOLD = 0.4                # Duplicate face suppression
VIS_THRESHOLD = 0.5                # Minimum face visibility
```

### Threading Configuration

```python
OMP_NUM_THREADS = 2                # OpenMP threads
MKL_NUM_THREADS = 2                # Intel MKL threads
OPENBLAS_NUM_THREADS = 2           # OpenBLAS threads
VECLIB_MAXIMUM_THREADS = 2         # macOS Accelerate
NUMEXPR_NUM_THREADS = 2            # NumExpr threads
```

### Performance Profiling

```python
ENABLE_PROFILING = False           # Detailed timing reports
PROFILING_OUTPUT_DIR = None        # None = Desktop
```

### Logging

```python
LOG_LEVEL = "INFO"                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = None                    # None = console only
```

## Performance Tuning

### Memory Optimization

**For 8GB RAM:**
```python
BATCH_SIZE = 50
MEMORY_CHECKPOINT_INTERVAL = 5
NUM_THREADS = 4
```

**For 16GB+ RAM:**
```python
BATCH_SIZE = 100
MEMORY_CHECKPOINT_INTERVAL = 10
NUM_THREADS = 6
```

### Speed vs. Accuracy Tradeoffs

**Maximum Speed (no blink detection):**
- `ENABLE_AU45_CALCULATION = False`
- Expected: ~14-28 FPS (5-7x speedup)
- Use when blink analysis not needed

**High Accuracy (with blink detection):**
- `ENABLE_AU45_CALCULATION = True`
- Expected: ~2-3 FPS
- Requires landmark detection every frame

### Device Selection

Application auto-detects best device:
1. CUDA (NVIDIA GPU) - if available
2. CPU + CoreML (Apple Silicon) - M1/M2/M3 Macs
3. CPU + ONNX (Intel) - Intel Macs

Force specific device:
```python
FORCE_DEVICE = 'cpu'  # or 'cuda' or None (auto)
```

## Architecture

### Processing Pipeline

```
Video Input
    ↓
Video Rotation Correction
    ↓
Frame Extraction (batched)
    ↓
Face Detection (RetinaFace)
    ↓
Landmark Detection (PFLD 68-point)
    ↓
CLNF Refinement (targeted)
    ↓
Face Splitting (left/right)
    ↓
Video Encoding (mirrored output)
    ↓
AU Extraction (PyFaceAU)
    ↓
CSV Output (per side)
```

### File Structure

```
S1 Face Mirror/
├── main.py                      # Entry point and GUI
├── config.py                    # Configuration settings
├── config_paths.py              # Path management
├── video_processor.py           # Core processing logic
├── pyfaceau_detector.py         # PyFaceAU integration
├── face_splitter.py             # Face splitting/mirroring
├── face_mirror.py               # Video mirroring logic
├── video_rotation.py            # Rotation detection
├── progress_window.py           # Progress GUI
├── performance_profiler.py      # Timing analysis
├── au45_calculator.py           # Blink detection
├── logger.py                    # Logging utilities
├── native_dialogs.py            # macOS file dialogs
├── splash_screen.py             # Startup screen
└── weights/                     # PyFaceAU model files
```

## Batch Processing

### Command Line Interface

```bash
# Process all videos in directory
python main.py --batch --mode 1 \
    --input ~/Videos/patients \
    --output ~/Documents/SplitFace/S1O\ Processed\ Files

# Face mirroring only
python main.py --batch --mode 2 \
    --input ~/Videos/patients \
    --output ~/Documents/SplitFace/S1O\ Processed\ Files

# AU extraction only
python main.py --batch --mode 3 \
    --input ~/Videos/patients \
    --output ~/Documents/SplitFace/S1O\ Processed\ Files
```

### Progress Tracking

The application automatically:
- Detects existing output files
- Skips already-processed videos
- Groups videos by patient ID
- Shows real-time FPS and ETA
- Reports memory usage

### Patient Grouping

Videos are grouped by patient ID extracted from filename:
- `patient123_action1.mp4` → Patient ID: `patient123`
- `patient123_action2.mp4` → Same patient
- `patient456_action1.mp4` → Different patient

All videos for a patient are processed together, with outputs organized by patient ID.

## Troubleshooting

### Memory Issues

**Symptoms:** Slow processing, system swap usage
**Solutions:**
- Reduce `BATCH_SIZE` to 50 or lower
- Reduce `NUM_THREADS` to 4 or lower
- Lower `MEMORY_CHECKPOINT_INTERVAL` to 5
- Close other applications
- Process fewer videos at once

### Low FPS

**With AU45 enabled:** 2-3 FPS is expected
**Without AU45:** Should achieve 14-28 FPS

**Solutions:**
- Disable AU45: `ENABLE_AU45_CALCULATION = False`
- Increase `NUM_THREADS` (if CPU cores available)
- Ensure running on Apple Silicon (M-series Mac)
- Check no other processes consuming CPU

### Face Detection Failures

**Symptoms:** "No face detected" warnings
**Solutions:**
- Lower `CONFIDENCE_THRESHOLD` to 0.3-0.4
- Check video quality and lighting
- Verify face is clearly visible and frontal
- Check video isn't corrupted

### Video Won't Process

**Symptoms:** Crashes, encoding errors
**Solutions:**
- Check video codec (MP4/H.264 recommended)
- Verify video file isn't corrupted
- Ensure sufficient disk space
- Check video has valid frames

### GUI Not Appearing

**Symptoms:** Application starts but no window
**Solutions:**
- Check tkinter is installed: `python -m tkinter`
- Verify Python has GUI permissions on macOS
- Try batch mode: `python main.py --batch --mode 1`

## Performance Metrics

Based on validation testing on Apple Silicon:

| Operation | Speed | Notes |
|-----------|-------|-------|
| Face Mirroring | 22.5 FPS | 6-thread parallel processing |
| AU Extraction | 60.8 FPS | Single-threaded, CoreML accelerated |
| Full Pipeline (AU45 disabled) | 14-28 FPS | Typical usage |
| Full Pipeline (AU45 enabled) | 2-3 FPS | Landmark detection every frame |

**Accuracy:** r = 0.864 mean correlation with OpenFace 2.2 (see VALIDATION.md)

## Citation

If you use S1 Face Mirror in your research, please cite:

```bibtex
@article{wilson2025splitface,
  title={A Split-Face Computer Vision/Machine Learning Assessment of Facial Paralysis Using Facial Action Units},
  author={Wilson IV, John and Rosenberg, Joshua and Gray, Mingyang L and Razavi, Christopher R},
  journal={Facial Plastic Surgery \& Aesthetic Medicine},
  year={2025},
  publisher={Mary Ann Liebert, Inc.}
}
```

Also cite PyFaceAU and OpenFace 2.2 (see S0 PyfaceAU README).

## See Also

- [Main Project README](../README.md) - Complete workflow and installation
- [VALIDATION.md](VALIDATION.md) - PyFaceAU accuracy validation
- [S2 Action Coder](../S2%20Action%20Coder/README.md) - Next step in workflow
- [S3 Data Analysis](../S3%20Data%20Analysis/README.md) - Paralysis detection
