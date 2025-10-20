# S1 Face Mirror

**Automated facial action unit (AU) extraction tool for behavioral research.**

S1 Face Mirror is a high-performance video processing tool that automatically detects faces, creates left/right hemisphere mirrored videos, and extracts 18 facial action units using OpenFace 3.0. Optimized for Apple Silicon, NVIDIA CUDA, and Intel CPUs.

## What Does This Do?

**Input:** A video of a patient performing facial expressions
**Output:**
- Mirror videos showing left and right sides reflected (source, left, right)
- Diagnostic visualization showing facial landmarks and midline
- CSV files with detailed Action Unit (AU) measurements for quantitative analysis

## Quick Start Guide

### Step 1: Install Python

You need Python 3.8 or newer. Check if you have it:

```bash
python3 --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/)

### Step 2: Install Required Software

Open Terminal (Mac) or Command Prompt (Windows) and run:

```bash
# Navigate to the S1 Face Mirror folder
cd "path/to/S1 Face Mirror"

# Install dependencies
pip install -r requirements.txt

# Download face detection models (this may take a few minutes)
openface download
```

**Note:** The `openface download` command will download about 200-300 MB of model files.

### Step 3: Run the Application

```bash
python main.py
```

The application workflow:
1. A file browser will appear - select one or more video files (.mp4, .mov, .avi, etc.)
2. If you've previously processed any files, you'll be prompted to skip or re-process them
3. A progress window will show real-time processing status for each video
4. The application automatically runs both face mirroring AND AU extraction
5. Results are saved to `../S1O Processed Files/` when complete

## Output Files

Processed files are saved to `../S1O Processed Files/` (one level above the project directory):

**Face Mirror Videos** (in `Face Mirror 1.0 Output/`):
| File | Description |
|------|-------------|
| `*_right_mirrored.mp4` | Right side of face mirrored across midline |
| `*_left_mirrored.mp4` | Left side of face mirrored across midline |
| `*_debug.mp4` | Debug visualization with landmarks and midline |

**Analysis Data** (in `Combined Data/`):
| File | Description |
|------|-------------|
| `*_source.*` | Original video (rotation-corrected if needed) |
| `*_left_mirrored.csv` | AU measurements for left mirrored video |
| `*_right_mirrored.csv` | AU measurements for right mirrored video |

Each CSV contains frame-by-frame AU intensity values, timestamps, and detection confidence scores.

## Understanding the Output Videos

### Mirror Videos
- **Right mirrored:** Shows what the face would look like if the right side were mirrored
- **Left mirrored:** Shows what the face would look like if the left side were mirrored
- These videos help visualize asymmetry by showing each side independently

### Debug Information
- The application uses facial landmarks to calculate an anatomical midline
- Mirroring is performed perpendicular to this midline with gradient blending
- Head rotation (yaw) is monitored but doesn't prevent processing
- For best results, have patients point their nose directly at the camera 

## Video Requirements

For best results:
- Patient should face the camera directly
- Adequate lighting on the face
- Minimal head rotation (keep head straight)
- Clear view of entire face including forehead and chin
- Avoid obstructions (hair, hands, etc. covering face)

## Processing Speed

The application automatically detects and uses GPU acceleration if available (CUDA). Processing consists of two stages:

**Stage 1: Face Mirroring** (2-4 fps on CPU, 5-10 fps on GPU)
**Stage 2: AU Extraction** (2-4 fps on CPU with 6 parallel threads, 8-12 fps on GPU)

Expected total processing times per video:
- 10 second video (300 frames) = ~2-4 minutes
- 30 second video (900 frames) = ~6-12 minutes
- 60 second video (1800 frames) = ~12-24 minutes

The progress window displays real-time updates for each video:
- Current stage (Reading, Processing, Writing, AU Extraction)
- Percentage complete with progress bars
- Frames processed and total frames
- Elapsed time and estimated time remaining
- Current processing rate (frames/second)
- Video number (e.g., "Video 2 of 5")

## Troubleshooting

### "No module named 'openface'"
Run: `pip install openface-test && openface download`

### "Model files not found" or "weights directory not found"
Run: `openface download` from the "S1 Face Mirror" directory

### Processing is very slow
- Normal speed is 2-4 fps on CPU for each stage
- GPU acceleration is automatically detected and enabled if you have a CUDA-capable graphics card
- Close other applications to free up system resources
- The application uses multi-threading (6 threads) and optimized memory management
- Apple Silicon (M1/M2/M3) Macs: MPS is detected but not used (OpenFace requires CPU mode)

### No faces detected
- Ensure adequate lighting
- Check that face is clearly visible and not obscured
- Make sure video is oriented correctly (use the rotation correction feature)

### Python is not recognized
- On Mac: Use `python3` instead of `python`
- On Windows: Add Python to your system PATH during installation

## File Organization

```
SplitFace Open3/
├── S1 Face Mirror/
│   ├── main.py                    # Run this to start the application
│   ├── requirements.txt           # List of required software packages
│   ├── README.md                  # This file
│   ├── weights/                   # Face detection model files (created by openface download)
│   └── *.py                       # Supporting Python modules
└── S1O Processed Files/           # Output directory (created automatically)
    ├── Face Mirror 1.0 Output/    # Mirrored videos (left, right, debug)
    └── Combined Data/             # Source videos + CSV files with AU measurements
```

## Advanced Options

### GPU Acceleration

GPU acceleration is **automatically detected and enabled** if available:

- **NVIDIA GPUs:** CUDA is automatically used if detected (3-5x speedup)
- **Apple Silicon (M1/M2/M3):** Falls back to CPU (OpenFace doesn't support MPS)
- **No GPU:** Uses optimized multi-threaded CPU processing (6 threads)

To manually install CUDA support for PyTorch:
1. Visit [pytorch.org](https://pytorch.org) and follow installation instructions
2. The application will automatically detect and use CUDA

### Batch Processing

- Select multiple videos at once in the file browser
- The application processes them sequentially
- If a video has already been processed, you'll be prompted to skip or re-process
- A summary is displayed when all videos are complete

### Smart Output Detection

The application automatically detects existing outputs:
- On startup, checks if selected videos have already been processed
- Gives options to: re-process all, process only new files, or cancel
- Prevents accidental re-processing and saves time

## Technical Details

### Face Detection & Landmark Tracking
- **OpenFace 3.0** with RetinaFace detector for face detection
- **STAR model** for 98 facial landmark points (WFLW format)
- Adaptive detection interval (detects less often on stable faces, more often when moving)
- Temporal smoothing with weighted history to reduce jitter
- Confidence threshold: 0.9 for face detection, 0.5 for landmarks

### Action Unit Extraction
- **Multitask neural network** extracts 8 base AUs from facial appearance
- **AU adapter** expands to 18 AUs using geometric calculations
- AU45 (Blink) calculated from eye landmark geometry when enabled
- Frame-by-frame CSV output with timestamps and confidence scores
- OpenFace 2.0-compatible CSV format

### Mirroring Method
- Calculates anatomical midline from medial eyebrow points (landmarks 38 & 50) to chin (landmark 16)
- Reflects pixels perpendicular to the midline
- Applies gradient blending along midline for smooth transitions
- Creates three mirrored videos: left, right, and debug visualization
- Source video (rotation-corrected) saved directly to Combined Data folder with CSV files

### Memory Management
- Aggressive memory cleanup between videos prevents crashes during batch processing
- PyTorch cache clearing after each video (GPU mode)
- Periodic deep garbage collection every 10 videos
- Memory usage monitoring and checkpoints

### Multi-threading
- Face mirroring: Single-threaded (GPU/CPU)
- AU extraction: 6-thread parallel processing on CPU, sequential on GPU
- Progress updates sent to GUI thread safely

## Support

For issues or questions about the software, please check:
1. This README troubleshooting section
2. Requirements.txt for correct dependency versions
3. Terminal/command prompt output for error messages

## Version Information

- OpenFace 3.0 for face detection and landmark tracking
- Python 3.8+ required
- Cross-platform (Mac, Windows, Linux)
