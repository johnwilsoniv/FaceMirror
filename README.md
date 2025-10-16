# SplitFace v2.0.0

A comprehensive suite for facial video processing, action coding, and paralysis detection using OpenFace 3.0 and machine learning.

## Overview

SplitFace consists of three integrated applications:

### S1: Face Mirror
Processes facial videos using OpenFace 3.0 to extract Action Units (AUs) and create mirrored comparison videos.

**Key Features:**
- OpenFace 3.0 facial landmark detection
- AU intensity extraction (18 AUs)
- Left/right face mirroring for asymmetry analysis
- Patient-based batch processing
- Progress tracking with existing output detection

**Input:** Raw video files
**Output:** `~/Documents/SplitFace/S1O Processed Files/`
- Mirrored videos (Face Mirror 1.0 Output/)
- AU CSV files for left and right sides (Combined Data/)

### S2: Action Coder
Interactive GUI for coding patient actions in videos with integrated Whisper speech recognition.

**Key Features:**
- PyQt5-based video player with timeline
- Automatic transcription using faster-whisper
- Smart action detection from speech
- Manual annotation and correction tools
- Undo/redo support

**Input:** Videos and CSVs from S1
**Output:** `~/Documents/SplitFace/S2O Coded Files/`
- Annotated videos with action overlays
- CSV files with action codes per frame

### S3: Data Analysis
Batch analysis tool for facial paralysis detection using machine learning classifiers.

**Key Features:**
- ML-based paralysis detection (Random Forest models)
- Zone-specific analysis (upper, mid, lower face)
- Asymmetry visualization and reporting
- Batch processing with GUI
- Comprehensive CSV output with predictions

**Input:** Coded files from S2
**Output:** `~/Documents/SplitFace/S3O Results/`
- Patient-specific analysis folders
- Combined results CSV
- Visualizations and plots

## Quick Start

### Running in Development Mode

Each application can be run directly with Python:

```bash
# S1 Face Mirror
cd "S1 Face Mirror"
python main.py

# S2 Action Coder
cd "S2 Action Coder"
python main.py

# S3 Data Analysis
cd "S3 Data Analysis"
python main.py
```

### Building Standalone Applications

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed build instructions.

**Quick build (all apps):**

macOS:
```bash
./build_macos.sh
```

Windows:
```cmd
build_windows.bat
```

## System Requirements

### Minimum Requirements
- **OS**: macOS 10.13+ or Windows 10+
- **CPU**: Intel i5 or equivalent (Apple Silicon supported)
- **RAM**: 8 GB (16 GB recommended for S1)
- **Storage**: 10 GB free space
- **Python**: 3.10 or 3.11 (for development/building)

### Recommended Hardware
- **CPU**: Multi-core processor (8+ cores)
- **RAM**: 16+ GB
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster S1 processing)

## Installation

### Development Environment

1. **Clone or download the repository**

2. **Create a virtual environment** (recommended)

   **Option A: Using Conda (Recommended)**
   ```bash
   conda create -n splitface python=3.11
   conda activate splitface
   ```

   **Option B: Using venv**
   ```bash
   python3 -m venv splitface_env
   source splitface_env/bin/activate  # macOS/Linux
   # or
   splitface_env\Scripts\activate  # Windows
   ```

3. **Install dependencies for each app**
   ```bash
   cd "S1 Face Mirror"
   pip install -r requirements.txt

   cd "../S2 Action Coder"
   pip install -r requirements.txt

   cd "../S3 Data Analysis"
   pip install -r requirements.txt
   ```

### Building Standalone Applications

To build standalone applications, you'll also need PyInstaller:

**Using Conda (Recommended):**
```bash
conda install -c conda-forge pyinstaller
```

**Using pip:**
```bash
pip install pyinstaller
```

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for complete build instructions.

### External Dependencies

#### FFmpeg (Required for S2)
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

## Workflow

The typical workflow through all three applications:

```
1. S1 Face Mirror
   └─> Process raw videos with OpenFace 3.0
       └─> Extract AUs and create mirrored videos

2. S2 Action Coder
   └─> Load S1 outputs (videos + CSVs)
       └─> Code patient actions with Whisper assistance

3. S3 Data Analysis
   └─> Load S2 outputs (coded CSVs)
       └─> Analyze paralysis with ML models
           └─> Generate reports and visualizations
```

## Output Directory Structure

All applications output to a standardized location:

```
~/Documents/SplitFace/
├── S1O Processed Files/
│   ├── Face Mirror 1.0 Output/
│   │   └── [patient_id]_source.mp4  (mirrored videos)
│   └── Combined Data/
│       ├── [patient_id]_left_mirrored.csv
│       └── [patient_id]_right_mirrored.csv
│
├── S2O Coded Files/
│   ├── [patient_id]_left_mirrored_coded.csv
│   ├── [patient_id]_right_mirrored_coded.csv
│   └── [patient_id]_source_coded.mp4
│
└── S3O Results/
    ├── combined_results.csv
    ├── paralysis_statistics.csv
    └── [patient_id]/
        ├── [action]_analysis.png
        └── [action]_frame.jpg
```

## Key Technologies

- **OpenFace 3.0**: Facial landmark detection and AU extraction
- **faster-whisper**: Speech-to-text for action detection
- **PyQt5**: GUI framework for S2
- **tkinter**: GUI framework for S3
- **scikit-learn**: Machine learning for paralysis detection
- **PyTorch**: Deep learning models in OpenFace
- **OpenCV**: Video processing
- **pandas**: Data manipulation
- **matplotlib**: Visualization

## Architecture

### Cross-Platform Design

All applications use a unified `config_paths.py` module that:
- Detects runtime environment (development vs. bundled)
- Determines platform (Windows vs. macOS)
- Provides standardized paths for data files and outputs
- Handles resource bundling for PyInstaller

### Version Management

All applications share the same version number (2.0.0), defined in their respective `config_paths.py` modules.

## Troubleshooting

### Common Issues

**Memory issues with S1**
- Reduce batch size
- Process fewer patients at once
- Close other applications
- See memory management notes in S1 README

**S2 won't open videos**
- Verify FFmpeg is installed and accessible
- Check video codec compatibility
- Ensure matching CSV files are present

**S3 models not found**
- Verify `models/` directory contains all `.pkl` files
- Check file permissions
- Rebuild if using bundled version

### Getting Help

1. Check application-specific READMEs:
   - [S1 Face Mirror/README.md](S1%20Face%20Mirror/README.md)
   - [S2 Action Coder/README.md](S2%20Action%20Coder/README.md)
   - [S3 Data Analysis/README.md](S3%20Data%20Analysis/README.md)

2. Review build instructions: [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)

3. Check console logs for detailed error messages

## Development Notes

### Code Structure

Each application follows a modular structure:
- `main.py`: Entry point
- `config_paths.py`: Cross-platform path management
- Application-specific modules for core functionality
- Separate GUI components where applicable

### Testing Changes

Always test in development mode before building:
```bash
python main.py
```

### Building for Distribution

1. Test all changes in development mode
2. Update version numbers in `config_paths.py`
3. Run build scripts
4. Test built applications on clean systems

## Citation

If you use SplitFace in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Changelog

### v2.0.0 (Current)
- Upgraded to OpenFace 3.0
- Cross-platform build support (Windows & macOS)
- Standardized output directory structure
- Improved memory management
- Enhanced GUI for S2
- ML-based paralysis detection in S3

### v1.0.0 (Previous)
- Initial release with OpenFace 2.2
- Basic S1, S2, S3 functionality

## Contributors

[Contributor information to be added]

## Acknowledgments

- OpenFace team for the facial analysis toolkit
- Hugging Face for transformer models
- faster-whisper contributors for speech recognition
