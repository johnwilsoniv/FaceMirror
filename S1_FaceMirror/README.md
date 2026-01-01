# Face Mirror (S1)

Video processing tool for extracting facial landmarks and action units from video recordings. Processes videos to generate frame-by-frame facial analysis data for downstream analysis.

## Features

- **Video Processing**: Extract facial data from MP4, MOV, and AVI video files
- **Face Detection**: MTCNN-based face detection with fallback handling
- **Landmark Detection**: 68-point facial landmark extraction
- **Action Unit Extraction**: Intensity estimation for 17 facial action units
- **Face Mirroring**: Generate left and right mirrored face data for split-face analysis
- **Batch Processing**: Process multiple videos in sequence
- **Progress Tracking**: Real-time progress display during processing

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Processing

```bash
# GUI mode
python main.py

# Process a single video
python main.py --video /path/to/video.mp4
```

## Output

Results are saved to `~/Documents/SplitFace/S1O Processed Files/`:

- `Combined Data/video_name_landmarks.csv` - 68-point facial landmarks per frame
- `Combined Data/video_name_action_units.csv` - AU intensities per frame
- `Combined Data/video_name.mp4` - Copy of source video (if needed by S2)

## Architecture

| File | Description |
|------|-------------|
| `main.py` | Application entry point and GUI |
| `video_processor.py` | Core video processing pipeline |
| `pyfaceau_detector.py` | PyFaceAU integration for AU extraction |
| `face_mirror.py` | Face mirroring and split-face logic |
| `face_splitter.py` | Left/right face splitting |
| `config.py` | Configuration settings |

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

Core dependencies:
- opencv-python - Video processing
- numpy - Numerical operations
- PyQt5 - GUI framework
- PyFaceAU - Action unit extraction (separate package)

## Citation

If you use Face Mirror in your research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## Acknowledgments

This tool integrates PyFaceAU for action unit extraction, which is based on:

> Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit. IEEE International Conference on Automatic Face and Gesture Recognition.

## License

See the main repository LICENSE file for terms of use.
