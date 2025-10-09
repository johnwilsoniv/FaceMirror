# Face Mirror - Video Processing for Facial Asymmetry Analysis

This application processes facial assessment videos to create mirror-image visualizations for analyzing facial symmetry. It automatically detects facial landmarks and creates mirrored videos showing what each side of the face would look like if it were symmetrical.

## What Does This Do?

**Input:** A video of a patient performing facial expressions
**Output:**
- Mirror videos showing left and right sides reflected
- Diagnostic visualization showing facial landmarks and midline
- Optional facial action unit analysis using OpenFace

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

A file browser will appear:
1. Select one or more video files (.mp4, .mov, .avi, etc.)
2. Click "Open"
3. Wait for processing to complete
4. Choose whether to run OpenFace analysis (optional)

## Output Files

All processed files are saved in the `output` folder:

| File | Description |
|------|-------------|
| `*_right_mirrored.mp4` | Right side of face mirrored across midline |
| `*_left_mirrored.mp4` | Left side of face mirrored across midline |
| `*_debug.mp4` | Shows detected landmarks and facial midline |
| `*_rotated.mp4` | Original video corrected for rotation (if needed) |

After OpenFace processing, files are moved to `S1O Processed Files` with CSV data.

## Understanding the Output Videos

### Mirror Videos
- **Right mirrored:** Shows what the face would look like if the right side were mirrored
- **Left mirrored:** Shows what the face would look like if the left side were mirrored
- These videos help visualize asymmetry by showing each side independently

### Debug Video
- Red line shows the calculated facial midline
- Green banner = optimal head position for analysis
- Yellow banner = acceptable head position
- Red banner = head rotation too extreme (may affect accuracy)
- Keep in mind that the banner is just a rough guesstimate and the video may process just fine despite head rotation issues. It is too complicated and resource intensive to get a very accurate estimate of head "yaw" which is actually what affects the output quality. It's always a good idea to have your patients point their nose at the camera sensor to minimize yaw. 

## Video Requirements

For best results:
- Patient should face the camera directly
- Adequate lighting on the face
- Minimal head rotation (keep head straight)
- Clear view of entire face including forehead and chin
- Avoid obstructions (hair, hands, etc. covering face)

## Processing Speed

Typical processing speed: **5-7 frames per second**

Expected processing times:
- 10 second video (300 frames) = ~1 minute
- 30 second video (900 frames) = ~3 minutes
- 60 second video (1800 frames) = ~6 minutes

## Troubleshooting

### "No module named 'openface'"
Run: `pip install openface-test && openface download`

### "Model files not found" or "weights directory not found"
Run: `openface download` from the "S1 Face Mirror" directory

### Processing is very slow
- Normal speed is 5-7 fps on CPU
- For faster processing, you can enable GPU acceleration (requires CUDA-capable graphics card)
- Close other applications to free up system resources

### No faces detected
- Ensure adequate lighting
- Check that face is clearly visible and not obscured
- Make sure video is oriented correctly (use the rotation correction feature)

### Python is not recognized
- On Mac: Use `python3` instead of `python`
- On Windows: Add Python to your system PATH during installation

## File Organization

```
S1 Face Mirror/
├── main.py                    # Run this to start the application
├── requirements.txt           # List of required software packages
├── README.md                  # This file
├── output/                    # Processed videos are saved here
├── weights/                   # Face detection model files (created by openface download)
└── S1O Processed Files/       # Final OpenFace-processed files (created after processing)
```

## Advanced Options

### GPU Acceleration (Optional)

If you have an NVIDIA GPU with CUDA support, you can enable GPU acceleration for faster processing:

1. Install PyTorch with CUDA support (see [pytorch.org](https://pytorch.org))
2. Edit `face_splitter.py` line 17:
   - Change `device='cpu'` to `device='cuda'`

This can provide 3-5x speedup for landmark detection.

### Batch Processing

You can select multiple videos at once. The application will process them sequentially and provide a summary when complete.

## OpenFace Integration

After creating mirror videos, you can optionally run OpenFace analysis to extract facial action units (AUs):

**Options:**
- **Run all files:** Process all videos in the output folder
- **Run session files only:** Process only videos from the current session
- **Do not run:** Skip OpenFace processing

OpenFace processing adds 30-60 seconds per video and produces CSV files with detailed AU measurements.

## Technical Details

### Face Detection
- Uses OpenFace 3.0 with RetinaFace detector
- Detects 98 facial landmark points
- Adaptive detection interval (faster on stable faces)
- Temporal smoothing to reduce jitter

### Mirroring Method
- Calculates anatomical midline from medial eyebrow points (landmarks 38 & 50) to chin (landmark 16)
- Reflects pixels perpendicular to the midline
- Applies gradient blending along midline for smooth transitions

### Performance Statistics
After processing, you'll see performance metrics:
- Total frames processed
- Detection frequency
- Processing time per frame
- Estimated FPS

## Support

For issues or questions about the software, please check:
1. This README troubleshooting section
2. Requirements.txt for correct dependency versions
3. Terminal/command prompt output for error messages

## Version Information

- OpenFace 3.0 for face detection and landmark tracking
- Python 3.8+ required
- Cross-platform (Mac, Windows, Linux)
