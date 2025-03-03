# Stable Face Splitter

This application processes video files to create symmetrical face visualizations for facial asymmetry assessment. It detects facial landmarks, calculates the anatomical midline, and creates mirrored face visualizations. It can also run OpenFace processing on the resulting videos to extract facial action units.

## File Structure

The codebase is organized into several modular components:

- `main.py` - Application entry point with UI interface
- `face_splitter.py` - Main class that coordinates all components
- `face_landmarks.py` - Face detection and landmark tracking
- `face_mirror.py` - Mirroring and visualization functionality
- `video_processor.py` - Video file handling and processing
- `video_rotation.py` - Video rotation detection and correction
- `openface_integration.py` - OpenFace processing functionality

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- dlib
- SciPy
- tkinter

You also need the dlib face landmark predictor file:
- `shape_predictor_68_face_landmarks.dat`

## Usage

1. Run the main application:
```
python main.py
```

2. Select one or more video files when prompted.

3. The application will process each video and create the following outputs in the `output` directory:
   - `*_right_mirrored.mp4` - Video with the right side of the face mirrored
   - `*_left_mirrored.mp4` - Video with the left side of the face mirrored
   - `*_debug.mp4` - Debug visualization showing landmarks and midline
   - `*_rotated.mp4` - Rotated video if rotation correction was needed
   
4. After processing, you'll be prompted to run OpenFace processing on the mirrored videos:
   - If you choose "Yes", OpenFace will analyze the mirrored videos to extract facial action units
   - The processed files will be automatically moved to a `1.5_Processed_Files` directory
   - The OpenFace results will be stored as CSV files

## Technical Details

### FaceLandmarkDetector

Handles face detection and landmark tracking with temporal smoothing:
- Uses dlib's frontal face detector and 68-point landmark predictor
- Implements temporal smoothing to reduce jitter
- Calculates anatomical midline based on facial landmarks

### FaceMirror

Creates mirrored face visualizations:
- Reflects the face along the anatomical midline
- Implements gradient blending along the midline for smooth transitions
- Creates debug visualizations for analysis

### VideoProcessor

Handles video file processing:
- Manages video reading, writing, and format conversion
- Tracks processing progress
- Integrates with video rotation correction

### StableFaceSplitter

Main class that coordinates all components:
- Instantiates and manages all component objects
- Provides a simple interface for video processing

### OpenFace Integration

Processes videos with OpenFace to extract facial action units:
- Processes videos with "mirrored" in the filename
- Ignores videos with "debug" in the filename
- Extracts facial action units using OpenFace
- Optionally moves processed files to a consolidated directory
