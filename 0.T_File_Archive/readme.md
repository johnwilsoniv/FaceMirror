# Stable Face Splitter (FSM-Tilt)

A facial mirroring application that creates symmetrical faces by reflecting along the anatomical midline, with specific handling for tilted faces.

## File Structure

The codebase has been split into multiple files for easier editing and maintenance:

- `main.py` - Main entry point and GUI interface
- `stable_face_core.py` - Core class definition and video processing logic
- `face_detection.py` - Face detection and landmark extraction
- `midline_calculation.py` - Algorithms for calculating facial midline
- `face_mirroring.py` - Logic for creating mirrored faces
- `visualization.py` - Visualization and debug rendering
- `stability_analysis.py` - Stability analysis and reporting
- `video_rotation.py` - Handles video rotation detection and correction

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- dlib
- scipy
- sklearn
- matplotlib (for stability analysis)
- pandas (for stability analysis)
- Tkinter (for GUI)

You'll also need the `shape_predictor_68_face_landmarks.dat` file from dlib for facial landmark detection.

## Usage

Run the application by executing:

```
python main.py
```

This will open a file selection dialog allowing you to select video files for processing.

## Features

- Anatomical midline detection with improved stability for tilted faces
- Right and left face mirroring based on anatomical midline
- Adaptive blending along the midline
- Debug visualization showing landmarks, midline, and head pose
- Stability analysis with metrics and reporting

## Output Files

For each input video, the application generates:
- Right mirrored video
- Left mirrored video
- Debug visualization video
- Midline log data (CSV)
- Stability analysis report (optional)
