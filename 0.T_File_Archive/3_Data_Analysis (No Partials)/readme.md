# Facial AU Analyzer

This application analyzes facial Action Units (AUs) from OpenFace data to assess facial symmetry and movement patterns.

## Features

- Analyze left and right facial movements from OpenFace CSV outputs
- Find frames with maximal intensity for specific facial actions
- Extract frames from video at points of maximal movement
- Generate visualizations of facial AU values
- Analyze facial symmetry between left and right sides
- Detect potential facial paralysis and identify affected side
- Identify which Action Units (AUs) show evidence of paralysis
- Detect synkinesis patterns (unwanted muscle co-activation)
- Process multiple patients in batch mode
- Generate combined reports across patients

## File Structure

- `main.py` - Main entry point for the application
- `facial_au_constants.py` - Constants and definitions
- `facial_au_analyzer.py` - Core analyzer for single patient
- `facial_au_batch_processor.py` - Batch processor for multiple patients
- `facial_au_gui.py` - GUI interface

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- opencv-python (cv2)
- tkinter (for GUI)

Install required packages:

```bash
pip install pandas numpy matplotlib opencv-python
```

Tkinter usually comes with Python installations.

## Usage

### GUI Mode

Launch the application with a graphical user interface:

```bash
python main.py
```

The GUI offers two modes:
1. **Single Patient Analysis** - Analyze one patient at a time
2. **Batch Analysis** - Process multiple patients at once

### Command Line Batch Mode

Run batch analysis from the command line:

```bash
python main.py --batch --data-dir /path/to/data --output-dir /path/to/output
```

Options:
- `--batch` - Run in batch mode (no GUI)
- `--data-dir` - Directory containing data files (required in batch mode)
- `--output-dir` - Directory to save output files (default: 'output')
- `--no-frames` - Do not extract frames from videos

## File Naming Conventions

The application expects files to follow these naming patterns:

- Left CSV: `PATIENTID_left*.csv` 
- Right CSV: `PATIENTID_right*.csv`
- Video: `PATIENTID_rotated_annotated.mp4` (or .avi, .mov, .MOV)

Where `PATIENTID` is a common prefix identifying the patient.

If the video doesn't follow the exact pattern, the application will try to find any video file that starts with the same PATIENTID.

## Output

The application generates:

1. Frame images at points of maximal action
2. Visualizations of AU values for each action
3. Symmetry charts comparing left vs right side movements (with highlighting for paralysis)
4. Summary CSV files with analysis results for each patient
5. Combined analysis across all patients
6. Paralysis detection reports identifying affected sides and AUs
7. Synkinesis detection reports showing unwanted muscle co-activation patterns

## How It Works

For each facial action (e.g., smiling, eyebrow raising):

1. The application finds the frame with maximal intensity of key Action Units on either the left or right side
2. At that frame, it captures all AU values from both sides
3. It extracts the video frame and creates visualizations
4. It calculates symmetry ratios between left and right sides
5. It detects potential facial paralysis by analyzing asymmetry patterns
6. It identifies possible synkinesis (like ocular-oral synkinesis, when eye closure triggers unwanted mouth movement)
7. For batch processing, it consolidates data across all patients and generates statistical reports

### Paralysis Detection

The application looks for consistent patterns of asymmetry across multiple actions to detect potential facial paralysis:
- Identifies which side shows minimal movement
- Determines which AUs are most affected
- Highlights the potentially paralyzed side in visualizations

### Synkinesis Detection

The application detects unwanted muscle co-activation patterns:
- Ocular-Oral Synkinesis: When eye closure triggers unwanted mouth movement
- Oral-Ocular Synkinesis: When mouth movement triggers unwanted eye narrowing or closure
