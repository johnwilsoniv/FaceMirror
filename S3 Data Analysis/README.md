# Data Analysis - Facial Paralysis Detection

This application analyzes facial action unit (AU) data to automatically detect and quantify facial paralysis using machine learning. Designed to work with PyFaceAU output data.

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
# GUI mode
python main.py

# Batch mode
python main.py --batch --data-dir /path/to/data

# Batch mode without visual outputs
python main.py --batch --data-dir /path/to/data --skip-visuals
```

## What It Does

Analyzes facial AU data from mirrored left/right video recordings to detect paralysis severity (None/Partial/Complete) across three facial regions:

- **Upper Face**: Eyebrows, forehead (AU01, AU02)
- **Mid Face**: Eyes, eyelids (AU45, AU07)
- **Lower Face**: Mouth, lips (AU12, AU25)

## PyFaceAU Compatibility

This system is configured for **PyFaceAU** (Pure Python OpenFace 2.2 implementation), which provides 17 functional Action Units with r=0.8640 correlation to original OpenFace 2.2.

### Available AUs in PyFaceAU
- AU01 (Inner Brow Raiser)
- AU02 (Outer Brow Raiser)
- AU04 (Brow Lowerer)
- AU05 (Upper Lid Raiser)
- AU06 (Cheek Raiser)
- AU07 (Lid Tightener)
- AU09 (Nose Wrinkler)
- AU10 (Upper Lip Raiser)
- AU12 (Lip Corner Puller/Smile)
- AU14 (Dimpler)
- AU15 (Lip Corner Depressor)
- AU17 (Chin Raiser)
- AU20 (Lip Stretcher)
- AU23 (Lip Tightener)
- AU25 (Lips Part)
- AU26 (Jaw Drop)
- AU45 (Blink)

### AU Mappings for Detection
- **Mid-face detection (eyes)**: AU07 (Lid Tightener), AU45 (Blink)
- **Close Eyes Tightly/Big Smile**: AU07 (Lid Tightener) for eye component
- **Wrinkle Nose**: AU09 (Nose Wrinkler)
- **Pucker Lips**: Median frame fallback (AU17 detection challenging)
- **Blow Cheeks**: Median frame fallback (AU23 detection challenging)

## Analyzed Actions

The system analyzes the following facial actions for peak frame detection:

**Priority Actions (fully supported):**
- RE (Raise Eyebrows)
- ES (Close Eyes Softly)
- ET (Close Eyes Tightly)
- SS (Soft Smile)
- BS (Big Smile)
- BK (Blink)

**Additional Actions:**
- SE (Say E)
- SO (Say O)
- FR (Frown)
- WN (Wrinkle Nose)
- LT (Lower Teeth)
- BL (Baseline)

**Median Frame Fallback:**
- PL (Pucker Lips)
- BC (Blow Cheeks)

## Input Data Format

Expects CSV files from S2 (Action Coder) with the following structure:
- Left/right mirrored pairs: `*_left_mirrored_coded.csv` and `*_right_mirrored_coded.csv`
- Required columns: `frame`, `action`, AU columns (AU01_r through AU45_r)
- Patient ID derived from filename pattern

## Output

Results saved to `../S3O Results/` directory:
- `combined_results.csv` - Complete analysis with paralysis classifications and AU measurements
- `patient_frames/` - Extracted peak frames for each action (if visual outputs enabled)
- `patient_plots/` - AU comparison plots (if visual outputs enabled)

### Output Columns
- Patient ID
- Paralysis classifications (Left/Right × Upper/Mid/Lower Face)
- AU measurements for each action (raw and normalized)
- Peak frame numbers for each action
- Paralysis detection flag

## Machine Learning Models

Pre-trained Random Forest classifiers in `models/`:
- `upper_face_model.pkl` - Upper face paralysis detection
- `mid_face_model.pkl` - Mid face paralysis detection
- `lower_face_model.pkl` - Lower face paralysis detection

Models use AU measurements and asymmetry features to classify paralysis severity.

## Configuration

Key parameters in `facial_au_constants.py`:

### Facial Zones
```python
FACIAL_ZONES = {
    'upper': ['AU01_r', 'AU02_r'],
    'mid': ['AU45_r', 'AU07_r'],
    'lower': ['AU12_r', 'AU25_r']
}
```

### Action-to-AU Mappings
Defines which AUs are used to find peak frames for each action. See `ACTION_TO_AUS` in `facial_au_constants.py`.

### Detection Thresholds
- `PARALYSIS_THRESHOLDS` - Asymmetry ratios and minimal movement thresholds
- `ASYMMETRY_THRESHOLDS` - Percent difference and ratio cutoffs
- `CONFIDENCE_THRESHOLDS` - ML model confidence requirements

**Note:** Models must be retrained with PyFaceAU data to use corrected AU07 mappings for mid-face detection.

## Architecture

- `main.py` - Entry point (GUI/batch mode)
- `facial_au_analyzer.py` - Core analysis engine
- `facial_au_batch_processor.py` - Batch processing controller
- `facial_au_gui.py` - GUI interface
- `facial_au_constants.py` - Configuration and constants
- `facial_au_frame_extractor.py` - Video frame extraction
- `facial_au_visualizer.py` - Plot generation
- `paralysis_detector.py` - ML-based paralysis detection
- `facial_paralysis_detection.py` - Rule-based paralysis detection

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- opencv-python (for frame extraction)
- matplotlib, seaborn (for visualizations)
- tkinter (for GUI)

## Workflow

1. **Load Data**: Read left/right CSV pairs for each patient
2. **Identify Actions**: Detect available actions in the data
3. **Find Peak Frames**: Locate maximum activation frame for each action
4. **Extract Baseline**: Get resting state AU values
5. **Calculate Normalized AUs**: Subtract baseline from action AUs
6. **Detect Paralysis**: Apply ML models to classify severity per facial zone
7. **Generate Outputs**: Save results CSV and visual outputs
8. **Aggregate**: Combine all patient results into summary report

## Notes

- Designed for Bell's palsy and facial nerve paralysis research
- Handles asymmetric facial movements between left/right sides
- Supports both single-patient (GUI) and batch processing modes
- Frame extraction optional for faster processing
