# S3 Data Analysis - Facial Paralysis Detection

Machine learning-based analysis of facial action unit (AU) data for automated detection and quantification of facial paralysis severity. This module processes AU data from coded video recordings to classify paralysis as None, Partial, or Complete across three facial regions.

## Citation

If you use this software for research purposes, please cite:

> Wilson, J., et al. (2025). A Split-Face Computer Vision Machine Learning Assessment of Facial Paralysis Using Facial Action Units. *[Journal Name]*.

The full paper is available at: `wilson-et-al-2025-a-split-face-computer-vision-machine-learning-assessment-of-facial-paralysis-using-facial-action-units.pdf`

## Model Performance

The trained models achieve the following accuracy on held-out test data:

| Facial Zone | Accuracy | Target |
|-------------|----------|--------|
| Upper Face  | 82.55%   | 83%    |
| Mid Face    | 93.52%   | 93%    |
| Lower Face  | 84.68%   | 84%    |

Models use ordinal classification to distinguish between None, Partial, and Complete paralysis severity levels.

## Acknowledgments

This system builds upon the Facial Action Coding System (FACS) and leverages Action Unit intensity estimation. The AU extraction pipeline is based on:

> Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit. IEEE International Conference on Automatic Face and Gesture Recognition.

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

### Training Models

```bash
# Launch training GUI
python training_gui.py

# Command-line training for a specific zone
python paralysis_training_pipeline.py mid
```

## Facial Zones and Action Units

The system analyzes paralysis across three facial regions:

| Zone | Action Units | Diagnostic Actions |
|------|--------------|-------------------|
| Upper Face | AU01, AU02 | Raise Eyebrows (RE) |
| Mid Face | AU45, AU07, AU06 | Eye closure (ES, ET), Blink (BK) |
| Lower Face | AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26 | Smile (SS, BS), Say E/O (SE, SO) |

### Supported Actions

**Primary Actions:**
- RE (Raise Eyebrows)
- ES (Close Eyes Softly)
- ET (Close Eyes Tightly)
- SS (Soft Smile)
- BS (Big Smile)
- BK (Blink)

**Secondary Actions:**
- SE (Say E)
- SO (Say O)
- FR (Frown)
- WN (Wrinkle Nose)
- LT (Lower Teeth)
- BL (Baseline)
- PL (Pucker Lips)
- BC (Blow Cheeks)

## Input Data Format

Expects CSV files from S2 (Action Coder) with the following structure:

- Left/right mirrored pairs: `*_left_mirrored_coded.csv` and `*_right_mirrored_coded.csv`
- Required columns: `frame`, `action`, AU columns (AU01_r through AU45_r)
- Patient ID derived from filename pattern

## Output

Results saved to `~/Documents/SplitFace/S3O Results/`:

- `combined_results.csv` - Complete analysis with paralysis classifications and AU measurements
- `paralysis_statistics.csv` - Aggregate statistics across all patients
- Per-patient directories with extracted frames and plots (if visual outputs enabled)

### Classification Output

For each patient and facial zone, the system outputs:
- Paralysis classification (None/Partial/Complete) for left and right sides
- Prediction confidence scores
- Raw and normalized AU measurements at peak frames

## Architecture

### Core Modules

| File | Description |
|------|-------------|
| `main.py` | Entry point for GUI and batch processing |
| `facial_au_analyzer.py` | Core analysis engine |
| `facial_au_batch_processor.py` | Batch processing controller |
| `paralysis_detector.py` | ML-based paralysis detection |
| `paralysis_config.py` | Zone configuration and parameters |

### Feature Extraction

| File | Description |
|------|-------------|
| `upper_face_features.py` | Upper face feature extraction |
| `mid_face_features.py` | Mid face feature extraction |
| `lower_face_features.py` | Lower face feature extraction |
| `paralysis_utils.py` | Shared feature computation utilities |

### Training Pipeline

| File | Description |
|------|-------------|
| `training_gui.py` | Interactive training interface |
| `paralysis_training_pipeline.py` | Command-line training pipeline |
| `paralysis_model_trainer.py` | Model training with ordinal classification |
| `paralysis_training_helpers.py` | Training utilities and metrics |

### Visualization

| File | Description |
|------|-------------|
| `facial_au_gui.py` | Analysis GUI interface |
| `facial_au_visualizer.py` | Plot and chart generation |
| `facial_au_frame_extractor.py` | Video frame extraction |

## Configuration

Key parameters are defined in `paralysis_config.py`:

- `ZONE_CONFIG` - Per-zone action units, actions, and file paths
- Feature extraction parameters (normalization, asymmetry calculation)
- Model hyperparameter search spaces for Optuna optimization

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

Core dependencies:
- pandas, numpy - Data processing
- scikit-learn, xgboost - Machine learning
- optuna - Hyperparameter optimization
- imbalanced-learn - Class balancing (SMOTE)
- opencv-python - Video frame extraction
- matplotlib, seaborn - Visualization

## Workflow

1. **Load Data**: Read left/right CSV pairs for each patient
2. **Identify Actions**: Detect available actions in the data
3. **Find Peak Frames**: Locate maximum activation frame for each action
4. **Extract Baseline**: Get resting state AU values from BL action
5. **Calculate Normalized AUs**: Subtract baseline from action AUs
6. **Extract Features**: Compute asymmetry metrics and derived features
7. **Detect Paralysis**: Apply ML models to classify severity per zone
8. **Generate Outputs**: Save results and optional visualizations

## License

See the main repository LICENSE file for terms of use.
