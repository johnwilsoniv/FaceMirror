# Data Analysis - Facial Asymmetry Detection and Quantification

This application analyzes facial action unit (AU) data from coded videos to automatically detect and quantify facial paralysis and synkinesis. It uses machine learning models to classify paralysis severity and the presence of different patterns of synkinesis. Please note that the synkinesis detection algorithms are not ready for publication.

## What Does This Do?

**Input:**
- Coded CSV files from S2 Action Coder (with action labels and AU measurements)
- Patient ID and side affected information

**Output:**
- Combined results CSV with paralysis and synkinesis classifications
- Performance metrics (confusion matrices, classification reports)
- Statistical analysis of AU patterns across conditions
- Saved in `S3O Results` directory

## Quick Start Guide

### Step 1: Install Python

You need Python 3.8 or newer. Check if you have it:

```bash
python3 --version
```

If you don't have Python, download it from [python.org](https://python.org/downloads/)

### Step 2: Install Dependencies

```bash
# Navigate to the S3 Data Analysis folder
cd "path/to/S3 Data Analysis"

# Install required packages
pip install -r requirements.txt
```

### Step 3: Prepare Your Data

Organize your coded CSV files in a directory structure:
```
Data Directory/
├── Patient001/
│   ├── [timestamp]_coded.csv
│   └── [timestamp]_coded.csv
├── Patient002/
│   └── [timestamp]_coded.csv
└── ...
```

### Step 4: Run the Analysis

**GUI Mode (Interactive):**
```bash
python main.py
```

**Batch Mode (Command Line):**
```bash
python main.py --batch --data-dir /path/to/data
```

**Skip visualizations (faster):**
```bash
python main.py --batch --data-dir /path/to/data --skip-visuals
```

## Understanding the Analysis

### Paralysis Detection

The system detects three severity levels for each facial region:

| Region | Severity Levels | Key Actions Analyzed |
|--------|----------------|---------------------|
| **Upper Face** | None, Partial, Complete | Raise Eyebrows (RE), Brow Cock (BC) |
| **Mid Face** | None, Partial, Complete | Close Eyes Softly/Tightly (ES/ET) |
| **Lower Face** | None, Partial, Complete | Smile (SS/BS), Pucker Lips (PL) |

**Detection Method:**
- Analyzes asymmetry in AU activations between left and right sides
- Uses trained machine learning models (Random Forest classifiers)
- Compares affected vs. unaffected side measurements

### Synkinesis Detection

Identifies abnormal muscle coupling patterns:

| Type | Description | Example |
|------|-------------|---------|
| **Ocular-Oral** | Eye actions cause mouth movement | Closing eyes triggers smile |
| **Oral-Ocular** | Mouth actions cause eye narrowing | Smiling causes eye squinting |
| **Snarl-Smile** | Upper lip retraction during smile | Showing upper teeth when smiling |
| **Mentalis** | Chin dimpling during mouth movements | Chin muscle activation with smile |
| **Brow Cocked** | Persistent eyebrow elevation | One brow remains raised |
| **Hypertonicity** | Excessive resting muscle tone | Baseline AU elevation |

**Detection Method:**
- Calculates correlation between trigger and coupled AUs
- Compares asymmetry patterns during specific actions
- Uses statistical thresholds and ML classification

## Output Files

Results are saved in **`S3O Results/`**:

| File | Description |
|------|-------------|
| `combined_results.csv` | Main output with all classifications |
| `analysis_results/` | Performance metrics and confusion matrices |
| `logs/` | Detailed processing logs |

### combined_results.csv Columns

**Patient Information:**
- Patient ID, Side Affected, File Name, Action, Frame

**AU Measurements:**
- Individual AU intensities (AU01_r, AU02_r, etc.)
- Left/Right values for bilateral AUs

**Paralysis Classifications:**
- Upper_Face_Paralysis, Mid_Face_Paralysis, Lower_Face_Paralysis
- Values: None, Partial, Complete

**Synkinesis Classifications:**
- Ocular-Oral, Oral-Ocular, Snarl-Smile, Mentalis, Brow_Cocked, Hypertonicity
- Values: Present (1) or Absent (0)

## GUI Mode

The interactive GUI allows you:
1. **Select patient directories** individually
2. **View real-time progress** for each patient
3. **Extract and view frames** for visual verification
4. **Generate plots** for AU patterns
5. **Save results** incrementally

### GUI Features:
- Patient list display
- Progress tracking
- Frame extraction toggle
- Visual output generation

## Batch Mode

For processing large datasets:

```bash
# Process all patients in directory
python main.py --batch --data-dir /path/to/patients

# Skip visual outputs (faster processing)
python main.py --batch --data-dir /path/to/data --skip-visuals
```

**Advantages:**
- Faster processing (no GUI overhead)
- Scriptable for automation
- Better for large datasets

## Machine Learning Models

Pre-trained models are stored in `models/` directory:

### Paralysis Models:
- `upper_face_model.pkl` - Upper face paralysis classifier
- `mid_face_model.pkl` - Mid face paralysis classifier
- `lower_face_model.pkl` - Lower face paralysis classifier
- Corresponding scalers (`*_scaler.pkl`) for feature normalization

### Synkinesis Models:
Located in `models/synkinesis/[type]/`:
- `model.pkl` - Binary classifier for each synkinesis type
- `scaler.pkl` - Feature normalization
- `features.list` - List of features used by the model

## Advanced Features

### Model Training/Retraining

If you have expert-labeled training data:

**Paralysis Models:**
```bash
python paralysis_training_pipeline.py
```

**Synkinesis Models:**
```bash
python synkinesis_training.py
```

### Performance Evaluation

Evaluate model performance on test data:

**Paralysis:**
```bash
python paralysis_performance.py
```

**Synkinesis:**
```bash
python synkinesis_performance.py
```

Outputs confusion matrices and classification reports to `analysis_results/` and `performance_analysis_results/`.

### Custom Configuration

Edit configuration files to adjust thresholds:
- `paralysis_config.py` - Paralysis detection settings
- `synkinesis_config.py` - Synkinesis detection settings
- `facial_au_constants.py` - AU definitions and mappings

## Troubleshooting

### "No patients detected"
- Ensure CSV files have "_coded" suffix
- Check directory structure matches expected format
- Verify CSV files contain required columns (Patient ID, Frame, Action, AUs)

### "Model files not found"
- Ensure `models/` directory exists with trained models
- Re-run training scripts if models are missing
- Check file paths in config files

### Memory errors with large datasets
- Use `--skip-visuals` flag to reduce memory usage
- Process patients in smaller batches
- Increase system RAM or use machine with more memory

### Statistical tests failing
- Install scikit-posthocs: `pip install scikit-posthocs`
- Check for sufficient data in each group
- Verify CSV contains expected condition columns

### ImportError for project modules
- Ensure you're running from the S3 Data Analysis directory
- Check sys.path configuration in main.py
- Verify all .py files are in the same directory

## File Organization

```
S3 Data Analysis/
├── main.py                          # Main entry point
├── facial_au_analyzer.py            # Core analysis logic
├── facial_au_batch_processor.py     # Batch processing
├── facial_au_gui.py                 # GUI interface
├── facial_au_constants.py           # AU and action definitions
├── paralysis_detector.py            # Paralysis classification
├── paralysis_config.py              # Paralysis settings
├── paralysis_utils.py               # Paralysis helper functions
├── synkinesis_detector.py           # Synkinesis classification
├── synkinesis_config.py             # Synkinesis settings
├── *_features.py                    # Feature extraction modules
├── paralysis_training_pipeline.py   # Train paralysis models
├── synkinesis_training.py           # Train synkinesis models
├── paralysis_performance.py         # Evaluate paralysis models
├── synkinesis_performance.py        # Evaluate synkinesis models
├── models/                          # Trained ML models
├── logs/                            # Processing logs
├── analysis_results/                # Performance metrics
└── README.md                        # This file
```

## Performance Notes

- **Processing speed**: ~10-50 patients per minute (depending on frame extraction)
- **With visuals**: Slower due to frame extraction and plot generation
- **Memory usage**: ~100-500 MB per patient (with visuals)
- **Recommended**: Use batch mode with `--skip-visuals` for large datasets

## Support

For issues or questions:
1. Check this README troubleshooting section
2. Review log files in `logs/` directory
3. Ensure all dependencies are installed
4. Verify input CSV files match expected format

## Version Information

- Python 3.8+ required
- scikit-learn for machine learning
- pandas for data manipulation
- matplotlib/seaborn for visualization
- Cross-platform (Mac, Windows, Linux)
