# Data Analysis - Facial Paralysis Detection

This application analyzes facial action unit (AU) data to automatically detect and quantify facial paralysis using machine learning.

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
```

## What It Does

Detects facial paralysis severity (None/Partial/Complete) across three regions:
- **Upper Face**: Eyebrows, forehead
- **Mid Face**: Eyes, cheeks
- **Lower Face**: Mouth, lips

## Output

Results saved to `S3O Results/combined_results.csv` with:
- Paralysis classifications per facial region
- AU measurements for each action
- Performance statistics

## Models

Pre-trained Random Forest classifiers in `models/`:
- `upper_face_model.pkl`
- `mid_face_model.pkl`
- `lower_face_model.pkl`

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- opencv-python, matplotlib, seaborn
