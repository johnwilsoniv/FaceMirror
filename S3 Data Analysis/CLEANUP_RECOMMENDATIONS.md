# Cleanup Recommendations

## Files That Can Be Safely Deleted

### 1. Training Log Files (in main directory)
These are temporary logs from model retraining sessions:
- `retrain_all_zones.log`
- `retrain_corrected.log`
- `retrain_optimized.log`
- `retrain_output.log`
- `retrain_sklearn_1.6.1.log`
- `training_output.log`
- `main_output.log`

### 2. Python Cache Files
- `__pycache__/` directory and all contents
- All `.pyc` files

### 3. Temporary Cleanup Script
- `cleanup_comments.py` (this was just used for the one-time cleanup)

### 4. Analysis Log Files (optional - keep if you need historical records)
These are in `analysis_results/` and subdirectories:
- `analysis_results/analysis_results/*.log`
- `analysis_results/brow_cocked/*.log`
- `analysis_results/hypertonicity/*.log`
- `analysis_results/lower/*.log`
- `analysis_results/mid/*.log`
- `analysis_results/upper/*.log`
- `analysis_results/mentalis/*.log`
- `analysis_results/ocular_oral/*.log`
- `analysis_results/oral_ocular/*.log`
- `analysis_results/snarl_smile/*.log`
- And various `*_analysis.log` and `*_analysis_thresh.log` files in root analysis_results/

### 5. Performance Analysis Logs (optional - keep if you need historical records)
- `performance_analysis_results/performance_analysis.log`

### 6. Training Logs (optional - keep if you need historical records)
These are in `logs/` directory:
- `logs/brow_cocked_training.log`
- `logs/hypertonicity_training.log`
- `logs/lower_face_training.log`
- `logs/mentalis_training.log`
- `logs/mid_face_training.log`
- `logs/ocular_oral_training.log`
- `logs/oral_ocular_training.log`
- `logs/snarl_smile_training.log`
- `logs/upper_face_training.log`
- `logs/synkinesis_main_pipeline_orchestrator.log`
- `logs/synkinesis_pipeline_main.log`

**KEEP these logs:**
- `logs/facial_au_analyzer.log` (active runtime log)
- `logs/predictive_au_analysis.log` (analysis results)
- `logs/predictive_au_analysis_per_action.log` (analysis results)

## Files to KEEP (Active/Important)

### Core Python Modules (all active)
1. `main.py` - Main entry point
2. `facial_au_analyzer.py` - Core analysis logic
3. `facial_au_batch_processor.py` - Batch processing
4. `facial_au_constants.py` - Configuration constants
5. `facial_au_frame_extractor.py` - Frame extraction
6. `facial_au_gui.py` - GUI interface
7. `facial_au_visualizer.py` - Visualization
8. `facial_paralysis_detection.py` - Paralysis detection interface
9. `paralysis_detector.py` - Paralysis detection logic
10. `paralysis_config.py` - Configuration
11. `paralysis_utils.py` - Utility functions
12. `paralysis_training_pipeline.py` - Training pipeline
13. `paralysis_model_trainer.py` - Model training
14. `paralysis_training_helpers.py` - Training helpers
15. `paralysis_performance.py` - Performance analysis
16. `upper_face_features.py` - Feature extraction
17. `mid_face_features.py` - Feature extraction
18. `lower_face_features.py` - Feature extraction

### Model Files (all active - DO NOT DELETE)
- `models/*.pkl` - All trained model artifacts
- `models/*.list` - Feature lists
- `models/synkinesis/*` - Synkinesis model artifacts

### Data Files (all active - DO NOT DELETE)
- `analysis_results/*.csv` - Analysis results
- `analysis_results/*.xlsx` - Excel reports
- `analysis_results/*.png` - Confusion matrices and plots

## Recommended Cleanup Commands

```bash
# 1. Remove main directory log files
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis"
rm -f retrain_*.log training_output.log main_output.log

# 2. Remove Python cache
rm -rf __pycache__

# 3. Remove cleanup script
rm -f cleanup_comments.py

# 4. (Optional) Remove analysis logs but keep results
find analysis_results -name "*.log" -delete
find performance_analysis_results -name "*.log" -delete

# 5. (Optional) Remove training logs but keep the important ones
cd logs
rm -f *_training.log synkinesis_*.log
# Keep: facial_au_analyzer.log, predictive_au_analysis*.log
```

## Summary

### Must Delete (Safe):
- Training logs in main directory (6 files)
- `__pycache__/` directory
- `cleanup_comments.py`

### Optional Delete (Historical logs):
- Analysis logs in analysis_results/ (~26 files)
- Training logs in logs/ (~11 files)
- Performance analysis logs (1 file)

**Total potential cleanup: ~45 files**
**Disk space saved: Minimal (logs are typically small)**

### Do NOT Delete:
- Any `.py` files (except cleanup_comments.py)
- Any `.pkl` files (trained models)
- Any `.list` files (feature lists)
- Any `.csv` or `.xlsx` files (analysis results)
- Any `.png` files (visualizations)
- `logs/facial_au_analyzer.log` (active)
- `logs/predictive_au_analysis*.log` (important results)
