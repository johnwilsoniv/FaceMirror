# analyze_mid_face_features.py

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import necessary functions and config ---
# Need extract_features and its helpers, plus config for actions/aus
try:
    from mid_face_features import extract_features, process_targets, calculate_ratio, calculate_percent_diff, calculate_single_ratio, calculate_single_percent_diff
    from mid_face_config import LOG_DIR, CLASS_NAMES, FEATURE_CONFIG, MID_FACE_ACTIONS, MID_FACE_AUS
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}. Ensure config and features files exist.")
    exit()
# --- End Imports ---

# Configure Logging and Output Directory
ANALYSIS_OUTPUT_DIR = 'analysis_results/feature_analysis_midface'
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'midface_feature_analysis.log'), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper to standardize labels (same as in performance analysis) ---
def standardize_expert_labels(val):
    if val is None or pd.isna(val): return 'None'
    val_str = str(val).strip().lower()
    if val_str in ['none', 'no', 'n/a', '0', '0.0', 'normal', '']: return 'None'
    if val_str in ['partial', 'mild', 'moderate', '1', '1.0']: return 'Partial'
    if val_str in ['complete', 'severe', '2', '2.0']: return 'Complete'
    return 'None'
# --- End Helper ---

def analyze_features(results_file='combined_results.csv', expert_file='FPRS FP Key.csv'):
    """
    Loads data, generates mid-face features, and analyzes their distribution
    across expert-graded paralysis classes.
    """
    logger.info("--- Mid-Face Feature Distribution Analysis ---")

    # Load datasets
    logger.info("Loading datasets...")
    try:
        results_df = pd.read_csv(results_file, low_memory=False)
        expert_df = pd.read_csv(expert_file)
    except Exception as e:
        logger.error(f"Error loading data: {e}"); return

    # Rename expert columns & Standardize Labels
    expert_df = expert_df.rename(columns={
        'Patient': 'Patient ID',
        'Paralysis - Left Mid Face': 'Expert_Left_Mid_Face',
        'Paralysis - Right Mid Face': 'Expert_Right_Mid_Face'
    })
    expert_df['Expert_Left_Mid_Face'] = expert_df['Expert_Left_Mid_Face'].apply(standardize_expert_labels)
    expert_df['Expert_Right_Mid_Face'] = expert_df['Expert_Right_Mid_Face'].apply(standardize_expert_labels)

    # Ensure Patient IDs are strings
    results_df['Patient ID'] = results_df['Patient ID'].astype(str)
    expert_df['Patient ID'] = expert_df['Patient ID'].astype(str)

    # Merge
    merged_df = pd.merge(results_df, expert_df[['Patient ID', 'Expert_Left_Mid_Face', 'Expert_Right_Mid_Face']], on='Patient ID', how='inner')
    logger.info(f"Merged data contains {len(merged_df)} patients")
    if merged_df.empty: logger.error("Merge failed."); return

    # Generate Full Feature Sets for Both Sides
    logger.info("Generating full feature set for analysis...")
    left_features_df = extract_features(merged_df, 'Left')
    right_features_df = extract_features(merged_df, 'Right')

    # Add Expert Labels and Side Indicator
    left_features_df['Expert_Label'] = merged_df['Expert_Left_Mid_Face']
    right_features_df['Expert_Label'] = merged_df['Expert_Right_Mid_Face']
    left_features_df['Side'] = 'Left'
    right_features_df['Side'] = 'Right'

    # Combine into one DataFrame for analysis
    analysis_df = pd.concat([left_features_df, right_features_df], ignore_index=True)

    # Handle potential NaN/Inf from feature generation (should be minimal now)
    analysis_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    analysis_df.fillna(0, inplace=True) # Fill NaNs with 0 for stats/plotting

    logger.info(f"Analysis DataFrame shape: {analysis_df.shape}")

    # --- Analyze Key Features ---
    key_features = [
        'ET_AU45_norm', 'ET_AU07_norm', 'ET_AU06_norm', # Core AUs during forceful closure
        'BS_AU06_norm',                                 # Cheek raise during smile
        'ET_AU45_ratio', 'ET_AU07_percent_diff',        # Asymmetry during forceful closure
        'BS_AU06_percent_diff',                         # Asymmetry during smile
        'ES_ET_AU45_ratio'                              # Ratio between soft/hard closure
    ]

    # Ensure selected key features exist in the dataframe
    key_features = [f for f in key_features if f in analysis_df.columns]
    if not key_features:
        logger.error("None of the specified key features were found in the generated feature set.")
        return

    logger.info("\n--- Descriptive Statistics for Key Features by Class & Side ---")
    try:
        stats = analysis_df.groupby(['Side', 'Expert_Label'])[key_features].agg(['mean', 'median', 'std'])
        # Improve formatting for display
        pd.set_option('display.float_format', '{:.3f}'.format)
        print(stats)
        pd.reset_option('display.float_format') # Reset formatting
        # Save stats to CSV
        stats.to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, 'key_feature_stats.csv'))
        logger.info(f"Saved key feature stats to {ANALYSIS_OUTPUT_DIR}")

    except KeyError as e:
         logger.error(f"KeyError accessing columns for stats: {e}. Available columns: {analysis_df.columns.tolist()}")
    except Exception as e:
         logger.error(f"Error calculating statistics: {e}")


    logger.info("\n--- Generating Box Plots for Key Features ---")
    plot_order = ['None', 'Partial', 'Complete']
    for feature in key_features:
        plt.figure(figsize=(10, 6))
        try:
            sns.boxplot(data=analysis_df, x='Expert_Label', y=feature, hue='Side', order=plot_order)
            plt.title(f'Distribution of {feature} by Paralysis Class and Side')
            plt.xlabel("Expert Graded Paralysis")
            plt.ylabel("Feature Value")
            plt.tight_layout()
            plot_filename = os.path.join(ANALYSIS_OUTPUT_DIR, f"{feature}_distribution.png")
            plt.savefig(plot_filename)
            plt.close()
            logger.info(f"Saved plot: {plot_filename}")
        except Exception as e:
            logger.error(f"Error creating plot for {feature}: {e}")
            plt.close() # Ensure plot is closed even if error occurs


    logger.info("--- Feature analysis complete ---")


if __name__ == "__main__":
    # Ensure the correct config is used by the imported functions
    logger.info(f"Using Actions: {FEATURE_CONFIG.get('actions')}")
    logger.info(f"Using AUs: {FEATURE_CONFIG.get('aus')}")
    analyze_features()