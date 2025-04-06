# facial_au_analyzer.py
# Analyzes facial Action Units from OpenFace output for a patient.
# - Integrated Hypertonicity detection
# - Updated output generation with synkinesis consolidation
# - Includes debug logging capabilities

import pandas as pd
import numpy as np
import os
import logging
import glob
import re
import json # Needed for logging dicts
import cv2 # Needed for frame extraction

# Attempt to import constants, provide fallbacks if missing
try:
    from facial_au_constants import (
        ACTION_TO_AUS, ACTION_DESCRIPTIONS, ALL_AU_COLUMNS,
        BASELINE_AU_ACTIVATIONS, ZONE_SPECIFIC_ACTIONS,
        FACIAL_ZONES, FACIAL_ZONE_WEIGHTS,
        SYNKINESIS_TYPES, INCLUDED_ACTIONS, HYPERTONICITY_AUS
    )
except ImportError:
    logging.error("Could not import constants from facial_au_constants. Using placeholders.")
    ACTION_TO_AUS = {'BS': ['AU12_r'], 'SS': ['AU12_r'], 'RE': ['AU01_r', 'AU02_r'], 'ES': ['AU06_r', 'AU07_r'], 'ET': ['AU06_r', 'AU07_r', 'AU45_r'], 'SE': ['AU10_r', 'AU12_r'], 'SO': ['AU25_r', 'AU26_r'], 'BL': []}
    ACTION_DESCRIPTIONS = {'BS': 'Big Smile', 'SS': 'Soft Smile', 'RE': 'Raise Eyebrows', 'ES': 'Close Eyes Softly', 'ET': 'Close Eyes Tightly', 'SE': 'Say E', 'SO': 'Say O', 'BL': 'Baseline'}
    ALL_AU_COLUMNS = [f'AU{i:02d}_r' for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]
    BASELINE_AU_ACTIVATIONS = {au: 0.1 for au in ALL_AU_COLUMNS} # Simple default
    FACIAL_ZONES = {'upper': ['AU01_r', 'AU02_r', 'AU04_r'], 'mid': ['AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU45_r'], 'lower': ['AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r']}
    ZONE_SPECIFIC_ACTIONS = {'upper': ['RE'], 'mid': ['ES', 'ET'], 'lower': ['BS', 'SS', 'SE', 'SO']}
    FACIAL_ZONE_WEIGHTS = {'upper': 1, 'mid': 1, 'lower': 1}
    SYNKINESIS_TYPES = ['Ocular-Oral', 'Oral-Ocular', 'Mentalis', 'Snarl-Smile']
    INCLUDED_ACTIONS = ['BS', 'SS', 'RE', 'ES', 'ET', 'SE', 'SO']
    HYPERTONICITY_AUS = ['AU12_r', 'AU14_r', 'AU06_r', 'AU07_r']


# --- Import Detectors ---
try: from facial_paralysis_detection import detect_side_paralysis
except ImportError: logging.error("Failed import: detect_side_paralysis"); detect_side_paralysis = None
try: from facial_synkinesis_detection import SynkinesisDetector
except ImportError: logging.error("Failed import: SynkinesisDetector"); SynkinesisDetector = None
try: from hypertonicity_detector import HypertonicityDetector
except ImportError: logging.error("Failed import: HypertonicityDetector"); HypertonicityDetector = None
# --- End Imports ---

from facial_au_visualizer import FacialAUVisualizer

# Get logger instance (assuming configured by main application)
logger = logging.getLogger(__name__)

class FacialAUAnalyzer:
    """ Analyzes facial Action Units from OpenFace output for a patient. """

    def __init__(self):
        """ Initializes the analyzer. """
        self.action_to_aus = ACTION_TO_AUS
        self.action_descriptions = ACTION_DESCRIPTIONS
        self.results = {} # Holds action-specific results + patient_summary
        self.patient_id = None
        self.left_data = None
        self.right_data = None
        self.frame_paths = {}
        self.baseline_values = {'left': {}, 'right': {}}
        self.output_dir = None
        self.facial_zones = FACIAL_ZONES
        self.facial_zone_weights = FACIAL_ZONE_WEIGHTS
        self.segmented_actions = {}
        self.ml_results = {} # Holds the dict passed to ML models

        logger.info("Instantiating ML detectors...")
        try:
            self.synkinesis_detector = SynkinesisDetector() if SynkinesisDetector else None
            if not self.synkinesis_detector: logger.error("SynkinesisDetector could not be instantiated.")
        except Exception as e:
            logger.error(f"Error instantiating SynkinesisDetector: {e}", exc_info=True)
            self.synkinesis_detector = None

        try:
            self.hypertonicity_detector = HypertonicityDetector() if HypertonicityDetector else None
            if not self.hypertonicity_detector: logger.error("HypertonicityDetector could not be instantiated.")
        except Exception as e:
            logger.error(f"Error instantiating HypertonicityDetector: {e}", exc_info=True)
            self.hypertonicity_detector = None

        logger.info("ML detectors instantiated (if available).")

        try:
            self.visualizer = FacialAUVisualizer()
            logger.info("Visualizer initialized.")
        except Exception as e:
            logger.error(f"Error instantiating FacialAUVisualizer: {e}", exc_info=True)
            self.visualizer = None


    def load_data(self, left_csv_path, right_csv_path):
        """ Loads and prepares left/right mirrored OpenFace data. """
        action_col_name = 'action'
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_Load"
        logger.info(f"({pid_for_log}) Loading data: Left='{os.path.basename(left_csv_path)}', Right='{os.path.basename(right_csv_path)}'")
        try:
            self.left_data = pd.read_csv(left_csv_path, low_memory=False)
            self.right_data = pd.read_csv(right_csv_path, low_memory=False)
            logger.debug(f"({pid_for_log}) Columns in loaded left_data: {self.left_data.columns.tolist()}")
            logger.debug(f"({pid_for_log}) Columns in loaded right_data: {self.right_data.columns.tolist()}")

            for df, side_name, csv_path in [(self.left_data, 'left', left_csv_path), (self.right_data, 'right', right_csv_path)]:
                if action_col_name not in df.columns:
                    logger.error(f"({pid_for_log}) '{action_col_name}' column missing in {side_name} CSV ({os.path.basename(csv_path)}). Cannot proceed.")
                    return False
                else:
                    nan_count = df[action_col_name].isna().sum()
                    if nan_count > 0:
                        logger.warning(f"({pid_for_log}) Found {nan_count} NaN values in '{action_col_name}' column ({side_name}). Filling with 'Unknown'.")
                    # Fill NaNs, convert to string, strip whitespace, uppercase
                    df[action_col_name] = df[action_col_name].fillna('Unknown').astype(str).str.strip().str.upper()
                    logger.debug(f"({pid_for_log}) Unique values in {side_name} '{action_col_name}' after processing: {df[action_col_name].unique()}")

            self._calculate_baseline_values()
            logger.info(f"({self.patient_id}) Data loaded and baseline calculated successfully.")
            return True
        except FileNotFoundError:
            logger.error(f"({pid_for_log}) File not found during loading: {left_csv_path} or {right_csv_path}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"({self.patient_id}) Error loading data: {e}", exc_info=True)
            return False

    def _calculate_baseline_values(self):
        """Calculates median baseline AU values from 'BL' action or initial frames."""
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_Baseline"
        action_col_name = 'action'
        # Initialize with defaults
        for au in ALL_AU_COLUMNS:
            self.baseline_values['left'][au] = BASELINE_AU_ACTIVATIONS.get(au, 0.1)
            self.baseline_values['right'][au] = BASELINE_AU_ACTIVATIONS.get(au, 0.1)

        if self.left_data is None or self.right_data is None:
            logger.warning(f"({pid_for_log}) Data not loaded. Using default baseline values.")
            return

        try:
            if action_col_name not in self.left_data.columns or action_col_name not in self.right_data.columns:
                logger.warning(f"({pid_for_log}) '{action_col_name}' column missing. Using default baseline values.")
                return

            # Try using 'BL' frames
            left_baseline = self.left_data[self.left_data[action_col_name] == 'BL']
            right_baseline = self.right_data[self.right_data[action_col_name] == 'BL']

            if not left_baseline.empty and not right_baseline.empty:
                 logger.info(f"({pid_for_log}) Calculating baseline from 'BL' frames ({len(left_baseline)} left, {len(right_baseline)} right)...")
                 for au in ALL_AU_COLUMNS:
                     if au in left_baseline.columns:
                         median_val = left_baseline[au].median()
                         self.baseline_values['left'][au] = float(median_val) if pd.notna(median_val) else 0.0
                     if au in right_baseline.columns:
                         median_val = right_baseline[au].median()
                         self.baseline_values['right'][au] = float(median_val) if pd.notna(median_val) else 0.0
                 return # Success using BL frames

            logger.warning(f"({pid_for_log}) No 'BL' action found or empty.")
            # Fallback to first 5 frames if BL is missing
            if len(self.left_data) >= 5 and len(self.right_data) >= 5:
                 logger.info(f"({pid_for_log}) Calculating baseline from first 5 frames.")
                 left_init = self.left_data.iloc[:5]
                 right_init = self.right_data.iloc[:5]
                 for au in ALL_AU_COLUMNS:
                     if au in left_init.columns:
                         median_val = left_init[au].median()
                         self.baseline_values['left'][au] = float(median_val) if pd.notna(median_val) else 0.0
                     if au in right_init.columns:
                         median_val = right_init[au].median()
                         self.baseline_values['right'][au] = float(median_val) if pd.notna(median_val) else 0.0
                 return # Success using first 5 frames

            logger.warning(f"({pid_for_log}) Insufficient data for baseline calculation (<5 frames and no 'BL'). Using default values.")
        except Exception as e:
             logger.error(f"({pid_for_log}) Error calculating baseline: {e}. Using default values.", exc_info=True)
             # Reset to defaults on error
             for au in ALL_AU_COLUMNS:
                self.baseline_values['left'][au] = BASELINE_AU_ACTIVATIONS.get(au, 0.1)
                self.baseline_values['right'][au] = BASELINE_AU_ACTIVATIONS.get(au, 0.1)

    def _normalize_au_value(self, side, au, value):
        """Normalizes a single AU value by subtracting the baseline."""
        baseline = self.baseline_values[side].get(au, BASELINE_AU_ACTIVATIONS.get(au, 0.1))
        baseline = float(baseline) if pd.notna(baseline) else 0.0
        value_float = float(value) if pd.notna(value) else 0.0
        normalized = max(0, value_float - baseline)
        return float(normalized)

    def analyze_maximal_intensity(self):
        """Finds the frame of maximal intensity for each action based on key AUs."""
        if self.left_data is None or self.right_data is None:
            logger.error("Data not loaded. Cannot analyze maximal intensity."); return None
        self.results = {} # Reset results
        self.ml_results = {} # Reset ML results storage

        # Initialize patient_summary within self.results
        self.results['patient_summary'] = {
            'hypertonicity': { # Default state for hypertonicity
                'detected': False, 'left': False, 'right': False,
                'confidence_left': 0.0, 'confidence_right': 0.0,
                'contributing_aus_left': set(), 'contributing_aus_right': set()
            }
            # Add other patient-level summaries here if needed later
        }

        action_col_name = 'action'
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_Intensity"
        self.segmented_actions = {} # Reset segmented actions
        all_actions_present = pd.concat([self.left_data[action_col_name], self.right_data[action_col_name]]).unique()
        logger.debug(f"({pid_for_log}) Actions present in data: {all_actions_present}")
        # Ensure INCLUDED_ACTIONS and SYNKINESIS_TYPES are valid lists
        local_included_actions = INCLUDED_ACTIONS if INCLUDED_ACTIONS else []
        local_synk_types = SYNKINESIS_TYPES if SYNKINESIS_TYPES else []
        actions_to_segment = set(local_included_actions) | {'BL'}
        logger.debug(f"({pid_for_log}) Actions to segment: {actions_to_segment}")

        # Segment data by action first
        for action in actions_to_segment:
            left_action_data = self.left_data[self.left_data[action_col_name] == action].copy()
            right_action_data = self.right_data[self.right_data[action_col_name] == action].copy()
            if left_action_data.empty and right_action_data.empty:
                logger.warning(f"({pid_for_log}) No data found for action '{action}'. Skipping segmentation.")
                continue
            self.segmented_actions[action] = {'left': left_action_data, 'right': right_action_data}
            logger.debug(f"({pid_for_log}) Segmented action '{action}': Left frames={len(left_action_data)}, Right frames={len(right_action_data)}")

        # Find max intensity frame for each segmented action
        for action, segment_data in self.segmented_actions.items():
            left_action_data = segment_data['left']
            right_action_data = segment_data['right']

            # Initialize the results structure for this action
            base_results_struct = {
                'max_side': 'N/A', 'max_frame': np.nan, 'max_value': np.nan,
                'left': {'idx': None, 'frame': np.nan, 'au_values': {}, 'normalized_au_values': {}},
                'right': {'idx': None, 'frame': np.nan, 'au_values': {}, 'normalized_au_values': {}},
                'paralysis': {}, # Populated later
                'synkinesis': { # Initialize synkinesis structure
                     'detected': False, 'types': [],
                     'side_specific': {st: {'left': False, 'right': False} for st in local_synk_types},
                     'confidence': {st: {'left': 0.0, 'right': 0.0} for st in local_synk_types},
                     'contributing_aus': {}
                 }
            }
            self.results[action] = base_results_struct

            # Special handling for Baseline ('BL') - find median frame
            if action == 'BL':
                 median_frame_left = int(left_action_data['frame'].median()) if not left_action_data.empty and 'frame' in left_action_data else np.nan
                 median_frame_right = int(right_action_data['frame'].median()) if not right_action_data.empty and 'frame' in right_action_data else np.nan
                 # Use nanmedian to handle cases where one side might be missing BL frames
                 valid_frames = [f for f in [median_frame_left, median_frame_right] if pd.notna(f)]
                 final_frame_num = int(np.nanmedian(valid_frames)) if valid_frames else 0

                 # Find closest index in original dataframes
                 left_final_idx = None
                 if not left_action_data.empty and 'frame' in left_action_data:
                     left_final_idx = (left_action_data['frame'] - final_frame_num).abs().idxmin()

                 right_final_idx = None
                 if not right_action_data.empty and 'frame' in right_action_data:
                     right_final_idx = (right_action_data['frame'] - final_frame_num).abs().idxmin()

                 self.results[action]['max_frame'] = final_frame_num
                 self.results[action]['left']['idx'] = left_final_idx
                 self.results[action]['left']['frame'] = final_frame_num
                 self.results[action]['right']['idx'] = right_final_idx
                 self.results[action]['right']['frame'] = final_frame_num
                 self.results[action]['max_value'] = np.nan # Max value not applicable for BL median
                 logger.debug(f"({pid_for_log}) Processed BL action, using median frame {final_frame_num}.")
                 continue # Skip max intensity calculation for BL

            # Max intensity logic for other actions
            key_aus = self.action_to_aus.get(action)
            if not key_aus:
                logger.warning(f"({pid_for_log}) No key AUs defined for action '{action}'. Skipping max intensity.")
                continue

            # Check required columns exist
            required_cols = ['frame'] + key_aus
            missing_left = [col for col in required_cols if col != 'frame' and col not in left_action_data.columns]
            missing_right = [col for col in required_cols if col != 'frame' and col not in right_action_data.columns]
            frame_missing_left = 'frame' not in left_action_data.columns and not left_action_data.empty
            frame_missing_right = 'frame' not in right_action_data.columns and not right_action_data.empty

            if missing_left or missing_right or frame_missing_left or frame_missing_right:
                missing_log = []
                if missing_left: missing_log.append(f"Left missing AUs: {missing_left}")
                if missing_right: missing_log.append(f"Right missing AUs: {missing_right}")
                if frame_missing_left: missing_log.append("Left missing 'frame'")
                if frame_missing_right: missing_log.append("Right missing 'frame'")
                logger.warning(f"({pid_for_log}) Skipping max intensity for '{action}' due to missing columns: {'; '.join(missing_log)}.")
                continue

            # Calculate average intensity of key AUs, handling potential missing AUs within rows
            valid_key_aus_left = [au for au in key_aus if au in left_action_data.columns]
            valid_key_aus_right = [au for au in key_aus if au in right_action_data.columns]

            if valid_key_aus_left: left_action_data['key_au_avg'] = left_action_data[valid_key_aus_left].mean(axis=1, skipna=True)
            else: left_action_data['key_au_avg'] = np.nan
            if valid_key_aus_right: right_action_data['key_au_avg'] = right_action_data[valid_key_aus_right].mean(axis=1, skipna=True)
            else: right_action_data['key_au_avg'] = np.nan

            # Drop rows where average couldn't be calculated (e.g., all key AUs were NaN)
            left_action_data = left_action_data.dropna(subset=['key_au_avg'])
            right_action_data = right_action_data.dropna(subset=['key_au_avg'])

            # Find the side and index with the overall maximum average intensity
            max_side, max_idx, max_value, max_frame_num = 'N/A', None, np.nan, np.nan
            left_max_value, right_max_value = -np.inf, -np.inf # Initialize to handle all-zero cases
            left_max_idx, right_max_idx = None, None

            if not left_action_data.empty:
                left_max_idx = left_action_data['key_au_avg'].idxmax()
                left_max_value = left_action_data.loc[left_max_idx, 'key_au_avg']
            if not right_action_data.empty:
                right_max_idx = right_action_data['key_au_avg'].idxmax()
                right_max_value = right_action_data.loc[right_max_idx, 'key_au_avg']

            # Determine max side based on calculated max values
            if left_max_idx is not None and right_max_idx is not None:
                if left_max_value >= right_max_value:
                    max_side, max_idx, max_value = 'left', left_max_idx, left_max_value
                    max_frame_num = left_action_data.loc[max_idx, 'frame']
                else:
                    max_side, max_idx, max_value = 'right', right_max_idx, right_max_value
                    max_frame_num = right_action_data.loc[max_idx, 'frame']
            elif left_max_idx is not None: # Only left had data
                max_side, max_idx, max_value = 'left', left_max_idx, left_max_value
                max_frame_num = left_action_data.loc[max_idx, 'frame']
            elif right_max_idx is not None: # Only right had data
                max_side, max_idx, max_value = 'right', right_max_idx, right_max_value
                max_frame_num = right_action_data.loc[max_idx, 'frame']
            else:
                logger.warning(f"({pid_for_log}) No valid intensity data found for action '{action}' after dropping NaNs. Skipping max intensity.")
                continue

            # Find the corresponding indices in the original segmented data for the max frame
            left_action_data_orig = segment_data['left']
            right_action_data_orig = segment_data['right']
            left_final_idx, right_final_idx = None, None # Reset for this action

            if pd.notna(max_frame_num):
                 matching_left_indices = left_action_data_orig.index[left_action_data_orig['frame'] == max_frame_num].tolist()
                 matching_right_indices = right_action_data_orig.index[right_action_data_orig['frame'] == max_frame_num].tolist()
                 left_final_idx = matching_left_indices[0] if matching_left_indices else None
                 right_final_idx = matching_right_indices[0] if matching_right_indices else None

                 # Fallback: If exact frame not found, use the closest frame
                 if left_final_idx is None and not left_action_data_orig.empty:
                     closest_idx_left = (left_action_data_orig['frame'] - max_frame_num).abs().idxmin()
                     left_final_idx = closest_idx_left
                     logger.warning(f"({pid_for_log} - {action}) Exact frame {int(max_frame_num)} not found in left data. Using closest index: {left_final_idx}.")
                 if right_final_idx is None and not right_action_data_orig.empty:
                     closest_idx_right = (right_action_data_orig['frame'] - max_frame_num).abs().idxmin()
                     right_final_idx = closest_idx_right
                     logger.warning(f"({pid_for_log} - {action}) Exact frame {int(max_frame_num)} not found in right data. Using closest index: {right_final_idx}.")
            else:
                 logger.error(f"({pid_for_log}) Could not determine max frame number for action '{action}'. Skipping index assignment.")
                 continue # Skip assignment if max_frame_num is NaN

            if left_final_idx is None and right_final_idx is None:
                logger.error(f"({pid_for_log}) Could not determine any final frame indices for action '{action}'. Skipping max intensity store."); continue

            # Store max intensity results
            final_frame_num_int = int(max_frame_num) if pd.notna(max_frame_num) else np.nan
            self.results[action]['max_side'] = max_side
            self.results[action]['max_frame'] = final_frame_num_int
            self.results[action]['max_value'] = float(max_value) if pd.notna(max_value) else np.nan
            self.results[action]['left']['idx'] = left_final_idx
            self.results[action]['left']['frame'] = final_frame_num_int
            self.results[action]['right']['idx'] = right_final_idx
            self.results[action]['right']['frame'] = final_frame_num_int
            logger.debug(f"({pid_for_log} - {action}) Max intensity frame: {final_frame_num_int}, Max side: {max_side}, Left Idx: {left_final_idx}, Right Idx: {right_final_idx}")

        # Populate AU values for the determined max frames across all actions
        for action, action_data in self.results.items():
            if action == 'patient_summary': continue # Skip the summary entry

            left_final_idx = action_data['left'].get('idx')
            right_final_idx = action_data['right'].get('idx')

            for au in ALL_AU_COLUMNS:
                for side_loop, idx, data_source in [('left', left_final_idx, self.left_data), ('right', right_final_idx, self.right_data)]:
                    raw_value = np.nan
                    if idx is not None and au in data_source.columns and idx in data_source.index:
                        raw_value = data_source.loc[idx, au]
                        raw_value = float(raw_value) if pd.notna(raw_value) else np.nan
                    # Store raw value
                    action_data[side_loop]['au_values'][au] = raw_value
                    # Normalize value (handles NaN internally) and store
                    action_data[side_loop]['normalized_au_values'][au] = self._normalize_au_value(side_loop, au, raw_value)

        logger.info(f"Completed max intensity analysis for patient {self.patient_id}")
        return self.results

    def generate_ml_input_dict(self):
        """
        Generates a flat dictionary from the analysis results suitable for ML model input.
        Includes raw and normalized values for max intensity frames.
        """
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_MLDict"
        if not self.results:
            logger.error(f"({pid_for_log}) No results available to generate ML input dictionary.")
            return None

        ml_input_dict = {'Patient ID': self.patient_id}
        logger.debug(f"({pid_for_log}) Generating ML input dict...")

        # Define actions to include (ensure BL is always included if present)
        local_included_actions = INCLUDED_ACTIONS if INCLUDED_ACTIONS else []
        actions_to_process = set(local_included_actions) | {'BL'}

        for action in actions_to_process:
            prefix = f'{action}_'
            action_info = self.results.get(action)

            # Handle missing or invalid action data gracefully
            if not action_info or not isinstance(action_info, dict):
                logger.warning(f"({pid_for_log}) Data missing or invalid for action '{action}'. Filling ML dict keys with 0.0 or NaN.")
                is_bl = (action == 'BL')
                ml_input_dict[prefix + 'Max Side'] = 'N/A' if is_bl else 'NA'
                ml_input_dict[prefix + 'Max Frame'] = np.nan
                ml_input_dict[prefix + 'Max Value'] = np.nan
                for side_key in ['left', 'right']:
                    side_label = side_key.capitalize()
                    for au in ALL_AU_COLUMNS:
                        ml_input_dict[f'{prefix}{side_label} {au}'] = 0.0 # Use 0 for raw missing
                        ml_input_dict[f'{prefix}{side_label} {au} (Normalized)'] = 0.0 # Use 0 for norm missing
                continue # Move to next action

            # Populate max intensity info
            is_bl = (action == 'BL')
            ml_input_dict[prefix + 'Max Side'] = 'N/A' if is_bl else action_info.get('max_side', 'NA')
            ml_input_dict[prefix + 'Max Frame'] = action_info.get('max_frame', np.nan)
            ml_input_dict[prefix + 'Max Value'] = action_info.get('max_value', np.nan)

            # Populate AU values (Raw and Normalized)
            for side_key in ['left', 'right']:
                side_label = side_key.capitalize()
                au_values = action_info.get(side_key, {}).get('au_values', {})
                norm_au_values = action_info.get(side_key, {}).get('normalized_au_values', {})
                for au in ALL_AU_COLUMNS:
                    raw_val = au_values.get(au, np.nan) # Default to NaN if AU missing for frame
                    norm_val = norm_au_values.get(au, 0.0) # Default to 0 if norm missing

                    # Convert NaN raw_val to 0.0 for ML input
                    raw_ml_val = 0.0
                    if pd.notna(raw_val): raw_ml_val = float(raw_val)

                    # Convert NaN norm_val to 0.0
                    norm_ml_val = 0.0
                    if pd.notna(norm_val): norm_ml_val = float(norm_val)

                    # Populate the dictionary
                    ml_input_dict[f'{prefix}{side_label} {au}'] = raw_ml_val
                    ml_input_dict[f'{prefix}{side_label} {au} (Normalized)'] = norm_ml_val

        logger.debug(f"({pid_for_log}) Generated ML input dict with {len(ml_input_dict)} keys.")
        # Check for NaNs that might still be present
        nan_keys = [k for k, v in ml_input_dict.items() if pd.isna(v)]
        if nan_keys:
            logger.warning(f"({pid_for_log}) ML input dict contains NaN values for keys: {nan_keys[:10]}...")

        return ml_input_dict

    def _detect_hypertonicity(self, ml_input_dict):
        """
        Detects hypertonicity using the dedicated detector.
        Updates self.results['patient_summary']['hypertonicity'].
        """
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_DetectHyperT"
        if not self.hypertonicity_detector:
            logger.warning(f"({pid_for_log}) Hypertonicity detector not available. Skipping detection.")
            return # Exit if detector not available
        if not ml_input_dict:
            logger.error(f"({pid_for_log}) ML input dictionary is missing. Cannot run hypertonicity detection.")
            return # Exit if no input dict

        logger.info(f"({pid_for_log}) Starting ML hypertonicity detection...")
        # Initialize results structure
        hypertonicity_results = {
            'detected': False, 'left': False, 'right': False,
            'confidence_left': 0.0, 'confidence_right': 0.0,
            'contributing_aus_left': set(), 'contributing_aus_right': set() # Placeholder
        }

        # --- ADDED: Log the input dict received by the detector in the pipeline ---
        logger.debug(f"HyperT PIPELINE Detect ({pid_for_log}) - ml_input_dict passed to detector:")
        try:
            # Log first ~20 items for brevity
            first_few_items = {k: ml_input_dict[k] for k in list(ml_input_dict)[:20]}
            logger.debug(json.dumps(first_few_items, indent=2, default=str))
        except Exception as log_e:
            logger.warning(f"HyperT PIPELINE Detect ({pid_for_log}) - Error logging ml_input_dict: {log_e}")
        # --- END ADDED LOGGING ---

        # Run detection for both sides
        for side in ['left', 'right']:
            side_label = side.capitalize()
            try:
                # The call below triggers logging inside hypertonicity_features.extract_features_for_detection
                is_detected, confidence = self.hypertonicity_detector.detect_hypertonicity(ml_input_dict, side_label)
                if is_detected:
                    hypertonicity_results['detected'] = True
                    hypertonicity_results[side] = True
                # Store confidence regardless of detection
                hypertonicity_results[f'confidence_{side}'] = confidence
                logger.debug(f"({pid_for_log}) Hypertonicity check {side}: Detected={is_detected}, Conf={confidence:.3f}")

            except Exception as e_hyper:
                 logger.error(f"({pid_for_log}) Exception during detect_hypertonicity call for {side}: {e_hyper}", exc_info=True)
                 # Mark detection as False on error and confidence as 0
                 hypertonicity_results[side] = False
                 hypertonicity_results[f'confidence_{side}'] = 0.0

        # Store the results in the patient summary section of self.results
        # Ensure 'patient_summary' exists before trying to update it
        if 'patient_summary' not in self.results:
            self.results['patient_summary'] = {}
            logger.warning(f"({pid_for_log}) 'patient_summary' key was missing in self.results before hypertonicity storage.")
        self.results['patient_summary']['hypertonicity'] = hypertonicity_results # Update/overwrite the entry
        logger.info(f"({pid_for_log}) Hypertonicity detection complete. Detected={hypertonicity_results['detected']} (L={hypertonicity_results['left']}, R={hypertonicity_results['right']})")


    def analyze_all_actions(self, run_ml_paralysis=True, run_ml_synkinesis=True, run_ml_hypertonicity=True):
        """Performs full analysis including intensity, ML detections."""
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_AnalyzeAll"
        logger.info(f"({pid_for_log}) Starting full analysis...")

        # Run max intensity analysis first, it initializes self.results including patient_summary
        if self.analyze_maximal_intensity() is None: # analyze_maximal_intensity returns self.results or None on failure
            logger.error(f"({pid_for_log}) Maximal intensity analysis failed. Cannot proceed."); return None

        # Generate ML input based on max intensity results
        ml_input_dict = self.generate_ml_input_dict()
        if ml_input_dict is None:
            logger.warning(f"({pid_for_log}) Failed to generate ML input dict. ML detections will be skipped.")
            self.ml_results = {} # Ensure it's an empty dict
        else:
            self.ml_results = ml_input_dict # Store the generated dict

        # --- Run Detectors ---
        # Note: These methods modify self.results internally
        if run_ml_paralysis:
             if detect_side_paralysis:
                 self.detect_paralysis(patient_row_data_dict=self.ml_results) # Pass the generated dict
             else: logger.warning(f"({pid_for_log}) Paralysis detection function unavailable. Skipping.")
        else: logger.info(f"({pid_for_log}) Skipping ML paralysis detection as per flag.")

        if run_ml_synkinesis:
             if self.synkinesis_detector:
                 self.detect_synkinesis(patient_row_data_dict=self.ml_results) # Pass the generated dict
             else: logger.warning(f"({pid_for_log}) Synkinesis detector unavailable. Skipping.")
        else: logger.info(f"({pid_for_log}) Skipping ML synkinesis detection as per flag.")

        # --- Call Hypertonicity Detection ---
        if run_ml_hypertonicity:
            if self.hypertonicity_detector:
                self._detect_hypertonicity(self.ml_results) # Pass the generated dict
            else: logger.warning(f"({pid_for_log}) Hypertonicity detector unavailable. Skipping.")
        else: logger.info(f"({pid_for_log}) Skipping ML hypertonicity detection as per flag.")
        # --- End Call ---

        logger.info(f"({pid_for_log}) Full analysis complete.")
        return self.results # Return the populated results structure


    def detect_paralysis(self, patient_row_data_dict):
        """Dispatches paralysis detection to the appropriate module."""
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_DetectParalysis"
        if not self.results: logger.warning(f"({pid_for_log}) No results available for paralysis detection."); return
        if detect_side_paralysis is None: logger.error(f"({pid_for_log}) Paralysis detection function is not available."); return
        if not patient_row_data_dict: logger.error(f"({pid_for_log}) ML input dictionary is missing. Cannot run paralysis detection."); return

        logger.info(f"({pid_for_log}) Starting ML paralysis detection...")
        # Initialize summaries
        zone_paralysis_summary = {'left': {'upper': 'None', 'mid': 'None', 'lower': 'None'}, 'right': {'upper': 'None', 'mid': 'None', 'lower': 'None'}}
        affected_aus_summary = {'left': {'upper': set(), 'mid': set(), 'lower': set()}, 'right': {'upper': set(), 'mid': set(), 'lower': set()}}

        for zone in self.facial_zones.keys():
            aus_for_zone = self.facial_zones.get(zone)
            if not aus_for_zone: logger.warning(f"({pid_for_log}) No AUs defined for zone '{zone}'. Skipping paralysis check."); continue

            # Find the most appropriate action data to use for this zone
            action_to_use = None
            for act in ZONE_SPECIFIC_ACTIONS.get(zone, []):
                 # Check if action exists in results and is a dictionary (not just patient_summary)
                 if act in self.results and isinstance(self.results[act], dict):
                      action_to_use = act; break
            if not action_to_use: # Fallback if zone-specific action is missing
                first_action = next((act for act, info in self.results.items() if isinstance(info, dict) and act not in ['patient_summary', 'BL']), None)
                if first_action:
                    action_to_use = first_action
                    logger.debug(f"({pid_for_log}) No primary action data for zone '{zone}'. Using fallback '{action_to_use}'.")
                else:
                    logger.error(f"({pid_for_log}) No action data available for paralysis detection in zone '{zone}'. Setting zone result to Error.")
                    zone_paralysis_summary['left'][zone] = 'Error'
                    zone_paralysis_summary['right'][zone] = 'Error'
                    continue # Skip to next zone if no action data available

            action_info = self.results[action_to_use]
            logger.debug(f"({pid_for_log}) Using action '{action_to_use}' data for zone '{zone}' paralysis detection.")

            # Perform detection for both sides
            for side in ['left', 'right']:
                logger.debug(f"({pid_for_log}) Detecting paralysis for {side} {zone} zone...")
                other_side = 'right' if side == 'left' else 'left'
                if not isinstance(action_info, dict):
                    logger.error(f"({pid_for_log}) Invalid action_info for '{action_to_use}'. Skip {side} {zone}.")
                    zone_paralysis_summary[side][zone] = 'Error'; continue
                side_data = action_info.get(side, {}); other_side_data = action_info.get(other_side, {})
                if not isinstance(side_data, dict) or not isinstance(other_side_data, dict):
                     logger.error(f"({pid_for_log}) Missing side data for '{action_to_use}'. Skip {side} {zone}.")
                     zone_paralysis_summary[side][zone] = 'Error'; continue

                # Extract relevant values (handle potential missing keys gracefully)
                values_raw = side_data.get('au_values', {}); other_values_raw = other_side_data.get('au_values', {})
                values_normalized = side_data.get('normalized_au_values', {}); other_values_normalized = other_side_data.get('normalized_au_values', {})
                # Calculate averages based on available normalized values
                side_zone_aus_vals = [values_normalized.get(au, 0.0) for au in aus_for_zone if pd.notna(values_normalized.get(au, np.nan))]
                other_side_zone_aus_vals = [other_values_normalized.get(au, 0.0) for au in aus_for_zone if pd.notna(other_values_normalized.get(au, np.nan))]
                side_avg = float(np.mean(side_zone_aus_vals)) if side_zone_aus_vals else 0.0; other_avg = float(np.mean(other_side_zone_aus_vals)) if other_side_zone_aus_vals else 0.0

                try:
                    # Call the external detection function
                    detect_side_paralysis(
                        analyzer_instance=self, info=action_info, zone=zone, side=side, aus=aus_for_zone,
                        values=values_raw, other_values=other_values_raw,
                        values_normalized=values_normalized, other_values_normalized=other_values_normalized,
                        side_avg=side_avg, other_avg=other_avg,
                        zone_paralysis_summary=zone_paralysis_summary, affected_aus_summary=affected_aus_summary,
                        row_data=patient_row_data_dict # Pass the consolidated ML input dict
                    )
                    logger.debug(f"({pid_for_log}) Completed paralysis check for {side} {zone}.")
                except Exception as e_detect:
                    logger.error(f"({pid_for_log}) Exception during detect_side_paralysis call for {side} {zone}: {e_detect}", exc_info=True)
                    zone_paralysis_summary[side][zone] = 'Error'

        # Consolidate paralysis results into self.results structure (apply to all actions for consistency)
        overall_paralysis_detected = any(zone_paralysis_summary[s][z] not in ['None', 'Error'] for s in ['left', 'right'] for z in self.facial_zones.keys())
        for action, info in self.results.items():
             if action == 'patient_summary': continue # Skip patient summary key
             if isinstance(info, dict):
                 # Use setdefault to ensure 'paralysis' key exists
                 info.setdefault('paralysis', {})['detected'] = overall_paralysis_detected
                 info['paralysis'].setdefault('zones', {'left': {}, 'right': {}})
                 info['paralysis'].setdefault('affected_aus', {'left': set(), 'right': set()})
                 # Populate zone results for this action
                 for side_loop in ['left', 'right']:
                      info['paralysis']['zones'].setdefault(side_loop, {})
                      info['paralysis']['affected_aus'].setdefault(side_loop, set())
                      side_affected_set = set()
                      for zone_loop in self.facial_zones.keys():
                          info['paralysis']['zones'][side_loop][zone_loop] = zone_paralysis_summary[side_loop].get(zone_loop, 'Error')
                          side_affected_set.update(affected_aus_summary[side_loop].get(zone_loop, set()))
                      info['paralysis']['affected_aus'][side_loop] = side_affected_set
             else:
                 logger.warning(f"({pid_for_log}) Skipping paralysis update for action '{action}' because results item is not a dictionary.")

        # Log final findings
        detected_zones_log = [f'{s} {z}={zone_paralysis_summary[s][z]}' for s in zone_paralysis_summary for z in zone_paralysis_summary[s] if zone_paralysis_summary[s][z] not in ['None', 'Error']]
        if detected_zones_log:
            logger.info(f"({pid_for_log}) Paralysis detection complete. Final findings: {'; '.join(detected_zones_log)}")
        else:
            logger.info(f"({pid_for_log}) Paralysis detection complete. No paralysis found or errors occurred.")


    def detect_synkinesis(self, patient_row_data_dict):
        """Dispatches synkinesis detection to the SynkinesisDetector."""
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_DetectSynk"
        if not self.results: logger.warning(f"({pid_for_log}) No results available for synkinesis detection."); return
        if self.synkinesis_detector is None: logger.error(f"({pid_for_log}) Synkinesis detector is not available."); return
        if not patient_row_data_dict: logger.error(f"({pid_for_log}) ML input dictionary is missing. Cannot run synkinesis detection."); return

        logger.info(f"({pid_for_log}) Starting ML synkinesis detection...")
        try:
            # Ensure synkinesis structure exists for all actions before calling detector
            local_synk_types = SYNKINESIS_TYPES if SYNKINESIS_TYPES else []
            for action in self.results:
                if action == 'patient_summary': continue # Skip patient summary
                if isinstance(self.results[action], dict):
                    self.results[action].setdefault('synkinesis', { # Use setdefault to create if missing
                         'detected': False, 'types': [],
                         'side_specific': {st: {'left': False, 'right': False} for st in local_synk_types},
                         'confidence': {st: {'left': 0.0, 'right': 0.0} for st in local_synk_types},
                         'contributing_aus': {}
                     })

            # Call the detector (which modifies self.results directly)
            self.synkinesis_detector.detect_synkinesis(self.results, patient_row_data_dict)

            # Summarize results after detection
            overall_synk_detected = False; detected_types_log = set()
            for action, info in self.results.items():
                if action == 'patient_summary': continue
                if isinstance(info, dict):
                     synk_info = info.get('synkinesis', {})
                     if synk_info.get('detected', False):
                         overall_synk_detected = True
                         detected_types_log.update(synk_info.get('types', []))
                else:
                    logger.warning(f"({pid_for_log}) Skipping synkinesis summary update for action '{action}' because results item is not a dictionary.")

            detected_log_str = ', '.join(sorted(list(detected_types_log))) if detected_types_log else 'None Detected'
            logger.info(f"({pid_for_log}) Synkinesis detection complete. Detected types: {detected_log_str}")

        except Exception as e:
            logger.error(f"({pid_for_log}) Exception during ML synkinesis detection call: {e}", exc_info=True)
            # Mark as error in results if exception occurs?
            for action, info in self.results.items():
                 if action == 'patient_summary': continue
                 if isinstance(info, dict):
                     info.setdefault('synkinesis', {})['detected'] = 'Error'


    def extract_frames(self, video_path, output_dir, generate_visuals=False):
        """
        Extracts frames from video based on analysis results and saves them.
        Populates self.frame_paths with paths to the ORIGINAL extracted PNGs.
        """
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_Extract"
        self.frame_paths = {} # Reset frame paths

        if not generate_visuals:
            logger.info(f"({pid_for_log}) generate_visuals is False. Skipping frame extraction.")
            return True, {} # Return success but empty paths

        if not self.results:
            logger.error(f"({pid_for_log}) No analysis results available for frame extraction.")
            return False, {}
        if not video_path or not os.path.exists(video_path):
            logger.error(f"({pid_for_log}) Video path invalid or not found: '{video_path}'.")
            return False, {}

        # Ensure output directory exists
        patient_output_dir = self.output_dir # Use the one set during processing
        if not patient_output_dir:
            logger.error(f"({pid_for_log}) Output directory not set in analyzer. Cannot extract frames.")
            return False, {}
        try:
            os.makedirs(patient_output_dir, exist_ok=True)
        except Exception as e_create:
            logger.error(f"({pid_for_log}) Failed to create output directory {patient_output_dir}: {e_create}")
            return False, {}

        logger.info(f"({pid_for_log}) Extracting frames from {os.path.basename(video_path)} into {patient_output_dir}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"({pid_for_log}) Could not open video file {video_path}")
            return False, {}

        total_extracted = 0
        extracted_frames_set = set() # Keep track of frames already extracted

        try:
            for action, info in self.results.items():
                if action == 'patient_summary': continue # Skip patient summary

                if not isinstance(info, dict) or 'max_frame' not in info:
                    logger.warning(f"({pid_for_log}) Skipping frame extraction for action '{action}': missing data or not a dictionary.")
                    continue

                frame_num = info['max_frame']
                if frame_num is None or pd.isna(frame_num):
                    logger.warning(f"({pid_for_log}) Skipping frame extraction for action '{action}': invalid frame number (NaN or None).")
                    continue
                frame_num = int(frame_num) # Convert valid frame number to int

                # Skip if this frame number has already been extracted (e.g., multiple actions max out at same frame)
                if frame_num in extracted_frames_set:
                    logger.debug(f"({pid_for_log}) Frame {frame_num} already extracted for a previous action. Skipping duplicate extraction for action '{action}'.")
                    # Still store the path if needed for dashboard etc.
                    action_desc_dup = self.action_descriptions.get(action, action)
                    base_label_dup = f"{pid_for_log}_{action_desc_dup}_frame{frame_num}"
                    original_output_path_dup = os.path.join(patient_output_dir, f"{base_label_dup}_original.png")
                    if os.path.exists(original_output_path_dup): # Check if the expected file from previous extraction exists
                         self.frame_paths[action] = original_output_path_dup
                    continue

                # Set frame position and read
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"({pid_for_log}) Could not read frame {frame_num} for action '{action}'.")
                    continue

                # Prepare filenames
                action_desc = self.action_descriptions.get(action, action) # Use description if available
                base_label = f"{pid_for_log}_{action_desc}_frame{frame_num}"
                original_output_path = os.path.join(patient_output_dir, f"{base_label}_original.png")
                output_path_jpg = os.path.join(patient_output_dir, f"{base_label}.jpg")

                # Save original PNG (no compression for potential later use)
                save_success_png = cv2.imwrite(original_output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Save labeled JPG
                labeled_frame = frame.copy()
                cv2.putText(labeled_frame, base_label.replace("_"," "), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                save_success_jpg = cv2.imwrite(output_path_jpg, labeled_frame)

                if save_success_png and save_success_jpg:
                    logger.info(f"({pid_for_log}) Saved frames {action} (Frame {frame_num}): {os.path.basename(output_path_jpg)}, {os.path.basename(original_output_path)}")
                    self.frame_paths[action] = original_output_path # Store path to ORIGINAL PNG for dashboard etc.
                    total_extracted += 1
                    extracted_frames_set.add(frame_num) # Mark frame as extracted
                else:
                    logger.error(f"({pid_for_log}) Failed save frame {action} (Frame {frame_num}). PNG Save Success={save_success_png}, JPG Save Success={save_success_jpg}")

        except Exception as e_extract:
            logger.error(f"({pid_for_log}) Error during frame extraction: {e_extract}", exc_info=True)
            cap.release()
            return False, self.frame_paths # Return partially extracted paths if error occurred mid-loop
        finally:
            cap.release() # Ensure video is released

        if total_extracted > 0:
            logger.info(f"({pid_for_log}) Extracted {total_extracted} frames successfully.")
            return True, self.frame_paths
        else:
            logger.error(f"({pid_for_log}) Failed to extract any frames.")
            return False, self.frame_paths


    def generate_summary_data(self, results=None):
        """
        Generates a dictionary summarizing patient analysis results,
        consolidating synkinesis findings across actions.
        """
        analysis_results = results if results is not None else self.results
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_Summary"
        if not analysis_results:
            logger.error(f"({pid_for_log}) No results available to generate summary data.")
            return {}

        logger.info(f"Generating summary data dictionary for patient {pid_for_log}")
        patient_summary = {'Patient ID': pid_for_log}

        # --- Define expected keys and initialize defaults ---
        paralysis_keys = [f'{s} {z} Face Paralysis' for s in ['Left', 'Right'] for z in ['Upper', 'Mid', 'Lower']]
        local_synk_types = SYNKINESIS_TYPES if SYNKINESIS_TYPES else []
        synk_keys = [f'{st} {s}' for st in local_synk_types for s in ['Left', 'Right']]
        conf_keys = [f'{st} {s} Confidence' for st in local_synk_types for s in ['Left', 'Right']]
        hyper_keys = ['Hypertonicity Left', 'Hypertonicity Right']
        hyper_conf_keys = ['Hypertonicity Left Confidence', 'Hypertonicity Right Confidence']
        summary_flags = ['Paralysis Detected', 'Synkinesis Detected', 'Hypertonicity Detected']

        for key in paralysis_keys: patient_summary[key] = 'None'
        for key in synk_keys: patient_summary[key] = 'No'
        for key in conf_keys: patient_summary[key] = np.nan
        for key in hyper_keys: patient_summary[key] = 'No'
        for key in hyper_conf_keys: patient_summary[key] = np.nan
        for key in summary_flags: patient_summary[key] = 'No'
        # --- End Initialization ---

        # --- Extract Consolidated Results ---
        # --- Paralysis: Use first action info (as it's determined across zones) ---
        first_action_info = next((info for action, info in analysis_results.items() if action not in ['patient_summary', 'BL'] and isinstance(info, dict)), None)
        if not first_action_info: first_action_info = analysis_results.get('BL', {}) # Fallback to BL if no other action data
        if not isinstance(first_action_info, dict): first_action_info = {} # Ensure it's dict

        paralysis_info = first_action_info.get('paralysis', {})
        overall_paralysis = paralysis_info.get('detected', False)
        final_paralysis_zones = paralysis_info.get('zones', {'left': {}, 'right': {}})
        # --- End Paralysis ---

        # --- Synkinesis: Consolidate across ALL actions ---
        overall_synkinesis = False
        final_synk_sides = {st: {'left': False, 'right': False} for st in local_synk_types}
        final_synk_conf = {st: {'left': 0.0, 'right': 0.0} for st in local_synk_types}

        logger.debug(f"SUMMARY_GEN ({pid_for_log}): Consolidating Synkinesis results across actions...")
        for action, action_info in analysis_results.items():
            if action == 'patient_summary' or not isinstance(action_info, dict): continue # Skip non-action entries

            synkinesis_info = action_info.get('synkinesis', {})
            if synkinesis_info.get('detected', False): # Check if any synkinesis was detected for this action
                overall_synkinesis = True # Set the overall flag if true for any action
                action_synk_sides = synkinesis_info.get('side_specific', {})
                action_synk_conf = synkinesis_info.get('confidence', {})

                for synk_type in local_synk_types:
                    # Check if this synkinesis type exists in the action's results
                    if synk_type in action_synk_sides and synk_type in action_synk_conf:
                        for side in ['left', 'right']:
                            # If this action detected this type/side...
                            if action_synk_sides.get(synk_type, {}).get(side, False):
                                # Mark as detected in the final summary
                                final_synk_sides[synk_type][side] = True
                                # Keep the highest confidence score found across all actions
                                current_conf = action_synk_conf.get(synk_type, {}).get(side, 0.0)
                                if current_conf > final_synk_conf[synk_type][side]:
                                    final_synk_conf[synk_type][side] = current_conf
                    else:
                         logger.warning(f"SUMMARY_GEN ({pid_for_log}): Synkinesis type '{synk_type}' not found in side_specific/confidence dict for action '{action}'. Skipping consolidation for this type/action.")


        logger.debug(f"SUMMARY_GEN ({pid_for_log}): Consolidated Synkinesis Data:")
        logger.debug(f"  final_synk_sides: {json.dumps(final_synk_sides, indent=2, default=str)}")
        logger.debug(f"  final_synk_conf: {json.dumps(final_synk_conf, indent=2, default=str)}")
        # --- End Synkinesis Consolidation ---

        # --- Hypertonicity: Retrieve directly from patient_summary ---
        hypertonicity_info = analysis_results.get('patient_summary', {}).get('hypertonicity', {})
        overall_hypertonicity = hypertonicity_info.get('detected', False)
        final_hyper_left = hypertonicity_info.get('left', False)
        final_hyper_right = hypertonicity_info.get('right', False)
        final_hyper_conf_left = hypertonicity_info.get('confidence_left', 0.0)
        final_hyper_conf_right = hypertonicity_info.get('confidence_right', 0.0)
        # --- End Hypertonicity ---

        # --- Populate Summary Dictionary ---
        # Overall Flags
        patient_summary['Paralysis Detected'] = 'Yes' if overall_paralysis else 'No'
        patient_summary['Synkinesis Detected'] = 'Yes' if overall_synkinesis else 'No' # Use consolidated flag
        patient_summary['Hypertonicity Detected'] = 'Yes' if overall_hypertonicity else 'No'

        # Paralysis Details
        for side in ['Left', 'Right']:
            for zone in ['Upper', 'Mid', 'Lower']:
                 summary_key = f'{side} {zone} Face Paralysis'
                 # Safely access nested dictionary
                 severity = final_paralysis_zones.get(side.lower(), {}).get(zone.lower(), 'None')
                 patient_summary[summary_key] = str(severity) if pd.notna(severity) and severity not in ['Error'] else 'None'

        # Synkinesis Details (Using CONSOLIDATED results)
        for synk_type in local_synk_types:
            for side in ['Left', 'Right']:
                 detected = final_synk_sides.get(synk_type, {}).get(side.lower(), False) # Use consolidated data
                 patient_summary[f'{synk_type} {side}'] = 'Yes' if detected else 'No'
                 conf_key = f'{synk_type} {side} Confidence'
                 confidence = final_synk_conf.get(synk_type, {}).get(side.lower(), 0.0) # Use consolidated data
                 try: patient_summary[conf_key] = float(confidence) if pd.notna(confidence) else np.nan
                 except (ValueError, TypeError): patient_summary[conf_key] = np.nan

        # Hypertonicity Details
        patient_summary['Hypertonicity Left'] = 'Yes' if final_hyper_left else 'No'
        patient_summary['Hypertonicity Right'] = 'Yes' if final_hyper_right else 'No'
        patient_summary['Hypertonicity Left Confidence'] = float(final_hyper_conf_left) if pd.notna(final_hyper_conf_left) else np.nan
        patient_summary['Hypertonicity Right Confidence'] = float(final_hyper_conf_right) if pd.notna(final_hyper_conf_right) else np.nan
        # --- End Detail Population ---

        # --- Populate action-specific AU values ---
        local_included_actions = INCLUDED_ACTIONS if INCLUDED_ACTIONS else []
        actions_to_summarize = set(local_included_actions) | {'BL'}
        for action in actions_to_summarize:
             prefix = f'{action}_'; info = analysis_results.get(action)

             # Populate defaults if action data is missing
             if not info or not isinstance(info, dict):
                 is_bl = (action == 'BL')
                 patient_summary[prefix + 'Max Side'] = 'N/A' if is_bl else 'NA'
                 patient_summary[prefix + 'Max Frame'] = np.nan
                 patient_summary[prefix + 'Max Value'] = np.nan
                 for side_key in ['left', 'right']:
                     side_label = side_key.capitalize()
                     for au in ALL_AU_COLUMNS:
                         patient_summary[f'{prefix}{side_label} {au}'] = np.nan
                         patient_summary[f'{prefix}{side_label} {au} (Normalized)'] = np.nan
                 continue # Skip to next action

             # Populate from available data
             is_bl = (action == 'BL')
             patient_summary[prefix + 'Max Side'] = 'N/A' if is_bl else info.get('max_side', 'NA')
             patient_summary[prefix + 'Max Frame'] = int(info['max_frame']) if pd.notna(info.get('max_frame')) else np.nan
             patient_summary[prefix + 'Max Value'] = float(info['max_value']) if pd.notna(info.get('max_value')) else np.nan

             for side_key in ['left', 'right']:
                 side_label = side_key.capitalize();
                 au_values = info.get(side_key, {}).get('au_values', {});
                 norm_au_values = info.get(side_key, {}).get('normalized_au_values', {})
                 for au in ALL_AU_COLUMNS:
                     raw_val = au_values.get(au, np.nan);
                     norm_val = norm_au_values.get(au, np.nan) # Use NaN default here too
                     # Populate Raw AU column
                     patient_summary[f'{prefix}{side_label} {au}'] = float(raw_val) if pd.notna(raw_val) else np.nan
                     # Populate Normalized AU column
                     patient_summary[f'{prefix}{side_label} {au} (Normalized)'] = float(norm_val) if pd.notna(norm_val) else np.nan

        # --- End AU Value Population ---

        logger.info(f"({pid_for_log}) Generated summary data dictionary.")
        return patient_summary


    def cleanup_extracted_frames(self):
        """Removes temporary _original.png AND corresponding .jpg files after visualization."""
        pid_for_log = self.patient_id if self.patient_id else "UnknownPatient_Cleanup"
        if not hasattr(self, 'frame_paths') or not self.frame_paths:
            logger.debug(f"({pid_for_log}) No frame paths stored, skipping cleanup.")
            return
        if not isinstance(self.frame_paths, dict):
            logger.warning(f"({pid_for_log}) frame_paths is not a dictionary ({type(self.frame_paths)}). Skipping cleanup.")
            return

        logger.info(f"({pid_for_log}) Cleaning up temporary extracted frame files (*_original.png and *.jpg)...")
        removed_png_count = 0
        removed_jpg_count = 0
        output_dir_to_check = self.output_dir

        if not output_dir_to_check or not os.path.isdir(output_dir_to_check):
            logger.warning(f"({pid_for_log}) Output directory '{output_dir_to_check}' not found or invalid during cleanup.")
            self.frame_paths = {} # Clear paths as we cannot verify/remove files
            return

        # Collect all potential files to remove based on stored paths and glob patterns
        files_to_consider = set()
        # From stored paths
        for action, frame_path_png in self.frame_paths.items():
            if isinstance(frame_path_png, str) and frame_path_png.endswith("_original.png"):
                if os.path.exists(frame_path_png):
                    files_to_consider.add(frame_path_png)
                # Construct expected JPG path
                jpg_equivalent = frame_path_png.replace("_original.png", ".jpg")
                if os.path.exists(jpg_equivalent):
                    files_to_consider.add(jpg_equivalent)

        # From Glob (safer approach to catch any strays if frame_paths was incomplete)
        try:
            glob_pattern_png = os.path.join(output_dir_to_check, f"{pid_for_log}*_original.png")
            found_pngs = glob.glob(glob_pattern_png)
            files_to_consider.update(found_pngs)
            logger.debug(f"({pid_for_log}) Found {len(found_pngs)} PNG files via glob: {[os.path.basename(f) for f in found_pngs]}")

            glob_pattern_jpg = os.path.join(output_dir_to_check, f"{pid_for_log}*.jpg")
            found_jpgs = glob.glob(glob_pattern_jpg)
            # Filter out any potential JPGs that ARE original image backups (unlikely)
            jpgs_to_remove = [f for f in found_jpgs if not f.endswith("_original.jpg")]
            files_to_consider.update(jpgs_to_remove)
            logger.debug(f"({pid_for_log}) Found {len(jpgs_to_remove)} relevant JPG files via glob: {[os.path.basename(f) for f in jpgs_to_remove]}")

        except Exception as glob_e:
            logger.error(f"({pid_for_log}) Error globbing for cleanup files in {output_dir_to_check}: {glob_e}")

        unique_files_to_remove = sorted(list(files_to_consider))
        logger.debug(f"({pid_for_log}) Unique files targeted for removal: {[os.path.basename(f) for f in unique_files_to_remove]}")

        # Remove the files
        for file_to_remove in unique_files_to_remove:
            try:
                if os.path.exists(file_to_remove): # Double-check existence
                    os.remove(file_to_remove)
                    if file_to_remove.endswith("_original.png"):
                        removed_png_count += 1
                        logger.debug(f"({pid_for_log}) Removed original frame PNG: {os.path.basename(file_to_remove)}")
                    elif file_to_remove.endswith(".jpg"):
                        removed_jpg_count += 1
                        logger.debug(f"({pid_for_log}) Removed frame JPG: {os.path.basename(file_to_remove)}")
            except OSError as e:
                logger.error(f"({pid_for_log}) Error removing frame file {os.path.basename(file_to_remove)}: {e}")

        # Log summary
        log_parts = []
        if removed_png_count > 0: log_parts.append(f"{removed_png_count} original PNG file(s)")
        if removed_jpg_count > 0: log_parts.append(f"{removed_jpg_count} JPG file(s)")
        if log_parts:
            logger.info(f"({pid_for_log}) Removed " + " and ".join(log_parts) + ".")
        else:
            logger.info(f"({pid_for_log}) No temporary frame files found/removed during cleanup.")

        self.frame_paths = {} # Clear stored paths after cleanup attempt


    # --- Wrapper methods for visualization ---
    def create_au_visualization(self, *args, **kwargs):
        if not self.visualizer:
            logger.error(f"({self.patient_id}) Visualizer not initialized. Cannot create AU visualization.")
            return
        try:
            self.visualizer.create_au_visualization(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"({self.patient_id}) Error during AU visualization: {e}", exc_info=True)

    def create_symmetry_visualization(self, patient_output_dir):
        if not self.visualizer:
            logger.error(f"({self.patient_id}) Visualizer not initialized. Cannot create symmetry visualization.")
            return None
        try:
            return self.visualizer.create_symmetry_visualization(self, patient_output_dir, self.patient_id, self.results, self.action_descriptions)
        except Exception as e:
            logger.error(f"({self.patient_id}) Error during symmetry visualization: {e}", exc_info=True)
            return None

    def create_patient_dashboard(self):
        if not self.visualizer:
            logger.error(f"({self.patient_id}) Visualizer not initialized. Cannot create dashboard.")
            return None
        try:
            # Ensure output_dir is set
            if not self.output_dir:
                logger.error(f"({self.patient_id}) Output directory not set. Cannot determine dashboard save location.")
                return None
            # Pass frame paths to the dashboard function
            return self.visualizer.create_patient_dashboard(self, self.output_dir, self.patient_id, self.results, self.action_descriptions, self.frame_paths)
        except Exception as e:
            logger.error(f"({self.patient_id}) Error during dashboard creation: {e}", exc_info=True)
            return None