# --- START OF FILE facial_au_analyzer.py ---

# facial_au_analyzer.py
# Updated to use generic ParalysisDetector and FacialSynkinesisDispatcher
# Handles passing patient_row_data for ML detection.
# V1.6 Fixes: Refine find_peak_frame to check column existence before use, handle lowercase 'action', ensure peak frames int, handle analyze_action errors, fix frame extraction call.
# V1.7 Fix: Corrected path logic in extract_frames call to prevent nested patient folders.
# V1.8 Fix: Correct synkinesis consolidation logic in generate_summary_data.
# V1.9: Reverted direct BrowCockedDetector handling; relies on Dispatcher now.

import pandas as pd
import numpy as np
import logging
import os
import re
import glob
import json # Added for logging potentially

# Import project constants and NEW detectors/dispatchers
from facial_au_constants import (
    ACTION_TO_AUS, ACTION_DESCRIPTIONS, ALL_AU_COLUMNS,
    FACIAL_ZONES, FACIAL_ZONE_WEIGHTS, PARALYSIS_SEVERITY_LEVELS,
    AU_NAMES, BASELINE_AU_ACTIVATIONS, ZONE_SPECIFIC_ACTIONS,
    PATIENT_SUMMARY_COLUMNS, SEVERITY_ABBREVIATIONS, SEVERITY_ABBREVIATIONS_CONTRADICTION,
    USE_ML_FOR_LOWER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_UPPER_FACE, # Keep paralysis flags
    SYNKINESIS_TYPES # Keep for structure initialization
)
# --- Refactored Imports ---
from paralysis_detector import ParalysisDetector # Generic paralysis detector
from facial_paralysis_detection import detect_side_paralysis # The dispatcher function for paralysis
from facial_synkinesis_detection import FacialSynkinesisDispatcher # Synkinesis dispatcher
# --- Removed Synkinesis Sub-Detector Imports ---
# --- REMOVED Direct Brow Cocked Detector Import ---

logger = logging.getLogger(__name__)

class FacialAUAnalyzer:
    """Analyzes facial AU data for expressions, paralysis, and synkinesis."""

    def __init__(self, output_dir="../3.5_Results"):
        self.left_data = None; self.right_data = None; self.combined_data = None
        self.patient_id = "UnknownPatient"; self.output_dir = output_dir
        self.results = {}; self.frame_paths = {}; self.action_peak_frames = {}
        self.patient_baseline_values = {'left': {}, 'right': {}}
        self.patient_row_data = {} # Store the combined peak data for the patient

        # --- Instantiate Generic Paralysis Detectors ---
        self.paralysis_detectors = {}
        if USE_ML_FOR_LOWER_FACE or USE_ML_FOR_MIDFACE or USE_ML_FOR_UPPER_FACE:
            logger.info("Initializing ML Paralysis Detectors...")
            try:
                if ParalysisDetector:
                     for zone in ['lower', 'mid', 'upper']:
                         if (zone == 'lower' and USE_ML_FOR_LOWER_FACE) or \
                            (zone == 'mid' and USE_ML_FOR_MIDFACE) or \
                            (zone == 'upper' and USE_ML_FOR_UPPER_FACE):
                             try: self.paralysis_detectors[zone] = ParalysisDetector(zone=zone)
                             except ValueError as ve: logger.error(f"Failed init ParalysisDetector for {zone}: {ve}")
                             except Exception as e_pd: logger.error(f"Failed init ParalysisDetector for {zone}: {e_pd}", exc_info=True)
                else: logger.error("ParalysisDetector class not found.")
            except Exception as e: logger.error(f"Error initializing paralysis detectors: {e}", exc_info=True)

        # --- Instantiate Synkinesis Dispatcher ---
        # The dispatcher is now expected to handle ALL configured synkinesis types, including 'brow_cocked'
        self.synkinesis_dispatcher = None
        logger.info("Initializing Synkinesis Dispatcher...")
        try:
             if FacialSynkinesisDispatcher: self.synkinesis_dispatcher = FacialSynkinesisDispatcher()
             else: logger.error("FacialSynkinesisDispatcher class not found.")
        except Exception as e: logger.error(f"Error initializing synkinesis dispatcher: {e}", exc_info=True)

        # --- REMOVED Direct Brow Cocked Detector Instantiation ---
        self.brow_cocked_detector = None # Ensure this is always None

    def load_data(self, left_csv_path, right_csv_path):
        """Loads and preprocesses data from left and right CSV files."""
        logger.info(f"Loading data: Left='{left_csv_path}', Right='{right_csv_path}'")
        try:
            self.left_data = pd.read_csv(left_csv_path, low_memory=False).fillna(0)
            self.right_data = pd.read_csv(right_csv_path, low_memory=False).fillna(0)

            action_col_name = 'action'
            target_action_col = 'Action'

            if 'frame' not in self.left_data.columns or 'frame' not in self.right_data.columns:
                raise ValueError("'frame' column missing in one or both input CSVs.")
            if action_col_name not in self.left_data.columns:
                 raise ValueError(f"Required column '{action_col_name}' missing in left CSV.")
            if action_col_name not in self.right_data.columns:
                 logger.warning(f"'{action_col_name}' column missing in right CSV. Using Action from left CSV.")

            self.combined_data = pd.merge(self.left_data, self.right_data, on='frame', suffixes=('_Left', '_Right'))

            suffixed_action_col = f"{action_col_name}_Left"
            if suffixed_action_col in self.combined_data.columns:
                self.combined_data[target_action_col] = self.combined_data[suffixed_action_col]
                right_suffixed_action_col = f"{action_col_name}_Right"
                cols_to_drop = [suffixed_action_col]
                if right_suffixed_action_col in self.combined_data.columns:
                    if not self.combined_data[suffixed_action_col].equals(self.combined_data[right_suffixed_action_col]):
                        logger.warning(f"{suffixed_action_col} and {right_suffixed_action_col} columns differ. Using {suffixed_action_col}.")
                    cols_to_drop.append(right_suffixed_action_col)
                self.combined_data = self.combined_data.drop(columns=cols_to_drop)
            elif action_col_name in self.combined_data.columns and target_action_col != action_col_name:
                 logger.debug(f"Renaming column '{action_col_name}' to '{target_action_col}'.")
                 self.combined_data = self.combined_data.rename(columns={action_col_name: target_action_col})
            elif target_action_col in self.combined_data.columns:
                 logger.debug(f"Target column '{target_action_col}' already exists after merge.")
            else:
                 raise ValueError(f"Failed to find or create '{target_action_col}' column after merge.")

            if target_action_col not in self.combined_data.columns:
                raise ValueError(f"Target '{target_action_col}' column still missing after processing.")

            logger.info(f"Data loaded successfully. Combined shape: {self.combined_data.shape}")
            logger.debug(f"Combined data columns: {self.combined_data.columns.tolist()}")
            return True
        except FileNotFoundError as e: logger.error(f"File not found: {e}"); return False
        except ValueError as e: logger.error(f"Data loading error: {e}"); return False
        except Exception as e: logger.error(f"Error loading/merging data: {e}", exc_info=True); return False

    def identify_actions(self):
        """Identifies unique actions present in the data."""
        if self.combined_data is None or 'Action' not in self.combined_data.columns:
            logger.error("Cannot identify actions: Combined data not loaded or 'Action' column missing.")
            return []
        present_actions = self.combined_data['Action'].astype(str).unique().tolist()
        # Check against ACTION_TO_AUS keys *and* 'BL'
        valid_actions = [act for act in present_actions if act in ACTION_TO_AUS or act == 'BL']
        skipped_actions = set(present_actions) - set(valid_actions)
        if skipped_actions:
             logger.warning(f"Skipping unrecognized or unconfigured action identifiers found in data: {list(skipped_actions)}")
        logger.info(f"Actions identified and valid for analysis: {valid_actions}")
        return valid_actions # Return only valid actions

    def find_peak_frame(self, action_data, action):
        """Finds the peak frame for a given action, ensuring integer return."""
        if action_data.empty: return None

        def get_median_frame(df):
            """Safely calculates median frame, returns integer or None."""
            if df.empty: return None
            median_val = df['frame'].median()
            if pd.isna(median_val): return None
            if isinstance(median_val, float):
                 if df.empty: return None
                 closest_idx = (df['frame'] - median_val).abs().idxmin()
                 if closest_idx in df.index: return int(df.loc[closest_idx, 'frame'])
                 else: return int(df['frame'].iloc[0]) if not df.empty else None
            return int(median_val)

        key_aus = ACTION_TO_AUS.get(action, [])
        use_median_fallback = False
        peak_frame = None

        if not key_aus:
            use_median_fallback = True
            logger.warning(f"No key AUs defined for action '{action}'. Using median frame.")
        else:
            # Check if required columns exist *before* trying to use them
            norm_key_au_cols_left = [f"{au}_Left (Normalized)" for au in key_aus if f"{au}_Left (Normalized)" in action_data.columns]
            norm_key_au_cols_right = [f"{au}_Right (Normalized)" for au in key_aus if f"{au}_Right (Normalized)" in action_data.columns]
            raw_key_au_cols_left = [f"{au}_Left" for au in key_aus if f"{au}_Left" in action_data.columns]
            raw_key_au_cols_right = [f"{au}_Right" for au in key_aus if f"{au}_Right" in action_data.columns]

            # Check if *any* columns for the key AUs exist for this action
            if not (norm_key_au_cols_left or norm_key_au_cols_right or raw_key_au_cols_left or raw_key_au_cols_right):
                use_median_fallback = True
                logger.warning(f"Key AUs {key_aus} defined for action '{action}', but corresponding columns not found in data. Using median frame.")
            else:
                # Try normalized first if columns exist
                if norm_key_au_cols_left or norm_key_au_cols_right:
                    try:
                        scores_norm = action_data[norm_key_au_cols_left + norm_key_au_cols_right].sum(axis=1)
                        if not scores_norm.empty:
                            peak_idx_norm = scores_norm.idxmax()
                            if peak_idx_norm in action_data.index:
                                 peak_frame_norm = int(action_data.loc[peak_idx_norm, 'frame'])
                                 max_activation_norm = scores_norm[peak_idx_norm]
                                 logger.debug(f"Peak frame for {action} (Normalized): Frame {peak_frame_norm} (Score: {max_activation_norm:.2f})")
                                 if max_activation_norm > 0.1: peak_frame = peak_frame_norm # Potential peak found
                            else: logger.warning(f"Normalized peak index {peak_idx_norm} invalid for {action}.")
                    except Exception as e_norm: logger.warning(f"Error calculating normalized peak score for {action}: {e_norm}.")

                # Try raw if normalized failed or wasn't significant, and if raw columns exist
                if peak_frame is None and (raw_key_au_cols_left or raw_key_au_cols_right):
                     try:
                         scores_raw = action_data[raw_key_au_cols_left + raw_key_au_cols_right].sum(axis=1)
                         if not scores_raw.empty:
                             peak_idx_raw = scores_raw.idxmax()
                             if peak_idx_raw in action_data.index:
                                 peak_frame_raw = int(action_data.loc[peak_idx_raw, 'frame'])
                                 max_activation_raw = scores_raw[peak_idx_raw]
                                 logger.debug(f"Peak frame for {action} (Raw): Frame {peak_frame_raw} (Score: {max_activation_raw:.2f})")
                                 peak_frame = peak_frame_raw # Use raw peak
                             else: logger.warning(f"Raw peak index {peak_idx_raw} invalid for {action}.")
                     except Exception as e_raw: logger.error(f"Error finding peak frame (Raw) for {action}: {e_raw}")

                # If still no peak frame found using AUs, trigger median fallback
                if peak_frame is None:
                     use_median_fallback = True
                     logger.warning(f"Could not find peak for action '{action}' using key AUs {key_aus}. Using median frame.")

        # Apply median fallback if needed
        if use_median_fallback:
            peak_frame = get_median_frame(action_data)
            if peak_frame is None: logger.error(f"Could not determine median frame for action '{action}'.")

        return peak_frame # Will be None if all methods fail


    def get_data_at_frame(self, frame):
        """Retrieves data for a specific frame."""
        if self.combined_data is None: return None
        try: frame_int = int(frame)
        except (ValueError, TypeError): logger.error(f"Invalid frame number type: {frame} ({type(frame)}). Cannot retrieve frame data."); return None
        frame_data = self.combined_data[self.combined_data['frame'] == frame_int]
        if frame_data.empty: logger.warning(f"No data found for frame {frame_int}."); return None
        return frame_data.iloc[0].to_dict()

    def calculate_normalized_values(self, current_values, baseline_values):
        """Calculates baseline-subtracted normalized AU values."""
        normalized = {}
        for au, value in current_values.items():
            baseline = baseline_values.get(au, 0)
            try: norm_val = max(0, float(value) - float(baseline))
            except (ValueError, TypeError): norm_val = max(0, float(value))
            normalized[au] = norm_val
        return normalized

    def analyze_action(self, action, peak_frame, patient_row_data):
        """Analyzes a single action at its peak frame, ensures basic structure."""
        action_results = {'action': action, 'max_frame': None, 'left': {'au_values': {}, 'normalized_au_values': {}}, 'right': {'au_values': {}, 'normalized_au_values': {}}}
        if isinstance(action, str): self.results[action] = action_results
        else: logger.warning(f"Attempted to analyze non-string action '{action}', skipping analysis."); return None

        if peak_frame is None: logger.warning(f"No peak frame for action '{action}'. Skipping full analysis details."); return action_results
        if not isinstance(peak_frame, (int, np.integer)): logger.error(f"Peak frame {peak_frame} for {action} is not an integer. Skipping data retrieval."); return action_results

        peak_frame_int = int(peak_frame)
        action_results['max_frame'] = peak_frame_int

        action_peak_data = self.get_data_at_frame(peak_frame_int)
        if action_peak_data is None:
            logger.error(f"Could not get data for peak frame {peak_frame_int} of action {action}. Some results missing.")
            return action_results

        action_results['left']['au_values'] = {au: action_peak_data.get(f"{au}_Left", 0.0) for au in ALL_AU_COLUMNS}
        action_results['right']['au_values'] = {au: action_peak_data.get(f"{au}_Right", 0.0) for au in ALL_AU_COLUMNS}
        action_results['left']['normalized_au_values'] = self.calculate_normalized_values(action_results['left']['au_values'], self.patient_baseline_values['left'])
        action_results['right']['normalized_au_values'] = self.calculate_normalized_values(action_results['right']['au_values'], self.patient_baseline_values['right'])

        self.results[action] = action_results
        return action_results


    def analyze_all_actions(self, run_ml_paralysis=True, run_ml_synkinesis=True, run_ml_hypertonicity=True):
        """Analyzes all identified actions."""
        if self.combined_data is None: logger.error("Data not loaded."); return None
        actions = self.identify_actions(); self.results = {}; self.action_peak_frames = {}

        # 1. Find Peak Frames for all valid actions
        for action in actions:
            action_data = self.combined_data[self.combined_data['Action'] == action]
            if action_data.empty: logger.warning(f"No data found for action '{action}'. Skipping peak frame search."); continue
            peak_frame = self.find_peak_frame(action_data, action)
            if peak_frame is not None: self.action_peak_frames[action] = peak_frame; logger.info(f"Peak frame for {action}: {peak_frame}")
            else: logger.error(f"Could not find peak frame for action {action}.")

        # 2. Get Baseline values
        baseline_action = 'BL'
        if baseline_action in self.action_peak_frames:
             baseline_frame = self.action_peak_frames[baseline_action]
             if baseline_frame is not None:
                 baseline_data = self.get_data_at_frame(baseline_frame)
                 if baseline_data:
                     self.patient_baseline_values['left'] = {au: baseline_data.get(f"{au}_Left", 0.0) for au in ALL_AU_COLUMNS}
                     self.patient_baseline_values['right'] = {au: baseline_data.get(f"{au}_Right", 0.0) for au in ALL_AU_COLUMNS}
                     logger.info(f"Baseline values extracted from frame {baseline_frame}.")
                 else: logger.error(f"Failed to extract baseline data from frame {baseline_frame}. Normalization will use 0 baseline.")
             else: logger.error(f"Peak frame for baseline action '{baseline_action}' is None or invalid. Normalization will use 0 baseline.")
        else: logger.warning("Baseline ('BL') action peak frame not found. Normalization will use 0 baseline.")

        # --- 3. Construct patient_row_data (Single row summary for ML) ---
        self.patient_row_data = {'Patient ID': self.patient_id}
        logger.info("Constructing patient_row_data dictionary for ML input...")
        for action in actions:
             peak_frame = self.action_peak_frames.get(action)
             if peak_frame is None: continue
             action_peak_data = self.get_data_at_frame(peak_frame)
             if action_peak_data is None: continue
             action_raw_left = {au: action_peak_data.get(f"{au}_Left", 0.0) for au in ALL_AU_COLUMNS}
             action_raw_right = {au: action_peak_data.get(f"{au}_Right", 0.0) for au in ALL_AU_COLUMNS}
             action_norm_left = self.calculate_normalized_values(action_raw_left, self.patient_baseline_values['left'])
             action_norm_right = self.calculate_normalized_values(action_raw_right, self.patient_baseline_values['right'])
             for au in ALL_AU_COLUMNS:
                 self.patient_row_data[f"{action}_Left {au}"] = action_raw_left.get(au, 0.0)
                 self.patient_row_data[f"{action}_Right {au}"] = action_raw_right.get(au, 0.0)
                 self.patient_row_data[f"{action}_Left {au} (Normalized)"] = action_norm_left.get(au, 0.0)
                 self.patient_row_data[f"{action}_Right {au} (Normalized)"] = action_norm_right.get(au, 0.0)
        logger.info(f"Constructed patient_row_data with {len(self.patient_row_data)} keys.")
        # --- End Construct patient_row_data ---

        # 4. Analyze each action's peak frame individually
        for action in actions:
             peak_frame = self.action_peak_frames.get(action)
             self.analyze_action(action, peak_frame, self.patient_row_data)

        # --- Initialize patient_summary in results here ---
        self.results.setdefault('patient_summary', {})

        # 5. Run ML Detections using the patient_row_data summary
        if run_ml_paralysis and self.paralysis_detectors:
            logger.info(f"({self.patient_id}) Running ML Paralysis Detection...")
            self._run_ml_paralysis_detection(self.results, self.patient_row_data)
        else: logger.info(f"({self.patient_id}) Skipping ML Paralysis Detection.")

        # --- Synkinesis Dispatcher handles ALL synkinesis types now ---
        if run_ml_synkinesis and self.synkinesis_dispatcher:
            logger.info(f"({self.patient_id}) Running ML Synkinesis Detection (via Dispatcher)...")
            try:
                 # Dispatcher now updates patient_summary for hypertonicity directly
                 # Assumes the dispatcher iterates through ALL configured types in SYNKINESIS_CONFIG, including 'brow_cocked'
                self.synkinesis_dispatcher.detect_synkinesis(self.results, patient_row_data_dict=self.patient_row_data)
            except Exception as e_synk_disp: logger.error(f"({self.patient_id}) Error during synkinesis dispatch: {e_synk_disp}", exc_info=True)
        elif run_ml_synkinesis: logger.warning(f"({self.patient_id}) ML Synkinesis detection requested but dispatcher unavailable.")
        else: logger.info(f"({self.patient_id}) Skipping ML Synkinesis Detection.")

        # --- REMOVE Direct Brow Cocked Detection Call ---
        # if self.brow_cocked_detector:
        #    ... (removed block) ...
        # else:
        #    logger.info(f"({self.patient_id}) Skipping Brow Cocked detection (detector not available).") # This is now handled by dispatcher check
        # --- END REMOVE ---

        # 6. Generate Patient Summary (Reads consolidated results including Brow Cocked from dispatcher)
        logger.info(f"({self.patient_id}) Generating final patient summary data...")
        patient_summary_data = self.generate_summary_data(results=self.results)
        self.results.setdefault('patient_summary', {}).update(patient_summary_data if patient_summary_data else {})

        return self.results


    def _run_ml_paralysis_detection(self, results, patient_row_data):
        """ Internal helper to run ML paralysis detection for all zones. """
        patient_level_paralysis = {'left': {}, 'right': {}}
        affected_aus_summary = {'left': {}, 'right': {}} # Currently unused but keep if needed

        for zone, aus in FACIAL_ZONES.items():
            detector = self.paralysis_detectors.get(zone)
            if not detector:
                 logger.debug(f"No paralysis detector available or ML disabled for zone '{zone}'. Skipping.")
                 patient_level_paralysis['left'][zone] = 'N/A' # Initialize with N/A
                 patient_level_paralysis['right'][zone] = 'N/A'
                 continue

            logger.debug(f"Running paralysis detection for zone: {zone}")
            rep_action = ZONE_SPECIFIC_ACTIONS.get(zone, [None])[0]
            action_info = {}
            rep_action_result = results.get(rep_action)
            if rep_action and rep_action_result is not None and isinstance(rep_action_result, dict):
                 action_info = rep_action_result
            else:
                 logger.warning(f"No representative action ('{rep_action}') found in results for zone '{zone}'. Using empty context for detect_side_paralysis.")
                 # Still create the structure in action_info for detect_side_paralysis
                 action_info['paralysis'] = {'zones': {'left': {}, 'right': {}}, 'detection_details': {}, 'confidence': {'left': {}, 'right': {}}}

            for side in ['left', 'right']:
                 other_side = 'right' if side == 'left' else 'left'
                 # Pass empty dicts if action_info is missing details (handled by get)
                 values = action_info.get(side, {}).get('au_values', {})
                 other_values = action_info.get(other_side, {}).get('au_values', {})
                 values_norm = action_info.get(side, {}).get('normalized_au_values', {})
                 other_values_norm = action_info.get(other_side, {}).get('normalized_au_values', {})

                 # Initialize the zone entry for the side if not present
                 if side not in patient_level_paralysis: patient_level_paralysis[side] = {}
                 if zone not in patient_level_paralysis[side]: patient_level_paralysis[side][zone] = 'None'

                 detect_side_paralysis(
                     analyzer_instance=self, info=action_info, zone=zone, side=side, aus=aus,
                     values=values, other_values=other_values, values_normalized=values_norm,
                     other_values_normalized=other_values_norm, side_avg=0, other_avg=0, # These averages might be unused by ML detector
                     zone_paralysis_summary=patient_level_paralysis, # Pass the dict to be updated
                     affected_aus_summary=affected_aus_summary,
                     row_data=patient_row_data
                 )

        # Update paralysis info in *all* action results dictionaries
        for action, info in results.items():
             if action == 'patient_summary': continue
             if isinstance(info, dict):
                 # Ensure the structure exists before assigning
                 info.setdefault('paralysis', {})['zones'] = patient_level_paralysis
             else:
                 logger.warning(f"Skipping paralysis zone update for action '{action}' as info is not a dictionary.")


    def generate_summary_data(self, results=None):
        """Generates a dictionary containing summary metrics for the patient."""
        if results is None: results = self.results
        if not results: logger.error("Cannot generate summary: No results available."); return None
        logger.info(f"({self.patient_id}) Generating summary data...")
        summary = {'Patient ID': self.patient_id}

        # --- Paralysis Summary ---
        final_paralysis = {'left': {'upper': 'NA', 'mid': 'NA', 'lower': 'NA'},
                           'right': {'upper': 'NA', 'mid': 'NA', 'lower': 'NA'}}
        paralysis_source_action = None
        for act, act_info in results.items():
            if act != 'patient_summary' and isinstance(act_info, dict) and 'paralysis' in act_info and 'zones' in act_info['paralysis']:
                 paralysis_data = act_info['paralysis'].get('zones')
                 if isinstance(paralysis_data, dict) and paralysis_data: # Check if not empty
                      final_paralysis = paralysis_data
                      paralysis_source_action = act
                      break # Found valid paralysis data
        if paralysis_source_action:
            logger.debug(f"Using paralysis data from action '{paralysis_source_action}' for summary.")
        else:
            logger.warning(f"({self.patient_id}) Could not find any action results with valid paralysis info to extract final paralysis summary.")

        paralysis_detected = False
        for side in ['Left', 'Right']:
             for zone_short, zone_long in {'upper': 'Upper', 'mid': 'Mid', 'lower': 'Lower'}.items():
                 key = f"{side} {zone_long} Face Paralysis"
                 severity = final_paralysis.get(side.lower(), {}).get(zone_short, 'NA')
                 # Ensure severity is a valid string for comparison/storage
                 severity_str = str(severity) if severity is not None else 'NA'
                 summary[key] = severity_str
                 if severity_str not in ['None', 'NA', 'Error']: paralysis_detected = True
        summary['Paralysis Detected'] = 'Yes' if paralysis_detected else 'No'


        # --- Synkinesis Summary ---
        # Use canonical types directly from constants
        local_synk_types = SYNKINESIS_TYPES if isinstance(SYNKINESIS_TYPES, list) else []
        if not local_synk_types:
            logger.error(f"({self.patient_id}) SYNKINESIS_TYPES from constants is empty. Cannot generate synkinesis summary.")
            return summary # Return partial summary

        # --- Find Canonical Keys for Patient-Level Types ---
        # Use .get() with a default to avoid KeyError if mapping fails for some reason
        hyper_canonical_key = self.synkinesis_dispatcher.config_key_to_canonical.get('hypertonicity', 'Hypertonicity') # Default to expected canonical
        bc_canonical_key = self.synkinesis_dispatcher.config_key_to_canonical.get('brow_cocked', 'Brow Cocked')     # Default to expected canonical

        patient_level_canonical_keys = [hyper_canonical_key, bc_canonical_key]

        # --- Consolidate Action-Specific Synkinesis ---
        # Initialize with canonical keys, default to False
        final_synk = {st: {'left': False, 'right': False} for st in local_synk_types}
        overall_non_patient_synk_detected = False

        for action, info in results.items():
            if action == 'patient_summary' or not isinstance(info, dict): continue
            if 'synkinesis' in info:
                synk_info = info.get('synkinesis', {})
                side_specific_data = synk_info.get('side_specific', {})
                if isinstance(side_specific_data, dict):
                    # Iterate through canonical keys stored in action results
                    for canonical_synk_type, side_data in side_specific_data.items():
                        # Only consolidate if it's a known type and NOT patient-level
                        if canonical_synk_type in final_synk and canonical_synk_type not in patient_level_canonical_keys:
                            # Consolidate using 'or'
                            final_synk[canonical_synk_type]['left'] = final_synk[canonical_synk_type]['left'] or side_data.get('left', False)
                            final_synk[canonical_synk_type]['right'] = final_synk[canonical_synk_type]['right'] or side_data.get('right', False)
                            # If either side is now True for this type, mark overall detection
                            if final_synk[canonical_synk_type]['left'] or final_synk[canonical_synk_type]['right']:
                                overall_non_patient_synk_detected = True


        # --- Get Patient-Level Results Directly from patient_summary using CANONICAL keys ---
        patient_summary_dict = results.get('patient_summary', {}) if results else {}

        # Hypertonicity
        hyper_info = patient_summary_dict.get(hyper_canonical_key, {}) # Use canonical key
        # Fetch boolean values directly
        hyper_left = hyper_info.get('left', False)
        hyper_right = hyper_info.get('right', False)
        # Ensure 'detected' reflects actual left/right status if present, default to False
        hyper_detected_overall = hyper_info.get('detected', hyper_left or hyper_right)
        # Store the boolean result back into our consolidated structure
        final_synk.setdefault(hyper_canonical_key, {})['left'] = hyper_left
        final_synk.setdefault(hyper_canonical_key, {})['right'] = hyper_right

        # Brow Cocked
        brow_cocked_summary = patient_summary_dict.get(bc_canonical_key, {}) # Use canonical key
        # Fetch boolean values directly
        bc_left = brow_cocked_summary.get('left', False)
        bc_right = brow_cocked_summary.get('right', False)
        bc_detected_overall = bc_left or bc_right
        # Store the boolean result back into our consolidated structure
        final_synk.setdefault(bc_canonical_key, {})['left'] = bc_left
        final_synk.setdefault(bc_canonical_key, {})['right'] = bc_right


        # --- Set Overall Synkinesis Detected Flag ---
        # Considers action-level OR patient-level detections
        synk_detected = overall_non_patient_synk_detected or hyper_detected_overall or bc_detected_overall
        summary['Synkinesis Detected'] = 'Yes' if synk_detected else 'No'

        # --- Populate Summary Dictionary using Canonical Keys ---
        # This loop now iterates through all types including Hyper/BrowCocked
        for canonical_synk_type in local_synk_types:
             # Use the boolean values stored in final_synk
             detected_left = final_synk.get(canonical_synk_type, {}).get('left', False)
             detected_right = final_synk.get(canonical_synk_type, {}).get('right', False)

             # Convert boolean to 'Yes'/'No' string for the summary
             summary[f"{canonical_synk_type} Left"] = 'Yes' if detected_left else 'No'
             summary[f"{canonical_synk_type} Right"] = 'Yes' if detected_right else 'No'

             # Add overall 'Detected' key specifically for Hypertonicity if needed elsewhere (optional)
             if canonical_synk_type == hyper_canonical_key:
                  summary[f"{canonical_synk_type} Detected"] = 'Yes' if hyper_detected_overall else 'No'


        # --- AU Value Summary (No changes needed here) ---
        for action, frame in self.action_peak_frames.items():
            if frame is not None:
                 try: summary[f"{action}_Max Frame"] = int(frame)
                 except (ValueError, TypeError): summary[f"{action}_Max Frame"] = np.nan

                 action_res = results.get(action, {})
                 if isinstance(action_res, dict):
                     for side in ['Left', 'Right']:
                         side_data = action_res.get(side.lower(), {})
                         if isinstance(side_data, dict):
                             for au in ALL_AU_COLUMNS:
                                 raw_key = f"{action}_{side} {au}"
                                 summary[raw_key] = side_data.get('au_values', {}).get(au, np.nan)
                                 norm_key = f"{action}_{side} {au} (Normalized)"
                                 summary[norm_key] = side_data.get('normalized_au_values', {}).get(au, np.nan)
                         else:
                             logger.warning(f"Data for side '{side.lower()}' in action '{action}' is not a dictionary. Skipping AU values.")
                             for au in ALL_AU_COLUMNS: summary[f"{action}_{side} {au}"] = np.nan; summary[f"{action}_{side} {au} (Normalized)"] = np.nan
                 else:
                     logger.warning(f"Results for action '{action}' is not a dictionary. Skipping AU values.")
                     for side in ['Left', 'Right']:
                         for au in ALL_AU_COLUMNS: summary[f"{action}_{side} {au}"] = np.nan; summary[f"{action}_{side} {au} (Normalized)"] = np.nan
            else:
                summary[f"{action}_Max Frame"] = np.nan
                logger.warning(f"Peak frame for action '{action}' was None. Skipping AU value summary for this action.")
                for side in ['Left', 'Right']:
                     for au in ALL_AU_COLUMNS: summary[f"{action}_{side} {au}"] = np.nan; summary[f"{action}_{side} {au} (Normalized)"] = np.nan

        logger.info(f"({self.patient_id}) Summary generation complete.")
        return summary

    def extract_frames(self, video_path, output_dir, generate_visuals=True):
        """Extracts key frames if video path is provided."""
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video path invalid or missing ('{video_path}'). Cannot extract frames."); return False, "Video path missing or invalid."
        if not self.action_peak_frames:
             logger.warning("No peak frames identified. Cannot extract specific frames."); return False, "No peak frames."
        if not generate_visuals:
             logger.info("Frame extraction skipped as generate_visuals is False."); return True, "Skipped by flag."

        try:
            from facial_au_frame_extractor import FacialFrameExtractor
            extractor = FacialFrameExtractor()
            valid_peak_frames = {action: int(frame) for action, frame in self.action_peak_frames.items()
                                 if frame is not None and isinstance(frame, (int, float, np.number)) and not pd.isna(frame)}
            if not valid_peak_frames:
                logger.warning("No valid integer peak frames to extract."); return False, "No valid peak frames."

            results_for_extractor = {}
            for action, frame_num in valid_peak_frames.items():
                action_data = self.results.get(action)
                if action_data and isinstance(action_data, dict):
                    results_for_extractor[action] = action_data.copy()
                    results_for_extractor[action]['max_frame'] = frame_num
                else:
                     results_for_extractor[action] = {'action': action, 'max_frame': frame_num}
                     logger.warning(f"Minimal results structure created for frame extractor for action '{action}' as full results were missing/invalid.")

            success, self.frame_paths = extractor.extract_frames(
                 analyzer=self, video_path=video_path,
                 output_dir=output_dir, # Use the passed patient-specific dir
                 patient_id=self.patient_id, results=results_for_extractor,
                 action_descriptions=ACTION_DESCRIPTIONS
            )
            return success, "Frames extracted successfully." if success else "Frame extraction failed (check logs)."
        except ImportError: logger.error("facial_au_frame_extractor module not found. Cannot extract frames."); return False, "Module not found."
        except Exception as e: logger.error(f"Error extracting frames from {video_path}: {e}", exc_info=True); return False, f"Error: {e}"

    def cleanup_extracted_frames(self):
        """Removes individual extracted frame images."""
        cleaned_count = 0;
        skipped_count = 0
        if not self.frame_paths: logger.debug("No frame paths recorded, skipping cleanup."); return

        frame_keys_to_clean = list(self.frame_paths.keys())
        logger.debug(
            f"Attempting cleanup for {len(frame_keys_to_clean)} frame paths: {list(self.frame_paths.values())}")
        for action in frame_keys_to_clean:
            frame_path_png = self.frame_paths.get(action)

            if frame_path_png and isinstance(frame_path_png, str) and frame_path_png.endswith("_original.png"):
                jpg_path = frame_path_png.replace("_original.png", ".jpg")
                paths_to_remove = []
                if os.path.exists(frame_path_png): paths_to_remove.append(frame_path_png)
                if os.path.exists(jpg_path): paths_to_remove.append(jpg_path)

                removed_current = False
                for p in paths_to_remove:
                    try:
                        os.remove(p)
                        logger.debug(f"Removed frame file: {p}")
                        removed_current = True
                    except OSError as e:
                        logger.warning(f"Could not remove frame file {p}: {e}")
                if removed_current:
                    cleaned_count += len(paths_to_remove)
                    if action in self.frame_paths:
                        del self.frame_paths[action]

            elif frame_path_png and isinstance(frame_path_png, str):
                logger.debug(f"Skipping cleanup for non-standard or non-'_original.png' path: {frame_path_png}")
                skipped_count += 1
            elif frame_path_png is not None:
                logger.debug(
                    f"Frame path exists in dict but is not a valid string path: {frame_path_png}")
                skipped_count += 1

        if cleaned_count > 0: logger.info(f"Cleaned up {cleaned_count} extracted frame image files.")
        if skipped_count > 0: logger.debug(f"Skipped cleanup for {skipped_count} items.")

# --- END OF FILE facial_au_analyzer.py ---