# facial_au_batch_processor.py (Cleanup Call ADDED after all visuals)

import os
import numpy as np
import pandas as pd
import logging
import glob
import re
from facial_au_analyzer import FacialAUAnalyzer
from facial_au_constants import INCLUDED_ACTIONS, ALL_AU_COLUMNS, SYNKINESIS_TYPES

# Setup logger
logger = logging.getLogger(__name__)

class FacialAUBatchProcessor:
    """ Process multiple patients' facial AU data in batch mode. """

    def __init__(self, output_dir="../3.5_Results"):
        self.output_dir = output_dir
        self.patients = []
        self.summary_data = None
        try:
            # Create the main output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Batch Processor initialized. Main output directory: {os.path.abspath(self.output_dir)}")
        except OSError as e:
             logger.error(f"Could not create main output directory '{output_dir}'. Error: {e}")
             raise # Raise error if main dir cannot be created

    def add_patient(self, left_csv, right_csv, video_path=None):
        # ... (remains the same) ...
        patient_id_match = re.search(r'(IMG_\d+)', os.path.basename(left_csv))
        patient_id = patient_id_match.group(1) if patient_id_match else os.path.basename(left_csv).split('_')[0]
        # Basic validation
        if not os.path.exists(left_csv): logger.warning(f"Left CSV not found: {left_csv}"); # Don't add if essential file missing? Or let load_data handle it.
        if not os.path.exists(right_csv): logger.warning(f"Right CSV not found: {right_csv}");
        if video_path and not os.path.exists(video_path): logger.warning(f"Video file specified but not found: {video_path}"); video_path = None # Set to None if not found

        self.patients.append({'left_csv': left_csv, 'right_csv': right_csv, 'video_path': video_path, 'patient_id': patient_id})
        logger.debug(f"Added patient for processing: {patient_id}")
        return True


    def auto_detect_patients(self, data_dir):
        """ Auto-detect patient files in a directory. """
        # ... (logic remains the same) ...
        if not os.path.isdir(data_dir): logger.error(f"Data directory not found: {data_dir}"); return 0
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"));
        # Make matching case-insensitive for flexibility
        left_files = sorted([f for f in csv_files if "_left" in os.path.basename(f).lower()])
        patients_added = 0; detected_ids = set();
        logger.info(f"Found {len(csv_files)} CSV files. Searching for '_left' pairs in {data_dir}")

        for left_file in left_files:
            # Extract base ID using regex, looking for standard pattern first
            base_name_match = re.search(r'(IMG_\d+)', os.path.basename(left_file), re.IGNORECASE);
            if not base_name_match:
                 # Fallback: try splitting if no standard ID found
                 base_name_simple = os.path.basename(left_file).lower().replace('_left.csv', '')
                 logger.warning(f"No standard ID (IMG_xxxx) in {os.path.basename(left_file)}. Trying base '{base_name_simple}'.")
                 base_name = base_name_simple # Use the derived name
                 if not base_name: # Skip if even fallback fails
                      logger.warning(f"Could not derive base name for {os.path.basename(left_file)}. Skipping.")
                      continue
            else:
                 base_name = base_name_match.group(1) # Use the standard ID

            if base_name in detected_ids: continue # Already processed this ID

            # Find corresponding right file (case-insensitive search)
            right_pattern = os.path.join(data_dir, f"{base_name}_right*.csv");
            # Use glob with case sensitivity potentially handled by OS, or filter manually if needed
            right_files = glob.glob(right_pattern) # Simple glob first
            # If simple glob fails, try iterating through all CSVs for case-insensitive match
            if not right_files:
                right_files = [f for f in csv_files if f"{base_name}_right".lower() in os.path.basename(f).lower()]

            if right_files:
                right_file = right_files[0] # Take the first match
                if len(right_files) > 1:
                    logger.warning(f"Multiple right files found for {base_name}, using: {os.path.basename(right_file)}")

                # Find corresponding video file (common formats, case-insensitive extension)
                video_path = None
                video_found = False
                for ext in ['mp4', 'avi', 'mov', 'wmv', 'mkv']: # Add more common formats if needed
                     # Search for patterns like ID.ext, ID_rotated.ext, ID_annotated.ext, ID_anything.ext
                     vid_patterns = [
                         os.path.join(data_dir, f"{base_name}.{ext}"),
                         os.path.join(data_dir, f"{base_name}_*.{ext}") # Generic pattern
                     ]
                     for pattern in vid_patterns:
                         # Use glob and potentially filter for case-insensitivity if glob doesn't handle it
                         vid_files = glob.glob(pattern)
                         # Case-insensitive check (example)
                         vid_files_filtered = [vf for vf in vid_files if os.path.basename(vf).lower().startswith(base_name.lower()) and vf.lower().endswith(f".{ext}")]
                         if vid_files_filtered:
                             video_path = vid_files_filtered[0] # Take first match
                             if len(vid_files_filtered) > 1: logger.warning(f"Multiple video files found for {base_name} with ext {ext}, using: {os.path.basename(video_path)}")
                             video_found = True
                             break # Stop searching extensions for this pattern
                     if video_found: break # Stop searching patterns

                # Add the patient if essential files are present
                if self.add_patient(left_file, right_file, video_path):
                     detected_ids.add(base_name); patients_added += 1
                     video_status = f"Video: {os.path.basename(video_path)}" if video_path else "Video NOT found"
                     logger.info(f"Detected patient: {base_name} (L: {os.path.basename(left_file)}, R: {os.path.basename(right_file)}, {video_status})")
                else:
                     logger.error(f"Failed to add patient {base_name} even though files were found.") # Should not happen based on add_patient logic
            else:
                logger.warning(f"No matching right CSV file found for base name '{base_name}' derived from {os.path.basename(left_file)}.")

        logger.info(f"Automatic patient detection complete. Total patients added for processing: {patients_added}")
        return patients_added


    def process_all(self, extract_frames=True, **kwargs):
        """
        Process all added patients. Includes analysis, ML detection, visualization,
        and cleanup.
        Args:
            extract_frames (bool): Flag to control generation of frames and visual plots.
        """
        if not self.patients: logger.error("No patients added to the batch processor. Cannot process."); return None
        all_patient_summaries = []
        total_patients = len(self.patients)
        logger.info(f"Starting batch processing for {total_patients} patients...")
        logger.info(f"Visual output generation is {'ENABLED' if extract_frames else 'DISABLED'}.")

        for i, patient_info in enumerate(self.patients):
            patient_id = patient_info['patient_id']
            logger.info(f"--- Processing patient {i + 1}/{total_patients}: {patient_id} ---")
            analyzer = None # Ensure analyzer is fresh for each patient
            frames_extracted_successfully = False # Track frame extraction status

            try:
                 analyzer = FacialAUAnalyzer()
                 # Define and create the specific output directory for this patient
                 # Place it inside the main batch output directory
                 patient_output_dir = os.path.join(self.output_dir, patient_id)
                 os.makedirs(patient_output_dir, exist_ok=True)
                 analyzer.output_dir = patient_output_dir # Set analyzer's output dir attribute

                 # 1. Load Data
                 if not analyzer.load_data(patient_info['left_csv'], patient_info['right_csv']):
                     logger.error(f"({patient_id}) Failed to load data. Skipping this patient.")
                     continue # Skip to next patient

                 # 2. Analyze Intensity
                 analysis_results = analyzer.analyze_maximal_intensity()
                 if analysis_results is None or not analysis_results:
                     logger.error(f"({patient_id}) Failed to analyze maximal intensity or no results. Skipping subsequent steps for this patient.")
                     continue # Skip to next patient

                 # 3. Generate ML Input
                 ml_input_dict = analyzer.generate_ml_input_dict()
                 if ml_input_dict is None:
                      logger.warning(f"({patient_id}) Failed to generate ML input dictionary. ML detection will be skipped.")
                      ml_input_dict = {} # Use empty dict to prevent errors later, but detection won't run effectively

                 # 4. Run ML Detections (if input dict is available)
                 if ml_input_dict and 'Patient ID' in ml_input_dict: # Check if dict seems valid
                     logger.info(f"({patient_id}) Running ML detections...")
                     analyzer.detect_paralysis(patient_row_data_dict=ml_input_dict)
                     analyzer.detect_synkinesis(patient_row_data_dict=ml_input_dict)
                 else:
                     logger.warning(f"({patient_id}) Skipping ML detection steps due to missing or invalid ML input dictionary.")
                     # Ensure default 'Not Detected' states if ML skipped? Analyzer should handle defaults.

                 # 5. Visualizations (Frames, Plots, Dashboard) - Conditionally
                 if extract_frames:
                     if patient_info['video_path']: # Video path exists (validated in add_patient)
                         logger.info(f"({patient_id}) Generating visual outputs...")
                         # Call analyzer's frame extraction (which also triggers AU visuals)
                         frames_extracted_successfully, _ = analyzer.extract_frames(
                              patient_info['video_path'],
                              self.output_dir, # Pass main output dir, analyzer handles subdir
                              generate_visuals=True # Explicitly True here
                         )

                         if frames_extracted_successfully:
                             # Create other visualizations AFTER frames are extracted
                             # These might use the results or just analyzer state
                             #logger.info(f"({patient_id}) Generating patient dashboard...")
                             #analyzer.create_patient_dashboard() # Uses analyzer.output_dir

                             logger.info(f"({patient_id}) Generating symmetry visualization...")
                             analyzer.create_symmetry_visualization(patient_output_dir) # Pass specific dir

                             # --- MOVED CLEANUP CALL HERE ---
                             # Cleanup originals *after* all visualizations that might need them
                             logger.info(f"({patient_id}) Cleaning up temporary original frame files...")
                             analyzer.cleanup_extracted_frames()
                             # --- END MOVED CLEANUP CALL ---

                         else:
                              logger.warning(f"({patient_id}) Frame extraction or AU visualization failed. Skipping dashboard, symmetry plot, and cleanup.")
                              # No cleanup needed if extraction failed

                     else: # No video path provided for this patient
                         logger.warning(f"({patient_id}) Visual outputs requested but no valid video path found. Skipping visuals.")
                 else: # extract_frames is False
                     logger.info(f"({patient_id}) Skipping visual output generation as requested.")
                 # --- End Visualization Block ---

                 # 6. Generate Summary Row
                 logger.info(f"({patient_id}) Generating summary data row...")
                 patient_summary = analyzer.generate_summary_data()
                 if patient_summary:
                     all_patient_summaries.append(patient_summary)
                     logger.info(f"({patient_id}) Summary data generated.")
                 else:
                     logger.error(f"({patient_id}) Summary data generation failed.")

            except Exception as e_patient:
                 logger.error(f"--- UNHANDLED ERROR processing patient {patient_id}: {e_patient} ---", exc_info=True)
                 # Attempt to add a placeholder summary indicating error if possible
                 error_summary = {'Patient ID': patient_id, 'Processing Status': 'Error'}
                 all_patient_summaries.append(error_summary)

            finally:
                 if analyzer: del analyzer # Explicitly delete analyzer instance to free memory
                 logger.info(f"--- Finished processing patient {patient_id} ---")

        # --- Combine and Save Summary CSV ---
        if all_patient_summaries:
            logger.info(f"Generating combined summary CSV for {len(all_patient_summaries)} processed patients.")
            try:
                 # Use pandas to handle potentially different keys (e.g., error rows) gracefully
                 summary_df = pd.DataFrame(all_patient_summaries)

                 # Define expected column order (add 'Processing Status' if used for errors)
                 id_cols = ['Patient ID', 'Processing Status']
                 high_level_cols = ['Paralysis Detected', 'Synkinesis Detected']
                 paralysis_cols = sorted([c for c in summary_df.columns if 'Face Paralysis' in c])
                 synk_base_cols = sorted([c for c in summary_df.columns if any(st in c for st in SYNKINESIS_TYPES) and 'Confidence' not in c and c not in high_level_cols])
                 synk_conf_cols = sorted([c for c in summary_df.columns if any(st in c for st in SYNKINESIS_TYPES) and 'Confidence' in c])
                 # Dynamically get remaining columns assumed to be action-related AU data
                 all_known_cols = set(id_cols + high_level_cols + paralysis_cols + synk_base_cols + synk_conf_cols)
                 action_cols = sorted([c for c in summary_df.columns if c not in all_known_cols])

                 # Define the desired order, ensuring all actual columns are included
                 expected_cols = [col for col in (id_cols + high_level_cols + paralysis_cols + synk_base_cols + synk_conf_cols + action_cols) if col in summary_df.columns]

                 # Reindex to ensure consistent order and include all columns found
                 summary_df = summary_df.reindex(columns=expected_cols)

                 output_filename = "combined_results.csv";
                 output_path = os.path.join(self.output_dir, output_filename) # Save in the main batch output dir
                 summary_df.to_csv(output_path, index=False, na_rep='NA'); # Use NA for missing values
                 logger.info(f"Saved combined results to {output_path}");
                 self.summary_data = summary_df; # Store for potential later analysis
                 return output_path
            except Exception as e_save:
                 logger.error(f"Failed to create or save the combined results CSV: {e_save}", exc_info=True);
                 return None # Indicate failure
        else:
            logger.error("No patient summaries were generated. Cannot create combined CSV.");
            return None


    def analyze_asymmetry_across_patients(self):
        # ... (remains the same - relies on self.summary_data) ...
        if self.summary_data is None or self.summary_data.empty:
            logger.error("No summary data loaded or generated. Run process_all() first or load data to analyze aggregate stats.")
            return None
        logger.info("Analyzing aggregate statistics from summary data...")
        self.analyze_paralysis_and_synkinesis() # Call the method with indentation fixes
        logger.info("Aggregate analysis complete.")
        return None # Or return some aggregate stats if needed later

    def analyze_paralysis_and_synkinesis(self):
        """ Analyzes paralysis and synkinesis stats from the self.summary_data DataFrame. """
        # ... (remains the same - includes previous indentation fixes) ...
        if self.summary_data is None or self.summary_data.empty:
            logger.error("Summary data missing for analysis.")
            return

        # Filter out error rows if they exist before calculating stats
        if 'Processing Status' in self.summary_data.columns:
             df = self.summary_data[self.summary_data['Processing Status'] != 'Error'].copy()
             num_errors = len(self.summary_data) - len(df)
             if num_errors > 0: logger.warning(f"Excluding {num_errors} rows with processing errors from statistical analysis.")
        else:
             df = self.summary_data.copy() # Work on a copy

        num_patients = len(df)
        if num_patients == 0:
            logger.warning("Summary data contains no successfully processed patients. Cannot analyze stats.")
            return
        logger.info(f"Analyzing paralysis/synkinesis stats for {num_patients} successfully processed patients.")

        # --- Paralysis Analysis ---
        paralysis_col = 'Paralysis Detected'
        if paralysis_col not in df.columns:
            logger.warning(f"'{paralysis_col}' column missing in summary data. Skipping paralysis stats.")
            paralysis_stats_available = False
        else:
            paralysis_stats_available = True
            df[paralysis_col] = df[paralysis_col].fillna('No').astype(str)
            paralysis_count = (df[paralysis_col] == 'Yes').sum()
            paralysis_perc = (paralysis_count / num_patients * 100) if num_patients > 0 else 0
            logger.info(f"Paralysis Detected: {paralysis_count}/{num_patients} ({paralysis_perc:.1f}%).")

            if paralysis_count > 0:
                stats = {
                    'Total Analyzed Patients': num_patients,
                    'Patients with Paralysis': paralysis_count,
                    'Paralysis %': paralysis_perc
                }
                paralysis_detail_cols_found = False
                for side in ['Left', 'Right']:
                    for zone in ['Upper', 'Mid', 'Lower']:
                        col = f'{side} {zone} Face Paralysis'
                        if col in df.columns:
                            paralysis_detail_cols_found = True
                            df[col] = df[col].fillna('None').astype(str) # Ensure consistent type
                            counts = df[col].value_counts()
                            # Calculate stats for specific levels
                            for level in ['Partial', 'Complete']: # Defined levels
                                count = counts.get(level, 0)
                                level_perc = (count / num_patients * 100) if num_patients > 0 else 0
                                stats[f'{side} {zone} {level} Count'] = count
                                stats[f'{side} {zone} {level} %'] = level_perc
                            # Optionally report 'Error' counts if that state exists
                            error_count = counts.get('Error', 0)
                            if error_count > 0: stats[f'{side} {zone} Error Count'] = error_count

                        # else: Column not found (logged during summary generation ideally)

                if not paralysis_detail_cols_found:
                     logger.warning("Paralysis detected overall, but detailed zone columns ('[Side] [Zone] Face Paralysis') not found. Cannot provide detailed stats.")

                try:
                    # Only save if detail columns were found
                    if paralysis_detail_cols_found:
                         stats_df = pd.DataFrame([stats]) # Create DataFrame from the stats dict
                         path = os.path.join(self.output_dir, "paralysis_statistics_ML.csv")
                         stats_df.to_csv(path, index=False, na_rep='NA')
                         logger.info(f"Paralysis stats saved: {path}")
                    else:
                         logger.info("Skipping saving detailed paralysis stats file as no detail columns were found.")
                except Exception as e:
                    logger.error(f"Save paralysis stats failed: {e}", exc_info=True)
            elif paralysis_stats_available : # Paralysis column exists but count is 0
                 logger.info("No patients detected with paralysis based on summary data.")


        # --- Synkinesis Analysis ---
        synk_col = 'Synkinesis Detected'
        if synk_col not in df.columns:
            logger.warning(f"'{synk_col}' column missing in summary data. Skipping synkinesis stats.")
            synkinesis_stats_available = False
        else:
            synkinesis_stats_available = True
            df[synk_col] = df[synk_col].fillna('No').astype(str)
            synkinesis_count = (df[synk_col] == 'Yes').sum()
            synkinesis_perc = (synkinesis_count / num_patients * 100) if num_patients > 0 else 0
            logger.info(f"Synkinesis Detected: {synkinesis_count}/{num_patients} ({synkinesis_perc:.1f}%).")

            if synkinesis_count > 0:
                stats = {
                    'Total Analyzed Patients': num_patients,
                    'Patients with Synkinesis': synkinesis_count,
                    'Synkinesis %': synkinesis_perc
                }
                # Find relevant synkinesis columns (excluding confidence, ID, overall flag)
                synk_base_cols = [c for c in df.columns if any(st in c for st in SYNKINESIS_TYPES) and 'Confidence' not in c and 'Detected' not in c and 'Patient ID' not in c and 'Processing Status' not in c]
                synk_detail_cols_found = False
                if synk_base_cols:
                     synk_detail_cols_found = True
                     for col in synk_base_cols:
                         # This check should always pass now since cols derived from df.columns
                         if col in df.columns:
                             df[col] = df[col].fillna('No').astype(str) # Standardize Yes/No
                             count = (df[col] == 'Yes').sum()
                             col_perc = (count / num_patients * 100) if num_patients > 0 else 0
                             stats[f'{col} Count'] = count
                             stats[f'{col} %'] = col_perc
                         # No else needed here

                if not synk_detail_cols_found:
                     logger.warning("Synkinesis detected overall, but detailed type columns ('[Type] [Side]') not found. Cannot provide detailed stats.")

                try:
                     # Only save if detail columns were found
                    if synk_detail_cols_found:
                         stats_df = pd.DataFrame([stats]) # Create DataFrame from the stats dict
                         path = os.path.join(self.output_dir, "synkinesis_statistics_ML.csv")
                         stats_df.to_csv(path, index=False, na_rep='NA')
                         logger.info(f"Synkinesis stats saved: {path}")
                    else:
                         logger.info("Skipping saving detailed synkinesis stats file as no detail columns were found.")
                except Exception as e:
                    logger.error(f"Save synkinesis stats failed: {e}", exc_info=True)
            elif synkinesis_stats_available: # Synkinesis column exists but count is 0
                 logger.info("No patients detected with synkinesis based on summary data.")