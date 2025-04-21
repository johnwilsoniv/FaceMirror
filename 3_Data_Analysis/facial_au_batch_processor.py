# facial_au_batch_processor.py
# V1.18 Update: Added keep_default_na=False when loading expert key.

import os
import numpy as np
import pandas as pd
import logging
import glob
import re
from facial_au_analyzer import FacialAUAnalyzer
from facial_au_visualizer import FacialAUVisualizer
from facial_au_constants import (
    INCLUDED_ACTIONS, ALL_AU_COLUMNS, SYNKINESIS_TYPES,
    ACTION_DESCRIPTIONS, ACTION_TO_AUS, ZONE_SPECIFIC_ACTIONS,
    EXPERT_KEY_MAPPING, PARALYSIS_FINDINGS_KEYS, BOOL_FINDINGS_KEYS,
    standardize_paralysis_label, standardize_binary_label
)

logger = logging.getLogger(__name__)


class FacialAUBatchProcessor:
    """ Process multiple patients' facial AU data in batch mode. """

    def __init__(self, output_dir="../3.5_Results"):
        self.output_dir = output_dir
        self.patients = []
        self.summary_data = None
        self.errors = {}
        self.data_dir_path = None
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Batch Processor initialized. Main output directory: {os.path.abspath(self.output_dir)}")
        except OSError as e:
             logger.error(f"Could not create main output directory '{output_dir}'. Error: {e}")
             raise

    def add_patient(self, left_csv, right_csv, video_path=None, patient_id=None):
        if not patient_id:
            id_match = re.search(r'(IMG_\d+)', os.path.basename(left_csv), re.IGNORECASE)
            if id_match: patient_id = id_match.group(1)
            else: patient_id = os.path.basename(left_csv).split('_')[0]; logger.warning(f"Could not find standard ID in '{os.path.basename(left_csv)}'. Using fallback ID: '{patient_id}'")
        if not os.path.exists(left_csv): logger.warning(f"Left CSV not found: {left_csv}")
        if not os.path.exists(right_csv): logger.warning(f"Right CSV not found: {right_csv}")
        if video_path and not os.path.exists(video_path): logger.warning(f"Video file not found: {video_path}"); video_path = None
        self.patients.append({'patient_id': patient_id, 'left': left_csv, 'right': right_csv, 'video': video_path})
        logger.debug(f"Added patient for processing: {patient_id}")
        return True


    def auto_detect_patients(self, data_dir):
        self.data_dir_path = data_dir
        if not os.path.isdir(data_dir): logger.error(f"Data directory not found: {data_dir}"); return 0
        logger.debug(f"Searching for CSVs using glob pattern: {os.path.join(data_dir, '*.csv')}")
        try: csv_files = list(glob.iglob(os.path.join(data_dir, "*.csv")))
        except Exception as e: logger.error(f"Error during glob search for CSV files in {data_dir}: {e}"); return 0
        logger.debug(f"Glob found {len(csv_files)} total CSV files.")
        potential_pairs = {}
        pair_regex = re.compile(r'^(.*?)(?:_left|_right)(_.*?)?\.csv$', re.IGNORECASE)
        img_regex = re.compile(r'^(IMG_\d+)(?:_left|_right)(_.*?)?\.csv$', re.IGNORECASE)
        for file_path in csv_files:
            filename = os.path.basename(file_path); base_name = None; suffix = ""
            img_match = img_regex.match(filename)
            if img_match: base_name = img_match.group(1); suffix = img_match.group(2) if img_match.group(2) else ""
            else:
                pair_match = pair_regex.match(filename)
                if pair_match: base_name = pair_match.group(1); suffix = pair_match.group(2) if pair_match.group(2) else "";
            if base_name:
                pair_key = (base_name, suffix)
                if pair_key not in potential_pairs: potential_pairs[pair_key] = {'left': None, 'right': None}
                if '_left' in filename.lower() and potential_pairs[pair_key]['left'] is None: potential_pairs[pair_key]['left'] = file_path
                elif '_right' in filename.lower() and potential_pairs[pair_key]['right'] is None: potential_pairs[pair_key]['right'] = file_path
        patients_added_list = []
        for (base_name, suffix), files in sorted(potential_pairs.items()):
            if files['left'] and files['right']:
                patient_id = base_name; video_path = None
                for ext in ['mp4', 'avi', 'mov', 'wmv', 'mkv']:
                    exact_vid_path = os.path.join(data_dir, f"{base_name}.{ext}")
                    if os.path.exists(exact_vid_path): video_path = exact_vid_path; break
                if not video_path:
                     for ext in ['mp4', 'avi', 'mov', 'wmv', 'mkv']:
                         pattern = os.path.join(data_dir, f"{base_name}*.{ext}")
                         try:
                             vid_files = glob.glob(pattern)
                             filtered_vids = [vf for vf in vid_files if re.match(rf"^{re.escape(base_name)}([_.]|$).*", os.path.basename(vf), re.IGNORECASE)]
                             if filtered_vids: filtered_vids.sort(key=len); video_path = filtered_vids[0]; break
                         except Exception as e_glob_vid: logger.error(f"Error globbing videos for {base_name}*.{ext}: {e_glob_vid}")
                files['video'] = video_path
                patient_data = {'patient_id': patient_id, 'left': files['left'], 'right': files['right'], 'video': files['video']}
                patients_added_list.append(patient_data)
                video_status = f"Video: {os.path.basename(files['video'])}" if files['video'] else "Video NOT found"
                logger.info(f"Detected patient: {patient_id} (L: {os.path.basename(files['left'])}, R: {os.path.basename(files['right'])}, {video_status})")
            else: logger.warning(f"Incomplete pair found for base '{base_name}' suffix '{suffix}'. Skipping.")
        self.patients = patients_added_list
        logger.info(f"Automatic patient detection complete. Total patients added for processing: {len(self.patients)}")
        return len(self.patients)


    def process_patient(self, patient_info, extract_frames=True, expert_data_df=None, debug_mode=False):
        patient_id = patient_info['patient_id']; left_csv = patient_info['left']; right_csv = patient_info['right']; video_path = patient_info['video']
        logger.info(f"--- Processing patient: {patient_id} ---")
        analyzer = None; results = None; patient_output_dir = os.path.join(self.output_dir, patient_id)
        frames_extracted_successfully = False;
        patient_contradictions = {} # Initialize dict to store contradictions for this patient

        try:
            os.makedirs(patient_output_dir, exist_ok=True)
            logger.info(f"Output directory for patient {patient_id}: {os.path.abspath(patient_output_dir)}")

            analyzer = FacialAUAnalyzer(); analyzer.patient_id = patient_id; analyzer.output_dir = patient_output_dir
            visualizer = FacialAUVisualizer()

            if not analyzer.load_data(left_csv, right_csv):
                logger.error(f"({patient_id}) Failed to load data. Skipping."); self.errors[patient_id] = "Data Loading Failed"
                return {'Patient ID': patient_id, 'Processing Status': 'Error: Data Loading Failed'}

            logger.info(f"({patient_id}) Analyzing actions...")
            results = analyzer.analyze_all_actions(run_ml_paralysis=True, run_ml_synkinesis=True, run_ml_hypertonicity=True)
            if not results:
                 logger.error(f"({patient_id}) Action analysis failed."); self.errors[patient_id] = "Action Analysis Failed"
                 return {'Patient ID': patient_id, 'Processing Status': 'Error: Action Analysis Failed'}

            patient_summary = analyzer.generate_summary_data(results=results)
            if not patient_summary:
                 logger.error(f"({patient_id}) Failed to generate summary."); self.errors[patient_id] = "Summary Generation Failed"
                 return {'Patient ID': patient_id, 'Processing Status': 'Error: Summary Generation Failed'}

            # Debug Comparison - Build the patient_contradictions dictionary
            if debug_mode and expert_data_df is not None and not expert_data_df.empty:
                logger.info(f"({patient_id}) Debug Mode: Comparing results with loaded expert key...")
                if patient_id in expert_data_df.index:
                    expert_row = expert_data_df.loc[patient_id]
                    logger.debug(f"({patient_id}) Found expert data row.")
                    for algo_key, expert_key in EXPERT_KEY_MAPPING.items():
                        logger.debug(f"({patient_id}) Comparing Algo Key: '{algo_key}' <-> Expert Key: '{expert_key}'")
                        if algo_key not in patient_summary: logger.warning(f"({patient_id}) Algo key '{algo_key}' missing for comparison."); continue
                        if expert_key not in expert_row.index: logger.warning(f"({patient_id}) Expert key '{expert_key}' missing."); continue

                        algo_val = patient_summary[algo_key];
                        # Retrieve expert value - it should be a string now if it was "None" due to keep_default_na=False
                        expert_val = expert_row[expert_key]
                        logger.debug(f"({patient_id}) Raw Values - Algo: '{algo_val}' (Type: {type(algo_val)}), Expert: '{expert_val}' (Type: {type(expert_val)})")

                        # Check for actual missing values (like empty strings if they weren't handled by keep_default_na=False)
                        # pandas might still make truly empty cells NaN depending on exact CSV format/options
                        if pd.isna(expert_val) or (isinstance(expert_val, str) and expert_val.strip() == ''):
                             logger.debug(f"({patient_id}) Skip compare '{algo_key}': Expert value is effectively missing (NaN or empty string).")
                             continue

                        contradiction = False; std_algo, std_expert = None, None
                        # Standardize both values using the appropriate function
                        if algo_key in PARALYSIS_FINDINGS_KEYS:
                            std_algo = standardize_paralysis_label(algo_val)
                            std_expert = standardize_paralysis_label(expert_val) # Will handle string "None" correctly
                            logger.debug(f"({patient_id}) Standardized Paralysis - Algo: '{std_algo}', Expert: '{std_expert}'")
                            if std_algo != std_expert: contradiction = True
                        elif algo_key in BOOL_FINDINGS_KEYS:
                            std_algo = standardize_binary_label(algo_val)
                            std_expert = standardize_binary_label(expert_val) # Will handle string "None" correctly now
                            logger.debug(f"({patient_id}) Standardized Binary - Algo: '{std_algo}', Expert: '{std_expert}'")
                            if std_algo is not None and std_expert is not None and std_algo != std_expert: contradiction = True
                            elif std_algo is None or std_expert is None: logger.debug(f"({patient_id}) Skipping comparison for '{algo_key}' due to None after standardization.")
                        else: logger.warning(f"({patient_id}) Unknown finding type for '{algo_key}'."); continue

                        # Store STANDARDIZED EXPERT value on contradiction
                        if contradiction:
                            patient_contradictions[algo_key] = std_expert
                            logger.info(f"({patient_id}) CONTRADICTION for '{algo_key}': Algo='{algo_val}'(std:'{std_algo}'), Expert='{expert_val}'(std:'{std_expert}')")
                        else:
                            logger.debug(f"({patient_id}) MATCH for '{algo_key}': Algo='{algo_val}'(std:'{std_algo}'), Expert='{expert_val}'(std:'{std_expert}')")
                else: logger.warning(f"({patient_id}) Patient ID not found in expert key file index. Skipping comparison.")
            elif debug_mode and expert_data_df is None:
                 logger.warning(f"({patient_id}) Debug mode ON, but expert file not loaded. Skipping comparison.")
            else: logger.debug(f"({patient_id}) Debug mode OFF. Skipping expert comparison.")
            # End Debug Comparison

            # Frame Extraction
            if extract_frames and video_path:
                 logger.info(f"({patient_id}) Extracting relevant frames...")
                 frames_extracted_successfully, _ = analyzer.extract_frames(video_path=video_path, output_dir=patient_output_dir, generate_visuals=True)
                 if not frames_extracted_successfully: logger.warning(f"({patient_id}) Frame extraction failed/skipped.")
            elif extract_frames and not video_path: logger.warning(f"({patient_id}) Frame extraction requested but no video path provided.")
            else: logger.info(f"({patient_id}) Frame extraction skipped.")

            # Visualizations (Pass contradictions dictionary)
            if extract_frames:
                logger.info(f"({patient_id}) Generating visual outputs...")
                generated_plot_paths = {}
                action_keys = sorted([k for k in results.keys() if k != 'patient_summary'], key=lambda x: (x != 'BL', x))
                for action in action_keys:
                    action_results = results.get(action, {})
                    if not isinstance(action_results, dict): logger.warning(f"Skip viz for '{action}': invalid results."); continue
                    frame_path_png = analyzer.frame_paths.get(action); final_frame_path_to_use = None
                    if frame_path_png and isinstance(frame_path_png, str):
                        jpg_path = frame_path_png.replace("_original.png", ".jpg");
                        if os.path.exists(jpg_path): final_frame_path_to_use = jpg_path
                        elif os.path.exists(frame_path_png): final_frame_path_to_use = frame_path_png
                        else: logger.warning(f"({patient_id}-{action}) Original frame PNG path stored '{frame_path_png}' but file not found.")
                    else: logger.warning(f"({patient_id}-{action}) No frame path stored or invalid type for action '{action}'.")

                    frame_num = action_results.get('max_frame', 'N/A')
                    try:
                        plot_path = None
                        if action == 'BL':
                            plot_path = visualizer.create_baseline_visualization(
                                analyzer=analyzer, au_values_left=action_results.get('left', {}).get('au_values', {}),
                                au_values_right=action_results.get('right', {}).get('au_values', {}), frame_num=frame_num,
                                patient_output_dir=patient_output_dir, frame_path=final_frame_path_to_use,
                                action_descriptions=ACTION_DESCRIPTIONS, results=results,
                                contradictions=patient_contradictions # Pass contradictions
                             )
                        else:
                            if not final_frame_path_to_use: logger.warning(f"({patient_id}-{action}) Frame missing for AU plot."); continue
                            plot_path = visualizer.create_au_visualization(
                                analyzer=analyzer, au_values_left=action_results.get('left', {}).get('au_values', {}),
                                au_values_right=action_results.get('right', {}).get('au_values', {}),
                                norm_au_values_left=action_results.get('left', {}).get('normalized_au_values', {}),
                                norm_au_values_right=action_results.get('right', {}).get('normalized_au_values', {}),
                                action=action, frame_num=frame_num, patient_output_dir=patient_output_dir,
                                frame_path=final_frame_path_to_use, action_descriptions=ACTION_DESCRIPTIONS,
                                action_to_aus=ACTION_TO_AUS, results=results,
                                contradictions=patient_contradictions # Pass contradictions
                            )
                        if plot_path: generated_plot_paths[action] = plot_path
                    except Exception as e_viz_action: logger.error(f"({patient_id}) Viz error {action}: {e_viz_action}", exc_info=True)
                # Generate Symmetry and Dashboard Plots (Pass contradictions)
                try:
                     sym_path = visualizer.create_symmetry_visualization(
                         analyzer=analyzer, patient_output_dir=patient_output_dir, patient_id=patient_id,
                         results=results, action_descriptions=ACTION_DESCRIPTIONS)
                     generated_plot_paths['symmetry'] = sym_path if sym_path else None
                except Exception as e_viz_sym: logger.error(f"({patient_id}) Symmetry viz error: {e_viz_sym}", exc_info=True)
                try:
                     dash_path = visualizer.create_patient_dashboard(
                         analyzer=analyzer, patient_output_dir=patient_output_dir, patient_id=patient_id,
                         results=results, action_descriptions=ACTION_DESCRIPTIONS,
                         frame_paths=analyzer.frame_paths, contradictions=patient_contradictions)
                     generated_plot_paths['dashboard'] = dash_path if dash_path else None
                except Exception as e_viz_dash: logger.error(f"({patient_id}) Dashboard viz error: {e_viz_dash}", exc_info=True)
                if frames_extracted_successfully and hasattr(analyzer, 'cleanup_extracted_frames'): logger.info(f"({patient_id}) Cleaning up frames..."); analyzer.cleanup_extracted_frames()
            else: logger.info(f"({patient_id}) Skipping visual output generation.")

            patient_summary['Processing Status'] = 'Success'
            return patient_summary

        except Exception as e_patient:
            logger.error(f"--- UNHANDLED ERROR processing patient {patient_id}: {e_patient} ---", exc_info=True)
            self.errors[patient_id] = str(e_patient); summary_error = {'Patient ID': patient_id, 'Processing Status': f'Error: {e_patient}'}
            if results and isinstance(results, dict):
                 first_info = next((info for act, info in results.items() if act not in ['patient_summary'] and isinstance(info, dict)), None)
                 if first_info: summary_error['Paralysis Detected'] = first_info.get('paralysis', {}).get('detected', 'Error'); summary_error['Synkinesis Detected'] = first_info.get('synkinesis', {}).get('detected', 'Error')
                 summary_error['Hypertonicity Detected'] = results.get('patient_summary', {}).get('hypertonicity', {}).get('detected', 'Error')
            return summary_error

        finally:
            if analyzer:
                 if extract_frames and frames_extracted_successfully and hasattr(analyzer, 'cleanup_extracted_frames'):
                     try: logger.info(f"({patient_id}) Cleaning up frames in finally block..."); analyzer.cleanup_extracted_frames()
                     except Exception as e_cleanup: logger.error(f"({patient_id}) Error during final frame cleanup: {e_cleanup}")
                 del analyzer
            logger.info(f"--- Finished processing patient {patient_id} ---")


    def process_all(self, extract_frames=True, debug_mode=False, progress_callback=None):
        """ Processes all added patients and saves a combined summary. """
        if not self.patients: logger.error("No patients added."); return None
        expert_data_df = None
        if debug_mode:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                expert_key_path = os.path.join(script_dir, "FPRS FP Key.csv")
                logger.info(f"Debug Mode ON: Attempting to load expert key from: {expert_key_path}")

                if os.path.exists(expert_key_path):
                    # *** Use keep_default_na=False ***
                    expert_data_df = pd.read_csv(expert_key_path, keep_default_na=False)
                    # *** End Modification ***
                    if 'Patient' not in expert_data_df.columns: logger.error(f"Expert key missing 'Patient' column: {expert_key_path}"); expert_data_df = None
                    else: expert_data_df = expert_data_df.set_index('Patient'); logger.info(f"Successfully loaded expert key: {expert_key_path}")
                else: logger.warning(f"Debug mode enabled, but expert key file not found at expected location: {expert_key_path}")
            except Exception as e_load_key: logger.error(f"Failed load expert key '{expert_key_path}': {e_load_key}", exc_info=True); expert_data_df = None

        all_patient_summaries = []; total_patients = len(self.patients)
        logger.info(f"Starting batch processing for {total_patients} patients..."); logger.info(f"Visuals: {extract_frames}, Debug: {debug_mode}.")

        for i, patient_info in enumerate(self.patients):
             logger.info(f"--- Progress: Processing patient {i+1}/{total_patients} ({patient_info['patient_id']}) ---")
             if progress_callback:
                 try: progress_callback(i + 1, total_patients)
                 except Exception as e_cb: logger.error(f"Error in progress callback: {e_cb}")
             # Pass expert_data_df AND debug_mode flag to process_patient
             patient_summary = self.process_patient(
                 patient_info,
                 extract_frames=extract_frames,
                 expert_data_df=expert_data_df,
                 debug_mode=debug_mode
             )
             if patient_summary: all_patient_summaries.append(patient_summary)

        if all_patient_summaries:
            logger.info(f"Generating combined summary CSV for {len(all_patient_summaries)} processed patients.")
            try:
                 summary_df = pd.DataFrame(all_patient_summaries)
                 all_cols = summary_df.columns.tolist(); id_group = [c for c in ['Patient ID', 'Processing Status'] if c in all_cols]
                 detection_flags = [c for c in ['Paralysis Detected', 'Synkinesis Detected', 'Hypertonicity Detected'] if c in all_cols]
                 paralysis_details = sorted([c for c in all_cols if 'Face Paralysis' in c])
                 synk_details = sorted([c for c in all_cols if any(st in c for st in SYNKINESIS_TYPES) and 'Confidence' not in c and c not in detection_flags and 'Detected' not in c])
                 synk_conf = sorted([c for c in all_cols if any(st in c for st in SYNKINESIS_TYPES) and 'Confidence' in c])
                 action_max = sorted([c for c in all_cols if 'Max Side' in c or 'Max Frame' in c or 'Max Value' in c])
                 action_aus = sorted([c for c in all_cols if re.search(r'^[A-Z]{2,3}_(Left|Right)\sAU\d{2}_r$', c)])
                 action_aus_norm = sorted([c for c in all_cols if re.search(r'^[A-Z]{2,3}_(Left|Right)\sAU\d{2}_r\s\(Normalized\)$', c)])
                 ordered_groups = id_group + detection_flags + paralysis_details + synk_details + synk_conf + action_max + action_aus + action_aus_norm
                 ordered_set = set(ordered_groups); remaining_cols = sorted([c for c in all_cols if c not in ordered_set])
                 final_col_order = ordered_groups + remaining_cols
                 summary_df = summary_df[[col for col in final_col_order if col in summary_df.columns]]
                 output_filename = "combined_results.csv"; output_path = os.path.join(self.output_dir, output_filename)
                 summary_df.to_csv(output_path, index=False, na_rep='NA')
                 logger.info(f"Saved combined results to {output_path}"); self.summary_data = summary_df
                 return output_path
            except Exception as e_save:
                 logger.error(f"Failed save combined results CSV: {e_save}", exc_info=True)
                 try: alt_path = os.path.join(self.output_dir, "combined_results_unordered.csv"); pd.DataFrame(all_patient_summaries).to_csv(alt_path, index=False, na_rep='NA'); logger.warning(f"Saved unordered results: {alt_path}")
                 except Exception as e_alt_save: logger.error(f"Failed save unordered results: {e_alt_save}")
                 return None
        else: logger.error("No patient summaries generated."); return None


    def analyze_asymmetry_across_patients(self):
        # (No changes needed in this method)
        if self.summary_data is None or self.summary_data.empty:
            logger.error("No summary data available for aggregate analysis."); summary_path = os.path.join(self.output_dir, "combined_results.csv")
            if os.path.exists(summary_path):
                logger.info(f"Loading summary data from {summary_path}...")
                try: self.summary_data = pd.read_csv(summary_path)
                except Exception as e: logger.error(f"Failed load summary: {e}"); return None
            else: logger.error("Summary data file not found."); return None
        logger.info("Analyzing aggregate statistics from summary data...")
        self.analyze_paralysis_synkinesis_hypertonicity()
        logger.info("Aggregate analysis complete.")
        return None


    def analyze_paralysis_synkinesis_hypertonicity(self):
        # (No changes needed in this method)
        if self.summary_data is None or self.summary_data.empty: logger.error("Summary data missing."); return
        if 'Processing Status' in self.summary_data.columns:
             df = self.summary_data.dropna(subset=['Processing Status']); df = df[df['Processing Status'] == 'Success'].copy()
             num_errors = len(self.summary_data) - len(df);
             if num_errors > 0: logger.warning(f"Excluding {num_errors} non-Success rows from stats.")
        else: df = self.summary_data.copy(); logger.warning("'Processing Status' column missing.")

        num_patients = len(df)
        if num_patients == 0: logger.warning("No successfully processed patients found. Skipping stats."); return
        logger.info(f"Analyzing stats for {num_patients} successfully processed patients.")

        paralysis_col = 'Paralysis Detected'
        if paralysis_col in df.columns:
            df[paralysis_col] = df[paralysis_col].fillna('No').astype(str)
            paralysis_count = (df[paralysis_col] == 'Yes').sum(); paralysis_perc = (paralysis_count / num_patients * 100) if num_patients > 0 else 0
            logger.info(f"Paralysis Detected: {paralysis_count}/{num_patients} ({paralysis_perc:.1f}%).")
            if paralysis_count > 0:
                stats = {'Total Analyzed Patients': num_patients, 'Patients with Paralysis': paralysis_count, 'Paralysis %': paralysis_perc}; detail_cols_found = False
                for side in ['Left', 'Right']:
                    for zone in ['Upper', 'Mid', 'Lower']:
                        col = f'{side} {zone} Face Paralysis';
                        if col in df.columns: detail_cols_found = True; df[col] = df[col].fillna('None').astype(str); counts = df[col].value_counts();
                        for level in ['Partial', 'Complete']: count = counts.get(level, 0); level_perc = (count / num_patients * 100) if num_patients > 0 else 0; stats[f'{side} {zone} {level} Count'] = count; stats[f'{side} {zone} {level} %'] = level_perc
                        if counts.get('Error', 0) > 0: stats[f'{side} {zone} Error Count'] = counts['Error']
                if not detail_cols_found: logger.warning("Paralysis detected, but detailed zone columns not found.")
                try:
                    if detail_cols_found: stats_df = pd.DataFrame([stats]); path = os.path.join(self.output_dir, "paralysis_statistics.csv"); stats_df.to_csv(path, index=False, na_rep='NA'); logger.info(f"Paralysis stats saved: {path}")
                except Exception as e: logger.error(f"Save paralysis stats failed: {e}", exc_info=True)
        else: logger.warning(f"'{paralysis_col}' column missing.")

        synk_col = 'Synkinesis Detected'; hyper_col = 'Hypertonicity Detected'
        synk_detected_flag = (df[synk_col] == 'Yes') if synk_col in df.columns else pd.Series([False]*len(df), index=df.index)
        hyper_detected_flag = (df[hyper_col] == 'Yes') if hyper_col in df.columns else pd.Series([False]*len(df), index=df.index)
        combined_detected = synk_detected_flag | hyper_detected_flag; combined_count = combined_detected.sum()
        combined_perc = (combined_count / num_patients * 100) if num_patients > 0 else 0
        logger.info(f"Synkinesis/Hypertonicity Detected (Any Type): {combined_count}/{num_patients} ({combined_perc:.1f}%).")

        if combined_count > 0:
            stats = {'Total Analyzed Patients': num_patients, 'Patients with Synkinesis/Hypertonicity': combined_count, 'Synkinesis/Hypertonicity %': combined_perc}
            local_synk_types = SYNKINESIS_TYPES
            synk_type_cols = [c for c in df.columns if any(c == f"{st} {side}" for st in local_synk_types for side in ['Left', 'Right'])]
            detail_cols_found = bool(synk_type_cols)
            if detail_cols_found:
                 for col in synk_type_cols:
                     if col in df.columns:
                         df[col] = df[col].fillna('No').astype(str); count = (df[col] == 'Yes').sum(); col_perc = (count / num_patients * 100) if num_patients > 0 else 0
                         stats[f'{col} Count'] = count; stats[f'{col} %'] = col_perc
            else: logger.warning("Synkinesis/Hypertonicity detected, but detailed type columns not found.")
            try:
                if detail_cols_found: stats_df = pd.DataFrame([stats]); path = os.path.join(self.output_dir, "synkinesis_hypertonicity_statistics.csv"); stats_df.to_csv(path, index=False, na_rep='NA'); logger.info(f"Synkinesis/Hypertonicity stats saved: {path}")
            except Exception as e: logger.error(f"Save synkinesis/hypertonicity stats failed: {e}", exc_info=True)
        if synk_col not in df.columns: logger.warning(f"'{synk_col}' column missing.")
        if hyper_col not in df.columns: logger.warning(f"'{hyper_col}' column missing.")