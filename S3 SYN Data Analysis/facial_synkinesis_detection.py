# --- START OF FILE facial_synkinesis_detection.py ---

# facial_synkinesis_detector.py (Refactored Dispatcher)
# V1.1 - Fix unpacking error when calling detector.detect
# V1.2 - Correctly store Hypertonicity result in patient_summary
# V1.3 - Standardize keys used for storing action-level synkinesis results
# V1.4 - Explicitly handle 'brow_cocked' and 'hypertonicity' outside the action loop.
# V1.5 - Fixed patient_summary initialization/access for Hypertonicity/Brow Cocked

import logging
import os

# --- Import the new generic detector and central config ---
try:
    from synkinesis_detector import SynkinesisDetector
    from synkinesis_config import SYNKINESIS_CONFIG, CLASS_NAMES
except ImportError:
    logging.critical("CRITICAL: Failed to import SynkinesisDetector or SYNKINESIS_CONFIG. Synkinesis detection will fail.")
    SynkinesisDetector = None
    SYNKINESIS_CONFIG = {}
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'} # Fallback

# --- Import constants ONLY for SYNKINESIS_TYPES needed for canonical keys ---
try:
    # Import SYNKINESIS_TYPES which should now include 'Brow Cocked'
    from facial_au_constants import SYNKINESIS_TYPES
except ImportError:
    logging.warning("Could not import SYNKINESIS_TYPES from facial_au_constants.");
    # Fallback: get types from config keys, ensure it's a list
    SYNKINESIS_TYPES = list(SYNKINESIS_CONFIG.keys()) if SYNKINESIS_CONFIG else []

logger = logging.getLogger(__name__)

class FacialSynkinesisDispatcher:
    """
    Dispatches synkinesis detection tasks to the generic SynkinesisDetector
    based on the action and configured synkinesis types.
    Also handles placing Hypertonicity and Brow Cocked results in the patient summary.
    """

    def __init__(self):
        """ Initialize the dispatcher. No models loaded here. """
        logger.info("FacialSynkinesisDispatcher initialized.")
        if SynkinesisDetector is None:
             logger.error("Generic SynkinesisDetector class is unavailable. Detection cannot proceed.")
        # Create a mapping from lowercase config keys to canonical capitalized keys
        if not SYNKINESIS_TYPES:
             logger.warning("SYNKINESIS_TYPES is empty, canonical key map will be empty.")
             self.config_key_to_canonical = {}
        else:
             # Use canonical types from constants (which should include Brow Cocked now)
             # Map config key (lowercase, underscore) to canonical key (as defined in constants)
             self.config_key_to_canonical = {}
             for stype_canonical in SYNKINESIS_TYPES:
                  config_key = stype_canonical.lower().replace('-', '_').replace(' ','_')
                  self.config_key_to_canonical[config_key] = stype_canonical
             logger.debug(f"Canonical key map created: {self.config_key_to_canonical}")
             # Explicitly check if critical patient-level types are mapped
             if 'hypertonicity' not in self.config_key_to_canonical: logger.error("Canonical key for 'hypertonicity' MISSING in map!")
             if 'brow_cocked' not in self.config_key_to_canonical: logger.error("Canonical key for 'brow_cocked' MISSING in map!")


    def detect_synkinesis(self, results, patient_row_data_dict):
        """
        Detects potential synkinesis by instantiating and calling the appropriate
        generic SynkinesisDetector for relevant actions and types defined in config.
        Handles Hypertonicity and Brow Cocked separately for patient_summary storage.

        Args:
            results (dict): Results dictionary with AU values per action (to be updated).
                            Also contains 'patient_summary'.
            patient_row_data_dict (dict): Full patient context row data.
        """
        if SynkinesisDetector is None: logger.error("Generic SynkinesisDetector unavailable. Skipping detection."); return
        if not results: logger.warning("No results passed to detect_synkinesis."); return
        if patient_row_data_dict is None: logger.error("patient_row_data_dict is None. Cannot perform synkinesis detection."); return
        if not SYNKINESIS_CONFIG: logger.error("SYNKINESIS_CONFIG is empty. Cannot determine relevant actions."); return
        if not self.config_key_to_canonical: logger.error("Canonical key map is empty. Patient-level storage will fail."); return

        logger.info("Detecting synkinesis using ML models (Dispatcher)...")

        # --- Initialize Patient Summary Structures ---
        # Ensure the main 'patient_summary' dictionary exists
        results.setdefault('patient_summary', {})

        # Explicitly ensure structures for patient-level types exist using CANONICAL keys
        patient_level_config_keys = ['hypertonicity', 'brow_cocked']
        for config_key in patient_level_config_keys:
            canonical_key = self.config_key_to_canonical.get(config_key)
            if not canonical_key:
                 logger.error(f"Failed to find canonical key for '{config_key}' during initialization. Storage WILL fail.")
                 # Optionally create with config_key as fallback, but log error prominently
                 canonical_key = config_key # Fallback, but expect errors later
                 results['patient_summary'].setdefault(canonical_key, {'error': f'Canonical key missing for {config_key}'})
            else:
                 # Use setdefault with the CANONICAL key to ensure the structure exists
                 default_structure = {
                     'left': False, 'right': False,
                     'conf_left': 0.0, 'conf_right': 0.0
                 }
                 # Add specific keys for hypertonicity if needed
                 if canonical_key == self.config_key_to_canonical.get('hypertonicity'):
                      default_structure['detected'] = False
                      # Keep confidence_left/right if other parts rely on it, otherwise remove redundancy
                      # default_structure['confidence_left'] = 0.0
                      # default_structure['confidence_right'] = 0.0
                      default_structure['contributing_aus_left'] = []
                      default_structure['contributing_aus_right'] = []

                 results['patient_summary'].setdefault(canonical_key, default_structure)
                 logger.debug(f"Ensured patient_summary structure exists for canonical key: '{canonical_key}'")
        # --- End Initialization ---

        # --- Build Action -> Action-Specific Synkinesis Type Mapping ---
        action_to_synk_types = {}
        action_specific_synk_types = [] # Store keys for action-specific types
        for synk_type_config_key, type_config in SYNKINESIS_CONFIG.items():
            # Exclude types handled ONLY at patient level (Hyper/BrowCocked)
            if synk_type_config_key in ['hypertonicity', 'brow_cocked']:
                continue
            action_specific_synk_types.append(synk_type_config_key)
            relevant_actions = type_config.get('relevant_actions', [])
            if not relevant_actions:
                logger.warning(f"No 'relevant_actions' defined for action-specific synkinesis type '{synk_type_config_key}' in config.")
            for act in relevant_actions:
                if act not in action_to_synk_types: action_to_synk_types[act] = []
                action_to_synk_types[act].append(synk_type_config_key)
        logger.debug(f"Action to Action-Specific Synkinesis Type mapping: {action_to_synk_types}")
        # --- End Mapping ---

        # --- Action-Specific Synkinesis Loop ---
        for action, info in results.items():
             if action == 'patient_summary': continue
             if not isinstance(info, dict):
                 logger.warning(f"Skipping action-specific synkinesis check for '{action}': info is not a dict.")
                 continue

             self._ensure_synkinesis_structure(info) # Ensure structure in action 'info' dict

             relevant_synk_config_types_for_action = action_to_synk_types.get(action, [])
             if not relevant_synk_config_types_for_action:
                 logger.debug(f"Action '{action}' does not trigger any action-specific synkinesis checks.")
                 continue

             logger.debug(f"Action '{action}' triggers checks for: {relevant_synk_config_types_for_action}")

             for synk_type_config_key in relevant_synk_config_types_for_action:
                 # Get the canonical (capitalized) key for storing results in ACTION info
                 canonical_synk_type = self.config_key_to_canonical.get(synk_type_config_key)
                 if not canonical_synk_type:
                     logger.warning(f"Could not find canonical key for config key '{synk_type_config_key}'. Skipping storage in action info for '{action}'.")
                     continue

                 logger.debug(f"Checking for {synk_type_config_key} during action '{action}' (storage key {canonical_synk_type})...")
                 try:
                     detector = SynkinesisDetector(synkinesis_type=synk_type_config_key)
                     if not all([detector.model, detector.scaler, detector.feature_names, detector.extract_features_func]):
                         logger.error(f"Failed to load artifacts/extractor for {detector.name}. Skipping check for '{action}'.")
                         continue

                     for side in ['left', 'right']:
                         side_label = side.capitalize()
                         try:
                            is_detected, confidence, positive_proba, details = detector.detect(patient_row_data_dict, side_label)

                            if details and details.get('error'):
                                logger.warning(f"Detection failed for {synk_type_config_key} ({action}, {side_label}): {details['error']}")
                            elif is_detected:
                                logger.info(f"ML {detector.name} DETECTED: Action={action}, Side={side_label}, Prob(+)={positive_proba:.3f}, Conf={confidence:.3f}")
                                info['synkinesis']['detected'] = True # Mark synkinesis present in this action
                                if canonical_synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(canonical_synk_type)
                                # Ensure structures exist before assignment
                                info['synkinesis']['side_specific'].setdefault(canonical_synk_type, {'left': False, 'right': False})
                                info['synkinesis']['confidence'].setdefault(canonical_synk_type, {'left': 0.0, 'right': 0.0})
                                # Update action-specific results
                                info['synkinesis']['side_specific'][canonical_synk_type][side] = True
                                current_conf = info['synkinesis']['confidence'][canonical_synk_type].get(side, 0.0)
                                info['synkinesis']['confidence'][canonical_synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"Stored action result: info['synkinesis']['side_specific']['{canonical_synk_type}']['{side}'] = True")
                            else: # Not detected
                                 logger.debug(f"ML {detector.name} NOT Detected: Action={action}, Side={side_label}, Prob(+)={positive_proba:.3f}, Conf={confidence:.3f}")
                                 # Ensure keys exist for consistent structure even if not detected
                                 info['synkinesis']['side_specific'].setdefault(canonical_synk_type, {'left': False, 'right': False})
                                 info['synkinesis']['confidence'].setdefault(canonical_synk_type, {'left': 0.0, 'right': 0.0})

                         except ValueError as ve_detect: logger.error(f"ValueError during detect for {synk_type_config_key} ({action}, {side}): {ve_detect}", exc_info=True)
                         except Exception as e_detect: logger.error(f"Exception during detect for {synk_type_config_key} ({action}, {side}): {e_detect}", exc_info=True)

                 except ValueError as ve_init: logger.error(f"Error initializing detector for type '{synk_type_config_key}': {ve_init}")
                 except Exception as e_init: logger.error(f"Error processing detector for type '{synk_type_config_key}' ({action}): {e_init}", exc_info=True)
        # --- End Action-Specific Loop ---

        # --- Handle Patient-Level Synkinesis (Hypertonicity & Brow Cocked) ---
        # Use the same config keys as initialization loop
        patient_level_config_keys = ['hypertonicity', 'brow_cocked']
        for synk_type_config_key in patient_level_config_keys:
            if synk_type_config_key not in SYNKINESIS_CONFIG:
                logger.warning(f"Patient-level synkinesis type '{synk_type_config_key}' not found in SYNKINESIS_CONFIG. Skipping.")
                continue

            # Get the canonical key again, using the map (should match initialization)
            canonical_synk_type = self.config_key_to_canonical.get(synk_type_config_key)
            if not canonical_synk_type:
                 # This should not happen if the map is correct and SYNKINESIS_TYPES is updated
                 logger.error(f"CRITICAL: Canonical key for '{synk_type_config_key}' missing in map during patient-level processing. Skipping.")
                 continue

            logger.debug(f"Checking for patient-level {synk_type_config_key} (storage key {canonical_synk_type})...")
            try:
                detector = SynkinesisDetector(synkinesis_type=synk_type_config_key)
                if not all([detector.model, detector.scaler, detector.feature_names, detector.extract_features_func]):
                    logger.error(f"Failed to load artifacts/extractor for {detector.name}. Skipping check.")
                    # Mark error in summary if possible
                    if 'patient_summary' in results and canonical_synk_type in results['patient_summary']:
                         results['patient_summary'][canonical_synk_type]['left'] = 'Error'
                         results['patient_summary'][canonical_synk_type]['right'] = 'Error'
                    continue

                # Get the specific summary sub-dictionary to update using the CANONICAL key
                # This .get() should now succeed because of the robust initialization block earlier
                summary_sub_dict = results['patient_summary'].get(canonical_synk_type)
                if summary_sub_dict is None:
                     # Log error if it still fails (indicates deeper issue)
                     logger.error(f"CRITICAL: Summary structure for '{canonical_synk_type}' STILL missing despite initialization attempt. Cannot store results.")
                     continue

                any_side_detected = False # Track if detected on either side for this patient-level type
                for side in ['left', 'right']:
                    side_label = side.capitalize()
                    try:
                        is_detected, confidence, positive_proba, details = detector.detect(patient_row_data_dict, side_label)

                        if details and details.get('error'):
                            logger.warning(f"Detection failed for {synk_type_config_key} ({side_label}): {details['error']}")
                            summary_sub_dict[side] = 'Error' # Store 'Error' string
                        elif is_detected:
                            logger.info(f"ML {detector.name} DETECTED: Side={side_label}, Prob(+)={positive_proba:.3f}, Conf={confidence:.3f}")
                            summary_sub_dict[side] = True # Store boolean True
                            summary_sub_dict[f'conf_{side}'] = confidence
                            any_side_detected = True
                        else: # Not detected
                            logger.debug(f"ML {detector.name} NOT Detected: Side={side_label}, Prob(+)={positive_proba:.3f}, Conf={confidence:.3f}")
                            summary_sub_dict[side] = False # Store boolean False
                            summary_sub_dict[f'conf_{side}'] = confidence

                    except ValueError as ve_detect: logger.error(f"ValueError during detect for {synk_type_config_key} ({side}): {ve_detect}", exc_info=True); summary_sub_dict[side] = 'Error'
                    except Exception as e_detect: logger.error(f"Exception during detect for {synk_type_config_key} ({side}): {e_detect}", exc_info=True); summary_sub_dict[side] = 'Error'

                # Set overall 'detected' flag for Hypertonicity if detected on either side
                # Check if the canonical key corresponds to Hypertonicity using the map
                hyper_canonical_check = self.config_key_to_canonical.get('hypertonicity')
                if hyper_canonical_check and canonical_synk_type == hyper_canonical_check and 'detected' in summary_sub_dict:
                    summary_sub_dict['detected'] = any_side_detected
                    logger.debug(f"Set overall '{canonical_synk_type}' detected flag to {any_side_detected}")

            except ValueError as ve_init: logger.error(f"Error initializing detector for patient-level type '{synk_type_config_key}': {ve_init}")
            except Exception as e_init: logger.error(f"Error processing detector for patient-level type '{synk_type_config_key}': {e_init}", exc_info=True)
        # --- End Patient-Level Loop ---


        self._log_detection_summary(results)


    def _ensure_synkinesis_structure(self, info):
        """ Ensures the synkinesis dictionary structure exists within the action info, using canonical keys. """
        synk_info = info.setdefault('synkinesis', {})
        synk_info.setdefault('detected', False)
        synk_info.setdefault('types', [])
        # Use the canonical keys from the mapping values (derived from SYNKINESIS_TYPES)
        all_canonical_types = list(self.config_key_to_canonical.values()) if self.config_key_to_canonical else SYNKINESIS_TYPES
        side_spec = synk_info.setdefault('side_specific', {})
        conf = synk_info.setdefault('confidence', {})

        # Get canonical keys for patient-level types to exclude them here
        hyper_canonical = self.config_key_to_canonical.get('hypertonicity')
        bc_canonical = self.config_key_to_canonical.get('brow_cocked')
        patient_level_canonical = [k for k in [hyper_canonical, bc_canonical] if k] # List of valid canonical patient keys

        for st_canonical in all_canonical_types:
            # Only create structure for types NOT handled solely in patient_summary
            if st_canonical not in patient_level_canonical:
                side_spec.setdefault(st_canonical, {'left': False, 'right': False})
                conf.setdefault(st_canonical, {'left': 0.0, 'right': 0.0})
        synk_info.setdefault('contributing_aus', {})


    def _log_detection_summary(self, results):
        """ Logs a summary including patient-level types from patient_summary. """
        action_specific_detected_summary = {}
        num_actions_processed = 0
        for action, info in results.items():
            if action == 'patient_summary' or not isinstance(info, dict): continue
            num_actions_processed += 1
            synk_info = info.get('synkinesis')
            if synk_info and synk_info.get('detected'):
                for canonical_synk_type, sides in synk_info.get('side_specific', {}).items():
                    # Get canonical keys for patient-level types to exclude them here
                    hyper_canonical = self.config_key_to_canonical.get('hypertonicity')
                    bc_canonical = self.config_key_to_canonical.get('brow_cocked')
                    patient_level_canonical = [k for k in [hyper_canonical, bc_canonical] if k]

                    if canonical_synk_type in patient_level_canonical: continue # Exclude types handled directly in patient_summary from this tally

                    if sides.get('left') or sides.get('right'):
                        action_specific_detected_summary.setdefault(canonical_synk_type, {'left': 0, 'right': 0})
                        if sides.get('left'): action_specific_detected_summary[canonical_synk_type]['left'] += 1
                        if sides.get('right'): action_specific_detected_summary[canonical_synk_type]['right'] += 1

        # Get patient-level results using canonical keys
        patient_summary = results.get('patient_summary', {})
        hyper_canonical = self.config_key_to_canonical.get('hypertonicity')
        bc_canonical = self.config_key_to_canonical.get('brow_cocked')

        hyper_info = patient_summary.get(hyper_canonical, {}) if hyper_canonical else {}
        hyper_detected = hyper_info.get('detected', False) # Overall flag for Hypertonicity

        bc_info = patient_summary.get(bc_canonical, {}) if bc_canonical else {}
        bc_detected = bc_info.get('left', False) or bc_info.get('right', False) # Overall flag for Brow Cocked

        if action_specific_detected_summary or hyper_detected or bc_detected:
            logger.info("========== DISPATCHER SYNKINESIS DETECTION SUMMARY ==========")
            logger.info(f"(Across {num_actions_processed} actions)")
            # Log patient-level findings first
            if hyper_detected:
                 # Use the canonical name
                 logger.info(f"  {hyper_canonical or 'Hypertonicity?'}: Detected=True (Left={hyper_info.get('left', False)}, Right={hyper_info.get('right', False)})")
            if bc_detected:
                 # Use the canonical name
                 logger.info(f"  {bc_canonical or 'Brow Cocked?'}: Detected=True (Left={bc_info.get('left', False)}, Right={bc_info.get('right', False)})")
            # Log action-specific findings counts
            for canonical_synk_type, side_counts in sorted(action_specific_detected_summary.items()):
                if side_counts['left'] > 0 or side_counts['right'] > 0:
                     logger.info(f"  {canonical_synk_type}: Left={side_counts['left']}, Right={side_counts['right']} (Action Count)")
        else:
            logger.info(f"Dispatcher: No synkinesis detected across {num_actions_processed} actions.")

# --- END OF FILE facial_synkinesis_detection.py ---