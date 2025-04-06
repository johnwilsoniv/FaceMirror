# facial_synkinesis_detection.py
# - Renamed class to SynkinesisDetector
# - Added setdefault check for 'types' list before appending
# - Integrated MentalisDetector

import logging
import os

# --- Imports for sub-detectors ---
try: from ocular_oral_detector import OcularOralDetector
except ImportError: logging.error("Failed to import OcularOralDetector"); OcularOralDetector = None
try: from oral_ocular_detector import OralOcularDetector
except ImportError: logging.error("Failed to import OralOcularDetector"); OralOcularDetector = None
try: from snarl_smile_detector import SnarlSmileDetector
except ImportError: logging.error("Failed to import SnarlSmileDetector"); SnarlSmileDetector = None
# --- NEW: Import Mentalis Detector ---
try: from mentalis_detector import MentalisDetector
except ImportError: logging.error("Failed to import MentalisDetector"); MentalisDetector = None
# --- END NEW ---
# --- END IMPORTS ---

# Import constants
try:
    # Ensure SYNKINESIS_PATTERNS is imported
    from facial_au_constants import SYNKINESIS_TYPES, SYNKINESIS_PATTERNS, INCLUDED_ACTIONS
except ImportError:
    logging.warning("Could not import synkinesis constants.");
    SYNKINESIS_TYPES = []; SYNKINESIS_PATTERNS = {}; INCLUDED_ACTIONS = []

logger = logging.getLogger(__name__)

class SynkinesisDetector:
    """ ML-based synkinesis dispatcher. Uses specialized ML models. """

    def __init__(self):
        """Initialize the synkinesis detector with specialized sub-detectors."""
        self.ocular_oral_detector = OcularOralDetector() if OcularOralDetector else None
        self.oral_ocular_detector = OralOcularDetector() if OralOcularDetector else None
        self.snarl_smile_detector = SnarlSmileDetector() if SnarlSmileDetector else None
        # --- NEW: Instantiate Mentalis Detector ---
        self.mentalis_detector = MentalisDetector() if MentalisDetector else None
        # --- END NEW ---

        logger.info("SynkinesisDetector initialized.")
        if not self.ocular_oral_detector: logger.warning("Ocular-Oral detector unavailable.")
        if not self.oral_ocular_detector: logger.warning("Oral-Ocular detector unavailable.")
        if not self.snarl_smile_detector: logger.warning("Snarl-Smile detector unavailable.")
        # --- NEW: Log Mentalis Status ---
        if not self.mentalis_detector: logger.warning("Mentalis detector unavailable.")
        # --- END NEW ---


    def detect_synkinesis(self, results, patient_row_data_dict):
        """
        Detect potential synkinesis using ML models. Updates results dict in place.
        Dispatches detection task to appropriate sub-detector based on action.

        Args:
            results (dict): Results dictionary to be updated.
            patient_row_data_dict (dict): Full patient context row data containing all AU values.
        """
        if not results: logger.warning("No results passed to detect_synkinesis."); return
        if patient_row_data_dict is None: logger.error("patient_row_data_dict is None. Cannot perform ML synkinesis detection."); return

        # --- Define relevant actions for each type directly here or load from constants ---
        # Using SYNKINESIS_PATTERNS from constants if available
        ocular_oral_actions = SYNKINESIS_PATTERNS.get('Ocular-Oral', {}).get('relevant_actions', [])
        oral_ocular_actions = SYNKINESIS_PATTERNS.get('Oral-Ocular', {}).get('relevant_actions', [])
        snarl_smile_actions = SYNKINESIS_PATTERNS.get('Snarl-Smile', {}).get('relevant_actions', [])
        # --- NEW: Get Mentalis trigger actions ---
        mentalis_trigger_actions = SYNKINESIS_PATTERNS.get('Mentalis', {}).get('trigger_actions', [])
        # --- END NEW ---


        for action, info in results.items():
             if not isinstance(info, dict):
                 logger.warning(f"Skipping synkinesis detection for action '{action}' - info is not a dictionary.")
                 continue
             # Ensure the base synkinesis structure exists and is initialized correctly
             synk_info = info.setdefault('synkinesis', {})
             synk_info.setdefault('detected', False)
             synk_info.setdefault('types', [])
             synk_info.setdefault('side_specific', {st: {'left': False, 'right': False} for st in SYNKINESIS_TYPES})
             synk_info.setdefault('confidence', {st: {'left': 0.0, 'right': 0.0} for st in SYNKINESIS_TYPES})
             synk_info.setdefault('contributing_aus', {}) # Keep this if rule-based logic adds AUs


             for side in ['left', 'right']:
                 side_label = side.capitalize() # e.g., 'Left', 'Right'

                 # --- Call Ocular-Oral Detector ---
                 if self.ocular_oral_detector and action in ocular_oral_actions:
                      try:
                           is_detected, confidence = self.ocular_oral_detector.detect_ocular_oral_synkinesis(patient_row_data_dict, side_label)
                           if is_detected:
                                synk_info['detected'] = True; synk_type = 'Ocular-Oral'
                                if synk_type not in synk_info['types']: synk_info['types'].append(synk_type)
                                synk_info['side_specific'][synk_type][side] = True
                                current_conf = synk_info['confidence'].get(synk_type, {}).get(side, 0.0)
                                synk_info['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Ocular-Oral DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Ocular-Oral detect error ({action}, {side_label}): {e}", exc_info=True)

                 # --- Call Oral-Ocular Detector ---
                 if self.oral_ocular_detector and action in oral_ocular_actions:
                      try:
                           is_detected, confidence = self.oral_ocular_detector.detect_oral_ocular_synkinesis(patient_row_data_dict, side_label)
                           if is_detected:
                                synk_info['detected'] = True; synk_type = 'Oral-Ocular'
                                if synk_type not in synk_info['types']: synk_info['types'].append(synk_type)
                                synk_info['side_specific'][synk_type][side] = True
                                current_conf = synk_info['confidence'].get(synk_type, {}).get(side, 0.0)
                                synk_info['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Oral-Ocular DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Oral-Ocular detect error ({action}, {side_label}): {e}", exc_info=True)

                 # --- Call Snarl-Smile Detector ---
                 if self.snarl_smile_detector and action in snarl_smile_actions:
                      try:
                           is_detected, confidence = self.snarl_smile_detector.detect_snarl_smile_synkinesis(patient_row_data_dict, side_label)
                           if is_detected:
                                synk_info['detected'] = True; synk_type = 'Snarl-Smile'
                                if synk_type not in synk_info['types']: synk_info['types'].append(synk_type)
                                synk_info['side_specific'][synk_type][side] = True
                                current_conf = synk_info['confidence'].get(synk_type, {}).get(side, 0.0)
                                synk_info['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Snarl-Smile DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Snarl-Smile detect error ({action}, {side_label}): {e}", exc_info=True)

                 # --- NEW: Call Mentalis Detector ---
                 if self.mentalis_detector and action in mentalis_trigger_actions:
                      try:
                           is_detected, confidence = self.mentalis_detector.detect_mentalis_synkinesis(patient_row_data_dict, side_label)
                           if is_detected:
                                synk_info['detected'] = True; synk_type = 'Mentalis'
                                # Ensure 'Mentalis' exists in side_specific and confidence before assignment
                                synk_info['side_specific'].setdefault(synk_type, {'left': False, 'right': False})
                                synk_info['confidence'].setdefault(synk_type, {'left': 0.0, 'right': 0.0})
                                # Now proceed
                                if synk_type not in synk_info['types']: synk_info['types'].append(synk_type)
                                synk_info['side_specific'][synk_type][side] = True
                                current_conf = synk_info['confidence'].get(synk_type, {}).get(side, 0.0)
                                synk_info['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Mentalis DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Mentalis detect error ({action}, {side_label}): {e}", exc_info=True)
                 # --- END NEW ---