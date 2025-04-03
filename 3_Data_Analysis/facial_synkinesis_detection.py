# facial_synkinesis_detection.py (Renamed from ml_synkinesis_detector.py)
# - Renamed class to SynkinesisDetector

import logging
import os

# --- Imports for sub-detectors ---
# These imports remain the same, pointing to the individual detector files
try: from ocular_oral_detector import OcularOralDetector
except ImportError: logging.error("Failed to import OcularOralDetector"); OcularOralDetector = None
try: from oral_ocular_detector import OralOcularDetector
except ImportError: logging.error("Failed to import OralOcularDetector"); OralOcularDetector = None
try: from snarl_smile_detector import SnarlSmileDetector
except ImportError: logging.error("Failed to import SnarlSmileDetector"); SnarlSmileDetector = None
# --- END IMPORTS ---

# Import constants
try: from facial_au_constants import SYNKINESIS_TYPES, INCLUDED_ACTIONS # Keep INCLUDED_ACTIONS if used for filtering?
except ImportError: logging.warning("Could not import synkinesis constants."); SYNKINESIS_TYPES = []; INCLUDED_ACTIONS = []

logger = logging.getLogger(__name__)

# --- RENAMED CLASS ---
class SynkinesisDetector:
    """ ML-based synkinesis dispatcher. Uses specialized ML models. """

    def __init__(self):
        """Initialize the synkinesis detector with specialized sub-detectors."""
        # --- Instantiation uses original detector class names ---
        self.ocular_oral_detector = OcularOralDetector() if OcularOralDetector else None
        self.oral_ocular_detector = OralOcularDetector() if OralOcularDetector else None
        self.snarl_smile_detector = SnarlSmileDetector() if SnarlSmileDetector else None
        # --- END Instantiation ---
        # --- Updated Log Message ---
        logger.info("SynkinesisDetector initialized.")
        if not self.ocular_oral_detector: logger.warning("Ocular-Oral detector unavailable.")
        if not self.oral_ocular_detector: logger.warning("Oral-Ocular detector unavailable.")
        if not self.snarl_smile_detector: logger.warning("Snarl-Smile detector unavailable.")
        # --- End Updated Log Message ---

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

        # Define action lists for routing (consider moving to constants if stable)
        ocular_oral_actions = ['ET', 'ES', 'RE', 'BL']
        oral_ocular_actions = ['BS', 'SS', 'SO', 'SE']
        snarl_smile_actions = ['BS', 'SS'] # Subset of oral_ocular

        for action, info in results.items():
             # Ensure the synkinesis structure exists
             if 'synkinesis' not in info:
                 info['synkinesis'] = {
                     'detected': False, 'types': [],
                     'side_specific': {st: {'left': False, 'right': False} for st in SYNKINESIS_TYPES},
                     'confidence': {st: {'left': 0.0, 'right': 0.0} for st in SYNKINESIS_TYPES},
                     'contributing_aus': {}
                 }

             for side in ['left', 'right']:
                 side_label = side.capitalize() # e.g., 'Left', 'Right'

                 # --- Call Ocular-Oral Detector ---
                 if self.ocular_oral_detector and action in ocular_oral_actions:
                      try:
                           is_detected, confidence = self.ocular_oral_detector.detect_ocular_oral_synkinesis(patient_row_data_dict, side_label) # Pass capitalized side
                           if is_detected:
                                info['synkinesis']['detected'] = True; synk_type = 'Ocular-Oral'
                                if synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(synk_type)
                                info['synkinesis']['side_specific'][synk_type][side] = True # Use lower-case side for dict key
                                current_conf = info['synkinesis']['confidence'].get(synk_type, {}).get(side, 0.0)
                                info['synkinesis']['confidence'][synk_type][side] = max(current_conf, confidence) # Use lower-case side for dict key
                                logger.debug(f"ML Ocular-Oral DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Ocular-Oral detect error ({action}, {side_label}): {e}", exc_info=True)

                 # --- Call Oral-Ocular Detector ---
                 if self.oral_ocular_detector and action in oral_ocular_actions:
                      try:
                           is_detected, confidence = self.oral_ocular_detector.detect_oral_ocular_synkinesis(patient_row_data_dict, side_label) # Pass capitalized side
                           if is_detected:
                                info['synkinesis']['detected'] = True; synk_type = 'Oral-Ocular'
                                if synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(synk_type)
                                info['synkinesis']['side_specific'][synk_type][side] = True # Use lower-case side
                                current_conf = info['synkinesis']['confidence'].get(synk_type, {}).get(side, 0.0)
                                info['synkinesis']['confidence'][synk_type][side] = max(current_conf, confidence) # Use lower-case side
                                logger.debug(f"ML Oral-Ocular DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Oral-Ocular detect error ({action}, {side_label}): {e}", exc_info=True)

                 # --- Call Snarl-Smile Detector ---
                 if self.snarl_smile_detector and action in snarl_smile_actions:
                      try:
                           is_detected, confidence = self.snarl_smile_detector.detect_snarl_smile_synkinesis(patient_row_data_dict, side_label) # Pass capitalized side
                           if is_detected:
                                info['synkinesis']['detected'] = True; synk_type = 'Snarl-Smile'
                                if synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(synk_type)
                                info['synkinesis']['side_specific'][synk_type][side] = True # Use lower-case side
                                current_conf = info['synkinesis']['confidence'].get(synk_type, {}).get(side, 0.0)
                                info['synkinesis']['confidence'][synk_type][side] = max(current_conf, confidence) # Use lower-case side
                                logger.debug(f"ML Snarl-Smile DETECTED: Action={action}, Side={side_label}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Snarl-Smile detect error ({action}, {side_label}): {e}", exc_info=True)