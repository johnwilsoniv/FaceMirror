# ml_synkinesis_detector.py (Corrected Imports/Instantiation)

import logging
import os

# --- UPDATED IMPORTS: Remove 'ml_' prefix ---
try: from ocular_oral_detector import OcularOralDetector # Renamed file/class
except ImportError: logging.error("Failed to import OcularOralDetector"); OcularOralDetector = None
try: from oral_ocular_detector import OralOcularDetector # Renamed file/class
except ImportError: logging.error("Failed to import OralOcularDetector"); OralOcularDetector = None
try: from snarl_smile_detector import SnarlSmileDetector # Renamed file/class
except ImportError: logging.error("Failed to import SnarlSmileDetector"); SnarlSmileDetector = None
# --- END UPDATED IMPORTS ---

# Import constants
try: from facial_au_constants import SYNKINESIS_TYPES, INCLUDED_ACTIONS
except ImportError: logging.warning("Could not import synkinesis constants."); SYNKINESIS_TYPES = []; INCLUDED_ACTIONS = []

logger = logging.getLogger(__name__)

class MLSynkinesisDetector: # Dispatcher class name remains unchanged
    """ ML-based synkinesis dispatcher. Uses specialized ML models. """

    def __init__(self):
        """Initialize the ML-based synkinesis detector with specialized sub-detectors."""
        # --- UPDATED INSTANTIATION: Use renamed classes ---
        self.ocular_oral_detector = OcularOralDetector() if OcularOralDetector else None
        self.oral_ocular_detector = OralOcularDetector() if OralOcularDetector else None
        self.snarl_smile_detector = SnarlSmileDetector() if SnarlSmileDetector else None
        # --- END UPDATED INSTANTIATION ---
        logger.info("MLSynkinesisDetector initialized.")
        if not self.ocular_oral_detector: logger.warning("Ocular-Oral detector unavailable.")
        if not self.oral_ocular_detector: logger.warning("Oral-Ocular detector unavailable.")
        if not self.snarl_smile_detector: logger.warning("Snarl-Smile detector unavailable.")

    def detect_synkinesis(self, results, patient_row_data_dict):
        """
        Detect potential synkinesis using ML models. Updates results dict in place.
        Args:
            results (dict): Results dictionary to be updated.
            patient_row_data_dict (dict): Full patient context row data.
        """
        if not results: logger.warning("No results passed to detect_synkinesis."); return
        if patient_row_data_dict is None: logger.error("patient_row_data_dict is None. Cannot perform ML synkinesis detection."); return

        ocular_oral_actions = ['ET', 'ES', 'RE', 'BL']
        oral_ocular_actions = ['BS', 'SS', 'SO', 'SE']
        snarl_smile_actions = ['BS', 'SS']

        for action, info in results.items():
             if 'synkinesis' not in info:
                 info['synkinesis'] = {
                     'detected': False, 'types': [],
                     'side_specific': {st: {'left': False, 'right': False} for st in SYNKINESIS_TYPES},
                     'confidence': {st: {'left': 0.0, 'right': 0.0} for st in SYNKINESIS_TYPES},
                     'contributing_aus': {}
                 }

             for side in ['left', 'right']:
                 # --- Call Ocular-Oral Detector ---
                 if self.ocular_oral_detector and action in ocular_oral_actions:
                      try:
                           is_detected, confidence = self.ocular_oral_detector.detect_ocular_oral_synkinesis(patient_row_data_dict, side)
                           if is_detected:
                                info['synkinesis']['detected'] = True; synk_type = 'Ocular-Oral'
                                if synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(synk_type)
                                info['synkinesis']['side_specific'][synk_type][side] = True
                                current_conf = info['synkinesis']['confidence'].get(synk_type, {}).get(side, 0.0)
                                info['synkinesis']['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Ocular-Oral DETECTED: Action={action}, Side={side}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Ocular-Oral detect error: {e}", exc_info=True)

                 # --- Call Oral-Ocular Detector ---
                 if self.oral_ocular_detector and action in oral_ocular_actions:
                      try:
                           is_detected, confidence = self.oral_ocular_detector.detect_oral_ocular_synkinesis(patient_row_data_dict, side)
                           if is_detected:
                                info['synkinesis']['detected'] = True; synk_type = 'Oral-Ocular'
                                if synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(synk_type)
                                info['synkinesis']['side_specific'][synk_type][side] = True
                                current_conf = info['synkinesis']['confidence'].get(synk_type, {}).get(side, 0.0)
                                info['synkinesis']['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Oral-Ocular DETECTED: Action={action}, Side={side}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Oral-Ocular detect error: {e}", exc_info=True)

                 # --- Call Snarl-Smile Detector ---
                 if self.snarl_smile_detector and action in snarl_smile_actions:
                      try:
                           is_detected, confidence = self.snarl_smile_detector.detect_snarl_smile_synkinesis(patient_row_data_dict, side)
                           if is_detected:
                                info['synkinesis']['detected'] = True; synk_type = 'Snarl-Smile'
                                if synk_type not in info['synkinesis']['types']: info['synkinesis']['types'].append(synk_type)
                                info['synkinesis']['side_specific'][synk_type][side] = True
                                current_conf = info['synkinesis']['confidence'].get(synk_type, {}).get(side, 0.0)
                                info['synkinesis']['confidence'][synk_type][side] = max(current_conf, confidence)
                                logger.debug(f"ML Snarl-Smile DETECTED: Action={action}, Side={side}, Conf={confidence:.3f}")
                      except Exception as e: logger.error(f"Snarl-Smile detect error: {e}", exc_info=True)