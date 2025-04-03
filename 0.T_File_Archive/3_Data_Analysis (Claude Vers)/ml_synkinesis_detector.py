"""
Main ML-based synkinesis detector implementation.
Integrates the specialized detectors for different synkinesis types.
"""

import logging
from ml_ocular_oral_detector import MLOcularOralDetector
from ml_oral_ocular_detector import MLOralOcularDetector
from ml_snarl_smile_detector import MLSnarlSmileDetector
from facial_au_constants import SYNKINESIS_TYPES

logger = logging.getLogger(__name__)

class MLSynkinesisDetector:
    """
    Machine learning-based synkinesis detector.
    Uses specialized ML models for different synkinesis types.
    """

    def __init__(self):
        """Initialize the ML-based synkinesis detector with specialized sub-detectors."""
        self.ocular_oral_detector = MLOcularOralDetector()
        self.oral_ocular_detector = MLOralOcularDetector()
        self.snarl_smile_detector = MLSnarlSmileDetector()

    def detect_synkinesis(self, results):
        """
        Detect potential synkinesis (unwanted co-activation of muscles).
        Uses ML models to detect each type of synkinesis.

        Args:
            results (dict): Results dictionary with AU values for each action

        Returns:
            None: Updates results dictionary in place
        """
        if not results:
            logger.warning("No results to analyze for synkinesis detection")
            return

        # Process each action for synkinesis detection
        for action, info in results.items():
            # Initialize synkinesis structure with Yes/No for each type and side
            if 'synkinesis' not in info:
                info['synkinesis'] = {
                    'detected': False,
                    'types': [],
                    'side_specific': {
                        synk_type: {'left': False, 'right': False}
                        for synk_type in SYNKINESIS_TYPES
                    },
                    'confidence': {
                        synk_type: {'left': 0, 'right': 0}
                        for synk_type in SYNKINESIS_TYPES
                    },
                    'contributing_aus': {}
                }

            # Make sure contributing_aus exists
            if 'contributing_aus' not in info['synkinesis']:
                info['synkinesis']['contributing_aus'] = {}

            # Check both sides for synkinesis
            sides_to_check = ['left', 'right']

            # Detect Ocular-Oral synkinesis (eye causing mouth movement)
            self._detect_ocular_oral(action, info, sides_to_check)

            # Detect Oral-Ocular synkinesis (mouth causing eye movement)
            self._detect_oral_ocular(action, info, sides_to_check)

            # Detect Snarl-Smile synkinesis (smile causing nose wrinkling)
            self._detect_snarl_smile(action, info, sides_to_check)

    def _detect_ocular_oral(self, action, info, sides_to_check):
        """
        Detect Ocular-Oral synkinesis using ML model.

        Args:
            action (str): Current action being analyzed
            info (dict): Results for this action
            sides_to_check (list): List of sides to check
        """
        synk_name = 'Ocular-Oral'
        for side in sides_to_check:
            # Use ML detector for detection
            is_detected, confidence_score, contributing_aus = self.ocular_oral_detector.detect_ocular_oral_synkinesis(
                action, info, side)

            # Update synkinesis information if detected
            if is_detected:
                info['synkinesis']['detected'] = True
                if synk_name not in info['synkinesis']['types']:
                    info['synkinesis']['types'].append(synk_name)
                info['synkinesis']['side_specific'][synk_name][side] = True

                # Store confidence score
                info['synkinesis']['confidence'][synk_name][side] = confidence_score

                # Ensure structure exists for contributing AUs
                if synk_name not in info['synkinesis']['contributing_aus']:
                    info['synkinesis']['contributing_aus'][synk_name] = {}
                if side not in info['synkinesis']['contributing_aus'][synk_name]:
                    info['synkinesis']['contributing_aus'][synk_name][side] = {}

                # Store contributing AUs information
                info['synkinesis']['contributing_aus'][synk_name][side] = contributing_aus

                logger.info(f"ML-based {synk_name} synkinesis detected on {side} side " +
                            f"during {action} (confidence: {confidence_score:.2f})")

    def _detect_oral_ocular(self, action, info, sides_to_check):
        """
        Detect Oral-Ocular synkinesis using ML model.

        Args:
            action (str): Current action being analyzed
            info (dict): Results for this action
            sides_to_check (list): List of sides to check
        """
        synk_name = 'Oral-Ocular'
        for side in sides_to_check:
            # Use ML detector for detection
            is_detected, confidence_score, contributing_aus = self.oral_ocular_detector.detect_oral_ocular_synkinesis(
                action, info, side)

            # Update synkinesis information if detected
            if is_detected:
                info['synkinesis']['detected'] = True
                if synk_name not in info['synkinesis']['types']:
                    info['synkinesis']['types'].append(synk_name)
                info['synkinesis']['side_specific'][synk_name][side] = True

                # Store confidence score
                info['synkinesis']['confidence'][synk_name][side] = confidence_score

                # Ensure structure exists for contributing AUs
                if synk_name not in info['synkinesis']['contributing_aus']:
                    info['synkinesis']['contributing_aus'][synk_name] = {}
                if side not in info['synkinesis']['contributing_aus'][synk_name]:
                    info['synkinesis']['contributing_aus'][synk_name][side] = {}

                # Store contributing AUs information
                info['synkinesis']['contributing_aus'][synk_name][side] = contributing_aus

                logger.info(f"ML-based {synk_name} synkinesis detected on {side} side " +
                            f"during {action} (confidence: {confidence_score:.2f})")

    def _detect_snarl_smile(self, action, info, sides_to_check):
        """
        Detect Snarl-Smile synkinesis using ML model.

        Args:
            action (str): Current action being analyzed
            info (dict): Results for this action
            sides_to_check (list): List of sides to check
        """
        synk_name = 'Snarl-Smile'
        for side in sides_to_check:
            # Use ML detector for detection
            is_detected, confidence_score, contributing_aus = self.snarl_smile_detector.detect_snarl_smile_synkinesis(
                action, info, side)

            # Update synkinesis information if detected
            if is_detected:
                info['synkinesis']['detected'] = True
                if synk_name not in info['synkinesis']['types']:
                    info['synkinesis']['types'].append(synk_name)
                info['synkinesis']['side_specific'][synk_name][side] = True

                # Store confidence score
                info['synkinesis']['confidence'][synk_name][side] = confidence_score

                # Ensure structure exists for contributing AUs
                if synk_name not in info['synkinesis']['contributing_aus']:
                    info['synkinesis']['contributing_aus'][synk_name] = {}
                if side not in info['synkinesis']['contributing_aus'][synk_name]:
                    info['synkinesis']['contributing_aus'][synk_name][side] = {}

                # Store contributing AUs information
                info['synkinesis']['contributing_aus'][synk_name][side] = contributing_aus

                logger.info(f"ML-based {synk_name} synkinesis detected on {side} side " +
                            f"during {action} (confidence: {confidence_score:.2f})")
