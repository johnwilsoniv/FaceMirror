"""
Facial synkinesis detection module.
Analyzes facial Action Units to detect possible synkinesis.
"""

import logging
from facial_au_constants import (
    SYNKINESIS_PATTERNS, SYNKINESIS_THRESHOLDS, SYNKINESIS_TYPES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialSynkinesisDetector:
    """
    Detects facial synkinesis (unwanted co-activation of muscles).
    """
    
    def __init__(self):
        """Initialize the facial synkinesis detector."""
        self.synkinesis_patterns = SYNKINESIS_PATTERNS
        self.synkinesis_thresholds = SYNKINESIS_THRESHOLDS
        self.SYNKINESIS_TYPES = SYNKINESIS_TYPES

    def detect_synkinesis(self, results):
        """
        Detect potential synkinesis (unwanted co-activation of muscles).
        Uses normalized values, requires multiple AU co-activation,
        and calculates confidence scores. Also tracks which specific AUs
        contributed to each synkinesis detection for better visualization.

        Args:
            results (dict): Results dictionary with AU values for each action

        Returns:
            None: Updates results dictionary in place
        """
        if not results:
            logger.warning("No results to analyze for synkinesis detection")
            return

        # First call specialized detection functions
        for action, info in results.items():
            # Initialize synkinesis structure with Yes/No for each type and side
            if 'synkinesis' not in info:
                info['synkinesis'] = {
                    'detected': False,
                    'types': [],
                    'side_specific': {
                        synk_type: {'left': False, 'right': False}
                        for synk_type in self.SYNKINESIS_TYPES
                    },
                    'confidence': {
                        synk_type: {'left': 0, 'right': 0}
                        for synk_type in self.SYNKINESIS_TYPES
                    },
                    'contributing_aus': {}  # Initialize the structure for tracking contributing AUs
                }

            # Make sure contributing_aus exists even if this is older data
            if 'contributing_aus' not in info['synkinesis']:
                info['synkinesis']['contributing_aus'] = {}

            # Detect Snarl Smile using specialized detection
            self._detect_snarl_smile(action, info)

        # Always check both sides for synkinesis
        sides_to_check = ['left', 'right']

        # Look for synkinesis in each relevant action
        for action, info in results.items():
            # For each defined synkinesis pattern (except Snarl-Smile which was handled specially)
            for synk_name, synk_pattern in self.synkinesis_patterns.items():
                # Skip Snarl-Smile as it was handled by specialized function
                if synk_name == 'Snarl-Smile':
                    continue

                # Only check patterns for relevant actions
                if action not in synk_pattern['relevant_actions']:
                    continue

                # Get thresholds for this synkinesis type
                thresholds = self.synkinesis_thresholds.get(synk_name, {
                    'trigger': 1.5,
                    'coupled': 1.0,
                    'ratio_lower': 0.3,
                    'ratio_upper': 1.0,
                    'min_coupled_aus': 2  # Require at least 2 coupled AUs to exceed threshold
                })

                trigger_aus = synk_pattern['trigger_aus']
                coupled_aus = synk_pattern['coupled_aus']

                # Check both sides
                for side in sides_to_check:
                    # Use normalized values to account for baseline activation
                    # Get trigger AU activations
                    trigger_values = {}
                    for au in trigger_aus:
                        if au in info[side]['normalized_au_values']:
                            trigger_values[au] = info[side]['normalized_au_values'][au]

                    # Get coupled AU activations (unwanted responses)
                    coupled_values = {}
                    for au in coupled_aus:
                        if au in info[side]['normalized_au_values']:
                            coupled_values[au] = info[side]['normalized_au_values'][au]

                    if trigger_values and coupled_values:
                        # Calculate average activation of trigger AUs
                        avg_trigger = sum(trigger_values.values()) / len(trigger_values)

                        # Track which specific AUs exceeded thresholds
                        active_trigger_aus = [au for au, val in trigger_values.items() if val > thresholds['trigger']]
                        active_coupled_aus = [au for au, val in coupled_values.items() if val > thresholds['coupled']]

                        # Count how many coupled AUs exceed the threshold
                        coupled_au_count = len(active_coupled_aus)

                        # Calculate average of only the AUs that exceed threshold
                        active_coupled_values = [val for val in coupled_values.values() if val > thresholds['coupled']]
                        avg_coupled = sum(active_coupled_values) / len(
                            active_coupled_values) if active_coupled_values else 0

                        # Calculate confidence score based on strength of coupled activation
                        confidence_score = 0
                        if avg_trigger > 0 and avg_coupled > 0:
                            trigger_factor = min(1, avg_trigger / thresholds['trigger'])
                            coupled_factor = min(1, avg_coupled / thresholds['coupled'])
                            confidence_score = (trigger_factor * 0.4) + (coupled_factor * 0.6)

                            # Adjust confidence based on number of coupled AUs active
                            proportion_active = coupled_au_count / len(coupled_values)
                            confidence_score *= (0.5 + (proportion_active * 0.5))

                        # If there's significant trigger activation AND enough coupled AUs exceed threshold
                        if (avg_trigger > thresholds['trigger'] and
                                avg_coupled > thresholds['coupled'] and
                                coupled_au_count >= thresholds.get('min_coupled_aus', 1)):

                            ratio = avg_coupled / avg_trigger if avg_trigger > 0 else 0

                            # If coupled activation is proportional to trigger, likely synkinesis
                            if thresholds['ratio_lower'] < ratio < thresholds['ratio_upper']:
                                # Update synkinesis information
                                info['synkinesis']['detected'] = True
                                if synk_name not in info['synkinesis']['types']:
                                    info['synkinesis']['types'].append(synk_name)
                                info['synkinesis']['side_specific'][synk_name][side] = True

                                # Store confidence score
                                info['synkinesis']['confidence'][synk_name][side] = confidence_score

                                # Make sure contributing_aus structure exists
                                if 'contributing_aus' not in info['synkinesis']:
                                    info['synkinesis']['contributing_aus'] = {}
                                if synk_name not in info['synkinesis']['contributing_aus']:
                                    info['synkinesis']['contributing_aus'][synk_name] = {}
                                if side not in info['synkinesis']['contributing_aus'][synk_name]:
                                    info['synkinesis']['contributing_aus'][synk_name][side] = {}

                                # Store which AUs contributed to detection
                                info['synkinesis']['contributing_aus'][synk_name][side] = {
                                    'trigger': active_trigger_aus,
                                    'response': active_coupled_aus
                                }

                                logger.info(f"Potential {synk_name} synkinesis detected on {side} side " +
                                            f"(confidence: {confidence_score:.2f}, coupled AUs: {coupled_au_count})")

    def _detect_snarl_smile(self, action, info):
        """
        Special detection for snarl smile synkinesis.
        Uses a weighted scoring approach instead of simple OR condition
        for more accurate detection and fewer false positives.
        Also tracks which specific AUs were responsible for detection.

        Args:
            action (str): The facial action being analyzed
            info (dict): Action data from results dictionary
        """
        # Only check for snarl smile during smile actions
        if action not in self.synkinesis_patterns['Snarl-Smile']['relevant_actions']:
            return

        # Get thresholds from updated thresholds
        thresholds = self.synkinesis_thresholds['Snarl-Smile']

        # Check both sides
        for side in ['left', 'right']:
            # Use normalized values to reduce false positives caused by baseline expressions
            # Get smile activation value - use AU12_r as primary trigger
            smile_value = info[side]['normalized_au_values'].get('AU12_r', 0)

            # Check if we have a significant smile (using the higher threshold)
            if smile_value > thresholds['trigger']:
                # Get individual component values
                nose_wrinkle = info[side]['normalized_au_values'].get('AU09_r', 0)
                upper_lip = info[side]['normalized_au_values'].get('AU10_r', 0)
                dimpler = info[side]['normalized_au_values'].get('AU14_r', 0)

                # Calculate weighted score based on importance of each component
                # Higher weights for more specific indicators
                weighted_score = (
                        (nose_wrinkle > thresholds['AU09_r']) * thresholds['AU09_weight'] +
                        (upper_lip > thresholds['AU10_r']) * thresholds['AU10_weight'] +
                        (dimpler > thresholds['AU14_r']) * thresholds['AU14_weight']
                )

                # Calculate confidence score (0-1) based on how much each AU exceeds its threshold
                confidence_score = 0
                component_count = 0

                # Track which AUs contributed to the synkinesis detection
                contributing_aus = []

                if nose_wrinkle > thresholds['AU09_r']:
                    confidence_score += min(1, (nose_wrinkle / thresholds['AU09_r'])) * thresholds['AU09_weight']
                    component_count += 1
                    contributing_aus.append('AU09_r')

                if upper_lip > thresholds['AU10_r']:
                    confidence_score += min(1, (upper_lip / thresholds['AU10_r'])) * thresholds['AU10_weight']
                    component_count += 1
                    contributing_aus.append('AU10_r')

                if dimpler > thresholds['AU14_r']:
                    confidence_score += min(1, (dimpler / thresholds['AU14_r'])) * thresholds['AU14_weight']
                    component_count += 1
                    contributing_aus.append('AU14_r')

                # Normalize confidence score
                if component_count > 0:
                    # Adjust confidence score based on smile intensity
                    smile_factor = min(1, smile_value / thresholds['trigger'])
                    confidence_score = confidence_score * smile_factor
                else:
                    confidence_score = 0

                # If the weighted score exceeds threshold AND at least 2 components are activated
                # This is more restrictive than the original OR condition
                if weighted_score >= thresholds['weighted_threshold'] and component_count >= 2:
                    # Update synkinesis information
                    info['synkinesis']['detected'] = True
                    if 'Snarl-Smile' not in info['synkinesis']['types']:
                        info['synkinesis']['types'].append('Snarl-Smile')
                    info['synkinesis']['side_specific']['Snarl-Smile'][side] = True

                    # Store confidence score
                    info['synkinesis']['confidence']['Snarl-Smile'][side] = confidence_score

                    # Make sure contributing_aus structure exists
                    if 'contributing_aus' not in info['synkinesis']:
                        info['synkinesis']['contributing_aus'] = {}
                    if 'Snarl-Smile' not in info['synkinesis']['contributing_aus']:
                        info['synkinesis']['contributing_aus']['Snarl-Smile'] = {}
                    if side not in info['synkinesis']['contributing_aus']['Snarl-Smile']:
                        info['synkinesis']['contributing_aus']['Snarl-Smile'][side] = {}

                    # Store which AUs contributed to detection
                    info['synkinesis']['contributing_aus']['Snarl-Smile'][side] = {
                        'trigger': ['AU12_r'],  # Smile is always the trigger
                        'response': contributing_aus
                    }

                    logger.info(
                        f"Potential Snarl-Smile synkinesis detected on {side} side (confidence: {confidence_score:.2f}, AUs: {contributing_aus})")