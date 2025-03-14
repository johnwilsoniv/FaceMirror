"""
Core Facial Action Unit Analyzer class.
Handles analysis of facial AUs for a single patient.
"""

import pandas as pd
import numpy as np
import os
import logging
import re
from facial_au_constants import (
    ACTION_TO_AUS, ACTION_DESCRIPTIONS, ALL_AU_COLUMNS, 
    BASELINE_AU_ACTIVATIONS
)
from facial_au_paralysis_detector import FacialParalysisDetector
from facial_au_synkinesis_detector import FacialSynkinesisDetector
from facial_au_visualizer import FacialAUVisualizer
from facial_au_frame_extractor import FacialFrameExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialAUAnalyzer:
    """
    Analyzes facial Action Units from OpenFace output for a patient.
    """
    
    def __init__(self):
        """Initialize the facial AU analyzer."""
        self.action_to_aus = ACTION_TO_AUS
        self.action_descriptions = ACTION_DESCRIPTIONS
        self.results = {}
        self.patient_id = None
        self.left_data = None
        self.right_data = None
        self.frame_paths = {}  # Store paths to extracted frames
        
        # Store baseline neutral values for normalization
        self.baseline_values = {
            'left': {},
            'right': {}
        }
        
        # Initialize specialized components
        self.paralysis_detector = FacialParalysisDetector()
        self.synkinesis_detector = FacialSynkinesisDetector()
        self.visualizer = FacialAUVisualizer()
        self.frame_extractor = FacialFrameExtractor()
        
    def load_data(self, left_csv_path, right_csv_path):
        """
        Load left and right CSV files for a patient.
        
        Args:
            left_csv_path (str): Path to left side CSV file
            right_csv_path (str): Path to right side CSV file
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            self.left_data = pd.read_csv(left_csv_path)
            self.right_data = pd.read_csv(right_csv_path)
            
            # Clean up NaN values in the action column
            self.left_data['action'] = self.left_data['action'].fillna('Unknown')
            self.right_data['action'] = self.right_data['action'].fillna('Unknown')
            
            # Extract patient ID from filename - keep the full IMG_XXXX format
            filename = os.path.basename(left_csv_path)
            match = re.search(r'(IMG_\d+)', filename)
            if match:
                self.patient_id = match.group(1)  # Extract the full IMG_XXXX format
            else:
                # Fallback in case the pattern doesn't match
                self.patient_id = os.path.dirname(left_csv_path).split('_')[0]
            
            logger.info(f"Loaded data for patient {self.patient_id}")
            logger.info(f"Left data shape: {self.left_data.shape}")
            logger.info(f"Right data shape: {self.right_data.shape}")
            
            # Calculate baseline values from neutral expressions if available
            self._calculate_baseline_values()
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _calculate_baseline_values(self):
        """
        Calculate baseline AU values from neutral expressions.
        Uses frames marked as 'BL' (baseline) if available, otherwise uses 
        the first few frames, or falls back to predefined values.
        """
        # Initialize with default baseline values
        for au in ALL_AU_COLUMNS:
            self.baseline_values['left'][au] = BASELINE_AU_ACTIVATIONS.get(au, 0.1)
            self.baseline_values['right'][au] = BASELINE_AU_ACTIVATIONS.get(au, 0.1)
            
        # Try to get actual baseline from 'BL' marked frames
        try:
            # Filter for baseline action frames
            left_baseline = self.left_data[self.left_data['action'] == 'BL']
            right_baseline = self.right_data[self.right_data['action'] == 'BL']
            
            # If BL frames exist, use them for baseline
            if len(left_baseline) > 0 and len(right_baseline) > 0:
                # Calculate median values for each AU to minimize outlier effects
                for au in ALL_AU_COLUMNS:
                    if au in left_baseline.columns:
                        self.baseline_values['left'][au] = left_baseline[au].median()
                    if au in right_baseline.columns:
                        self.baseline_values['right'][au] = right_baseline[au].median()
                        
                logger.info("Calculated baseline values from 'BL' marked frames")
                return
            
            # If no BL frames, try using the first few frames as baseline
            if len(self.left_data) > 5 and len(self.right_data) > 5:
                # Use the first 5 frames
                left_baseline = self.left_data.iloc[:5]
                right_baseline = self.right_data.iloc[:5]
                
                # Calculate median values for each AU
                for au in ALL_AU_COLUMNS:
                    if au in left_baseline.columns:
                        self.baseline_values['left'][au] = left_baseline[au].median()
                    if au in right_baseline.columns:
                        self.baseline_values['right'][au] = right_baseline[au].median()
                        
                logger.info("Calculated baseline values from initial frames")
                return
                
            # Fall back to default values if unable to calculate from data
            logger.info("Using default baseline values")
            
        except Exception as e:
            logger.warning(f"Error calculating baseline values: {str(e)}. Using defaults.")
    
    def _normalize_au_value(self, side, au, value):
        """
        Normalize AU value by subtracting the baseline.
        Ensures values don't go below zero.
        
        Args:
            side (str): 'left' or 'right'
            au (str): Action Unit name (e.g., 'AU01_r')
            value (float): Original AU value
            
        Returns:
            float: Normalized AU value
        """
        baseline = self.baseline_values[side].get(au, BASELINE_AU_ACTIVATIONS.get(au, 0.1))
        normalized = max(0, value - baseline)
        return normalized
    
    def analyze_maximal_intensity(self):
        """
        Find maximal intensity for key AUs for each action, looking at both sides.
        For each action, find the one frame that demonstrates the maximal value 
        for the key AUs on either side.
        
        Returns:
            dict: Results dictionary with analysis results
        """
        if self.left_data is None or self.right_data is None:
            logger.error("Data not loaded. Please run load_data() first")
            return None
            
        self.results = {}
        
        # Get unique actions in the data, filtering out NaN values and 'Unknown' actions
        unique_actions = [action for action in pd.unique(self.left_data['action']) 
                         if isinstance(action, str) and action in self.action_to_aus and action != 'Unknown']
        
        for action in unique_actions:
            # Filter data for this action
            left_action_data = self.left_data[self.left_data['action'] == action].copy()  # Create explicit copy
            right_action_data = self.right_data[self.right_data['action'] == action].copy()  # Create explicit copy
            
            # Get key AUs for this action
            key_aus = self.action_to_aus[action]
            
            # Special handling for 'BL' (Baseline) action - since it has no key AUs, we'll use all AUs or a subset
            if action == 'BL' and (not key_aus or len(key_aus) == 0):
                # For baseline, consider representative AUs from each facial zone
                key_aus = ['AU01_r', 'AU07_r', 'AU12_r']  # One from each zone: brow, eye, mouth
                logger.info("Using representative AUs for Baseline action analysis")
            
            # Calculate maximal intensity for left side
            if len(key_aus) == 1:
                left_action_data.loc[:, 'key_au_avg'] = left_action_data[key_aus].values
            else:
                left_action_data.loc[:, 'key_au_avg'] = left_action_data[key_aus].mean(axis=1)
            
            left_max_value = left_action_data['key_au_avg'].max()
            left_max_idx = left_action_data['key_au_avg'].idxmax()
            left_max_frame = left_action_data.loc[left_max_idx, 'frame']
            
            # Calculate maximal intensity for right side
            if len(key_aus) == 1:
                right_action_data.loc[:, 'key_au_avg'] = right_action_data[key_aus].values
            else:
                right_action_data.loc[:, 'key_au_avg'] = right_action_data[key_aus].mean(axis=1)
            
            right_max_value = right_action_data['key_au_avg'].max()
            right_max_idx = right_action_data['key_au_avg'].idxmax()
            right_max_frame = right_action_data.loc[right_max_idx, 'frame']
            
            # Determine which side had the maximum value
            if left_max_value >= right_max_value:
                max_side = 'left'
                max_frame = left_max_frame
                max_value = left_max_value
                max_idx = left_max_idx
            else:
                max_side = 'right'
                max_frame = right_max_frame
                max_value = right_max_value
                max_idx = right_max_idx
            
            # Find the corresponding frame in the other side
            if max_side == 'left':
                matching_right_idx = right_action_data[right_action_data['frame'] == max_frame].index
                if len(matching_right_idx) > 0:
                    matching_right_idx = matching_right_idx[0]
                else:
                    # If exact frame not found, find closest
                    right_frames = right_action_data['frame'].values
                    closest_frame_idx = np.abs(right_frames - max_frame).argmin()
                    matching_right_idx = right_action_data.iloc[closest_frame_idx].name
            else:
                matching_left_idx = left_action_data[left_action_data['frame'] == max_frame].index
                if len(matching_left_idx) > 0:
                    matching_left_idx = matching_left_idx[0]
                else:
                    # If exact frame not found, find closest
                    left_frames = left_action_data['frame'].values
                    closest_frame_idx = np.abs(left_frames - max_frame).argmin()
                    matching_left_idx = left_action_data.iloc[closest_frame_idx].name
            
            # Store results with all AU values from both sides at the max frame
            self.results[action] = {
                'max_side': max_side,
                'max_frame': max_frame,
                'max_value': max_value,
                'left': {
                    'idx': left_max_idx if max_side == 'left' else matching_left_idx,
                    'frame': max_frame,
                    'au_values': {},
                    'normalized_au_values': {}
                },
                'right': {
                    'idx': right_max_idx if max_side == 'right' else matching_right_idx,
                    'frame': max_frame,
                    'au_values': {},
                    'normalized_au_values': {}
                },
                'paralysis': {
                    'detected': False,
                    'zones': {
                        'left': {'upper': 'None', 'mid': 'None', 'lower': 'None'},
                        'right': {'upper': 'None', 'mid': 'None', 'lower': 'None'}
                    },
                    'affected_aus': {'left': [], 'right': []}
                },
                'synkinesis': {
                    'detected': False,
                    'types': [],
                    'side_specific': {
                        synk_type: {'left': False, 'right': False} 
                        for synk_type in self.synkinesis_detector.SYNKINESIS_TYPES
                    },
                    'confidence': {
                        synk_type: {'left': 0, 'right': 0}
                        for synk_type in self.synkinesis_detector.SYNKINESIS_TYPES
                    }
                }
            }
            
            # Get all AU values for both sides at the max frame and normalize them
            for au in ALL_AU_COLUMNS:
                try:
                    left_idx = self.results[action]['left']['idx']
                    right_idx = self.results[action]['right']['idx']
                    
                    if au in self.left_data.columns and left_idx in self.left_data.index:
                        orig_value = self.left_data.loc[left_idx, au]
                        self.results[action]['left']['au_values'][au] = orig_value
                        # Store normalized value
                        self.results[action]['left']['normalized_au_values'][au] = self._normalize_au_value('left', au, orig_value)
                    else:
                        self.results[action]['left']['au_values'][au] = np.nan
                        self.results[action]['left']['normalized_au_values'][au] = np.nan
                        
                    if au in self.right_data.columns and right_idx in self.right_data.index:
                        orig_value = self.right_data.loc[right_idx, au]
                        self.results[action]['right']['au_values'][au] = orig_value
                        # Store normalized value
                        self.results[action]['right']['normalized_au_values'][au] = self._normalize_au_value('right', au, orig_value)
                    else:
                        self.results[action]['right']['au_values'][au] = np.nan
                        self.results[action]['right']['normalized_au_values'][au] = np.nan
                except Exception as e:
                    logger.error(f"Error getting AU values for {au}: {str(e)}")
                    self.results[action]['left']['au_values'][au] = np.nan
                    self.results[action]['right']['au_values'][au] = np.nan
                    self.results[action]['left']['normalized_au_values'][au] = np.nan
                    self.results[action]['right']['normalized_au_values'][au] = np.nan
        
        logger.info(f"Completed analysis for patient {self.patient_id}")
        # After collecting all AU values, analyze for paralysis and synkinesis
        self.detect_paralysis()
        self.detect_synkinesis()
        
        return self.results

    def extract_frames(self, video_path, output_dir):
        """
        Extract frames from video at points of maximal expression
        and save them as images with appropriate labels.
        Delegates to frame extractor component.

        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frames

        Returns:
            bool: True if frames extracted successfully
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return False

        # Store output directory for future use
        self.output_dir = output_dir

        # Use the frame extractor component
        success, frame_paths = self.frame_extractor.extract_frames(
            self, video_path, output_dir, self.patient_id, self.results, self.action_descriptions
        )

        if success:
            self.frame_paths = frame_paths

            # Create visualizations for each action
            for action, info in self.results.items():
                if action in self.frame_paths:
                    self.create_au_visualization(
                        info['left']['au_values'],
                        info['right']['au_values'],
                        info['left']['normalized_au_values'],
                        info['right']['normalized_au_values'],
                        action,
                        int(info['max_frame']),
                        os.path.join(output_dir, self.patient_id)
                    )

            # Create symmetry visualization
            self.create_symmetry_visualization(output_dir)
            
            # Clean up extracted frames after all visualizations are complete
            self.cleanup_extracted_frames()

        return success
    
    def cleanup_extracted_frames(self):
        """
        Clean up extracted frame images after visualizations are created.
        Only removes the .jpg and .png frame images, not the visualization outputs.
        """
        if not hasattr(self, 'frame_paths') or not self.frame_paths:
            logger.info("No frame paths to clean up")
            return
            
        logger.info(f"Cleaning up extracted frames for patient {self.patient_id}")
        frames_removed = 0
        
        # Loop through the frame paths dictionary
        for action, frame_path in self.frame_paths.items():
            # Delete the original frame PNG
            if os.path.exists(frame_path):
                os.remove(frame_path)
                frames_removed += 1
                
            # Also delete the labeled frame JPG (has a different path)
            labeled_frame_path = frame_path.replace("_original.png", ".jpg")
            if os.path.exists(labeled_frame_path):
                os.remove(labeled_frame_path)
                frames_removed += 1
        
        logger.info(f"Removed {frames_removed} extracted frame images")
    
    def create_au_visualization(self, au_values_left, au_values_right, norm_au_values_left, 
                                norm_au_values_right, action, frame_num, output_dir):
        """
        Create visualization of AU values. Delegates to visualizer component.
        
        Args:
            au_values_left (dict): Left side AU values
            au_values_right (dict): Right side AU values
            norm_au_values_left (dict): Left side normalized AU values
            norm_au_values_right (dict): Right side normalized AU values
            action (str): Action code
            frame_num (int): Frame number
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        return self.visualizer.create_au_visualization(
            self, au_values_left, au_values_right, 
            norm_au_values_left, norm_au_values_right,
            action, frame_num, output_dir,
            self.frame_paths, self.action_descriptions,
            self.action_to_aus, self.results
        )
    
    def create_symmetry_visualization(self, output_dir):
        """
        Create visualization of facial symmetry. Delegates to visualizer component.
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return None
        
        return self.visualizer.create_symmetry_visualization(
            self, output_dir, self.patient_id, self.results,
            self.action_descriptions
        )

    def create_patient_dashboard(self):
        """
        Create a comprehensive dashboard visualization for the patient.
        Shows paralysis and synkinesis findings across all actions in a single view.

        Returns:
            str: Path to saved patient dashboard visualization
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return None

        patient_output_dir = os.path.join(self.output_dir, self.patient_id) if hasattr(self,
                                                                                       'output_dir') else os.path.join(
            ".", self.patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        # Create the dashboard using the visualizer
        output_path = self.visualizer.create_patient_dashboard(
            self, patient_output_dir, self.patient_id, self.results, self.action_descriptions
        )

        logger.info(f"Created patient dashboard visualization: {output_path}")
        return output_path
    
    def detect_paralysis(self):
        """
        Detect potential facial paralysis. Delegates to paralysis detector component.
        
        Returns:
            None: Updates self.results in place
        """
        if not self.results:
            logger.warning("No results to analyze for paralysis detection")
            return
        
        self.paralysis_detector.detect_paralysis(self.results)
    
    def detect_synkinesis(self):
        """
        Detect potential synkinesis. Delegates to synkinesis detector component.
        
        Returns:
            None: Updates self.results in place
        """
        if not self.results:
            logger.warning("No results to analyze for synkinesis detection")
            return
        
        self.synkinesis_detector.detect_synkinesis(self.results)

    def generate_summary_data(self):
        """
        Generate summary data for the patient analysis.
        Creates a single dictionary with all patient data, including
        paralysis and synkinesis information at the beginning followed
        by action-specific data with prefixed column names.

        Returns:
            dict: Dictionary with patient summary data
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return {}

        # Initialize summary information for paralysis and synkinesis across all actions
        patient_summary = {
            'Patient ID': self.patient_id,
            'Left Upper Face Paralysis': 'None',
            'Left Mid Face Paralysis': 'None',  # Explicitly set as string
            'Left Lower Face Paralysis': 'None',
            'Right Upper Face Paralysis': 'None',
            'Right Mid Face Paralysis': 'None',  # Explicitly set as string
            'Right Lower Face Paralysis': 'None',
            'Ocular-Oral Left': 'No',
            'Ocular-Oral Right': 'No',
            'Oral-Ocular Left': 'No',
            'Oral-Ocular Right': 'No',
            'Snarl-Smile Left': 'No',
            'Snarl-Smile Right': 'No',
            'Paralysis Detected': 'No',
            'Synkinesis Detected': 'No'
        }

        # Process each action
        for action, info in self.results.items():
            # Skip BL, WN, and PL actions
            if action in ['BL', 'WN', 'PL']:
                continue

            # Create action-specific data with prefixed column names
            action_data = {
                f'{action}_Max Side': info['max_side'],
                f'{action}_Max Frame': info['max_frame'],
                f'{action}_Max Value': info['max_value'],
            }

            # Add all AU values for both sides with action prefix
            for au in ALL_AU_COLUMNS:
                if au in info['left']['au_values']:
                    action_data[f'{action}_Left {au}'] = info['left']['au_values'][au]
                else:
                    action_data[f'{action}_Left {au}'] = np.nan

                if au in info['right']['au_values']:
                    action_data[f'{action}_Right {au}'] = info['right']['au_values'][au]
                else:
                    action_data[f'{action}_Right {au}'] = np.nan

                # Add normalized values too
                if au in info['left']['normalized_au_values']:
                    action_data[f'{action}_Left {au} (Normalized)'] = info['left']['normalized_au_values'][au]
                else:
                    action_data[f'{action}_Left {au} (Normalized)'] = np.nan

                if au in info['right']['normalized_au_values']:
                    action_data[f'{action}_Right {au} (Normalized)'] = info['right']['normalized_au_values'][au]
                else:
                    action_data[f'{action}_Right {au} (Normalized)'] = np.nan

            # Update patient_summary with action data
            patient_summary.update(action_data)

            # Update patient-level paralysis information
            if info['paralysis']['detected']:
                patient_summary['Paralysis Detected'] = 'Yes'

                # Update zone-specific paralysis information for both sides
                for side in ['left', 'right']:
                    for zone in ['upper', 'mid', 'lower']:
                        zone_severity = info['paralysis']['zones'][side][zone]

                        # CRITICAL FIX: Ensure severity is a string, not a number
                        if not isinstance(zone_severity, str):
                            if zone_severity is None or np.isnan(zone_severity):
                                zone_severity = 'None'
                            else:
                                zone_severity = str(zone_severity)

                        current_severity = patient_summary[f'{side.capitalize()} {zone.capitalize()} Face Paralysis']

                        # Use the most severe rating across all actions
                        if (zone_severity == 'Complete' or
                                (zone_severity == 'Partial' and current_severity == 'None')):
                            patient_summary[f'{side.capitalize()} {zone.capitalize()} Face Paralysis'] = zone_severity

            # Update patient-level synkinesis information
            if info['synkinesis']['detected']:
                patient_summary['Synkinesis Detected'] = 'Yes'

                # Update type/side-specific synkinesis information
                for synk_type in self.synkinesis_detector.SYNKINESIS_TYPES:
                    for side in ['left', 'right']:
                        if info['synkinesis']['side_specific'][synk_type][side]:
                            # Add basic detection
                            patient_summary[f'{synk_type} {side.capitalize()}'] = 'Yes'

                            # Add confidence score if available
                            if 'confidence' in info['synkinesis'] and synk_type in info['synkinesis']['confidence']:
                                if side in info['synkinesis']['confidence'][synk_type]:
                                    confidence = info['synkinesis']['confidence'][synk_type][side]
                                    patient_summary[f'{synk_type} {side.capitalize()} Confidence'] = confidence

        # Final check to make sure all paralysis values are strings
        for key in list(patient_summary.keys()):
            if "Paralysis" in key and not isinstance(patient_summary[key], str):
                # Convert any non-string paralysis values to strings
                if patient_summary[key] is None or (
                        isinstance(patient_summary[key], float) and np.isnan(patient_summary[key])):
                    patient_summary[key] = 'None'
                else:
                    patient_summary[key] = str(patient_summary[key])

        return patient_summary
