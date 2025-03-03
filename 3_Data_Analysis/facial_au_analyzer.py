"""
Core Facial Action Unit Analyzer class.
Handles analysis of facial AUs for a single patient.
"""

import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import logging
import re
from facial_au_constants import (
    ACTION_TO_AUS, ACTION_DESCRIPTIONS, ALL_AU_COLUMNS, 
    PARALYSIS_THRESHOLD, MINIMAL_MOVEMENT_THRESHOLD, SYNKINESIS_PATTERNS,
    AU_NAMES
)

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
                self.patient_id = os.path.basename(left_csv_path).split('_')[0]
            
            logger.info(f"Loaded data for patient {self.patient_id}")
            logger.info(f"Left data shape: {self.left_data.shape}")
            logger.info(f"Right data shape: {self.right_data.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
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
                    'au_values': {}
                },
                'right': {
                    'idx': right_max_idx if max_side == 'right' else matching_right_idx,
                    'frame': max_frame,
                    'au_values': {}
                },
                'paralysis': {
                    'detected': False,
                    'side': None,
                    'affected_aus': []
                },
                'synkinesis': {
                    'detected': False,
                    'types': [],  # Changed from single 'type' to list of 'types'
                    'details': {}  # Changed from string to dictionary keyed by synkinesis type
                }
            }
            
            # Get all AU values for both sides at the max frame
            for au in ALL_AU_COLUMNS:
                try:
                    left_idx = self.results[action]['left']['idx']
                    right_idx = self.results[action]['right']['idx']
                    
                    if au in self.left_data.columns and left_idx in self.left_data.index:
                        self.results[action]['left']['au_values'][au] = self.left_data.loc[left_idx, au]
                    else:
                        self.results[action]['left']['au_values'][au] = np.nan
                        
                    if au in self.right_data.columns and right_idx in self.right_data.index:
                        self.results[action]['right']['au_values'][au] = self.right_data.loc[right_idx, au]
                    else:
                        self.results[action]['right']['au_values'][au] = np.nan
                except Exception as e:
                    logger.error(f"Error getting AU values for {au}: {str(e)}")
                    self.results[action]['left']['au_values'][au] = np.nan
                    self.results[action]['right']['au_values'][au] = np.nan
        
        logger.info(f"Completed analysis for patient {self.patient_id}")
        # After collecting all AU values, analyze for paralysis and synkinesis
        self.detect_paralysis()
        self.detect_synkinesis()
        
        return self.results
    
    def extract_frames(self, video_path, output_dir):
        """
        Extract frames from video at points of maximal expression
        and save them as images with appropriate labels.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frames
            
        Returns:
            bool: True if frames extracted successfully
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return False
        
        # Create output directory if it doesn't exist
        patient_output_dir = os.path.join(output_dir, self.patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video FPS: {fps}")
        logger.info(f"Total frames: {frame_count}")
        
        # Extract frames for each action
        for action, info in self.results.items():
            frame_num = int(info['max_frame'])
            
            # Set video to the right frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Could not read frame {frame_num}")
                continue
            
            # Create image label with patient ID, action, and frame number
            action_desc = self.action_descriptions.get(action, action)
            label = f"{self.patient_id}_{action_desc}_frame{frame_num}"
            
            # Add label text to the image
            cv2.putText(
                frame, 
                label, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Save the image
            output_path = os.path.join(patient_output_dir, f"{label}.jpg")
            cv2.imwrite(output_path, frame)
            logger.info(f"Saved {output_path}")
            
            # Create combined visualization for both sides
            self.create_au_visualization(
                info['left']['au_values'],
                info['right']['au_values'],
                action,
                frame_num,
                patient_output_dir
            )
        
        # Release the video capture
        cap.release()
        return True
    
    def create_au_visualization(self, au_values_left, au_values_right, action, frame_num, output_dir):
        """
        Create a bar chart visualization of the AU values for both left and right sides.
        
        Args:
            au_values_left (dict): Dictionary of left side AU values
            au_values_right (dict): Dictionary of right side AU values
            action (str): Action code
            frame_num (int): Frame number
            output_dir (str): Directory to save visualization
            
        Returns:
            str: Path to saved visualization
        """
        plt.figure(figsize=(16, 10))
        
        # Combine AU data from both sides
        all_aus = set(au_values_left.keys()) | set(au_values_right.keys())
        
        # Filter out AU_c values (only keep AU_r values)
        all_aus = [au for au in all_aus if au.endswith('_r')]
        
        # Filter out AUs with very low values on both sides for clearer visualization
        significant_aus = [au for au in all_aus 
                          if (au_values_left.get(au, 0) > 0.01 or au_values_right.get(au, 0) > 0.01)]
        
        # Sort AUs by name for consistent ordering
        significant_aus = sorted(significant_aus)
        
        if not significant_aus:
            logger.warning(f"No significant AU values for {action}")
            plt.close()
            return None
        
        # Prepare data for plotting
        x = np.arange(len(significant_aus))
        width = 0.35
        
        # Get values for each side
        left_values = [au_values_left.get(au, 0) for au in significant_aus]
        right_values = [au_values_right.get(au, 0) for au in significant_aus]
        
        # Create bar chart with both sides
        bars1 = plt.bar(x - width/2, left_values, width, label='Left Side', color='skyblue')
        bars2 = plt.bar(x + width/2, right_values, width, label='Right Side', color='lightcoral')
        
        # Add value labels to the bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # Only add label if value is significant
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        # Add title and labels
        action_desc = self.action_descriptions.get(action, action)
        plt.title(f"Action Unit Values - {self.patient_id} - {action_desc} (Frame {frame_num})", fontsize=14)
        plt.xlabel("Action Units", fontsize=12)
        plt.ylabel("Intensity", fontsize=12)
        
        # Create x-tick labels with both code and descriptive name
        tick_labels = []
        for au in significant_aus:
            au_name = AU_NAMES.get(au, au)
            # Format the label to show code and name
            tick_labels.append(f"{au}\n{au_name}")
        
        plt.xticks(x, tick_labels, rotation=45, fontsize=9, ha='right')
        plt.legend(fontsize=10)
        
        # Add gridlines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Highlight key AUs for this action
        key_aus = self.action_to_aus[action]
        for i, au in enumerate(significant_aus):
            if au in key_aus:
                # Add highlight box behind key AUs
                plt.axvspan(i-width/1.5, i+width/1.5, alpha=0.2, color='yellow')
                # Add key AU label
                plt.text(i, -0.05, "Key AU", ha='center', va='top', 
                       rotation=45, fontsize=8, color='red')
        
        # Add paralysis info if detected
        if action in self.results and self.results[action]['paralysis']['detected']:
            paralyzed_side = self.results[action]['paralysis']['side']
            plt.figtext(0.5, 0.01, f"Potential {paralyzed_side} side paralysis detected", 
                      ha='center', fontsize=10, color='red',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Add synkinesis info if detected - with side information
        if action in self.results and self.results[action]['synkinesis']['detected']:
            synk_types = self.results[action]['synkinesis']['types']
            if synk_types:
                # Extract side information from details
                synk_sides = {"left": [], "right": []}
                for synk_type, details_list in self.results[action]['synkinesis']['details'].items():
                    for detail in details_list:
                        if "left side" in detail.lower():
                            synk_sides["left"].append(synk_type)
                        if "right side" in detail.lower():
                            synk_sides["right"].append(synk_type)
                
                # Create text with side information
                side_texts = []
                if synk_sides["left"]:
                    side_texts.append(f"LEFT: {', '.join(synk_sides['left'])}")
                if synk_sides["right"]:
                    side_texts.append(f"RIGHT: {', '.join(synk_sides['right'])}")
                
                synk_text = f"Potential synkinesis detected: {' | '.join(side_texts)}"
                plt.figtext(0.5, 0.04, synk_text, 
                          ha='center', fontsize=10, color='blue',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust layout to make room for annotations
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"{self.patient_id}_{action_desc}_frame{frame_num}_AUs.png")
        plt.savefig(output_path, dpi=150)  # Higher DPI for better quality
        plt.close()
        logger.info(f"Saved combined AU visualization: {output_path}")
        return output_path
    
    def generate_summary_data(self):
        """
        Generate summary data for the patient analysis.
        This is for inclusion in a combined CSV report.
        
        Returns:
            list: List of dictionaries with summary data for each action
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return []
        
        summary_data = []
        
        for action, info in self.results.items():
            # Create data row - removed Description, Key AUs, Symmetry Ratio, Asymmetry Detected, and Synkinesis Details
            row_data = {
                'Patient ID': self.patient_id,
                'Action': action,
                'Max Side': info['max_side'],
                'Max Frame': info['max_frame'],
                'Max Value': info['max_value'],
                'Paralysis Detected': info['paralysis']['detected'],
                'Paralyzed Side': info['paralysis']['side'] if info['paralysis']['detected'] else 'None',
                'Affected AUs': ', '.join(info['paralysis']['affected_aus']) if info['paralysis']['affected_aus'] else 'None',
                'Synkinesis Detected': info['synkinesis']['detected'],
                'Synkinesis Types': ', '.join(info['synkinesis']['types']) if info['synkinesis']['types'] else 'None'
            }
            
            # Add all AU values for both sides
            for au in ALL_AU_COLUMNS:
                if au in info['left']['au_values']:
                    row_data[f'Left {au}'] = info['left']['au_values'][au]
                else:
                    row_data[f'Left {au}'] = np.nan
                    
                if au in info['right']['au_values']:
                    row_data[f'Right {au}'] = info['right']['au_values'][au]
                else:
                    row_data[f'Right {au}'] = np.nan
            
            summary_data.append(row_data)
        
        return summary_data
    
    def detect_paralysis(self):
        """
        Detect potential facial paralysis by analyzing asymmetry patterns.
        Identifies which side may be paralyzed and which AUs show significant asymmetry.
        
        This is added to the results dictionary for each action.
        """
        if not self.results:
            logger.warning("No results to analyze for paralysis detection")
            return
        
        # Count how many actions show asymmetry patterns consistent with paralysis
        left_side_minimal = 0
        right_side_minimal = 0
        
        # Track which AUs show asymmetry
        affected_aus = set()
        
        for action, info in self.results.items():
            # Get key AUs for this action
            key_aus = self.action_to_aus[action]
            
            for au in key_aus:
                if au in info['left']['au_values'] and au in info['right']['au_values']:
                    left_value = info['left']['au_values'][au]
                    right_value = info['right']['au_values'][au]
                    
                    # Check if either side has minimal movement
                    left_minimal = left_value < MINIMAL_MOVEMENT_THRESHOLD
                    right_minimal = right_value < MINIMAL_MOVEMENT_THRESHOLD
                    
                    # If one side moves and the other doesn't, possible paralysis
                    if left_minimal and not right_minimal:
                        left_side_minimal += 1
                        affected_aus.add(au)
                        info['paralysis']['affected_aus'].append(au)
                    elif right_minimal and not left_minimal:
                        right_side_minimal += 1
                        affected_aus.add(au)
                        info['paralysis']['affected_aus'].append(au)
                    
                    # Even if both sides move, check for significant asymmetry
                    if left_value > 0 and right_value > 0:
                        ratio = min(left_value, right_value) / max(left_value, right_value)
                        if ratio < PARALYSIS_THRESHOLD:
                            if left_value < right_value:
                                left_side_minimal += 1
                                affected_aus.add(au)
                                if au not in info['paralysis']['affected_aus']:
                                    info['paralysis']['affected_aus'].append(au)
                            else:
                                right_side_minimal += 1
                                affected_aus.add(au)
                                if au not in info['paralysis']['affected_aus']:
                                    info['paralysis']['affected_aus'].append(au)
        
        # Determine if paralysis is detected and which side
        paralysis_threshold_count = len(self.results) * 0.5  # At least 50% of actions show asymmetry
        
        if left_side_minimal > paralysis_threshold_count or right_side_minimal > paralysis_threshold_count:
            paralysis_side = 'left' if left_side_minimal > right_side_minimal else 'right'
            
            # Update all actions with the paralysis information
            for action in self.results:
                self.results[action]['paralysis']['detected'] = True
                self.results[action]['paralysis']['side'] = paralysis_side
            
            logger.info(f"Potential facial paralysis detected on {paralysis_side} side")
            logger.info(f"Affected AUs: {', '.join(affected_aus)}")
    
    def _detect_snarl_smile(self, action, info):
        """
        Special detection for snarl smile synkinesis.
        Uses individual AU thresholds instead of averages for more accurate detection.
        
        Args:
            action (str): The facial action being analyzed
            info (dict): Action data from results dictionary
        """
        # Only check for snarl smile during smile actions
        if action not in ['BS', 'SS']:
            return
        
        # Check both sides
        for side in ['left', 'right']:
            # Get smile activation value
            smile_value = info[side]['au_values'].get('AU12_r', 0)
            
            # Check if we have a significant smile
            if smile_value > 0.4:
                # Check each snarl component individually
                nose_wrinkle = info[side]['au_values'].get('AU09_r', 0) 
                upper_lip = info[side]['au_values'].get('AU10_r', 0)
                dimpler = info[side]['au_values'].get('AU14_r', 0)
                
                # If ANY of these are activated significantly, we have a potential snarl smile
                if nose_wrinkle > 0.3 or upper_lip > 0.3 or dimpler > 0.4:
                    # Create list of detected snarl components
                    snarl_components = []
                    if nose_wrinkle > 0.3:
                        snarl_components.append('AU09_r (Nose Wrinkler)')
                    if upper_lip > 0.3:
                        snarl_components.append('AU10_r (Upper Lip Raiser)')
                    if dimpler > 0.4:
                        snarl_components.append('AU14_r (Dimpler)')
                    
                    # Add this synkinesis type if not already detected
                    if 'Snarl-Smile' not in info['synkinesis']['types']:
                        info['synkinesis']['detected'] = True
                        info['synkinesis']['types'].append('Snarl-Smile')
                        
                    # Create synkinesis detail message
                    detail_msg = (
                        f"Snarl-Smile on {side} side: AU12_r (Smile) triggers "
                        f"{', '.join(snarl_components)}"
                    )
                    
                    # Store details for this synkinesis type
                    if 'Snarl-Smile' not in info['synkinesis']['details']:
                        info['synkinesis']['details']['Snarl-Smile'] = []
                    info['synkinesis']['details']['Snarl-Smile'].append(detail_msg)
                    
                    logger.info(f"Potential Snarl-Smile synkinesis detected on {side} side")
    
    def detect_synkinesis(self):
        """
        Detect potential synkinesis (unwanted co-activation of muscles).
        Detects all possible types of synkinesis patterns and accumulates them rather than overwriting.
        This allows multiple types of synkinesis to be detected for the same action.
        
        This is added to the results dictionary for each action.
        """
        if not self.results:
            logger.warning("No results to analyze for synkinesis detection")
            return
        
        # First call specialized detection functions
        for action, info in self.results.items():
            # Detect Snarl Smile using specialized detection
            self._detect_snarl_smile(action, info)
        
        # Always check both sides for synkinesis
        sides_to_check = ['left', 'right']
        
        # Look for synkinesis in each relevant action
        for action, info in self.results.items():
            # Get key AUs for this action
            key_aus = self.action_to_aus[action]
            
            # For each defined synkinesis pattern (except Snarl-Smile which was handled specially)
            for synk_name, synk_pattern in SYNKINESIS_PATTERNS.items():
                # Skip Snarl-Smile as it was handled by specialized function
                if synk_name == 'Snarl-Smile':
                    continue
                    
                trigger_aus = synk_pattern['trigger_aus']
                coupled_aus = synk_pattern['coupled_aus']
                
                # Use different thresholds for Oral-Ocular during smile actions
                detection_threshold = 0.3
                if synk_name == 'Oral-Ocular' and action in ['BS', 'SS']:
                    detection_threshold = 0.25
                
                # Check if this action involves trigger AUs
                if any(au in key_aus for au in trigger_aus):
                    # Check both sides
                    for side in sides_to_check:
                        # Calculate average activation of trigger AUs
                        side_trigger_values = [
                            info[side]['au_values'].get(au, 0) 
                            for au in trigger_aus 
                            if au in info[side]['au_values'] and au in key_aus
                        ]
                        
                        # Get coupled AU activations (unwanted responses)
                        side_coupled_values = [
                            info[side]['au_values'].get(au, 0) 
                            for au in coupled_aus 
                            if au in info[side]['au_values']
                        ]
                        
                        if side_trigger_values and side_coupled_values:
                            avg_trigger = sum(side_trigger_values) / len(side_trigger_values)
                            avg_coupled = sum(side_coupled_values) / len(side_coupled_values)
                            
                            # If there's significant trigger activation and unwanted coupled activation
                            if avg_trigger > detection_threshold and avg_coupled > detection_threshold:
                                ratio = avg_coupled / avg_trigger if avg_trigger > 0 else 0
                                
                                # If coupled activation is proportional to trigger, likely synkinesis
                                if 0.3 < ratio < 1.5:
                                    # Add this synkinesis type if not already detected
                                    if synk_name not in info['synkinesis']['types']:
                                        info['synkinesis']['detected'] = True
                                        info['synkinesis']['types'].append(synk_name)
                                        
                                    # Create synkinesis detail message
                                    detail_msg = (
                                        f"{synk_name} on {side} side: {', '.join([au for au in trigger_aus if au in key_aus])} "
                                        f"triggers {', '.join(coupled_aus)} (ratio: {ratio:.2f})"
                                    )
                                    
                                    # Store details for this synkinesis type
                                    if synk_name not in info['synkinesis']['details']:
                                        info['synkinesis']['details'][synk_name] = []
                                    info['synkinesis']['details'][synk_name].append(detail_msg)
                                    
                                    logger.info(f"Potential {synk_name} synkinesis detected on {side} side")
    
    def create_symmetry_visualization(self, output_dir):
        """
        Create a visualization of the facial symmetry.
        
        Args:
            output_dir (str): Directory to save visualization
            
        Returns:
            str: Path to saved visualization
        """
        if not hasattr(self, 'results') or not self.results:
            logger.error("No analysis results. Please run analyze_maximal_intensity() first")
            return None
        
        patient_output_dir = os.path.join(output_dir, self.patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # Prepare data for visualization
        actions = []
        left_values = []
        right_values = []
        
        for action, info in self.results.items():
            action_desc = self.action_descriptions.get(action, action)
            actions.append(action_desc)
            
            # Calculate average of key AUs
            key_aus = self.action_to_aus[action]
            
            left_key_values = [info['left']['au_values'].get(au, 0) for au in key_aus]
            right_key_values = [info['right']['au_values'].get(au, 0) for au in key_aus]
            
            left_avg = np.mean(left_key_values) if left_key_values else 0
            right_avg = np.mean(right_key_values) if right_key_values else 0
            
            left_values.append(left_avg)
            right_values.append(right_avg)
        
        # Create bar chart
        x = np.arange(len(actions))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, left_values, width, label='Left')
        bars2 = plt.bar(x + width/2, right_values, width, label='Right')
        
        # Highlight bars for paralyzed side if paralysis is detected
        paralysis_detected = any(info['paralysis']['detected'] for info in self.results.values())
        if paralysis_detected:
            paralyzed_side = next((info['paralysis']['side'] for info in self.results.values() 
                                  if info['paralysis']['detected']), None)
            
            if paralyzed_side == 'left':
                for bar in bars1:
                    bar.set_color('red')
                plt.title(f'Left vs Right Facial Movement Intensity - Patient {self.patient_id} (Left side paralysis detected)')
            elif paralyzed_side == 'right':
                for bar in bars2:
                    bar.set_color('red')
                plt.title(f'Left vs Right Facial Movement Intensity - Patient {self.patient_id} (Right side paralysis detected)')
        else:
            plt.title(f'Left vs Right Facial Movement Intensity - Patient {self.patient_id}')
        
        plt.xlabel('Actions')
        plt.ylabel('Key AU Intensity')
        plt.xticks(x, actions, rotation=45, ha='right')
        plt.legend()
        
        # Add synkinesis info if detected
        synkinesis_detected = any(info['synkinesis']['detected'] for info in self.results.values())
        if synkinesis_detected:
            # Collect all unique synkinesis types across all actions
            all_synk_types = set()
            for info in self.results.values():
                if info['synkinesis']['detected']:
                    all_synk_types.update(info['synkinesis']['types'])
            
            if all_synk_types:
                synk_text = f"Synkinesis detected: {', '.join(all_synk_types)}"
                plt.figtext(0.5, 0.01, synk_text, ha='center', fontsize=10, 
                           bbox=dict(facecolor='yellow', alpha=0.2))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(patient_output_dir, f"{self.patient_id}_symmetry_chart.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved symmetry visualization: {output_path}")
        return output_path
