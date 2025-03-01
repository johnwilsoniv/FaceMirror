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
            
            # Extract patient ID from filename
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
        
        # Get unique actions in the data, filtering out NaN values
        unique_actions = [action for action in pd.unique(self.left_data['action']) 
                         if isinstance(action, str) and action in self.action_to_aus]
        
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
                    'type': None,
                    'details': ""
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
        
        # Add synkinesis info if detected
        if action in self.results and self.results[action]['synkinesis']['detected']:
            synk_type = self.results[action]['synkinesis']['type']
            plt.figtext(0.5, 0.04, f"Potential {synk_type} synkinesis detected", 
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
            action_desc = self.action_descriptions.get(action, action)
            key_aus = ', '.join(self.action_to_aus[action])
            
            # Calculate key AU values on both sides
            left_key_au_values = {k: v for k, v in info['left']['au_values'].items() 
                                if k in self.action_to_aus[action]}
            right_key_au_values = {k: v for k, v in info['right']['au_values'].items() 
                                  if k in self.action_to_aus[action]}
            
            left_max = np.mean(list(left_key_au_values.values())) if left_key_au_values else 0
            right_max = np.mean(list(right_key_au_values.values())) if right_key_au_values else 0
            
            # Calculate symmetry ratio
            if right_max != 0 and not np.isnan(right_max):
                symmetry_ratio = left_max / right_max
            else:
                symmetry_ratio = np.nan
                
            # Determine if there's asymmetry (this is a simple threshold, may need adjustment)
            asymmetry = abs(1 - symmetry_ratio) > 0.2 if not np.isnan(symmetry_ratio) else False
            
            # Create data row
            row_data = {
                'Patient ID': self.patient_id,
                'Action': action,
                'Description': action_desc,
                'Key AUs': key_aus,
                'Max Side': info['max_side'],
                'Max Frame': info['max_frame'],
                'Max Value': info['max_value'],
                'Symmetry Ratio': symmetry_ratio,
                'Asymmetry Detected': asymmetry,
                'Paralysis Detected': info['paralysis']['detected'],
                'Paralyzed Side': info['paralysis']['side'] if info['paralysis']['detected'] else 'None',
                'Affected AUs': ', '.join(info['paralysis']['affected_aus']) if info['paralysis']['affected_aus'] else 'None',
                'Synkinesis Detected': info['synkinesis']['detected'],
                'Synkinesis Type': info['synkinesis']['type'] if info['synkinesis']['detected'] else 'None',
                'Synkinesis Details': info['synkinesis']['details'] if info['synkinesis']['detected'] else 'None'
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
    
    def detect_synkinesis(self):
        """
        Detect potential synkinesis (unwanted co-activation of muscles).
        Focuses on ocular-oral and oral-ocular synkinesis patterns.
        
        This is added to the results dictionary for each action.
        """
        if not self.results:
            logger.warning("No results to analyze for synkinesis detection")
            return
        
        # Look for synkinesis in each relevant action
        for action, info in self.results.items():
            # Only check actions involving eye closure or mouth movements
            key_aus = self.action_to_aus[action]
            
            # For each defined synkinesis pattern
            for synk_name, synk_pattern in SYNKINESIS_PATTERNS.items():
                trigger_aus = synk_pattern['trigger_aus']
                coupled_aus = synk_pattern['coupled_aus']
                
                # Check if this action involves trigger AUs
                if any(au in key_aus for au in trigger_aus):
                    # Calculate average activation of trigger AUs and coupled AUs
                    trigger_activations = []
                    coupled_activations = []
                    
                    # Check both sides independently
                    for side in ['left', 'right']:
                        # Get trigger AU activations
                        side_trigger_values = [
                            info[side]['au_values'].get(au, 0) 
                            for au in trigger_aus 
                            if au in info[side]['au_values']
                        ]
                        
                        # Get coupled AU activations
                        side_coupled_values = [
                            info[side]['au_values'].get(au, 0) 
                            for au in coupled_aus 
                            if au in info[side]['au_values']
                        ]
                        
                        if side_trigger_values and side_coupled_values:
                            avg_trigger = sum(side_trigger_values) / len(side_trigger_values)
                            avg_coupled = sum(side_coupled_values) / len(side_coupled_values)
                            
                            trigger_activations.append((side, avg_trigger))
                            coupled_activations.append((side, avg_coupled))
                    
                    # If we have data for both sides
                    if trigger_activations and coupled_activations:
                        # Sort by trigger activation (highest first)
                        trigger_activations.sort(key=lambda x: x[1], reverse=True)
                        
                        # Check if there's unwanted co-activation on either side
                        for side, trigger_value in trigger_activations:
                            if trigger_value > 0.3:  # Significant trigger activation
                                # Find corresponding coupled activation
                                coupled_value = next((val for s, val in coupled_activations if s == side), 0)
                                
                                # If coupled activation is also significant, might be synkinesis
                                if coupled_value > 0.3:
                                    ratio = coupled_value / trigger_value if trigger_value > 0 else 0
                                    
                                    # If coupled activation is proportional to trigger, likely synkinesis
                                    if 0.3 < ratio < 1.5:
                                        info['synkinesis']['detected'] = True
                                        info['synkinesis']['type'] = synk_name
                                        info['synkinesis']['details'] += (
                                            f"{synk_name} on {side} side: {', '.join(trigger_aus)} "
                                            f"triggers {', '.join(coupled_aus)} (ratio: {ratio:.2f}). "
                                        )
                                        
                                        logger.info(f"Potential {synk_name} synkinesis detected on {side} side")
                                        break
    
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
            synk_types = set()
            for info in self.results.values():
                if info['synkinesis']['detected'] and info['synkinesis']['type']:
                    synk_types.add(info['synkinesis']['type'])
            
            if synk_types:
                synk_text = f"Synkinesis detected: {', '.join(synk_types)}"
                plt.figtext(0.5, 0.01, synk_text, ha='center', fontsize=10, 
                           bbox=dict(facecolor='yellow', alpha=0.2))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(patient_output_dir, f"{self.patient_id}_symmetry_chart.png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved symmetry visualization: {output_path}")
        return output_path
