"""
Batch processor for analyzing multiple patients' facial AU data.
"""

import os

import numpy as np
import pandas as pd
import logging
import glob
from facial_au_analyzer import FacialAUAnalyzer
from facial_au_constants import SUMMARY_COLUMNS, SYNKINESIS_TYPES, PARALYSIS_SEVERITY_LEVELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialAUBatchProcessor:
    """
    Process multiple patients' facial AU data in batch mode.
    """
    
    def __init__(self, output_dir="../3.5_Results"):
        """
        Initialize the batch processor.
        
        Args:
            output_dir (str): Directory to save output files
        """
        self.output_dir = output_dir
        self.patients = []
        self.summary_data = []
        os.makedirs(output_dir, exist_ok=True)
        
    def add_patient(self, left_csv, right_csv, video_path=None):
        """
        Add a patient to the batch.
        
        Args:
            left_csv (str): Path to left side CSV file
            right_csv (str): Path to right side CSV file
            video_path (str, optional): Path to video file
            
        Returns:
            bool: True if patient added successfully
        """
        self.patients.append({
            'left_csv': left_csv,
            'right_csv': right_csv,
            'video_path': video_path,
            'patient_id': os.path.basename(left_csv).split('_')[0]
        })
        return True
    
    def auto_detect_patients(self, data_dir):
        """
        Auto-detect patient files in a directory based on naming conventions.
        Looks for pairs of files with _left and _right in their names.
        
        Args:
            data_dir (str): Directory containing patient data files
            
        Returns:
            int: Number of patients detected
        """
        # Get all CSV files
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        # Find left files
        left_files = [f for f in csv_files if "_left" in f.lower()]
        
        patients_added = 0
        
        for left_file in left_files:
            # Extract base name without _left
            base_name = os.path.basename(left_file).split("_left")[0]
            
            # Look for matching right file
            right_pattern = os.path.join(data_dir, f"{base_name}_right*.csv")
            right_files = glob.glob(right_pattern)
            
            if right_files:
                right_file = right_files[0]
                
                # Look for matching video file with pattern IMG_XXXX_rotated_annotated.mp4
                video_pattern = os.path.join(data_dir, f"{base_name}_rotated_annotated.mp4")
                video_files = glob.glob(video_pattern)
                
                # If not found, try other video formats with the same pattern
                if not video_files:
                    video_pattern = os.path.join(data_dir, f"{base_name}_rotated_annotated.avi")
                    video_files = glob.glob(video_pattern)
                    
                if not video_files:
                    video_pattern = os.path.join(data_dir, f"{base_name}_rotated_annotated.mov")
                    video_files = glob.glob(video_pattern)
                    
                if not video_files:
                    video_pattern = os.path.join(data_dir, f"{base_name}_rotated_annotated.MOV")
                    video_files = glob.glob(video_pattern)
                
                # If still not found, try a more general pattern
                if not video_files:
                    video_pattern = os.path.join(data_dir, f"{base_name}*.mp4")
                    video_files = glob.glob(video_pattern)
                
                video_path = video_files[0] if video_files else None
                
                # Add patient to batch
                self.add_patient(left_file, right_file, video_path)
                patients_added += 1
                logger.info(f"Detected patient: {base_name} (CSV: {os.path.basename(left_file)}, {os.path.basename(right_file)})")
        
        return patients_added

    def process_all(self, extract_frames=True):
        """
        Process all patients in the batch.
        Creates one row per patient with actions as column prefixes.
        Places paralysis/synkinesis summary at the beginning of each row.
        Sets NA for missing actions and filters out BL, WN, and PL actions.

        Args:
            extract_frames (bool): Whether to extract frames from videos

        Returns:
            str: Path to combined summary CSV file
        """
        if not self.patients:
            logger.error("No patients added to batch")
            return None

        all_patient_data = []

        for i, patient in enumerate(self.patients):
            logger.info(f"Processing patient {i + 1}/{len(self.patients)}: {patient['patient_id']}")

            # Create analyzer for this patient
            analyzer = FacialAUAnalyzer()

            # Load data
            if not analyzer.load_data(patient['left_csv'], patient['right_csv']):
                logger.error(f"Failed to load data for patient {patient['patient_id']}")
                continue

            # Analyze data
            analyzer.analyze_maximal_intensity()

            # Extract frames if video is available and extraction is requested
            if extract_frames and patient['video_path']:
                analyzer.extract_frames(patient['video_path'], self.output_dir)
            else:
                # Even if we don't extract frames, create symmetry visualization
                analyzer.create_symmetry_visualization(self.output_dir)

            # Get summary data in new format (one row per patient)
            patient_summary = analyzer.generate_summary_data()

            # Ensure all paralysis and synkinesis fields are strings at this stage too
            for key in list(patient_summary.keys()):
                if ("Paralysis" in key or "Ocular" in key or "Smile" in key) and not isinstance(patient_summary[key],
                                                                                                str):
                    # Convert any non-string values to strings
                    if patient_summary[key] is None or (
                            isinstance(patient_summary[key], float) and np.isnan(patient_summary[key])):
                        patient_summary[key] = 'None'
                    else:
                        patient_summary[key] = str(patient_summary[key])

            all_patient_data.append(patient_summary)
            logger.info(f"Completed processing for patient {patient['patient_id']}")

        # Create combined summary CSV
        if all_patient_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_patient_data)

            # Define columns that should always be strings
            string_cols = [
                'Patient ID',
                'Left Upper Face Paralysis', 'Left Mid Face Paralysis', 'Left Lower Face Paralysis',
                'Right Upper Face Paralysis', 'Right Mid Face Paralysis', 'Right Lower Face Paralysis',
                'Ocular-Oral Left', 'Ocular-Oral Right',
                'Oral-Ocular Left', 'Oral-Ocular Right',
                'Snarl-Smile Left', 'Snarl-Smile Right',
                'Paralysis Detected', 'Synkinesis Detected'
            ]

            # First, ensure these columns exist and replace any NaN values
            for col in string_cols:
                if col in df.columns:
                    # Replace NaN with None string first
                    df[col] = df[col].fillna('None')

                    # Then convert to string type
                    df[col] = df[col].astype(str)

                    # Finally replace any 'nan' strings with 'None'
                    df[col] = df[col].replace({'nan': 'None', 'NaN': 'None', 'NAN': 'None'})

            # Log the column types to verify our fix worked
            logger.info("Column types after conversion:")
            for col in string_cols:
                if col in df.columns:
                    logger.info(f"{col}: {df[col].dtype}")

            # Reorder columns to follow the requested structure
            # First get the Patient ID and summary columns
            summary_cols = [
                'Patient ID',
                'Left Upper Face Paralysis', 'Left Mid Face Paralysis', 'Left Lower Face Paralysis',
                'Right Upper Face Paralysis', 'Right Mid Face Paralysis', 'Right Lower Face Paralysis',
                'Ocular-Oral Left', 'Ocular-Oral Right',
                'Oral-Ocular Left', 'Oral-Ocular Right',
                'Snarl-Smile Left', 'Snarl-Smile Right',
                'Paralysis Detected', 'Synkinesis Detected'
            ]

            # Then get the action-specific columns
            action_cols = [col for col in df.columns if col not in summary_cols]

            # Combine to get all columns in order
            ordered_cols = summary_cols + action_cols

            # Apply the column ordering if all columns exist
            available_cols = [col for col in ordered_cols if col in df.columns]
            df = df[available_cols]

            # Replace NaN with NA for missing values in other columns
            df = df.fillna('NA')
            
            # Force midface paralysis columns to be strings
            midface_cols = ['Left Mid Face Paralysis', 'Right Mid Face Paralysis']
            for col in midface_cols:
                if col in df.columns:
                    # Convert to string and replace NaN/None values
                    df[col] = df[col].astype(str)
                    df[col] = df[col].replace({'nan': 'None', 'NaN': 'None', 'NA': 'None'})
            
            # Log midface column types after the special handling
            logger.info("Midface column types after explicit conversion:")
            for col in midface_cols:
                if col in df.columns:
                    logger.info(f"{col}: {df[col].dtype}")

            # Save to CSV
            output_path = os.path.join(self.output_dir, "combined_results.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved combined results to {output_path}")

            self.summary_data = df
            return output_path

        return None
    
    def analyze_asymmetry_across_patients(self):
        """
        Analyze asymmetry patterns across all patients.
        Modified to work with enhanced zone-specific detection.
        
        Returns:
            None: Just saves the paralysis and synkinesis analysis
        """
        if not hasattr(self, 'summary_data') or self.summary_data.empty:
            logger.error("No summary data available. Please run process_all() first")
            return None
        
        # Perform paralysis and synkinesis analysis
        self.analyze_paralysis_and_synkinesis()
        
        return None
        
    def analyze_paralysis_and_synkinesis(self):
        """
        Analyze patterns of facial paralysis and synkinesis across all patients.
        Creates summary reports of these conditions with zone-specific information.
        """
        if not hasattr(self, 'summary_data') or self.summary_data.empty:
            logger.error("No summary data available for analysis")
            return
        
        if isinstance(self.summary_data, list):
            df = pd.DataFrame(self.summary_data)
        else:
            df = self.summary_data
        
        # Create separate summaries for paralysis and synkinesis
        
        # Paralysis summary - count total and by zone/side/severity
        paralysis_count = (df['Paralysis Detected'] == 'Yes').sum()
        
        if paralysis_count > 0:
            # Count zone-specific statistics
            zone_stats = {}
            for side in ['Left', 'Right']:
                for zone in ['Upper', 'Mid', 'Lower']:
                    col = f'{side} {zone} Face Paralysis'
                    
                    # Count by severity
                    for severity in ['Partial', 'Complete']:
                        count = (df[col] == severity).sum()
                        zone_stats[f'{side} {zone} Face {severity}'] = count
                        zone_stats[f'{side} {zone} Face {severity} Percentage'] = (count / paralysis_count) * 100
            
            paralysis_summary_stats = {
                'Total Patients': len(df),
                'Patients with Paralysis': paralysis_count,
                'Paralysis Percentage': (paralysis_count / len(df)) * 100,
                **zone_stats
            }
            
            # Save paralysis stats
            stats_df = pd.DataFrame([paralysis_summary_stats])
            stats_path = os.path.join(self.output_dir, "paralysis_statistics.csv")
            stats_df.to_csv(stats_path, index=False)
            
            logger.info(f"Saved paralysis statistics to {stats_path}")
        
        # Synkinesis summary - count by type and side
        synkinesis_count = (df['Synkinesis Detected'] == 'Yes').sum()
        
        if synkinesis_count > 0:
            # Calculate side-specific statistics for each type
            side_stats = {}
            for synk_type in SYNKINESIS_TYPES:
                for side in ['Left', 'Right']:
                    col = f'{synk_type} {side}'
                    if col in df.columns:
                        count = (df[col] == 'Yes').sum()
                        side_stats[f'{synk_type} {side} Count'] = count
                        side_stats[f'{synk_type} {side} Percentage'] = (count / synkinesis_count) * 100
            
            synkinesis_summary_stats = {
                'Total Patients': len(df),
                'Patients with Synkinesis': synkinesis_count,
                'Synkinesis Percentage': (synkinesis_count / len(df)) * 100,
                **side_stats
            }
            
            # Save synkinesis stats
            stats_df = pd.DataFrame([synkinesis_summary_stats])
            stats_path = os.path.join(self.output_dir, "synkinesis_statistics.csv")
            stats_df.to_csv(stats_path, index=False)
            
            logger.info(f"Saved synkinesis statistics to {stats_path}")
        
        return
