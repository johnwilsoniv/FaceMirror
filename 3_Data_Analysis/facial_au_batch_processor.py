"""
Batch processor for analyzing multiple patients' facial AU data.
"""

import os
import pandas as pd
import logging
import glob
from facial_au_analyzer import FacialAUAnalyzer
from facial_au_constants import SUMMARY_COLUMNS

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
    
    def __init__(self, output_dir="output"):
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
        
        Args:
            extract_frames (bool): Whether to extract frames from videos
            
        Returns:
            str: Path to combined summary CSV file
        """
        if not self.patients:
            logger.error("No patients added to batch")
            return None
        
        all_summary_data = []
        
        for i, patient in enumerate(self.patients):
            logger.info(f"Processing patient {i+1}/{len(self.patients)}: {patient['patient_id']}")
            
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
            
            # Create symmetry visualization
            analyzer.create_symmetry_visualization(self.output_dir)
            
            # Get summary data
            patient_summary = analyzer.generate_summary_data()
            all_summary_data.extend(patient_summary)
            
            logger.info(f"Completed processing for patient {patient['patient_id']}")
        
        # Create combined summary CSV
        if all_summary_data:
            df = pd.DataFrame(all_summary_data)
            
            # Ensure all expected columns are present (fill missing with NaN)
            for col in SUMMARY_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            
            # Reorder columns to match expected order
            df = df[SUMMARY_COLUMNS]
            
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
        
        Returns:
            pd.DataFrame: DataFrame with asymmetry analysis
        """
        if not hasattr(self, 'summary_data') or self.summary_data.empty:
            logger.error("No summary data available. Please run process_all() first")
            return None
        
        # Group by action and calculate asymmetry statistics
        if isinstance(self.summary_data, list):
            df = pd.DataFrame(self.summary_data)
        else:
            df = self.summary_data
            
        asymmetry_by_action = df.groupby(['Action', 'Description']).agg({
            'Patient ID': 'count',
            'Asymmetry Detected': 'sum',
            'Symmetry Ratio': ['mean', 'median', 'std']
        }).reset_index()
        
        # Calculate percentage of patients with asymmetry for each action
        asymmetry_by_action['Asymmetry Percentage'] = (
            asymmetry_by_action[('Asymmetry Detected', 'sum')] / 
            asymmetry_by_action[('Patient ID', 'count')] * 100
        )
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, "asymmetry_analysis.csv")
        asymmetry_by_action.to_csv(output_path)
        logger.info(f"Saved asymmetry analysis to {output_path}")
        
        # Also analyze paralysis and synkinesis across patients
        self.analyze_paralysis_and_synkinesis()
        
        return asymmetry_by_action
        
    def analyze_paralysis_and_synkinesis(self):
        """
        Analyze patterns of facial paralysis and synkinesis across all patients.
        Creates summary reports of these conditions.
        """
        if not hasattr(self, 'summary_data') or self.summary_data.empty:
            logger.error("No summary data available for analysis")
            return
        
        if isinstance(self.summary_data, list):
            df = pd.DataFrame(self.summary_data)
        else:
            df = self.summary_data
        
        # Count patients with paralysis and identify side
        paralysis_summary = df.groupby(['Patient ID']).agg({
            'Paralysis Detected': 'max',  # True if detected in any action
            'Paralyzed Side': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            'Affected AUs': lambda x: ', '.join(set([y for y in x if y != 'None']))
        }).reset_index()
        
        # Add summary of paralysis by side
        paralysis_count = paralysis_summary['Paralysis Detected'].sum()
        if paralysis_count > 0:
            side_counts = paralysis_summary[paralysis_summary['Paralysis Detected']]['Paralyzed Side'].value_counts()
            paralysis_summary_stats = {
                'Total Patients': len(df['Patient ID'].unique()),
                'Patients with Paralysis': paralysis_count,
                'Paralysis Percentage': (paralysis_count / len(df['Patient ID'].unique())) * 100,
                'Left Side Paralysis': side_counts.get('left', 0),
                'Right Side Paralysis': side_counts.get('right', 0)
            }
            
            # Save paralysis summary
            paralysis_path = os.path.join(self.output_dir, "paralysis_analysis.csv")
            paralysis_summary.to_csv(paralysis_path, index=False)
            
            # Save paralysis stats
            stats_df = pd.DataFrame([paralysis_summary_stats])
            stats_path = os.path.join(self.output_dir, "paralysis_statistics.csv")
            stats_df.to_csv(stats_path, index=False)
            
            logger.info(f"Saved paralysis analysis to {paralysis_path}")
        
        # Analyze synkinesis
        synkinesis_summary = df.groupby(['Patient ID']).agg({
            'Synkinesis Detected': 'max',  # True if detected in any action
            'Synkinesis Type': lambda x: ', '.join(set([y for y in x if y != 'None'])),
            'Synkinesis Details': lambda x: ' '.join(set([y for y in x if y != 'None']))
        }).reset_index()
        
        # Add summary of synkinesis by type
        synkinesis_count = synkinesis_summary['Synkinesis Detected'].sum()
        if synkinesis_count > 0:
            # Count each synkinesis type
            synkinesis_types = []
            for types in synkinesis_summary[synkinesis_summary['Synkinesis Detected']]['Synkinesis Type']:
                synkinesis_types.extend([t.strip() for t in types.split(',')])
            
            type_counts = pd.Series(synkinesis_types).value_counts()
            
            synkinesis_summary_stats = {
                'Total Patients': len(df['Patient ID'].unique()),
                'Patients with Synkinesis': synkinesis_count,
                'Synkinesis Percentage': (synkinesis_count / len(df['Patient ID'].unique())) * 100
            }
            
            # Add counts for each synkinesis type
            for synk_type in type_counts.index:
                synkinesis_summary_stats[f'{synk_type} Count'] = type_counts[synk_type]
                synkinesis_summary_stats[f'{synk_type} Percentage'] = (type_counts[synk_type] / synkinesis_count) * 100
            
            # Save synkinesis summary
            synkinesis_path = os.path.join(self.output_dir, "synkinesis_analysis.csv")
            synkinesis_summary.to_csv(synkinesis_path, index=False)
            
            # Save synkinesis stats
            stats_df = pd.DataFrame([synkinesis_summary_stats])
            stats_path = os.path.join(self.output_dir, "synkinesis_statistics.csv")
            stats_df.to_csv(stats_path, index=False)
            
            logger.info(f"Saved synkinesis analysis to {synkinesis_path}")
        
        return
