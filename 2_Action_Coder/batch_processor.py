# batch_processor.py - Handles batch processing of multiple video/CSV file sets
import os
import re
from PyQt5.QtCore import QObject, pyqtSignal

class BatchProcessor(QObject):
    """Handles sequential loading and processing of multiple video/CSV file sets."""
    
    # Signals
    file_loaded_signal = pyqtSignal(bool)  # Success/failure of loading operation
    
    def __init__(self):
        """Initialize the batch processor."""
        super().__init__()
        self.file_sets = []  # List of file sets to process
        self.current_index = -1  # Index of currently loaded file set
    
    def find_matching_files(self, directory):
        """
        Find matching video and CSV files in the given directory.
        
        Args:
            directory: Path to the directory containing files
            
        Returns:
            A list of dicts with matched file sets
        """
        if not os.path.isdir(directory):
            return []
        
        all_files = os.listdir(directory)
        video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        csv_files = [f for f in all_files if f.lower().endswith('.csv')]
        
        # Create a dictionary to group files by their base identifier
        file_groups = {}
        
        # Process video files
        for video_file in video_files:
            # Extract base identifier (e.g., "IMG_0234")
            match = re.match(r'([\w-]+)_', video_file)
            if match:
                base_id = match.group(1)
                if base_id not in file_groups:
                    file_groups[base_id] = {'video': None, 'csvs': []}
                file_groups[base_id]['video'] = os.path.join(directory, video_file)
        
        # Process CSV files
        for csv_file in csv_files:
            # Extract base identifier (e.g., "IMG_0234")
            match = re.match(r'([\w-]+)_', csv_file)
            if match:
                base_id = match.group(1)
                if base_id in file_groups:
                    file_groups[base_id]['csvs'].append(os.path.join(directory, csv_file))
                    
        # Filter out incomplete sets and build the final list
        result = []
        for base_id, files in file_groups.items():
            if files['video'] and files['csvs']:
                # Create a set with video and up to 2 CSVs
                file_set = {
                    'base_id': base_id,
                    'video': files['video'],
                    'csv1': files['csvs'][0] if len(files['csvs']) > 0 else None,
                    'csv2': files['csvs'][1] if len(files['csvs']) > 1 else None
                }
                result.append(file_set)
        
        return result
    
    def set_file_sets(self, file_sets):
        """
        Set the file sets to process.
        
        Args:
            file_sets: List of dicts containing file paths
        """
        self.file_sets = file_sets
        self.current_index = -1  # Reset the index
    
    def get_total_files(self):
        """Get the total number of file sets."""
        return len(self.file_sets)
    
    def get_current_index(self):
        """Get the index of the current file set."""
        return self.current_index
    
    def get_current_file_set(self):
        """Get the current file set."""
        if 0 <= self.current_index < len(self.file_sets):
            return self.file_sets[self.current_index]
        return None
    
    def load_next_file(self):
        """
        Load the next file set in the sequence.
        
        Returns:
            The next file set dict, or None if at the end
        """
        if self.current_index + 1 < len(self.file_sets):
            self.current_index += 1
            return self.file_sets[self.current_index]
        return None
    
    def load_previous_file(self):
        """
        Load the previous file set in the sequence.
        
        Returns:
            The previous file set dict, or None if at the beginning
        """
        if self.current_index > 0:
            self.current_index -= 1
            return self.file_sets[self.current_index]
        return None
    
    def load_first_file(self):
        """
        Load the first file set in the sequence.
        
        Returns:
            The first file set dict, or None if no files
        """
        if len(self.file_sets) > 0:
            self.current_index = 0
            return self.file_sets[0]
        return None
    
    def has_next_file(self):
        """Check if there is a next file in the sequence."""
        return self.current_index + 1 < len(self.file_sets)
    
    def has_previous_file(self):
        """Check if there is a previous file in the sequence."""
        return self.current_index > 0
