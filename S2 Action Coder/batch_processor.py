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

        print(f"Scanning directory: {directory}")
        all_files = os.listdir(directory)

        # Debug: Print all files found
        print(f"Found {len(all_files)} files in directory")

        # Case-insensitive file extension matching
        video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        csv_files = [f for f in all_files if f.lower().endswith('.csv')]

        print(f"Found {len(video_files)} video files and {len(csv_files)} CSV files")

        # Print the first few files of each type for debugging
        if video_files:
            print(f"Video file examples: {video_files[:3]}")
        if csv_files:
            print(f"CSV file examples: {csv_files[:3]}")

        # Extract base identifiers using a more robust method
        video_bases = {}
        for video_file in video_files:
            # First try to extract the base ID before any suffix (like "_rotated")
            # This should handle "IMG_0935_rotated.MOV" -> "IMG_0935"
            parts = video_file.split('_')
            if len(parts) >= 2:
                # Use the first two parts as the base (e.g., "IMG_0935")
                base_id = f"{parts[0]}_{parts[1]}"
                video_bases[base_id] = video_file
                print(f"Extracted base '{base_id}' from video file '{video_file}'")

        csv_groups = {}
        for csv_file in csv_files:
            # Extract the base ID in the same way
            parts = csv_file.split('_')
            if len(parts) >= 2:
                base_id = f"{parts[0]}_{parts[1]}"
                if base_id not in csv_groups:
                    csv_groups[base_id] = []
                csv_groups[base_id].append(csv_file)
                print(f"Extracted base '{base_id}' from CSV file '{csv_file}'")

        # Match video files with CSV files
        file_sets = []
        for base_id, video_file in video_bases.items():
            if base_id in csv_groups and csv_groups[base_id]:
                print(f"Matched base ID: {base_id}")
                print(f"  Video: {video_file}")
                print(f"  CSVs: {csv_groups[base_id]}")

                file_set = {
                    'base_id': base_id,
                    'video': os.path.join(directory, video_file),
                    'csv1': os.path.join(directory, csv_groups[base_id][0]) if len(csv_groups[base_id]) > 0 else None,
                    'csv2': os.path.join(directory, csv_groups[base_id][1]) if len(csv_groups[base_id]) > 1 else None
                }
                file_sets.append(file_set)

        print(f"Total matched file sets: {len(file_sets)}")
        return file_sets

    def find_matching_files_for_videos(self, video_files, search_directory=None):
        """
        Find matching CSV files for a list of video files.

        Args:
            video_files: List of full paths to video files
            search_directory: Optional directory to search for CSV files (defaults to video directories)

        Returns:
            A list of dicts with matched file sets
        """
        file_sets = []

        # Process each video file
        for video_file in video_files:
            video_dir = os.path.dirname(video_file)
            video_filename = os.path.basename(video_file)

            # Use the search directory if provided, otherwise use the video's directory
            csv_search_dir = search_directory if search_directory else video_dir

            print(f"Looking for CSV matches for video: {video_filename} in {csv_search_dir}")

            # Extract base identifier from video filename
            parts = video_filename.split('_')
            if len(parts) >= 2:
                # Use the first two parts as the base (e.g., "IMG_0935")
                base_id = f"{parts[0]}_{parts[1]}"
                print(f"Extracted base '{base_id}' from video file '{video_filename}'")

                # Look for matching CSV files in the search directory
                csv_files = []
                if os.path.isdir(csv_search_dir):
                    all_files = os.listdir(csv_search_dir)
                    for file in all_files:
                        if file.lower().endswith('.csv'):
                            file_parts = file.split('_')
                            if len(file_parts) >= 2 and f"{file_parts[0]}_{file_parts[1]}" == base_id:
                                csv_files.append(os.path.join(csv_search_dir, file))

                # If we found matching CSVs, create a file set
                if csv_files:
                    file_set = {
                        'base_id': base_id,
                        'video': video_file,
                        'csv1': csv_files[0] if len(csv_files) > 0 else None,
                        'csv2': csv_files[1] if len(csv_files) > 1 else None
                    }
                    file_sets.append(file_set)
                    print(f"Created match for {base_id}:")
                    print(f"  Video: {video_file}")
                    print(f"  CSVs: {csv_files[:2]}")

        print(f"Total matched file sets: {len(file_sets)}")
        return file_sets

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