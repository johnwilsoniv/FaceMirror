# csv_handler.py - Handles CSV file operations
import pandas as pd
import os

class CSVHandler:
    def __init__(self, input_file=None, second_input_file=None):
        """Initialize the CSV handler with the input file paths."""
        self.input_file = input_file
        self.second_input_file = second_input_file
        self.data = None
        self.second_data = None
        
        if input_file:
            self.load_data()
        if second_input_file:
            self.load_second_data()
    
    def load_data(self):
        """Load data from the first input CSV file."""
        try:
            self.data = pd.read_csv(self.input_file)
            print(f"Loaded first CSV with {len(self.data)} rows and {len(self.data.columns)} columns.")
            return True
        except Exception as e:
            print(f"Error loading first CSV: {e}")
            self.data = None
            return False
    
    def load_second_data(self):
        """Load data from the second input CSV file."""
        try:
            self.second_data = pd.read_csv(self.second_input_file)
            print(f"Loaded second CSV with {len(self.second_data)} rows and {len(self.second_data.columns)} columns.")
            return True
        except Exception as e:
            print(f"Error loading second CSV: {e}")
            self.second_data = None
            return False
    
    def set_input_file(self, input_file):
        """Set the first input file and load it."""
        self.input_file = input_file
        return self.load_data()
    
    def set_second_input_file(self, input_file):
        """Set the second input file and load it."""
        self.second_input_file = input_file
        return self.load_second_data()
    
    def get_frame_count(self):
        """Return the number of frames in the first CSV."""
        if self.data is not None:
            return len(self.data)
        return 0

    def add_action_column(self, action_data):
        """
        Add action column to both dataframes.

        Args:
            action_data: Dictionary mapping frame numbers to action codes
        """
        success = True

        if self.data is not None:
            try:
                # Initialize action column with empty strings
                self.data['action'] = ""

                # Fill in the actions - make sure we handle all frames
                for frame, action in action_data.items():
                    frame_idx = int(frame)  # Ensure we have an integer index
                    if 0 <= frame_idx < len(self.data):
                        self.data.at[frame_idx, 'action'] = action

                print(f"Added action column to first CSV with {sum(self.data['action'] != '')} action entries")
            except Exception as e:
                print(f"Error adding action column to first CSV: {e}")
                success = False
        else:
            print("No data loaded for first CSV.")
            success = False

        if self.second_data is not None:
            try:
                # Initialize action column with empty strings
                self.second_data['action'] = ""

                # Fill in the actions
                for frame, action in action_data.items():
                    frame_idx = int(frame)  # Ensure we have an integer index
                    if 0 <= frame_idx < len(self.second_data):
                        self.second_data.at[frame_idx, 'action'] = action

                print(f"Added action column to second CSV with {sum(self.second_data['action'] != '')} action entries")
            except Exception as e:
                print(f"Error adding action column to second CSV: {e}")
                success = False
        else:
            print("No data loaded for second CSV.")

        return success

    def save_data(self, output_file, second_output_file=None):
        """Save the modified data to new CSV files."""
        overall_success = True

        if self.data is not None:
            try:
                # Ensure directory exists
                output_dir = os.path.dirname(output_file)
                if output_dir:  # Only create if there's a directory path
                    os.makedirs(output_dir, exist_ok=True)

                self.data.to_csv(output_file, index=False)
                print(f"Saved first modified CSV to {output_file}")
            except Exception as e:
                print(f"Error saving first CSV: {e}")
                overall_success = False
        else:
            print("No data to save for first CSV.")
            overall_success = False

        # Save second CSV if available
        if self.second_data is not None and second_output_file:
            try:
                # Ensure directory exists
                output_dir = os.path.dirname(second_output_file)
                if output_dir:  # Only create if there's a directory path
                    os.makedirs(output_dir, exist_ok=True)

                self.second_data.to_csv(second_output_file, index=False)
                print(f"Saved second modified CSV to {second_output_file}")
            except Exception as e:
                print(f"Error saving second CSV: {e}")
                overall_success = False

        return overall_success
