# csv_handler.py

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

    def load_data(self): # (No change)
        """Load data from the first input CSV file."""
        try:
            self.data = pd.read_csv(self.input_file)
            print(f"Loaded first CSV with {len(self.data)} rows and {len(self.data.columns)} columns.")
            return True
        except Exception as e:
            print(f"Error loading first CSV: {e}")
            self.data = None
            return False

    def load_second_data(self): # (No change)
        """Load data from the second input CSV file."""
        try:
            self.second_data = pd.read_csv(self.second_input_file)
            print(f"Loaded second CSV with {len(self.second_data)} rows and {len(self.second_data.columns)} columns.")
            return True
        except Exception as e:
            print(f"Error loading second CSV: {e}")
            self.second_data = None
            return False

    def set_input_file(self, input_file): # (No change)
        """Set the first input file and load it."""
        self.input_file = input_file
        return self.load_data()

    def set_second_input_file(self, input_file): # (No change)
        """Set the second input file and load it."""
        self.second_input_file = input_file
        return self.load_second_data()

    def get_frame_count(self): # (No change)
        """Return the number of frames (rows) in the first CSV."""
        if self.data is not None:
            return len(self.data)
        return 0

    # --- Modified add_action_column ---
    def add_action_column(self, action_data_dict):
        """
        Add action column to both dataframes using a pre-generated dictionary.
        Uses blank string ('') as the default for uncoded frames.

        Args:
            action_data_dict: Dictionary mapping frame numbers (int) to action codes (str)
                              or blank string ('').
        """
        success = True

        if self.data is not None:
            try:
                num_rows = len(self.data)
                # --- MODIFICATION: Default to blank string ---
                action_col = [''] * num_rows
                action_count = 0
                bl_count = 0
                # Fill in actions from the dictionary
                for frame, action in action_data_dict.items():
                    frame_idx = int(frame)
                    if 0 <= frame_idx < num_rows:
                        action_code = action if action is not None else '' # Ensure blank if None
                        action_col[frame_idx] = action_code
                        if action_code and action_code != 'BL': action_count += 1
                        if action_code == 'BL': bl_count += 1
                    # else: # Optional: Warn if dict contains frames outside CSV range
                    #     print(f"CSVHandler WARN: Frame index {frame_idx} from action dict is out of bounds for CSV1 (len={num_rows}).")

                self.data['action'] = action_col
                print(f"Added action column to first CSV (len={num_rows}) with {action_count} action and {bl_count} BL entries.")
            except Exception as e:
                print(f"Error adding action column to first CSV: {e}")
                success = False
        else:
            print("No data loaded for first CSV.")
            # success = False # Don't mark as failed if only second CSV exists

        if self.second_data is not None:
            try:
                num_rows = len(self.second_data)
                 # --- MODIFICATION: Default to blank string ---
                action_col = [''] * num_rows
                action_count = 0
                bl_count = 0
                for frame, action in action_data_dict.items():
                    frame_idx = int(frame)
                    if 0 <= frame_idx < num_rows:
                        action_code = action if action is not None else '' # Ensure blank if None
                        action_col[frame_idx] = action_code
                        if action_code and action_code != 'BL': action_count += 1
                        if action_code == 'BL': bl_count += 1
                    # else:
                    #      print(f"CSVHandler WARN: Frame index {frame_idx} from action dict is out of bounds for CSV2 (len={num_rows}).")

                self.second_data['action'] = action_col
                print(f"Added action column to second CSV (len={num_rows}) with {action_count} action and {bl_count} BL entries.")
            except Exception as e:
                print(f"Error adding action column to second CSV: {e}")
                success = False
        # else: # Don't print if second CSV just wasn't loaded
        #     print("No data loaded for second CSV.")

        return success
    # --- End MODIFIED ---

    def save_data(self, output_file, second_output_file=None): # (No change)
        """Save the modified data to new CSV files."""
        overall_success = True
        if self.data is not None:
            try:
                output_dir = os.path.dirname(output_file)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                self.data.to_csv(output_file, index=False)
                print(f"Saved first modified CSV to {output_file}")
            except Exception as e:
                print(f"Error saving first CSV: {e}")
                overall_success = False
        else:
            print("No data to save for first CSV.")

        if self.second_data is not None and second_output_file:
            try:
                output_dir = os.path.dirname(second_output_file)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                self.second_data.to_csv(second_output_file, index=False)
                print(f"Saved second modified CSV to {second_output_file}")
            except Exception as e:
                print(f"Error saving second CSV: {e}")
                overall_success = False

        return overall_success

