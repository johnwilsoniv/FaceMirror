import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import sys
import platform

class FacialAUAnalyzer:
    def __init__(self):
        # Define the key AUs for each action
        self.action_to_aus = {
            'RE': ['AU01_r', 'AU02_r'],  # Raise Eyebrows
            'ES': ['AU45_r', 'AU07_r'],  # Close Eyes Softly
            'ET': ['AU45_r', 'AU07_r'],  # Close Eyes Tightly
            'SS': ['AU12_r'],            # Soft Smile
            'BS': ['AU12_r'],            # Big Smile
            'SO': ['AU28_c'],            # Say 'O' (Using AU28_c since AU28_r not in columns)
            'SE': ['AU12_r'],            # Say 'E'
            'BL': ['AU45_r', 'AU07_r'],  # Blink
            'WN': ['AU09_r'],            # Wrinkle Nose
            'PL': ['AU28_c']             # Pucker Lips (Using AU28_c since AU28_r not in columns)
        }
        
        # Action descriptions for labels
        self.action_descriptions = {
            'RE': 'Raise Eyebrows',
            'ES': 'Close Eyes Softly',
            'ET': 'Close Eyes Tightly',
            'SS': 'Soft Smile',
            'BS': 'Big Smile',
            'SO': 'Say O',
            'SE': 'Say E',
            'BL': 'Blink',
            'WN': 'Wrinkle Nose',
            'PL': 'Pucker Lips'
        }
        
    def load_data(self, left_csv_path, right_csv_path):
        """Load left and right CSV files for a patient"""
        self.left_data = pd.read_csv(left_csv_path)
        self.right_data = pd.read_csv(right_csv_path)
        
        # Extract patient ID from filename
        self.patient_id = os.path.basename(left_csv_path).split('_')[0]
        
        print(f"Loaded data for patient {self.patient_id}")
        print(f"Left data shape: {self.left_data.shape}")
        print(f"Right data shape: {self.right_data.shape}")
        
    def analyze_maximal_intensity(self):
        """Find maximal intensity for key AUs for each action on both sides"""
        self.results = {}
        
        # Get unique actions in the data
        unique_actions = pd.unique(self.left_data['action'])
        
        for action in unique_actions:
            if action not in self.action_to_aus:
                print(f"Warning: Action {action} not in predefined action list. Skipping.")
                continue
                
            # Filter data for this action
            left_action_data = self.left_data[self.left_data['action'] == action]
            right_action_data = self.right_data[self.right_data['action'] == action]
            
            # Get key AUs for this action
            key_aus = self.action_to_aus[action]
            
            # Calculate left side maximal intensity
            if len(key_aus) == 1:
                # If only one AU, find max value directly
                left_max_values = left_action_data[key_aus].max().values
                left_max_row_idx = left_action_data[key_aus].idxmax().values[0]
            else:
                # If multiple AUs, calculate average and find max
                left_action_data['avg_key_aus'] = left_action_data[key_aus].mean(axis=1)
                left_max_values = [left_action_data['avg_key_aus'].max()]
                left_max_row_idx = left_action_data['avg_key_aus'].idxmax()
            
            # Get the frame number for max intensity on left side
            left_max_frame = self.left_data.loc[left_max_row_idx, 'frame']
            
            # Calculate right side maximal intensity
            if len(key_aus) == 1:
                # If only one AU, find max value directly
                right_max_values = right_action_data[key_aus].max().values
                right_max_row_idx = right_action_data[key_aus].idxmax().values[0]
            else:
                # If multiple AUs, calculate average and find max
                right_action_data['avg_key_aus'] = right_action_data[key_aus].mean(axis=1)
                right_max_values = [right_action_data['avg_key_aus'].max()]
                right_max_row_idx = right_action_data['avg_key_aus'].idxmax()
            
            # Get the frame number for max intensity on right side
            right_max_frame = self.right_data.loc[right_max_row_idx, 'frame']
            
            # Store results
            self.results[action] = {
                'left': {
                    'max_value': left_max_values[0],
                    'frame': left_max_frame,
                    'row_idx': left_max_row_idx
                },
                'right': {
                    'max_value': right_max_values[0],
                    'frame': right_max_frame,
                    'row_idx': right_max_row_idx
                }
            }
        
        return self.results
    
    def extract_frames(self, video_path, output_dir):
        """
        Extract frames from video at points of maximal expression
        and save them as images with appropriate labels
        """
        if not hasattr(self, 'results'):
            print("Please run analyze_maximal_intensity() first")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video FPS: {fps}")
        print(f"Total frames: {frame_count}")
        
        # Extract frames for each action and side
        for action, sides in self.results.items():
            for side, info in sides.items():
                frame_num = int(info['frame'])
                
                # Set video to the right frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Error: Could not read frame {frame_num}")
                    continue
                
                # Create image label with patient ID, action, side, and frame number
                action_desc = self.action_descriptions.get(action, action)
                label = f"{self.patient_id}_{action_desc}_{side}_frame{frame_num}"
                
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
                output_path = os.path.join(output_dir, f"{label}.jpg")
                cv2.imwrite(output_path, frame)
                print(f"Saved {output_path}")
                
                # Also save information about the AUs
                key_aus = self.action_to_aus[action]
                au_values = {}
                
                # Get the values for each key AU at this frame
                for au in key_aus:
                    if side == 'left':
                        au_values[au] = self.left_data.loc[info['row_idx'], au]
                    else:
                        au_values[au] = self.right_data.loc[info['row_idx'], au]
                
                # Create a visualization of the AU values
                self.create_au_visualization(au_values, action, side, frame_num, output_dir)
        
        # Release the video capture
        cap.release()
        
    def create_au_visualization(self, au_values, action, side, frame_num, output_dir):
        """Create a bar chart visualization of the AU values"""
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        bars = plt.bar(range(len(au_values)), list(au_values.values()), tick_label=list(au_values.keys()))
        
        # Add value labels to the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Add title and labels
        action_desc = self.action_descriptions.get(action, action)
        plt.title(f"AU Values for {self.patient_id} - {action_desc} - {side}")
        plt.xlabel("Action Units")
        plt.ylabel("Intensity")
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"{self.patient_id}_{action_desc}_{side}_frame{frame_num}_AUs.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved AU visualization: {output_path}")
        
    def generate_summary_report(self, output_dir):
        """Generate a summary report of the analysis"""
        if not hasattr(self, 'results'):
            print("Please run analyze_maximal_intensity() first")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a DataFrame for the results
        summary_data = []
        
        for action, sides in self.results.items():
            action_desc = self.action_descriptions.get(action, action)
            key_aus = ', '.join(self.action_to_aus[action])
            
            # Add left side data
            left_max = sides['left']['max_value']
            left_frame = sides['left']['frame']
            
            # Add right side data
            right_max = sides['right']['max_value']
            right_frame = sides['right']['frame']
            
            # Calculate symmetry (ratio of left to right)
            if right_max != 0:
                symmetry_ratio = left_max / right_max
            else:
                symmetry_ratio = np.nan
                
            # Determine if there's asymmetry (this is a simple threshold, may need adjustment)
            asymmetry = abs(1 - symmetry_ratio) > 0.2
            
            summary_data.append({
                'Action': action,
                'Description': action_desc,
                'Key AUs': key_aus,
                'Left Max': left_max,
                'Left Frame': left_frame,
                'Right Max': right_max,
                'Right Frame': right_frame,
                'Symmetry Ratio (L/R)': symmetry_ratio,
                'Asymmetry Detected': asymmetry
            })
        
        # Create DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_data)
        output_path = os.path.join(output_dir, f"{self.patient_id}_summary.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"Saved summary report: {output_path}")
        
        # Also create a visualization of the symmetry
        self.create_symmetry_visualization(summary_df, output_dir)
        
    def create_symmetry_visualization(self, summary_df, output_dir):
        """Create a visualization of the facial symmetry"""
        plt.figure(figsize=(12, 8))
        
        # Create bar chart of left vs right maximal intensities
        actions = summary_df['Description']
        left_max = summary_df['Left Max']
        right_max = summary_df['Right Max']
        
        x = np.arange(len(actions))
        width = 0.35
        
        plt.bar(x - width/2, left_max, width, label='Left')
        plt.bar(x + width/2, right_max, width, label='Right')
        
        plt.xlabel('Actions')
        plt.ylabel('Maximal Intensity')
        plt.title(f'Left vs Right Facial Movement Intensity - Patient {self.patient_id}')
        plt.xticks(x, actions, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"{self.patient_id}_symmetry_chart.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved symmetry visualization: {output_path}")

# GUI application for file selection and analysis
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import threading
    
    class FacialAUAnalyzerGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("Facial AU Analyzer")
            self.root.geometry("800x600")
            
            # Initialize variables
            self.left_csv_path = tk.StringVar()
            self.right_csv_path = tk.StringVar()
            self.video_path = tk.StringVar()
            self.output_dir = tk.StringVar(value="output")
            
            # Create GUI elements
            self.create_widgets()
            
            # Initialize analyzer
            self.analyzer = FacialAUAnalyzer()
            self.analysis_thread = None
            
        def create_widgets(self):
            # Create main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # File selection frame
            file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
            file_frame.pack(fill=tk.X, pady=10)
            
            # Left CSV file
            ttk.Label(file_frame, text="Left CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Entry(file_frame, textvariable=self.left_csv_path, width=50).grid(row=0, column=1, padx=5, pady=5)
            ttk.Button(file_frame, text="Browse...", command=self.browse_left_csv).grid(row=0, column=2, padx=5, pady=5)
            
            # Right CSV file
            ttk.Label(file_frame, text="Right CSV File:").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Entry(file_frame, textvariable=self.right_csv_path, width=50).grid(row=1, column=1, padx=5, pady=5)
            ttk.Button(file_frame, text="Browse...", command=self.browse_right_csv).grid(row=1, column=2, padx=5, pady=5)
            
            # Video file
            ttk.Label(file_frame, text="Video File:").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=2, column=1, padx=5, pady=5)
            ttk.Button(file_frame, text="Browse...", command=self.browse_video).grid(row=2, column=2, padx=5, pady=5)
            
            # Output directory
            ttk.Label(file_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=3, column=1, padx=5, pady=5)
            ttk.Button(file_frame, text="Browse...", command=self.browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
            
            # Analysis and results frame
            analysis_frame = ttk.LabelFrame(main_frame, text="Analysis", padding="10")
            analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Progress bar
            self.progress_var = tk.DoubleVar()
            ttk.Label(analysis_frame, text="Progress:").pack(anchor=tk.W, pady=5)
            self.progress_bar = ttk.Progressbar(analysis_frame, variable=self.progress_var, maximum=100)
            self.progress_bar.pack(fill=tk.X, pady=5)
            
            # Status label
            self.status_var = tk.StringVar(value="Ready")
            ttk.Label(analysis_frame, textvariable=self.status_var).pack(anchor=tk.W, pady=5)
            
            # Results text
            ttk.Label(analysis_frame, text="Results:").pack(anchor=tk.W, pady=5)
            self.results_text = tk.Text(analysis_frame, height=10, wrap=tk.WORD)
            self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Scrollbar for results text
            scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.results_text.config(yscrollcommand=scrollbar.set)
            
            # Buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=10)
            
            # Run button
            ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.RIGHT, padx=5)
            
            # Auto-detect files button
            ttk.Button(button_frame, text="Auto-Detect Files", command=self.auto_detect_files).pack(side=tk.RIGHT, padx=5)
            
        def browse_left_csv(self):
            filepath = filedialog.askopenfilename(
                title="Select Left CSV File",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if filepath:
                self.left_csv_path.set(filepath)
                
        def browse_right_csv(self):
            filepath = filedialog.askopenfilename(
                title="Select Right CSV File",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if filepath:
                self.right_csv_path.set(filepath)
                
        def browse_video(self):
            filepath = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.MOV"), ("All Files", "*.*")]
            )
            if filepath:
                self.video_path.set(filepath)
                
        def browse_output_dir(self):
            dirpath = filedialog.askdirectory(title="Select Output Directory")
            if dirpath:
                self.output_dir.set(dirpath)
                
        def auto_detect_files(self):
            """Try to intelligently detect files based on naming patterns"""
            # Get current directory
            current_dir = os.getcwd()
            
            # Search for matching files
            left_files = []
            right_files = []
            video_files = []
            
            for file in os.listdir(current_dir):
                if file.endswith(".csv") and "left" in file.lower():
                    left_files.append(file)
                elif file.endswith(".csv") and "right" in file.lower():
                    right_files.append(file)
                elif file.endswith((".mp4", ".avi", ".mov", ".MOV")):
                    video_files.append(file)
            
            # Try to find matching pairs
            if left_files and right_files:
                # Find common prefix
                for left_file in left_files:
                    prefix = left_file.split("_left")[0]
                    matching_right = [rf for rf in right_files if rf.startswith(prefix)]
                    matching_video = [vf for vf in video_files if vf.startswith(prefix)]
                    
                    if matching_right:
                        self.left_csv_path.set(left_file)
                        self.right_csv_path.set(matching_right[0])
                        
                        if matching_video:
                            self.video_path.set(matching_video[0])
                        
                        self.log_message(f"Auto-detected files with prefix: {prefix}")
                        return
            
            # If no matching pairs found
            if left_files:
                self.left_csv_path.set(left_files[0])
                self.log_message(f"Found left CSV: {left_files[0]}")
            
            if right_files:
                self.right_csv_path.set(right_files[0])
                self.log_message(f"Found right CSV: {right_files[0]}")
                
            if video_files:
                self.video_path.set(video_files[0])
                self.log_message(f"Found video: {video_files[0]}")
                
            if not (left_files or right_files or video_files):
                messagebox.showinfo("Auto-Detect", "No matching files found in current directory.")
                
        def run_analysis(self):
            """Run the analysis in a separate thread"""
            # Check if files are selected
            if not self.left_csv_path.get() or not self.right_csv_path.get() or not self.video_path.get():
                messagebox.showerror("Error", "Please select all required files")
                return
            
            # Check if analysis is already running
            if self.analysis_thread and self.analysis_thread.is_alive():
                messagebox.showinfo("Info", "Analysis is already running")
                return
            
            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.progress_var.set(0)
            self.status_var.set("Starting analysis...")
            
            # Start analysis in a separate thread
            self.analysis_thread = threading.Thread(target=self.analyze_data)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
        def analyze_data(self):
            """Run the analysis and update GUI"""
            try:
                # Update status
                self.update_status("Loading data...")
                self.update_progress(10)
                
                # Load data
                self.analyzer.load_data(self.left_csv_path.get(), self.right_csv_path.get())
                self.log_message(f"Loaded data for patient {self.analyzer.patient_id}")
                self.update_progress(30)
                
                # Analyze maximal intensity
                self.update_status("Analyzing maximal intensity...")
                results = self.analyzer.analyze_maximal_intensity()
                self.update_progress(50)
                
                # Log results
                for action, sides in results.items():
                    action_desc = self.analyzer.action_descriptions.get(action, action)
                    self.log_message(f"\nAction: {action_desc} ({action})")
                    self.log_message(f"Left max: {sides['left']['max_value']:.4f} (frame {sides['left']['frame']})")
                    self.log_message(f"Right max: {sides['right']['max_value']:.4f} (frame {sides['right']['frame']})")
                
                # Extract frames
                self.update_status("Extracting frames from video...")
                self.analyzer.extract_frames(self.video_path.get(), self.output_dir.get())
                self.update_progress(80)
                
                # Generate report
                self.update_status("Generating summary report...")
                self.analyzer.generate_summary_report(self.output_dir.get())
                self.update_progress(100)
                
                # Complete
                self.update_status(f"Analysis complete. Results saved to {self.output_dir.get()}")
                messagebox.showinfo("Complete", f"Analysis complete. Results saved to {self.output_dir.get()}")
                
                # Open output folder
                if os.path.exists(self.output_dir.get()):
                    # Cross-platform way to open folder
                    if platform.system() == "Windows":
                        os.startfile(self.output_dir.get())
                    elif platform.system() == "Darwin":  # macOS
                        import subprocess
                        subprocess.run(['open', self.output_dir.get()])
                    else:  # Linux and other Unix-like
                        import subprocess
                        subprocess.run(['xdg-open', self.output_dir.get()])
                
            except Exception as e:
                self.update_status(f"Error: {str(e)}")
                self.log_message(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
                
        def update_status(self, message):
            """Update status label"""
            self.root.after(0, lambda: self.status_var.set(message))
            
        def update_progress(self, value):
            """Update progress bar"""
            self.root.after(0, lambda: self.progress_var.set(value))
            
        def log_message(self, message):
            """Add message to results text"""
            self.root.after(0, lambda: self.results_text.insert(tk.END, message + "\n"))
            self.root.after(0, lambda: self.results_text.see(tk.END))
    
    # Create and run GUI application
    root = tk.Tk()
    app = FacialAUAnalyzerGUI(root)
    root.mainloop()
