"""
GUI for the Facial AU Analyzer application.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import platform
import logging
from facial_au_analyzer import FacialAUAnalyzer
from facial_au_batch_processor import FacialAUBatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacialAUAnalyzerGUI:
    """
    GUI for Facial AU Analyzer with single patient and batch analysis modes.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root (tk.Tk): Root window
        """
        self.root = root
        self.root.title("Facial AU Analyzer")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.left_csv_path = tk.StringVar()
        self.right_csv_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="output")
        self.data_dir = tk.StringVar()
        
        # For batch mode
        self.batch_mode = tk.BooleanVar(value=False)
        self.extract_frames = tk.BooleanVar(value=True)
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize analyzer
        self.analyzer = FacialAUAnalyzer()
        self.batch_processor = FacialAUBatchProcessor()
        self.analysis_thread = None
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.single_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.single_tab, text="Single Patient Analysis")
        self.notebook.add(self.batch_tab, text="Batch Analysis")
        
        # Setup Single Patient tab
        self.setup_single_patient_tab()
        
        # Setup Batch Analysis tab
        self.setup_batch_analysis_tab()
    
    def setup_single_patient_tab(self):
        """Setup widgets for single patient analysis tab."""
        # File selection frame
        file_frame = ttk.LabelFrame(self.single_tab, text="File Selection", padding="10")
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
        analysis_frame = ttk.LabelFrame(self.single_tab, text="Analysis", padding="10")
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
        button_frame = ttk.Frame(self.single_tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Run button
        ttk.Button(button_frame, text="Run Analysis", command=self.run_single_analysis).pack(side=tk.RIGHT, padx=5)
        
        # Auto-detect files button
        ttk.Button(button_frame, text="Auto-Detect Files", command=self.auto_detect_files).pack(side=tk.RIGHT, padx=5)
    
    def setup_batch_analysis_tab(self):
        """Setup widgets for batch analysis tab."""
        # Directory selection frame
        dir_frame = ttk.LabelFrame(self.batch_tab, text="Directory Selection", padding="10")
        dir_frame.pack(fill=tk.X, pady=10)
        
        # Data directory
        ttk.Label(dir_frame, text="Data Directory (containing CSV and video files):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(dir_frame, textvariable=self.data_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse...", command=self.browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # Output directory for batch
        ttk.Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(dir_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(self.batch_tab, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Extract frames checkbox
        ttk.Checkbutton(
            options_frame, 
            text="Extract frames from videos", 
            variable=self.extract_frames
        ).pack(anchor=tk.W, pady=5)
        
        # Batch Analysis and results frame
        batch_analysis_frame = ttk.LabelFrame(self.batch_tab, text="Batch Analysis", padding="10")
        batch_analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Detected patients list
        ttk.Label(batch_analysis_frame, text="Detected Patients:").pack(anchor=tk.W, pady=5)
        
        # Create a frame with scrollbar for the patients list
        patients_frame = ttk.Frame(batch_analysis_frame)
        patients_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.patients_listbox = tk.Listbox(patients_frame, height=10)
        self.patients_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        patients_scrollbar = ttk.Scrollbar(patients_frame, command=self.patients_listbox.yview)
        patients_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.patients_listbox.config(yscrollcommand=patients_scrollbar.set)
        
        # Batch progress bar
        self.batch_progress_var = tk.DoubleVar()
        ttk.Label(batch_analysis_frame, text="Batch Progress:").pack(anchor=tk.W, pady=5)
        self.batch_progress_bar = ttk.Progressbar(batch_analysis_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X, pady=5)
        
        # Batch status label
        self.batch_status_var = tk.StringVar(value="Ready")
        ttk.Label(batch_analysis_frame, textvariable=self.batch_status_var).pack(anchor=tk.W, pady=5)
        
        # Batch Buttons frame
        batch_button_frame = ttk.Frame(self.batch_tab)
        batch_button_frame.pack(fill=tk.X, pady=10)
        
        # Scan for patients button
        ttk.Button(
            batch_button_frame, 
            text="Scan for Patients", 
            command=self.scan_for_patients
        ).pack(side=tk.LEFT, padx=5)
        
        # Run batch analysis button
        ttk.Button(
            batch_button_frame, 
            text="Run Batch Analysis", 
            command=self.run_batch_analysis
        ).pack(side=tk.RIGHT, padx=5)
        
        # Open results button
        ttk.Button(
            batch_button_frame, 
            text="Open Results Folder", 
            command=self.open_results_folder
        ).pack(side=tk.RIGHT, padx=5)
    
    def browse_left_csv(self):
        """Browse for left CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select Left CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filepath:
            self.left_csv_path.set(filepath)
            
    def browse_right_csv(self):
        """Browse for right CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select Right CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filepath:
            self.right_csv_path.set(filepath)
            
    def browse_video(self):
        """Browse for video file."""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.MOV"), ("All Files", "*.*")]
        )
        if filepath:
            self.video_path.set(filepath)
            
    def browse_output_dir(self):
        """Browse for output directory."""
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.output_dir.set(dirpath)
    
    def browse_data_dir(self):
        """Browse for data directory."""
        dirpath = filedialog.askdirectory(title="Select Data Directory")
        if dirpath:
            self.data_dir.set(dirpath)
            
    def auto_detect_files(self):
        """Try to intelligently detect files based on naming patterns."""
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
    
    def scan_for_patients(self):
        """Scan for patients in the data directory."""
        if not self.data_dir.get():
            messagebox.showerror("Error", "Please select a data directory")
            return
        
        # Clear listbox
        self.patients_listbox.delete(0, tk.END)
        
        # Create new batch processor
        self.batch_processor = FacialAUBatchProcessor(self.output_dir.get())
        
        # Detect patients
        num_patients = self.batch_processor.auto_detect_patients(self.data_dir.get())
        
        # Update listbox
        for patient in self.batch_processor.patients:
            self.patients_listbox.insert(tk.END, f"{patient['patient_id']} - {os.path.basename(patient['left_csv'])}")
        
        # Update status
        if num_patients > 0:
            self.batch_status_var.set(f"Found {num_patients} patients")
            messagebox.showinfo("Patient Detection", f"Found {num_patients} patients")
        else:
            self.batch_status_var.set("No patients found")
            messagebox.showinfo("Patient Detection", "No patients found in the selected directory")
    
    def run_single_analysis(self):
        """Run analysis for a single patient."""
        # Check if files are selected
        if not self.left_csv_path.get() or not self.right_csv_path.get():
            messagebox.showerror("Error", "Please select left and right CSV files")
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
        self.analysis_thread = threading.Thread(target=self.analyze_single_patient)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def run_batch_analysis(self):
        """Run batch analysis for all detected patients."""
        if not self.batch_processor or not self.batch_processor.patients:
            messagebox.showerror("Error", "No patients detected. Please scan for patients first.")
            return
        
        # Check if analysis is already running
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showinfo("Info", "Analysis is already running")
            return
        
        # Reset progress
        self.batch_progress_var.set(0)
        self.batch_status_var.set("Starting batch analysis...")
        
        # Start analysis in a separate thread
        self.analysis_thread = threading.Thread(target=self.analyze_batch)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def analyze_single_patient(self):
        """Analyze a single patient's data."""
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
            for action, info in results.items():
                action_desc = self.analyzer.action_descriptions.get(action, action)
                self.log_message(f"\nAction: {action_desc} ({action})")
                self.log_message(f"Max side: {info['max_side']}")
                self.log_message(f"Max frame: {info['max_frame']}")
                self.log_message(f"Max value: {info['max_value']:.4f}")
            
            # Extract frames if video path is provided
            if self.video_path.get():
                self.update_status("Extracting frames from video...")
                self.analyzer.extract_frames(self.video_path.get(), self.output_dir.get())
                self.update_progress(80)
            else:
                self.log_message("No video path provided, skipping frame extraction")
            
            # Create symmetry visualization
            self.update_status("Creating symmetry visualization...")
            self.analyzer.create_symmetry_visualization(self.output_dir.get())
            
            # Generate summary data
            self.update_status("Generating summary data...")
            summary_data = self.analyzer.generate_summary_data()
            self.update_progress(100)
            
            # Complete
            patient_dir = os.path.join(self.output_dir.get(), self.analyzer.patient_id)
            self.update_status(f"Analysis complete. Results saved to {patient_dir}")
            messagebox.showinfo("Complete", f"Analysis complete. Results saved to {patient_dir}")
            
            # Open output folder
            self.open_folder(patient_dir)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def analyze_batch(self):
        """Run batch analysis for all patients."""
        try:
            # Update status
            self.batch_status_var.set("Processing batch...")
            self.batch_progress_var.set(10)
            
            # Set the output directory for the batch processor
            self.batch_processor.output_dir = self.output_dir.get()
            
            # Process all patients
            output_path = self.batch_processor.process_all(extract_frames=self.extract_frames.get())
            
            if output_path:
                # Calculate asymmetry across patients
                self.batch_status_var.set("Analyzing asymmetry patterns...")
                self.batch_progress_var.set(90)
                self.batch_processor.analyze_asymmetry_across_patients()
                
                # Complete
                self.batch_status_var.set("Batch analysis complete")
                self.batch_progress_var.set(100)
                messagebox.showinfo("Complete", f"Batch analysis complete. Results saved to {self.output_dir.get()}")
                
                # Open output folder
                self.open_folder(self.output_dir.get())
            else:
                self.batch_status_var.set("No results generated")
                messagebox.showwarning("Warning", "No results were generated. Check logs for details.")
            
        except Exception as e:
            self.batch_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def open_results_folder(self):
        """Open the results folder."""
        if os.path.exists(self.output_dir.get()):
            self.open_folder(self.output_dir.get())
        else:
            messagebox.showinfo("Info", "Output directory does not exist yet")
    
    def open_folder(self, folder_path):
        """
        Open a folder in the file explorer.
        
        Args:
            folder_path (str): Path to folder
        """
        if os.path.exists(folder_path):
            # Cross-platform way to open folder
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                subprocess.run(['open', folder_path])
            else:  # Linux and other Unix-like
                import subprocess
                subprocess.run(['xdg-open', folder_path])
    
    def update_status(self, message):
        """
        Update status label.
        
        Args:
            message (str): Status message
        """
        self.root.after(0, lambda: self.status_var.set(message))
        
    def update_progress(self, value):
        """
        Update progress bar.
        
        Args:
            value (float): Progress value (0-100)
        """
        self.root.after(0, lambda: self.progress_var.set(value))
        
    def log_message(self, message):
        """
        Add message to results text.
        
        Args:
            message (str): Message to log
        """
        self.root.after(0, lambda: self.results_text.insert(tk.END, message + "\n"))
        self.root.after(0, lambda: self.results_text.see(tk.END))
