"""
GUI for the Facial AU Analyzer application.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import platform
import logging
import re
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
    GUI for Facial AU Analyzer with batch analysis mode.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root (tk.Tk): Root window
        """
        self.root = root
        self.root.title("Facial AU Analyzer")
        self.root.geometry("500x400")  # Smaller window size
        
        # Initialize variables
        self.data_dir = tk.StringVar()
        self.output_dir = "../3.5_Results"  # Fixed output directory
        
        # Create GUI elements
        self.create_widgets()
        
        # Initialize batch processor
        self.batch_processor = FacialAUBatchProcessor(self.output_dir)
        self.analysis_thread = None
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame with reduced padding
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Directory selection frame
        dir_frame = ttk.LabelFrame(main_frame, text="Select Data Directory", padding="5")
        dir_frame.pack(fill=tk.X, pady=5)
        
        # Data directory
        ttk.Label(dir_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(dir_frame, textvariable=self.data_dir, width=30).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse...", command=self.browse_data_dir).grid(row=0, column=2, padx=5, pady=2)
        
        # Batch Analysis frame
        batch_analysis_frame = ttk.LabelFrame(main_frame, text="Batch Analysis", padding="5")
        batch_analysis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Detected patients label
        ttk.Label(batch_analysis_frame, text="Detected Patients:").pack(anchor=tk.W, pady=2)
        
        # Create patients listbox directly in the batch_analysis_frame (without a separate frame)
        self.patients_listbox = tk.Listbox(batch_analysis_frame, height=6, borderwidth=1, relief=tk.SOLID)
        self.patients_listbox.pack(fill=tk.BOTH, expand=True, pady=2, padx=0)
        
        # Add scrollbar directly to the listbox
        scrollbar = ttk.Scrollbar(self.patients_listbox, command=self.patients_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.patients_listbox.config(yscrollcommand=scrollbar.set)
        
        # Progress label
        ttk.Label(batch_analysis_frame, text="Progress:").pack(anchor=tk.W, pady=2)
        
        # Batch progress bar
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(batch_analysis_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X, pady=2)
        
        # Batch status label
        self.batch_status_var = tk.StringVar(value="Ready")
        ttk.Label(batch_analysis_frame, textvariable=self.batch_status_var).pack(anchor=tk.W, pady=2)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Scan for patients button
        ttk.Button(
            button_frame, 
            text="Scan for Patients", 
            command=self.scan_for_patients
        ).pack(side=tk.LEFT, padx=5)
        
        # Run batch analysis button
        ttk.Button(
            button_frame, 
            text="Run Batch Analysis", 
            command=self.run_batch_analysis
        ).pack(side=tk.RIGHT, padx=5)
    
    def browse_data_dir(self):
        """Browse for data directory and automatically scan after selection."""
        # Determine Documents folder based on operating system
        if platform.system() == 'Windows':
            initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
        elif platform.system() == 'Darwin':  # macOS
            initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
        else:  # Linux and other Unix-like systems
            initial_dir = os.path.join(os.path.expanduser("~"), "Documents")
            
        # If Documents folder doesn't exist, fall back to home directory
        if not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
            
        dirpath = filedialog.askdirectory(title="Select Data Directory", initialdir=initial_dir)
        if dirpath:
            self.data_dir.set(dirpath)
            # Automatically scan for patients after directory selection
            self.scan_for_patients()
    
    def scan_for_patients(self):
        """Scan for patients in the data directory."""
        if not self.data_dir.get():
            messagebox.showerror("Error", "Please select a data directory")
            return
        
        # Clear listbox
        self.patients_listbox.delete(0, tk.END)
        
        # Create new batch processor
        self.batch_processor = FacialAUBatchProcessor(self.output_dir)
        
        # Detect patients
        num_patients = self.batch_processor.auto_detect_patients(self.data_dir.get())
        
        # Update listbox - show full patient IDs (IMG_XXXX format)
        for patient in self.batch_processor.patients:
            # Use the full patient ID without trying to extract just the number
            patient_id = patient['patient_id']
            
            # If it doesn't already have the IMG_ prefix, try to extract it from the filename
            if not patient_id.startswith("IMG_"):
                filename = os.path.basename(patient['left_csv'])
                match = re.search(r'(IMG_\d+)', filename)
                if match:
                    patient_id = match.group(1)
            
            self.patients_listbox.insert(tk.END, patient_id)
        
        # Update status without showing popup
        if num_patients > 0:
            self.batch_status_var.set(f"Found {num_patients} patients")
        else:
            self.batch_status_var.set("No patients found")
            messagebox.showinfo("Patient Detection", "No patients found in the selected directory")
    
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
    
    def analyze_batch(self):
        """Run batch analysis for all patients."""
        try:
            # Update status
            self.batch_status_var.set("Processing batch...")
            self.batch_progress_var.set(10)
            
            # Process all patients - always extract frames
            output_path = self.batch_processor.process_all(extract_frames=True)
            
            if output_path:
                # Calculate asymmetry across patients
                self.batch_status_var.set("Analyzing asymmetry patterns...")
                self.batch_progress_var.set(90)
                self.batch_processor.analyze_asymmetry_across_patients()
                
                # Complete - update status without showing popup
                self.batch_status_var.set(f"Batch analysis complete. Results saved to {self.output_dir}")
                self.batch_progress_var.set(100)
            else:
                self.batch_status_var.set("No results generated")
                messagebox.showwarning("Warning", "No results were generated. Check logs for details.")
            
        except Exception as e:
            self.batch_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
