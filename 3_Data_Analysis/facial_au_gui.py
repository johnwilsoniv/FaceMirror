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
# Removed time import as it wasn't used
from facial_au_analyzer import FacialAUAnalyzer
from facial_au_batch_processor import FacialAUBatchProcessor
from facial_au_constants import PARALYSIS_SEVERITY_LEVELS

# Configure logging
logger = logging.getLogger(__name__) # Assuming configured by main

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
        # Adjusted size slightly
        self.root.geometry("550x480")

        # Initialize variables
        self.data_dir = tk.StringVar()
        self.output_dir = "../3.5_Results"
        # --- Variable for visuals checkbox ---
        self.generate_visuals_var = tk.BooleanVar(value=True) # Default to True
        # --- END Variable ---

        # Create GUI elements
        self.create_widgets()

        # Initialize batch processor (will be recreated on scan)
        self.batch_processor = None
        self.analysis_thread = None

    def create_widgets(self):
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Directory Selection ---
        dir_frame = ttk.LabelFrame(main_frame, text="Select Data Directory", padding="10")
        dir_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dir_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dir_frame, textvariable=self.data_dir, width=40).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse...", command=self.browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        dir_frame.columnconfigure(1, weight=1)

        # --- Batch Analysis Section ---
        batch_frame = ttk.LabelFrame(main_frame, text="Batch Analysis", padding="10")
        batch_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Patient List
        list_frame = ttk.Frame(batch_frame)
        # Expand list box more
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        ttk.Label(list_frame, text="Detected Patients:").pack(anchor=tk.NW)
        self.patients_listbox = tk.Listbox(list_frame, height=10, exportselection=False) # Increased height
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.patients_listbox.yview)
        self.patients_listbox.config(yscrollcommand=scrollbar.set)
        self.patients_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Options Frame ---
        options_frame = ttk.Frame(batch_frame)
        options_frame.pack(fill=tk.X, pady=5)

        paralysis_info_label = f"Paralysis Levels Used: {', '.join(PARALYSIS_SEVERITY_LEVELS)}"
        ttk.Label(options_frame, text=paralysis_info_label, font=("TkDefaultFont", 8)).pack(anchor=tk.W)

        # --- ADDED: Visuals Checkbox ---
        ttk.Checkbutton(
            options_frame,
            text="Generate Visual Outputs (Frames & Figures)",
            variable=self.generate_visuals_var,
            onvalue=True,
            offvalue=False
        ).pack(anchor=tk.W, pady=(5,0))
        # --- END ADDED ---

        # Progress Section
        progress_frame = ttk.Frame(batch_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(progress_frame, text="Progress:").pack(anchor=tk.W)
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(progress_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X)
        self.batch_status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.batch_status_var, wraplength=450).pack(anchor=tk.W, pady=(2,0))

        # --- Buttons Frame ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 5))
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1) # Equal weight
        ttk.Button(button_frame, text="Scan for Patients", command=self.scan_for_patients).grid(row=0, column=0, padx=10, sticky=tk.EW)
        ttk.Button(button_frame, text="Run Batch Analysis", command=self.run_batch_analysis).grid(row=0, column=1, padx=10, sticky=tk.EW)

    def browse_data_dir(self):
        """Browse for data directory and automatically scan after selection."""
        # ... (logic remains the same) ...
        initial_dir = os.path.expanduser("~"); docs_dir = os.path.join(initial_dir, "Documents")
        if os.path.isdir(docs_dir): initial_dir = docs_dir
        dirpath = filedialog.askdirectory(title="Select Data Directory", initialdir=initial_dir)
        if dirpath: self.data_dir.set(dirpath); self.scan_for_patients()


    def scan_for_patients(self):
        """Scan for patients in the data directory."""
        # ... (logic remains the same) ...
        data_dir_path = self.data_dir.get()
        if not data_dir_path: messagebox.showerror("Error", "Select data directory."); return
        if not os.path.isdir(data_dir_path): messagebox.showerror("Error", f"Invalid directory:\n{data_dir_path}"); return
        logger.info(f"Scanning for patients in: {data_dir_path}"); self.patients_listbox.delete(0, tk.END)
        self.batch_status_var.set("Scanning..."); self.root.update_idletasks()
        try:
            self.batch_processor = FacialAUBatchProcessor(self.output_dir); num_patients = self.batch_processor.auto_detect_patients(data_dir_path)
            if self.batch_processor.patients:
                 for patient in self.batch_processor.patients: self.patients_listbox.insert(tk.END, patient['patient_id'])
                 self.batch_status_var.set(f"Found {num_patients} patients. Ready.")
                 logger.info(f"Scan complete. Found {num_patients} patients.")
            else: self.batch_status_var.set("No patients found."); logger.warning("Scan complete. No patients found."); messagebox.showinfo("Scan Results", "No patient data files found.")
        except Exception as e: logger.error(f"Scan error: {e}", exc_info=True); messagebox.showerror("Scan Error", f"Error scanning:\n{e}"); self.batch_status_var.set("Scan Error.")


    def run_batch_analysis(self):
        """Run batch analysis for all detected patients."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Busy", "Analysis already running.")
            return
        if not self.batch_processor or not self.batch_processor.patients:
            messagebox.showerror("Error", "No patients detected. Scan first.")
            return

        # --- Get checkbox state ---
        generate_visuals = self.generate_visuals_var.get()
        logger.info(f"Starting GUI batch analysis. Generate visuals: {generate_visuals}")
        # --- End Get checkbox ---

        self.batch_progress_var.set(0)
        self.batch_status_var.set("Starting batch analysis...")
        self.root.update_idletasks()

        # Pass the flag to the thread target
        self.analysis_thread = threading.Thread(target=self.analyze_batch, args=(generate_visuals,))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    # --- Modified to accept the flag ---
    def analyze_batch(self, generate_visuals):
        """Run batch analysis in a separate thread."""
        try:
            total_patients = len(self.batch_processor.patients)
            logger.info(f"analyze_batch started for {total_patients} patients. Generate visuals: {generate_visuals}")
            self.batch_status_var.set(f"Processing {total_patients} patients...")
            self.batch_progress_var.set(5)
            self.root.update_idletasks()

            # --- Pass generate_visuals to extract_frames parameter ---
            output_path = self.batch_processor.process_all(
                extract_frames=generate_visuals
                # No need to pass results_file/expert_file
            )
            # --- End pass flag ---

            if output_path:
                self.batch_status_var.set("Performing final analysis...")
                self.batch_progress_var.set(95)
                self.root.update_idletasks()
                self.batch_processor.analyze_asymmetry_across_patients()

                self.batch_status_var.set(f"Batch complete! Results: {output_path}")
                self.batch_progress_var.set(100)
                logger.info("Batch analysis thread finished successfully.")
                self.root.after(0, lambda: messagebox.showinfo("Analysis Complete", f"Batch analysis successful!\nResults saved to:\n{output_path}"))
                # Consider NOT closing automatically from GUI run
                # self.root.after(1000, self.root.destroy)
            else:
                self.batch_status_var.set("Processing failed. Check logs.")
                logger.error("Batch processing failed to produce output.")
                self.root.after(0, lambda: messagebox.showerror("Error", "Batch processing failed. Check logs."))

        except Exception as e:
            logger.error(f"Error in analyze_batch thread: {e}", exc_info=True)
            self.batch_status_var.set(f"Error occurred: {e}")
            self.root.after(0, lambda: messagebox.showerror("Batch Error", f"Error during batch processing:\n{e}\nCheck logs."))
        finally:
             self.root.after(0, self.batch_progress_var.set, 100) # Ensure progress bar completes