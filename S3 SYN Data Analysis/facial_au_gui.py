"""
GUI for the Facial AU Analyzer application.
V1.11 Update: Removed Scan button, added Debug Mode checkbox, fixed layout by removing fixed geometry.
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
        # Let Tkinter determine initial size
        # self.root.geometry("...")

        # Initialize variables
        self.data_dir = tk.StringVar()
        self.output_dir = "../S3O Results"
        self.generate_visuals_var = tk.BooleanVar(value=True)
        self.debug_mode_var = tk.BooleanVar(value=False)

        # Create GUI elements
        self.create_widgets()

        # Initialize batch processor
        self.batch_processor = None
        self.analysis_thread = None

    def create_widgets(self):
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Directory Selection
        dir_frame = ttk.LabelFrame(main_frame, text="Select Data Directory", padding="10")
        dir_frame.pack(fill=tk.X, pady=5, side=tk.TOP, anchor='n')
        ttk.Label(dir_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dir_frame, textvariable=self.data_dir, width=40).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse...", command=self.browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        dir_frame.columnconfigure(1, weight=1)

        # Batch Analysis Section
        batch_frame = ttk.LabelFrame(main_frame, text="Batch Analysis", padding="10")
        batch_frame.pack(fill=tk.BOTH, expand=True, pady=5, side=tk.TOP)

        # Patient List Frame
        list_frame = ttk.Frame(batch_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        ttk.Label(list_frame, text="Detected Patients:").pack(anchor=tk.NW)
        self.patients_listbox = tk.Listbox(list_frame, height=10, exportselection=False)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.patients_listbox.yview)
        self.patients_listbox.config(yscrollcommand=scrollbar.set)
        self.patients_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Options Frame
        options_frame = ttk.Frame(batch_frame)
        options_frame.pack(fill=tk.X, pady=5)
        paralysis_info_label = f"Paralysis Levels Used: {', '.join(PARALYSIS_SEVERITY_LEVELS)}"
        ttk.Label(options_frame, text=paralysis_info_label, font=("TkDefaultFont", 8)).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Generate Visual Outputs (Frames & Figures)", variable=self.generate_visuals_var, onvalue=True, offvalue=False).pack(anchor=tk.W, pady=(5,0))
        ttk.Checkbutton(options_frame, text="Debug Mode (Compare results with 'FPRS FP Key.csv' in data dir)", variable=self.debug_mode_var, onvalue=True, offvalue=False).pack(anchor=tk.W, pady=(2,0))

        # Progress Section
        progress_frame = ttk.Frame(batch_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        ttk.Label(progress_frame, text="Progress:").pack(anchor=tk.W)
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(progress_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X)
        self.batch_status_var = tk.StringVar(value="Ready (Select Data Directory)")
        ttk.Label(progress_frame, textvariable=self.batch_status_var, wraplength=450).pack(anchor=tk.W, pady=(2,0))

        # Button Frame (at the bottom)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 5), anchor='s')
        button_frame.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(button_frame, text="Run Batch Analysis", command=self.run_batch_analysis)
        self.run_button.grid(row=0, column=0, padx=10, pady=5, sticky=tk.EW)
        self.run_button.config(state=tk.DISABLED)

    def browse_data_dir(self):
        """Browse for data directory and automatically scan."""
        initial_dir = os.path.expanduser("~"); docs_dir = os.path.join(initial_dir, "Documents")
        if os.path.isdir(docs_dir): initial_dir = docs_dir
        dirpath = filedialog.askdirectory(title="Select Data Directory", initialdir=initial_dir)
        if dirpath:
            self.data_dir.set(dirpath)
            self.scan_for_patients_auto() # Auto-scan

    def scan_for_patients_auto(self):
        """Scan for patients automatically."""
        data_dir_path = self.data_dir.get()
        if not data_dir_path or not os.path.isdir(data_dir_path):
             self.run_button.config(state=tk.DISABLED)
             return

        logger.info(f"Scanning for patients in: {data_dir_path}")
        self.patients_listbox.delete(0, tk.END)
        self.batch_status_var.set("Scanning...")
        self.root.update_idletasks()

        try:
            self.batch_processor = FacialAUBatchProcessor(self.output_dir)
            num_patients = self.batch_processor.auto_detect_patients(data_dir_path)
            if self.batch_processor.patients:
                 for patient in self.batch_processor.patients: self.patients_listbox.insert(tk.END, patient['patient_id'])
                 self.batch_status_var.set(f"Found {num_patients} patients. Ready to Run.")
                 self.run_button.config(state=tk.NORMAL)
                 logger.info(f"Scan complete. Found {num_patients} patients.")
            else:
                 self.batch_status_var.set("No patient data pairs found in directory.")
                 self.run_button.config(state=tk.DISABLED)
                 logger.warning("Scan complete. No patients found.")
        except Exception as e:
            logger.error(f"Error during automatic scan: {e}", exc_info=True)
            messagebox.showerror("Scan Error", f"Error scanning directory:\n{e}")
            self.batch_status_var.set("Scan Error.")
            self.run_button.config(state=tk.DISABLED)


    def run_batch_analysis(self):
        """Run batch analysis for all detected patients."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Busy", "Analysis already running.")
            return
        if not self.batch_processor or not self.batch_processor.patients:
            messagebox.showerror("Error", "No patients detected. Select a valid data directory.")
            return

        generate_visuals = self.generate_visuals_var.get()
        debug_mode = self.debug_mode_var.get() # Get debug mode state
        logger.info(f"Starting GUI batch analysis. Visuals: {generate_visuals}, Debug: {debug_mode}")

        self.batch_progress_var.set(0)
        self.batch_status_var.set("Starting batch analysis...")
        self.run_button.config(state=tk.DISABLED)
        self.root.update_idletasks()

        # Pass flags to the thread target
        self.analysis_thread = threading.Thread(target=self.analyze_batch, args=(generate_visuals, debug_mode))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        self.root.after(100, self.check_analysis_thread)

    def analyze_batch(self, generate_visuals, debug_mode):
        """Run batch analysis in a separate thread."""
        try:
            total_patients = len(self.batch_processor.patients)
            logger.info(f"analyze_batch started for {total_patients} patients. Visuals: {generate_visuals}, Debug: {debug_mode}")

            def update_progress(current, total):
                 progress = int((current / total) * 90)
                 self.root.after(0, self.batch_progress_var.set, progress)
                 status_text = f"Processing patient {current}/{total}..."
                 self.root.after(0, self.batch_status_var.set, status_text)

            # Pass flags and callback to process_all
            output_path = self.batch_processor.process_all(
                extract_frames=generate_visuals,
                debug_mode=debug_mode, # Pass debug flag
                progress_callback=update_progress
            )

            if output_path:
                self.root.after(0, self.batch_status_var.set, "Performing final analysis...")
                self.root.after(0, self.batch_progress_var.set, 95)
                self.batch_processor.analyze_asymmetry_across_patients()

                self.root.after(0, self.batch_status_var.set, f"Batch complete! Results: {output_path}")
                self.root.after(0, self.batch_progress_var.set, 100)
                logger.info("Batch analysis thread finished successfully.")
                self.root.after(0, lambda: messagebox.showinfo("Analysis Complete", f"Batch analysis successful!\nResults saved to:\n{output_path}"))
            else:
                self.root.after(0, self.batch_status_var.set, "Processing failed. Check logs.")
                self.root.after(0, self.batch_progress_var.set, 100)
                logger.error("Batch processing failed to produce output.")
                self.root.after(0, lambda: messagebox.showerror("Error", "Batch processing failed. Check logs."))

        except Exception as e:
            logger.error(f"Error in analyze_batch thread: {e}", exc_info=True)
            self.root.after(0, self.batch_status_var.set, f"Error occurred: {e}")
            self.root.after(0, lambda: messagebox.showerror("Batch Error", f"Error during batch processing:\n{e}\nCheck logs."))
        finally:
             self.root.after(0, self.batch_progress_var.set, 100)
             self.root.after(0, self.run_button.configure, {'state': tk.NORMAL}) # Re-enable button


    def check_analysis_thread(self):
        """Periodically check if the analysis thread is still running."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.root.after(100, self.check_analysis_thread)
        else:
            # Ensure button is enabled once thread definitely finishes
            # Use configure() which is more standard than config() for widgets
            self.run_button.configure(state=tk.NORMAL)
            logger.debug("Analysis thread check: Thread finished.")