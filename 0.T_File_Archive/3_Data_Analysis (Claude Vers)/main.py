"""
Main entry point for the Facial AU Analyzer application.

This script launches the GUI application. It can also be used
to run batch analysis from the command line.

Usage:
    GUI mode:
        python main.py
        
    Command line batch mode:
        python main.py --batch --data-dir /path/to/data
"""

import argparse
import tkinter as tk
import logging
import sys
import os
from facial_au_gui import FacialAUAnalyzerGUI
from facial_au_batch_processor import FacialAUBatchProcessor
from facial_au_constants import PARALYSIS_SEVERITY_LEVELS

# Ensure log directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
LOG_LEVEL = logging.INFO  # Changed from DEBUG to INFO to reduce logging output

# Create a simplified formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)

# Create console handler - only showing WARNING and above to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
console_handler.setFormatter(formatter)

# Create file handler
file_handler = logging.FileHandler(os.path.join(log_dir, 'facial_au_analyzer.log'))
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Configure specific loggers
logger = logging.getLogger(__name__)
logger.info("================ STARTING APPLICATION ================")
logger.info(f"Logging level: {LOG_LEVEL}")

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial AU Analyzer')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode (no GUI)')
    parser.add_argument('--data-dir', type=str, help='Directory containing data files')
    
    args = parser.parse_args()
    
    # Default output directory
    output_dir = "../3.5_Results"
    
    # Run in batch mode if specified
    if args.batch:
        if not args.data_dir:
            logger.error("Data directory must be specified in batch mode")
            parser.print_help()
            return 1
        
        logger.info(f"Running in batch mode. Data directory: {args.data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using enhanced paralysis detection with severity levels: {', '.join(PARALYSIS_SEVERITY_LEVELS)}")
        
        # Create batch processor
        processor = FacialAUBatchProcessor(output_dir)
        
        # Detect patients
        num_patients = processor.auto_detect_patients(args.data_dir)
        logger.info(f"Detected {num_patients} patients")
        
        if num_patients == 0:
            logger.error("No patients detected")
            return 1
        
        # Process all patients - always extract frames
        output_path = processor.process_all(extract_frames=True)
        
        if output_path:
            # Analyze asymmetry across patients
            processor.analyze_asymmetry_across_patients()
            logger.info(f"Batch analysis complete. Results saved to {output_dir}")
            return 0
        else:
            logger.error("Failed to process patients")
            return 1
    
    # Run in GUI mode
    else:
        logger.info("Starting GUI application")
        root = tk.Tk()
        app = FacialAUAnalyzerGUI(root)
        root.mainloop()
        return 0

if __name__ == "__main__":
    sys.exit(main())
