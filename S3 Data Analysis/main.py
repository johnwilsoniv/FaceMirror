"""
Main entry point for the Facial AU Analyzer application.
This script launches the GUI application. It can also be used
to run batch analysis from the command line.

Usage:
    GUI mode:
        python main.py

    Command line batch mode:
        python main.py --batch --data-dir /path/to/data [--skip-visuals]
"""

import argparse
import tkinter as tk
import logging
import sys
import os
import config_paths

# Get the directory where the currently running script (main.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Verify this path matches your directory structure
# Construct the path to the directory containing facial_au_analyzer.py
# Assumption: facial_au_analyzer.py is in the SAME directory as main.py
analyzer_dir = script_dir
# OR if it's in a subdirectory named 'analysis_modules':
# analyzer_dir = os.path.join(script_dir, 'analysis_modules')
# OR if main.py is one level ABOVE the directory containing the analyzer:
# analyzer_dir = os.path.join(script_dir, '3_Data_Analysis') # Adjust '3_Data_Analysis' if needed

# Prepend this directory to the Python path if it's not already there
if os.path.isdir(analyzer_dir) and analyzer_dir not in sys.path:
    sys.path.insert(0, analyzer_dir)
    print(f"DEBUG: Added '{analyzer_dir}' to sys.path")
elif not os.path.isdir(analyzer_dir):
    print(f"DEBUG WARNING: Constructed analyzer_dir '{analyzer_dir}' does not exist. Check path logic.", file=sys.stderr)

try:
    from facial_au_gui import FacialAUAnalyzerGUI
    from facial_au_batch_processor import FacialAUBatchProcessor
    from facial_au_constants import PARALYSIS_SEVERITY_LEVELS
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary project modules: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1) # Exit if essential imports fail


# Ensure log directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
LOG_LEVEL = logging.INFO # Set back to DEBUG for detailed logs during testing

# Create a simplified formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s [%(filename)s:%(lineno)d] - %(message)s' # Added filename/lineno
)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG) # Show INFO and above on console (can change to DEBUG)
console_handler.setFormatter(formatter)

# Create file handler
log_file_path = os.path.join(log_dir, 'facial_au_analyzer.log')
file_handler = None # Initialize to None
try:
    # Use 'w' mode to overwrite the log file for each new run
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') # Added encoding
    file_handler.setLevel(LOG_LEVEL) # File handler gets the most detailed level
    file_handler.setFormatter(formatter)
except PermissionError:
    print(f"Error: Permission denied to write log file at {log_file_path}. Check directory permissions.", file=sys.stderr)
except Exception as e_log:
     print(f"Error creating file handler for logging: {e_log}", file=sys.stderr)


# Configure root logger
root_logger = logging.getLogger()
# Remove existing handlers to prevent duplicate messages
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
root_logger.setLevel(LOG_LEVEL) # Set root logger to the most detailed level
root_logger.addHandler(console_handler) # Add console handler
if file_handler:
    root_logger.addHandler(file_handler) # Add file handler if created successfully
else:
     print("Warning: File logging is disabled.", file=sys.stderr)

# Set matplotlib logger level higher to suppress font messages
logging.getLogger('matplotlib').setLevel(logging.INFO)

# Configure specific loggers
logger = logging.getLogger(__name__) # Logger for this main module
logger.info("=" * 60)
logger.info(f"{config_paths.APP_NAME} v{config_paths.VERSION}")
logger.info("=" * 60)
logger.info(f"Root logging level set to: {logging.getLevelName(LOG_LEVEL)}")
if file_handler: logger.info(f"Logging to file: {log_file_path}")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial AU Analyzer')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode (no GUI)')
    parser.add_argument('--data-dir', type=str, help='Directory containing patient data CSV/video files')
    parser.add_argument('--skip-visuals', action='store_true', help='Skip generation of frame images and plots')

    args = parser.parse_args()

    # Default output directory - use config_paths for cross-platform compatibility
    output_dir = str(config_paths.get_output_base_dir())

    # Run in batch mode if specified
    if args.batch:
        if not args.data_dir:
            logger.error("--data-dir must be specified in batch mode")
            parser.print_help()
            return 1
        if not os.path.isdir(args.data_dir):
             logger.error(f"Specified data directory does not exist: {args.data_dir}")
             return 1

        # Determine whether to generate visuals based on the flag
        generate_visuals_flag = not args.skip_visuals # True if --skip-visuals is NOT present
        logger.info(f"Running in batch mode. Data directory: {args.data_dir}")
        logger.info(f"Output directory: {os.path.abspath(output_dir)}") # Log absolute path
        logger.info(f"Generate Visual Outputs: {generate_visuals_flag}")
        logger.info(f"Paralysis detection levels defined: {', '.join(PARALYSIS_SEVERITY_LEVELS)}")

        # Create batch processor
        try:
             processor = FacialAUBatchProcessor(output_dir)
        except Exception as e:
             logger.error(f"Failed to initialize batch processor: {e}", exc_info=True)
             return 1

        # Detect patients
        num_patients = processor.auto_detect_patients(args.data_dir)
        logger.info(f"Detected {num_patients} patients")

        if num_patients == 0:
            logger.warning("No patients detected in the specified directory. Exiting.")
            return 1 # Exit if no patients found

        # Process all patients, passing the visual flag
        output_path = processor.process_all(
            extract_frames=generate_visuals_flag
        )

        if output_path:
            # Analyze results using the generated output file
            try:
                 processor.analyze_asymmetry_across_patients()
            except Exception as e_analysis:
                 logger.error(f"Error during post-processing analysis: {e_analysis}", exc_info=True)
            logger.info(f"Batch analysis complete. Final results saved to {output_path}")
            return 0
        else:
            logger.error("Batch processing failed to generate an output file.")
            return 1

    # Run in GUI mode
    else:
        logger.info("Starting GUI application")
        try:
            root = tk.Tk()
            app = FacialAUAnalyzerGUI(root)
            root.mainloop()
            return 0
        except Exception as e_gui:
             logger.error(f"Error starting GUI: {e_gui}", exc_info=True)
             print("\nError: Failed to start GUI. Check logs for details.", file=sys.stderr)
             return 1

if __name__ == "__main__":
    # Wrap main call in try-except block for better top-level error reporting
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e_main:
         logger.critical(f"Critical error in main execution: {e_main}", exc_info=True)
         print(f"\nA critical error occurred: {e_main}. Check logs.", file=sys.stderr)
         sys.exit(1)