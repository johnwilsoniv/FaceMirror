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

import sys
import os
from splash_screen import SplashScreen
import argparse
import tkinter as tk
import logging
import config_paths

# Get the directory where the currently running script (main.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Handle PyInstaller bundled app - add _MEIPASS to sys.path for local modules
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running as bundled app - local .py modules are in _MEIPASS
    meipass_dir = sys._MEIPASS
    if meipass_dir not in sys.path:
        sys.path.insert(0, meipass_dir)
        print(f"DEBUG: Added PyInstaller _MEIPASS '{meipass_dir}' to sys.path")

    # Also add parent Frameworks directory for macOS app bundles
    # This ensures dynamically imported modules (lower_face_features, etc.) can be found
    frameworks_dir = os.path.dirname(meipass_dir) if meipass_dir.endswith('MacOS') else meipass_dir
    if 'Frameworks' not in meipass_dir:
        # Check if Frameworks directory exists alongside
        potential_frameworks = os.path.join(os.path.dirname(meipass_dir), 'Frameworks')
        if os.path.isdir(potential_frameworks) and potential_frameworks not in sys.path:
            sys.path.insert(0, potential_frameworks)
            print(f"DEBUG: Added Frameworks directory '{potential_frameworks}' to sys.path")

    analyzer_dir = meipass_dir
else:
    # Running from source - use script directory
    analyzer_dir = script_dir
    if os.path.isdir(analyzer_dir) and analyzer_dir not in sys.path:
        sys.path.insert(0, analyzer_dir)
        print(f"DEBUG: Added '{analyzer_dir}' to sys.path")

try:
    from facial_au_gui import FacialAUAnalyzerGUI
    from facial_au_batch_processor import FacialAUBatchProcessor
    from facial_au_constants import PARALYSIS_SEVERITY_LEVELS, USE_ML_FOR_LOWER_FACE, USE_ML_FOR_MIDFACE, USE_ML_FOR_UPPER_FACE
    from paralysis_detector import ParalysisDetector
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary project modules: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1) # Exit if essential imports fail


# Ensure log directory exists (use config_paths for proper location)
log_dir = str(config_paths.get_logs_dir())

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


def preload_ml_detectors(splash=None):
    """
    Pre-load ML paralysis detectors during startup.
    Returns dict of {zone: ParalysisDetector} for enabled zones.
    """
    detectors = {}
    zones_to_load = []

    if USE_ML_FOR_LOWER_FACE:
        zones_to_load.append('lower')
    if USE_ML_FOR_MIDFACE:
        zones_to_load.append('mid')
    if USE_ML_FOR_UPPER_FACE:
        zones_to_load.append('upper')

    if not zones_to_load:
        logger.info("No ML detectors enabled, skipping pre-load")
        return detectors

    logger.info(f"Pre-loading ML detectors for zones: {zones_to_load}")

    for zone in zones_to_load:
        if splash:
            splash.update_status(f"Loading {zone} face model...")

        try:
            detectors[zone] = ParalysisDetector(zone=zone)
            logger.info(f"Successfully pre-loaded {zone} face detector")
        except ValueError as ve:
            logger.error(f"Failed to initialize ParalysisDetector for {zone}: {ve}")
        except Exception as e_pd:
            logger.error(f"Failed to initialize ParalysisDetector for {zone}: {e_pd}", exc_info=True)

    logger.info(f"Pre-loaded {len(detectors)}/{len(zones_to_load)} ML detectors")
    return detectors


def main():
    """Main entry point."""
    # Show splash screen (ONLY in main process)
    splash = SplashScreen("Data Analysis", "1.0.0")
    splash.show()

    # Stage 1: Loading frameworks
    splash.update_status("Loading frameworks...")

    # Stage 2: Pre-load ML models (actual work now!)
    preloaded_detectors = preload_ml_detectors(splash)

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
        # Close splash screen for batch mode
        splash.close()

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

        # Create batch processor with pre-loaded detectors
        try:
             processor = FacialAUBatchProcessor(output_dir, shared_detectors=preloaded_detectors)
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
            # Close splash screen completely (like S1 does)
            splash.close()

            # Create fresh Tk instance for main GUI (ensures native macOS styling)
            root = tk.Tk()

            app = FacialAUAnalyzerGUI(root, shared_detectors=preloaded_detectors)

            # Ensure window is visible
            root.deiconify()
            root.lift()
            root.update()
            root.focus_force()

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