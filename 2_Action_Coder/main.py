# --- START OF FILE main.py ---

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt # *** ADDED IMPORT ***
import os

# Exception logging function (keep here or move to utils)
def add_exception_logging():
    # (Code identical to previous version)
    import traceback, datetime
    os.makedirs("logs", exist_ok=True); original_excepthook = sys.excepthook
    def exception_handler(exc_type, exc_value, exc_traceback):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); log_file = os.path.join("logs", f"crash_{timestamp}.log")
        exception_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        with open(log_file, "w") as f: f.write(f"Timestamp: {timestamp}\nException: {exc_type.__name__}: {exc_value}\nTraceback:\n{exception_text}")
        original_excepthook(exc_type, exc_value, exc_traceback)
        try:
            if QApplication.instance():
                 from PyQt5.QtWidgets import QMessageBox
                 QMessageBox.critical( None, "Application Error", f"Unexpected error: {exc_type.__name__}: {exc_value}\n\nDetails logged to {log_file}")
        except Exception as msg_e: print(f"Error showing critical message box: {msg_e}")
    sys.excepthook = exception_handler; print("Exception logging enabled.")

# Import the main controller
from app_controller import ApplicationController
from batch_processor import BatchProcessor # Needed for monkey-patching

# Pydub/FFmpeg configuration (keep here or move to utils)
def configure_pydub():
    try:
        from pydub import AudioSegment
        PYDUB_AVAILABLE = True
        print("pydub library found.")
    except ImportError:
        PYDUB_AVAILABLE = False
        print("Warning: pydub missing. Snippet playback disabled.")
        return # Cannot configure if not available

    try:
        from app_controller import find_ffmpeg_path # Import helper
        import shutil
        ffmpeg_path = find_ffmpeg_path()
        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path and sys.platform == 'darwin' and os.path.exists('/opt/homebrew/bin/ffprobe'): ffprobe_path = '/opt/homebrew/bin/ffprobe'

        if ffmpeg_path: print(f"Config pydub: ffmpeg at {ffmpeg_path}"); AudioSegment.converter = ffmpeg_path
        else: print("WARN: ffmpeg not found for pydub.")
        if ffprobe_path: print(f"Config pydub: ffprobe at {ffprobe_path}"); AudioSegment.ffprobe = ffprobe_path
        else: print("WARN: ffprobe not found for pydub.")
    except Exception as e: print(f"WARN: Could not config ffmpeg/ffprobe for pydub: {e}")

# Monkey-patch BatchProcessor (keep here or move to utils)
def patch_batch_processor():
    original_find_files = BatchProcessor.find_matching_files
    original_find_files_for_videos = BatchProcessor.find_matching_files_for_videos

    # *** FIXED MUTABLE DEFAULT ARGUMENT ***
    def filtered_find_files(self, directory, rejected_suffixes=None):
        if rejected_suffixes is None:
            rejected_suffixes = ["_annotated", "_coded"]
        # *** END FIX ***
        all_file_sets = original_find_files(self, directory)
        valid_file_sets = []; processed_suffixes_lower = [s.lower() for s in rejected_suffixes]
        for fs in all_file_sets:
            is_rejected = False
            for key in ['video', 'csv1', 'csv2']:
                if fs.get(key) and any(suffix in os.path.basename(fs[key]).lower() for suffix in processed_suffixes_lower): is_rejected = True; break
            if not is_rejected: valid_file_sets.append(fs)
        return valid_file_sets

    # *** FIXED MUTABLE DEFAULT ARGUMENT ***
    def filtered_find_files_for_videos(self, video_files, search_directory=None, rejected_suffixes=None):
        if rejected_suffixes is None:
            rejected_suffixes = ["_annotated", "_coded"]
        # *** END FIX ***
        all_file_sets = original_find_files_for_videos(self, video_files, search_directory)
        valid_file_sets = []; processed_suffixes_lower = [s.lower() for s in rejected_suffixes]
        for fs in all_file_sets:
             is_rejected = False
             base_video_name = os.path.basename(fs['video']).lower() if fs.get('video') else ''; csv1_name = os.path.basename(fs['csv1']).lower() if fs.get('csv1') else ''; csv2_name = os.path.basename(fs['csv2']).lower() if fs.get('csv2') else ''
             if fs.get('video') and any(suffix in base_video_name for suffix in processed_suffixes_lower): is_rejected = True
             if not is_rejected and fs.get('csv1') and any(suffix in csv1_name for suffix in processed_suffixes_lower): is_rejected = True
             if not is_rejected and fs.get('csv2') and any(suffix in csv2_name for suffix in processed_suffixes_lower): is_rejected = True
             if not is_rejected: valid_file_sets.append(fs)
        return valid_file_sets

    BatchProcessor.find_matching_files = filtered_find_files
    BatchProcessor.find_matching_files_for_videos = filtered_find_files_for_videos
    print("BatchProcessor methods patched for file filtering.")


# --- Main Execution ---
if __name__ == "__main__":
    add_exception_logging()
    configure_pydub()
    patch_batch_processor()

    app = QApplication(sys.argv)
    # Set High DPI scaling based on Qt version
    # *** Qt is now imported, these should work ***
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    controller = ApplicationController(app) # Pass app instance

    exit_code = 1 # Default error code
    if controller.window: # Check if window was created successfully
        exit_code = app.exec_()
        controller.cleanup_on_exit() # Perform cleanup

    sys.exit(exit_code)

# --- END OF FILE main.py ---