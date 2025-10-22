
import sys
import os
from splash_screen import SplashScreen
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# ============================================================================
# PERFORMANCE PROFILING - Toggle this to enable/disable profiling
# ============================================================================
# Set to True to enable performance profiling (diagnose GUI choppiness)
# Set to False to run normally without profiling overhead
#
# When enabled:
#   - Monitors event loop latency (GUI responsiveness)
#   - Tracks signal/slot overhead
#   - Analyzes frame timing (video playback)
#   - Detects memory leaks
#   - Auto-generates report on exit with recommendations
#
# To use: Run app normally, close when done -> see analysis in terminal
# ============================================================================
ENABLE_PROFILING = False  # ← DISABLED for stability
# ============================================================================

# ============================================================================
# DIAGNOSTIC PROFILING - Component-level timing analysis
# ============================================================================
# Set to True to enable diagnostic profiling (detailed component timing)
# Set to False to run normally without diagnostic overhead
#
# When enabled:
#   - Tracks video frame extraction timing (seek, read, conversion)
#   - Monitors cache hit/miss rates (RGB + QImage caches)
#   - Records component-level breakdowns
#   - Auto-generates diagnostic report on exit
#
# To use: Run app normally, close when done -> see diagnostic_report_*.json
# ============================================================================
ENABLE_DIAGNOSTIC_PROFILING = False  # ← DISABLED for stability
# ============================================================================


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
        import config_paths
        import shutil
        ffmpeg_path = config_paths.get_ffmpeg_path()
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
    # Show splash screen with loading stages (ONLY in main process)
    splash = SplashScreen("Action Coder", "1.0.0")
    splash.show()

    # Stage 1: Loading frameworks
    splash.update_status("Loading frameworks...")

    # Stage 2: Initializing PyQt5
    splash.update_status("Initializing PyQt5...")

    add_exception_logging()
    configure_pydub()

    # Import BatchProcessor for monkey-patching (lightweight import)
    from batch_processor import BatchProcessor
    patch_batch_processor()

    # Set diagnostic profiling flag in config (so other modules can check it)
    import config
    config.ENABLE_DIAGNOSTIC_PROFILING = ENABLE_DIAGNOSTIC_PROFILING
    if ENABLE_DIAGNOSTIC_PROFILING:
        print("Diagnostic profiling ENABLED (component-level timing)")

    # Stage 3: Loading AI models (Whisper)
    splash.update_status("Loading AI models (Whisper)...")

    # Pre-load Whisper model during splash screen for better UX
    whisper_model_global = None
    try:
        # Check if faster-whisper is available
        from faster_whisper import WhisperModel
        import torch
        from pathlib import Path

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        # Check if model is already cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_exists = False
        if cache_dir.exists():
            # Look for any files containing "large-v3" in the cache
            model_files = list(cache_dir.glob("*large-v3*"))
            if model_files:
                model_cache_exists = True
                print(f"Whisper model found in cache: {len(model_files)} files")

        # Show appropriate message based on cache status
        if not model_cache_exists:
            splash.update_status("Downloading Whisper model (3GB)...\nThis may take 5-10 minutes on first run.\nSubsequent launches will be instant.")
        else:
            splash.update_status("Loading Whisper model from cache...")

        # Load the model (may download on first run)
        whisper_model_global = WhisperModel("large-v3", device=device, compute_type=compute_type)

        print(f"Whisper model pre-loaded successfully on {device}")
        splash.update_status("Whisper model loaded successfully!")

    except Exception as e:
        print(f"Warning: Could not pre-load Whisper model: {e}")
        print("Model will be loaded on-demand during processing.")
        whisper_model_global = None

    # Import ApplicationController (now that heavy imports are done)
    from app_controller import ApplicationController

    # Stage 4: Starting application
    splash.update_status("Starting application...")

    # Configure OpenGL for hardware acceleration (BEFORE QApplication)
    from PyQt5.QtGui import QSurfaceFormat
    gl_format = QSurfaceFormat()
    gl_format.setSwapInterval(1)  # Enable vsync for smooth rendering
    gl_format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)  # Double buffering
    QSurfaceFormat.setDefaultFormat(gl_format)

    # Set High DPI scaling BEFORE QApplication creation
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Enable OpenGL-based rendering for widgets
    if hasattr(Qt, 'AA_UseDesktopOpenGL'):
        QApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)

    app = QApplication(sys.argv)

    # === PERFORMANCE PROFILING SETUP ===
    profiler = None
    if ENABLE_PROFILING:
        try:
            from performance_profiler import PerformanceProfiler
            profiler = PerformanceProfiler(app)
            profiler.start_profiling()
            profiler.enable_frame_timing()
            print("Performance profiling ENABLED")
        except ImportError:
            print("Performance profiler not found (performance_profiler.py)")
            print("   Continuing without profiling...")
            profiler = None
        except Exception as e:
            print(f"Failed to start profiler: {e}")
            print("   Continuing without profiling...")
            profiler = None
    # === END PROFILING SETUP ===

    # Close splash screen before showing main window
    splash.close()

    controller = ApplicationController(app, whisper_model=whisper_model_global) # Pass app instance and pre-loaded model

    # === ATTACH PROFILER TO CONTROLLER ===
    if profiler:
        controller._profiler = profiler
        controller._frame_analyzer = profiler.frame_analyzer

        # Wrap frame update handler for timing
        original_handle_frame_update = controller._handle_frame_update
        def profiled_frame_update(frame_number, qimage):
            controller._frame_analyzer.mark_frame_start(frame_number)
            controller._frame_analyzer.detect_dropped_frame(frame_number)
            original_handle_frame_update(frame_number, qimage)
            controller._frame_analyzer.mark_frame_complete()
        controller._handle_frame_update = profiled_frame_update

        # Track critical signals
        try:
            profiler.track_signal(controller.playback_manager, 'frame_update_needed',
                                controller.playback_manager.frame_update_needed)
            profiler.track_signal(controller.processing_manager, 'processing_status_update',
                                controller.processing_manager.processing_status_update)
            profiler.track_signal(controller.action_tracker, 'action_ranges_changed',
                                controller.action_tracker.action_ranges_changed)
            print("Profiling integrated (3 signals tracked)")
        except Exception as e:
            print(f"Signal tracking failed: {e}")
    # === END PROFILER ATTACHMENT ===

    exit_code = 1 # Default error code
    if controller.window: # Check if window was created successfully
        exit_code = app.exec_()
        controller.cleanup_on_exit() # Perform cleanup

    # === GENERATE PERFORMANCE REPORTS ===
    if profiler:
        print("\n" + "="*70)
        print("Generating performance report...")
        print("="*70)
        profiler.stop_profiling()
        report = profiler.generate_report()

        # Auto-analyze if script is available
        try:
            import subprocess
            print("\nRunning automated analysis...")
            subprocess.run([sys.executable, "analyze_performance.py", "--latest"],
                         capture_output=False, text=True, timeout=30)
        except FileNotFoundError:
            print("\nTo analyze results, run:")
            print("  python analyze_performance.py --latest")
        except Exception as e:
            print(f"\nAuto-analysis failed: {e}")
            print("Run manually: python analyze_performance.py --latest")

        print("\n" + "="*70 + "\n")

    # === GENERATE DIAGNOSTIC PROFILER REPORT ===
    if ENABLE_DIAGNOSTIC_PROFILING:
        try:
            from diagnostic_profiler import get_diagnostic_profiler
            diagnostic_profiler = get_diagnostic_profiler()
            if diagnostic_profiler:
                print("\n" + "="*70)
                print("Generating diagnostic profiler report...")
                print("="*70)
                diagnostic_profiler.print_summary()
                diagnostic_profiler.save_report()
                print("="*70 + "\n")
        except ImportError:
            print("Diagnostic profiler not found (diagnostic_profiler.py)")
        except Exception as e:
            print(f"Failed to generate diagnostic report: {e}")
    # === END DIAGNOSTIC PROFILER REPORT ===
    # === END PERFORMANCE REPORTS ===

    sys.exit(exit_code)

