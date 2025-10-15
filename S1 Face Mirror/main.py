"""
S1 Face Mirror - Main Entry Point
Version 2.0 - Multi-Mode Processing

Provides three processing workflows:
1. Mirror + OpenFace: Complete pipeline (rotate, split, mirror, AU extraction)
2. Mirror Only: Video processing without AU extraction
3. OpenFace Only: Batch process existing mirrored videos for AU analysis
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from multiprocessing import cpu_count
import gc
import torch

# Import workflow components
from mode_selection_window import show_mode_selection
from face_splitter import StableFaceSplitter
from openface_integration import OpenFace3Processor
from progress_window import ProcessingProgressWindow, ProgressUpdate, OpenFaceProgressWindow, OpenFacePatientUpdate


def auto_detect_device():
    """
    Auto-detect best available device for processing

    Note: MPS is not supported by OpenFace 3.0 face detection models.
    Falls back to CPU on Apple Silicon.

    Returns:
        str: 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {device_name}")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Apple MPS detected, but not supported by face detection models")
        print("Falling back to CPU for compatibility")
        return 'cpu'
    else:
        print("Using CPU (no GPU detected)")
        return 'cpu'


def check_existing_outputs(input_path, output_dir, openface_output_dir=None):
    """
    Check if output files already exist for a given input video.

    Args:
        input_path: Path to input video file
        output_dir: Path to Face Mirror 1.0 Output directory
        openface_output_dir: Path to Combined Data directory (optional, for mode 1)

    Returns:
        dict with 'has_mirror_outputs' and 'has_openface_outputs' bools
    """
    input_file = Path(input_path)
    base_name = input_file.stem
    output_dir = Path(output_dir)

    # Check for mirror outputs (source, left, right)
    mirror_files_exist = True
    for suffix in ['_source', '_left_mirrored', '_right_mirrored']:
        found = False
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            if (output_dir / f"{base_name}{suffix}{ext}").exists():
                found = True
                break
        if not found:
            mirror_files_exist = False
            break

    # Check for OpenFace CSV outputs if directory provided
    openface_files_exist = False
    if openface_output_dir:
        openface_output_dir = Path(openface_output_dir)
        left_csv = openface_output_dir / f"{base_name}_left_mirrored.csv"
        right_csv = openface_output_dir / f"{base_name}_right_mirrored.csv"
        openface_files_exist = left_csv.exists() and right_csv.exists()

    return {
        'has_mirror_outputs': mirror_files_exist,
        'has_openface_outputs': openface_files_exist
    }


def filter_existing_outputs(input_paths, output_dir, openface_output_dir=None, mode_name=""):
    """
    Check which input files have existing outputs and prompt user.

    Args:
        input_paths: List of input video paths
        output_dir: Path to Face Mirror 1.0 Output directory
        openface_output_dir: Path to Combined Data directory (optional, for mode 1)
        mode_name: Name of the mode (for display purposes)

    Returns:
        List of input paths to process (filtered based on user choice)
    """
    # Check each input file for existing outputs
    files_with_outputs = []
    files_without_outputs = []

    for input_path in input_paths:
        status = check_existing_outputs(input_path, output_dir, openface_output_dir)

        # Determine if file has complete outputs
        if openface_output_dir:
            # Mode 1: needs both mirror and OpenFace outputs
            has_complete_outputs = status['has_mirror_outputs'] and status['has_openface_outputs']
        else:
            # Mode 2: only needs mirror outputs
            has_complete_outputs = status['has_mirror_outputs']

        if has_complete_outputs:
            files_with_outputs.append(input_path)
        else:
            files_without_outputs.append(input_path)

    # If no files have existing outputs, process all
    if not files_with_outputs:
        return list(input_paths)

    # Show dialog asking user what to do
    if files_without_outputs:
        # Some files have outputs, some don't
        message = (
            f"Found existing outputs for {len(files_with_outputs)} of {len(input_paths)} selected files.\n\n"
            f"Files with existing outputs:\n"
        )
        for f in files_with_outputs[:3]:
            message += f"  • {Path(f).name}\n"
        if len(files_with_outputs) > 3:
            message += f"  ... and {len(files_with_outputs) - 3} more\n"

        message += f"\n\nHow would you like to proceed?"
    else:
        # All files have outputs
        message = (
            f"All {len(input_paths)} selected files already have existing outputs.\n\n"
            f"Would you like to re-process them?"
        )

    # Create custom dialog
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel(root)
    dialog.title("Existing Outputs Detected")
    dialog.geometry("500x300")
    dialog.resizable(False, False)
    dialog.configure(bg='#f5f7fa')

    # Make modal
    dialog.transient(root)
    dialog.grab_set()

    user_choice = {'action': 'cancel'}

    # Message frame
    msg_frame = tk.Frame(dialog, bg='#ffffff', padx=20, pady=20)
    msg_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    tk.Label(
        msg_frame,
        text=message,
        font=("Helvetica Neue", 10),
        bg='#ffffff',
        fg='#1a1a1a',
        justify=tk.LEFT,
        wraplength=450
    ).pack(anchor=tk.W)

    # Button frame
    btn_frame = tk.Frame(dialog, bg='#f5f7fa', pady=10)
    btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

    def on_rerun_all():
        user_choice['action'] = 'rerun_all'
        dialog.destroy()
        root.destroy()

    def on_skip_completed():
        user_choice['action'] = 'skip_completed'
        dialog.destroy()
        root.destroy()

    def on_cancel():
        user_choice['action'] = 'cancel'
        dialog.destroy()
        root.destroy()

    tk.Button(
        btn_frame,
        text="Re-run All Files",
        command=on_rerun_all,
        font=("Helvetica Neue", 10, "bold"),
        bg='#0066cc',
        fg='white',
        padx=15,
        pady=8,
        relief=tk.RAISED
    ).pack(side=tk.LEFT, padx=(0, 5))

    if files_without_outputs:
        tk.Button(
            btn_frame,
            text="Skip Completed Files",
            command=on_skip_completed,
            font=("Helvetica Neue", 10),
            bg='#00a86b',
            fg='white',
            padx=15,
            pady=8,
            relief=tk.RAISED
        ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        btn_frame,
        text="Cancel",
        command=on_cancel,
        font=("Helvetica Neue", 10),
        bg='#e0e0e0',
        fg='#1a1a1a',
        padx=15,
        pady=8,
        relief=tk.RAISED
    ).pack(side=tk.LEFT, padx=5)

    # Center dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f'+{x}+{y}')

    dialog.wait_window()

    # Return filtered list based on user choice
    if user_choice['action'] == 'rerun_all':
        return list(input_paths)
    elif user_choice['action'] == 'skip_completed':
        return files_without_outputs
    else:
        return []  # Cancel - return empty list


def process_single_video(args):
    """
    Worker function for video processing with optional OpenFace extraction.

    Args:
        args: tuple of (input_path, output_dir, debug_mode, video_name, video_num,
                       total_videos, progress_window, run_openface, device)

    Returns:
        dict with processing results
    """
    input_path, output_dir, debug_mode, video_name, video_num, total_videos, progress_window, run_openface, device = args

    # Create progress callback function
    def progress_callback(stage, current, total, message, fps=0.0):
        """Callback to send progress updates to the GUI window and terminal"""
        # Update GUI progress window
        if progress_window:
            try:
                progress_window.update_progress(ProgressUpdate(
                    video_name=video_name,
                    video_num=video_num,
                    total_videos=total_videos,
                    stage=stage,
                    current=current,
                    total=total,
                    message=message,
                    fps=fps
                ))
            except Exception as e:
                pass  # Silently ignore progress update errors

        # Print terminal progress update
        if stage in ['reading', 'processing', 'writing']:
            if current > 0 and current % 50 == 0:
                # Update every 50 frames to avoid cluttering terminal
                percentage = (current / total * 100) if total > 0 else 0
                if fps > 0:
                    remaining_frames = total - current
                    eta_seconds = remaining_frames / fps if fps > 0 else 0
                    eta_str = f" | ETA: {int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                    fps_str = f" | {fps:.1f} fps"
                else:
                    eta_str = ""
                    fps_str = ""

                stage_names = {
                    'reading': 'Reading',
                    'processing': 'Processing',
                    'writing': 'Writing'
                }
                stage_name = stage_names.get(stage, stage.title())
                print(f"  [{stage_name}] {current:>6}/{total:>6} frames ({percentage:>5.1f}%){fps_str}{eta_str}", end='\r')

            # Print completion message
            elif current == total and total > 0:
                stage_names = {
                    'reading': 'Reading',
                    'processing': 'Processing',
                    'writing': 'Writing'
                }
                stage_name = stage_names.get(stage, stage.title())
                print(f"  [{stage_name}] {current:>6}/{total:>6} frames (100.0%) - Complete          ")

    try:
        # Create a new splitter instance for this worker with progress callback
        # Use GPU for face detection/landmark tracking if available
        splitter = StableFaceSplitter(
            debug_mode=debug_mode,
            device=device,
            progress_callback=progress_callback
        )
        outputs = splitter.process_video(input_path, output_dir)

        # Cleanup detector memory after face splitting
        splitter.landmark_detector.cleanup_memory()
        del splitter
        gc.collect()

        # Run OpenFace processing if requested (Mode 1: Mirror + OpenFace)
        openface_outputs = []
        if run_openface and outputs:
            # Get S1O base directory for OpenFace output
            output_dir_path = Path(output_dir)
            s1o_base = output_dir_path.parent
            openface_output_dir = s1o_base / 'Combined Data'
            openface_output_dir.mkdir(parents=True, exist_ok=True)

            # Find mirrored video files (exclude debug videos)
            mirrored_videos = [f for f in outputs if 'mirrored' in f and 'debug' not in f]

            if mirrored_videos:
                # Initialize OpenFace processor with same device as mirroring
                openface_processor = OpenFace3Processor(
                    device=device,
                    calculate_landmarks=False,  # Not needed for basic AU extraction
                    num_threads=6  # Use 6 threads for parallel CPU processing
                )

                # Process each mirrored video
                total_mirrored = len(mirrored_videos)
                for idx, mirrored_video_path in enumerate(mirrored_videos, 1):
                    mirrored_video = Path(mirrored_video_path)
                    csv_filename = mirrored_video.stem + '.csv'
                    output_csv_path = openface_output_dir / csv_filename

                    # Determine side info for progress display (1/2 or 2/2)
                    side_info = f"{idx}/{total_mirrored}"

                    # Create OpenFace progress callback
                    def openface_progress_callback(current_frame, total_frames, processing_fps):
                        """Callback for OpenFace frame processing progress"""
                        # Update GUI
                        if progress_window:
                            progress_window.update_progress(ProgressUpdate(
                                video_name=video_name,
                                video_num=video_num,
                                total_videos=total_videos,
                                stage='openface',
                                current=current_frame,
                                total=total_frames,
                                message=f"Extracting AUs: {mirrored_video.name}",
                                fps=processing_fps,
                                side_info=side_info
                            ))

                        # Update terminal (every 50 frames)
                        if current_frame > 0 and current_frame % 50 == 0:
                            percentage = (current_frame / total_frames * 100) if total_frames > 0 else 0
                            if processing_fps > 0:
                                remaining_frames = total_frames - current_frame
                                eta_seconds = remaining_frames / processing_fps
                                eta_str = f" | ETA: {int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                                fps_str = f" | {processing_fps:.1f} fps"
                            else:
                                eta_str = ""
                                fps_str = ""
                            print(f"  [AU Extraction] {current_frame:>6}/{total_frames:>6} frames ({percentage:>5.1f}%){fps_str}{eta_str}", end='\r')

                        # Print completion message
                        elif current_frame == total_frames and total_frames > 0:
                            print(f"  [AU Extraction] {current_frame:>6}/{total_frames:>6} frames (100.0%) - Complete          ")

                    # Process video with OpenFace
                    try:
                        frame_count = openface_processor.process_video(
                            mirrored_video,
                            output_csv_path,
                            progress_callback=openface_progress_callback
                        )
                        if frame_count > 0:
                            openface_outputs.append(str(output_csv_path))
                            print(f"\n✓ CSV saved: {output_csv_path.name}")
                    except Exception as e:
                        print(f"  Warning: OpenFace processing failed for {mirrored_video.name}: {e}")

                    # Memory cleanup after each mirrored video (important when processing left + right)
                    if idx < total_mirrored:  # Not the last video
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Cleanup OpenFace processor
                del openface_processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Send completion update
        if progress_window:
            progress_window.update_progress(ProgressUpdate(
                video_name=video_name,
                video_num=video_num,
                total_videos=total_videos,
                stage='complete',
                current=1,
                total=1,
                message="Video processing complete"
            ))

        return {
            'input': input_path,
            'success': True,
            'outputs': outputs,
            'openface_outputs': openface_outputs
        }
    except Exception as e:
        # Send error update
        if progress_window:
            progress_window.update_progress(ProgressUpdate(
                video_name=video_name,
                video_num=video_num,
                total_videos=total_videos,
                stage='error',
                current=0,
                total=1,
                error=str(e)
            ))

        return {
            'input': input_path,
            'success': False,
            'error': str(e)
        }


def workflow_mirror_openface():
    """
    Mode 1: Mirror + OpenFace
    Process new videos through complete pipeline (rotate, split, mirror + AU extraction)
    """
    print("\n=== MODE 1: MIRROR + OPENFACE ===")

    # Select input videos
    root = tk.Tk()
    root.withdraw()
    input_paths = filedialog.askopenfilenames(
        title="Select Video Files for Complete Processing",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    if not input_paths:
        root.destroy()
        return

    # Setup output directories
    s1o_base = Path.cwd().parent / 'S1O Processed Files'
    output_dir = s1o_base / 'Face Mirror 1.0 Output'
    openface_output_dir = s1o_base / 'Combined Data'
    output_dir.mkdir(parents=True, exist_ok=True)
    openface_output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing outputs and filter
    input_paths = filter_existing_outputs(
        input_paths,
        output_dir,
        openface_output_dir,
        mode_name="Mirror + OpenFace"
    )

    if not input_paths:
        print("No videos to process (user cancelled or all files skipped)")
        root.destroy()
        return

    debug_mode = True

    # Auto-detect GPU for entire pipeline
    device = auto_detect_device()

    print(f"\nProcessing {len(input_paths)} video(s) with complete pipeline...")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Using device: {device.upper()}")

    # Create progress window with OpenFace stage enabled
    progress_window = ProcessingProgressWindow(total_videos=len(input_paths), include_openface=True)

    # Process videos sequentially with OpenFace enabled
    results = []
    for video_num, input_path in enumerate(input_paths, 1):
        print(f"\n{'='*60}")
        print(f"VIDEO {video_num} of {len(input_paths)}: {Path(input_path).name}")
        print(f"{'='*60}")

        video_name = Path(input_path).name
        result = process_single_video((
            input_path, output_dir, debug_mode,
            video_name, video_num, len(input_paths), progress_window,
            True,  # run_openface=True (Mode 1)
            device  # Use GPU for both mirroring and OpenFace
        ))
        results.append(result)
        print(f"\nCompleted {video_num}/{len(input_paths)} videos")

        # Aggressive memory cleanup after each video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Memory cleanup completed for video {video_num}")

    # Close progress window
    progress_window.close()

    # Show results summary
    summary = "Processing Results:\n\n"
    for result in results:
        if result['success']:
            summary += f"✓ {Path(result['input']).name}\n"
            summary += f"  - Mirrored videos: {len([f for f in result['outputs'] if 'mirrored' in f and 'debug' not in f])}\n"
            summary += f"  - AU CSV files: {len(result.get('openface_outputs', []))}\n"
        else:
            summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

    messagebox.showinfo("Processing Complete", summary)
    root.destroy()


def workflow_mirror_only():
    """
    Mode 2: Mirror Only
    Process new videos with rotation, splitting, and mirroring (no AU extraction)
    """
    print("\n=== MODE 2: MIRROR ONLY ===")

    # Select input videos
    root = tk.Tk()
    root.withdraw()
    input_paths = filedialog.askopenfilenames(
        title="Select Video Files for Mirroring",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    if not input_paths:
        root.destroy()
        return

    # Setup output directories
    s1o_base = Path.cwd().parent / 'S1O Processed Files'
    output_dir = s1o_base / 'Face Mirror 1.0 Output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing outputs and filter
    input_paths = filter_existing_outputs(
        input_paths,
        output_dir,
        openface_output_dir=None,  # No OpenFace outputs to check
        mode_name="Mirror Only"
    )

    if not input_paths:
        print("No videos to process (user cancelled or all files skipped)")
        root.destroy()
        return

    debug_mode = True

    # Auto-detect GPU for face detection/mirroring
    device = auto_detect_device()

    print(f"\nProcessing {len(input_paths)} video(s) - mirroring only...")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Using device: {device.upper()}")

    # Create progress window WITHOUT OpenFace stage (Mirror Only mode)
    progress_window = ProcessingProgressWindow(total_videos=len(input_paths), include_openface=False)

    # Process videos sequentially WITHOUT OpenFace
    results = []
    for video_num, input_path in enumerate(input_paths, 1):
        print(f"\n{'='*60}")
        print(f"VIDEO {video_num} of {len(input_paths)}: {Path(input_path).name}")
        print(f"{'='*60}")

        video_name = Path(input_path).name
        result = process_single_video((
            input_path, output_dir, debug_mode,
            video_name, video_num, len(input_paths), progress_window,
            False,  # run_openface=False (Mode 2)
            device  # Use GPU for mirroring
        ))
        results.append(result)
        print(f"\nCompleted {video_num}/{len(input_paths)} videos")

        # Aggressive memory cleanup after each video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Memory cleanup completed for video {video_num}")

    # Close progress window
    progress_window.close()

    # Show results summary
    summary = "Processing Results:\n\n"
    for result in results:
        if result['success']:
            summary += f"✓ {Path(result['input']).name}\n"
            left_video = next((f for f in result['outputs'] if 'left_mirrored' in f), None)
            right_video = next((f for f in result['outputs'] if 'right_mirrored' in f), None)
            if left_video:
                summary += f"  - Left: {Path(left_video).name}\n"
            if right_video:
                summary += f"  - Right: {Path(right_video).name}\n"
        else:
            summary += f"✗ {Path(result['input']).name} - Error: {result['error']}\n"

    messagebox.showinfo("Processing Complete", summary)
    root.destroy()


def workflow_openface_only():
    """
    Mode 3: OpenFace Only
    Select source videos and automatically process their left/right mirrored versions
    """
    print("\n=== MODE 3: OPENFACE ONLY ===")

    # Setup directories
    s1o_base = Path.cwd().parent / 'S1O Processed Files'
    mirrored_videos_dir = s1o_base / 'Face Mirror 1.0 Output'
    openface_output_dir = s1o_base / 'Combined Data'

    # Use native file picker to select source videos
    root = tk.Tk()
    root.withdraw()

    # Show instruction dialog first
    messagebox.showinfo(
        "Select Videos for Processing",
        "Please select videos with names ending in '_source.mp4' (or .avi, .mov, .mkv).\n\n"
        "Example: 'patient123_source.mp4'\n\n"
        "The system will automatically find and process the corresponding left and right mirrored videos."
    )

    selected_source_files = filedialog.askopenfilenames(
        title="Select SOURCE Videos (files ending in '_source.mp4')",
        initialdir=mirrored_videos_dir,
        filetypes=[
            ("Source videos", "*_source.mp4 *_source.avi *_source.mov *_source.mkv"),
            ("All video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    if not selected_source_files:
        print("No videos selected for processing")
        root.destroy()
        return

    # Validate that user selected source files
    invalid_files = []
    valid_source_files = []

    for file_path in selected_source_files:
        file_name = Path(file_path).name
        # Check if file is a source video (not debug, not mirrored)
        if '_source' in file_name and '_debug' not in file_name and '_mirrored' not in file_name:
            valid_source_files.append(file_path)
        else:
            invalid_files.append(file_name)

    # Show error if user selected wrong file types
    if invalid_files:
        messagebox.showerror(
            "Invalid File Selection",
            f"Please select only un-mirrored videos (files ending in '_source.mp4').\n\n"
            f"The following files are not un-mirrored source videos:\n" +
            "\n".join(invalid_files[:5]) +
            (f"\n... and {len(invalid_files) - 5} more" if len(invalid_files) > 5 else "") +
            f"\n\nCorrect format examples:\n"
            f"  ✓ patient123_source.mp4\n"
            f"  ✗ patient123_left_mirrored.mp4 (mirrored video)\n"
            f"  ✗ patient123_debug.mp4 (debug file)"
        )

    if not valid_source_files:
        print("No valid source videos selected")
        root.destroy()
        return

    # For each source video, find corresponding left and right mirrored videos
    # Group by patient (each patient has left and right videos)
    patient_videos = []  # List of dicts: {'patient_name': str, 'left': Path, 'right': Path}
    missing_videos = []

    for source_path in valid_source_files:
        source_file = Path(source_path)
        base_name = source_file.stem.replace('_source', '')

        # Find left and right mirrored videos
        left_mirrored = None
        right_mirrored = None

        # Search for left mirrored video (any video extension)
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            candidate = mirrored_videos_dir / f"{base_name}_left_mirrored{ext}"
            if candidate.exists():
                left_mirrored = candidate
                break

        # Search for right mirrored video (any video extension)
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            candidate = mirrored_videos_dir / f"{base_name}_right_mirrored{ext}"
            if candidate.exists():
                right_mirrored = candidate
                break

        if left_mirrored and right_mirrored:
            patient_videos.append({
                'patient_name': base_name,
                'left': left_mirrored,
                'right': right_mirrored
            })
            print(f"Found pair: {left_mirrored.name} + {right_mirrored.name}")
        else:
            missing_videos.append(source_file.name)
            if not left_mirrored:
                print(f"Warning: Missing left mirrored video for {source_file.name}")
            if not right_mirrored:
                print(f"Warning: Missing right mirrored video for {source_file.name}")

    if missing_videos:
        messagebox.showwarning(
            "Missing Mirrored Videos",
            f"Could not find mirrored videos for:\n" + "\n".join(missing_videos[:5]) +
            (f"\n... and {len(missing_videos) - 5} more" if len(missing_videos) > 5 else "")
        )

    if not patient_videos:
        messagebox.showerror(
            "No Videos to Process",
            "No mirrored videos were found for the selected source videos."
        )
        root.destroy()
        return

    root.destroy()

    # Auto-detect GPU for OpenFace processing
    device = auto_detect_device()

    total_patients = len(patient_videos)
    print(f"\nProcessing {total_patients} patient(s) (left + right sides) with OpenFace...")
    print(f"Using device: {device.upper()}")

    # Create OpenFace output directory
    openface_output_dir.mkdir(parents=True, exist_ok=True)

    # Create progress window for OpenFace-only mode (simplified patient-based UI)
    progress_window = OpenFaceProgressWindow(total_patients=total_patients)

    # Initialize OpenFace processor with detected device
    openface_processor = OpenFace3Processor(
        device=device,
        calculate_landmarks=False,  # Not needed for basic AU extraction
        num_threads=6  # Use 6 threads for parallel CPU processing
    )

    # Process each patient (left + right sides)
    processed_count = 0
    for patient_num, patient in enumerate(patient_videos, 1):
        patient_name = patient['patient_name']

        print(f"\n{'='*60}")
        print(f"PATIENT {patient_num} of {total_patients}: {patient_name}")
        print(f"{'='*60}")

        # Process both sides for this patient
        for side, side_name in [('left', 'Left'), ('right', 'Right')]:
            video_path = patient[side]
            video_file = Path(video_path)
            csv_filename = video_file.stem + '.csv'
            output_csv_path = openface_output_dir / csv_filename

            print(f"\nProcessing {side_name} side: {video_file.name}")

            # Create progress callback for both GUI and terminal
            def progress_callback(current_frame, total_frames, processing_fps, side=side, patient_name=patient_name, patient_num=patient_num):
                """Progress callback for Mode 3 with patient-based GUI support"""
                # Update GUI progress window
                if progress_window:
                    try:
                        progress_window.update_progress(OpenFacePatientUpdate(
                            patient_name=patient_name,
                            patient_num=patient_num,
                            total_patients=total_patients,
                            side=side,
                            current_frame=current_frame,
                            total_frames=total_frames,
                            fps=processing_fps
                        ))
                    except Exception:
                        pass  # Silently ignore progress update errors

                # Update terminal (every 50 frames)
                if current_frame > 0 and current_frame % 50 == 0:
                    percentage = (current_frame / total_frames * 100) if total_frames > 0 else 0
                    if processing_fps > 0:
                        remaining_frames = total_frames - current_frame
                        eta_seconds = remaining_frames / processing_fps
                        eta_str = f" | ETA: {int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                        fps_str = f" | {processing_fps:.1f} fps"
                    else:
                        eta_str = ""
                        fps_str = ""
                    side_display = side_name
                    print(f"  [{side_display} Side] {current_frame:>6}/{total_frames:>6} frames ({percentage:>5.1f}%){fps_str}{eta_str}", end='\r')

                # Print completion message
                elif current_frame == total_frames and total_frames > 0:
                    side_display = side_name
                    print(f"  [{side_display} Side] {current_frame:>6}/{total_frames:>6} frames (100.0%) - Complete          ")

            try:
                frame_count = openface_processor.process_video(video_file, output_csv_path, progress_callback=progress_callback)

                if frame_count > 0:
                    processed_count += 1
                    print(f"\n✓ Successfully processed: {video_file.name}")
                    print(f"   CSV saved: {output_csv_path.name}")
                else:
                    print(f"✗ Failed to process: {video_file.name}")

                    # Send error update
                    if progress_window:
                        progress_window.update_progress(OpenFacePatientUpdate(
                            patient_name=patient_name,
                            patient_num=patient_num,
                            total_patients=total_patients,
                            side=side,
                            current_frame=0,
                            total_frames=1,
                            fps=0.0,
                            error="No frames processed"
                        ))

            except Exception as e:
                print(f"✗ Error processing {video_file.name}: {e}")

                # Send error update
                if progress_window:
                    progress_window.update_progress(OpenFacePatientUpdate(
                        patient_name=patient_name,
                        patient_num=patient_num,
                        total_patients=total_patients,
                        side=side,
                        current_frame=0,
                        total_frames=1,
                        fps=0.0,
                        error=str(e)
                    ))

            # Memory cleanup after each side (important when processing left + right)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Cleanup OpenFace processor
    del openface_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Close progress window
    progress_window.close()

    print(f"\nProcessing complete. {processed_count}/{total_patients * 2} files were processed.")

    # Show completion message
    messagebox.showinfo(
        "OpenFace Processing Complete",
        f"{processed_count} video(s) were processed with OpenFace.\n\n"
        f"CSV files saved to:\n{s1o_base / 'Combined Data'}"
    )


def main():
    """Main entry point - show mode selection and route to appropriate workflow"""

    # Define mode selection callback
    selected_mode = None

    def on_mode_selected(mode):
        nonlocal selected_mode
        selected_mode = mode

    # Show mode selection window
    show_mode_selection(on_mode_selected)

    # Route to appropriate workflow
    if selected_mode == 'mirror_openface':
        workflow_mirror_openface()
    elif selected_mode == 'mirror_only':
        workflow_mirror_only()
    elif selected_mode == 'openface_only':
        workflow_openface_only()
    else:
        print("No mode selected - exiting")


if __name__ == "__main__":
    main()
