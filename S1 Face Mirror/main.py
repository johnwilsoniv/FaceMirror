"""
S1 Face Mirror - Video processing pipeline with face mirroring and AU extraction
"""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import native_dialogs
from multiprocessing import cpu_count
import gc
import torch
import psutil
import time

# Import configuration and workflow components
import config_paths
from face_splitter import StableFaceSplitter
from openface_integration import OpenFace3Processor
from progress_window import ProcessingProgressWindow, ProgressUpdate


# Configuration Constants
NUM_THREADS = 6  # Thread count for parallel processing
MEMORY_CHECKPOINT_INTERVAL = 10  # Deep cleanup every N videos
PROGRESS_UPDATE_INTERVAL = 50  # Terminal progress update every N frames


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

    # Check for mirror outputs in Face Mirror 1.0 Output (left, right)
    mirror_files_exist = True
    for suffix in ['_left_mirrored', '_right_mirrored']:
        found = False
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            if (output_dir / f"{base_name}{suffix}{ext}").exists():
                found = True
                break
        if not found:
            mirror_files_exist = False
            break

    # Check for source video in Combined Data (if openface_output_dir provided)
    if mirror_files_exist and openface_output_dir:
        openface_output_dir_path = Path(openface_output_dir)
        source_found = False
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4']:
            if (openface_output_dir_path / f"{base_name}_source{ext}").exists():
                source_found = True
                break
        if not source_found:
            mirror_files_exist = False

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


def filter_existing_outputs(input_paths, output_dir, openface_output_dir=None):
    """
    Check which input files have existing outputs and prompt user.

    Args:
        input_paths: List of input video paths
        output_dir: Path to Face Mirror 1.0 Output directory
        openface_output_dir: Path to Combined Data directory (optional)

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

    # Build consolidated message with all info and options
    if files_without_outputs:
        # Some files have outputs, some don't
        message = (
            f"Found {len(files_with_outputs)} of {len(input_paths)} file(s) "
            f"already processed.\n\n"
            f"Files already processed:\n"
        )
        for f in files_with_outputs[:3]:
            message += f"  • {Path(f).name}\n"
        if len(files_with_outputs) > 3:
            message += f"  ... and {len(files_with_outputs) - 3} more\n"

        message += f"\nWhat would you like to do?"

        result = native_dialogs.ask_three_choice(
            "Existing Outputs Detected",
            message,
            f"Re-run All ({len(input_paths)})",
            f"Process Unprocessed ({len(files_without_outputs)})",
            "Cancel",
            default_button=2
        )

        if result == 1:  # Re-run all
            return list(input_paths)
        elif result == 2:  # Process unprocessed
            return files_without_outputs
        else:  # Cancel or None
            return []
    else:
        # All files have outputs
        message = (
            f"All {len(input_paths)} selected file(s) "
            f"already have existing outputs.\n\n"
            f"Files already processed:\n"
        )
        for f in files_with_outputs[:5]:
            message += f"  • {Path(f).name}\n"
        if len(files_with_outputs) > 5:
            message += f"  ... and {len(files_with_outputs) - 5} more\n"

        message += "\n\nRe-process all files? (This will overwrite existing outputs)"

        result = native_dialogs.ask_yes_no("Existing Outputs Detected", message, default_yes=False)

        if result:
            return list(input_paths)
        else:
            return []


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

    splitter = None
    openface_processor = None

    try:
        # Validate input file exists and is readable
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        if not input_file.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")

        # Check file size (warn if file is suspiciously small)
        file_size = input_file.stat().st_size
        if file_size < 1000:  # Less than 1KB
            raise ValueError(f"Input video file appears to be empty or corrupted (size: {file_size} bytes)")

        # Create a new splitter instance for this worker with progress callback
        # Use GPU for face detection/landmark tracking if available
        splitter = StableFaceSplitter(
            debug_mode=debug_mode,
            device=device,
            progress_callback=progress_callback
        )
        outputs = splitter.process_video(input_path, output_dir)

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
                    calculate_landmarks=True,  # Required for AU45 (blink) calculation
                    num_threads=NUM_THREADS
                )

                # Process each mirrored video
                total_mirrored = len(mirrored_videos)
                for idx, mirrored_video_path in enumerate(mirrored_videos, 1):
                    mirrored_video = Path(mirrored_video_path)
                    csv_filename = mirrored_video.stem + '.csv'
                    output_csv_path = openface_output_dir / csv_filename

                    # Determine side info for progress display (e.g., "Left 1/2" or "Right 2/2")
                    if 'left' in mirrored_video.name.lower():
                        side_name = "Left"
                    elif 'right' in mirrored_video.name.lower():
                        side_name = "Right"
                    else:
                        side_name = ""
                    side_info = f"{side_name} {idx}/{total_mirrored}" if side_name else f"{idx}/{total_mirrored}"

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
    finally:
        # Ensure resources are cleaned up even if errors occur
        # Memory cleanup is critical for batch processing to prevent crashes
        try:
            if splitter is not None:
                splitter.landmark_detector.cleanup_memory()
                del splitter
        except Exception as e:
            print(f"  Warning: Splitter cleanup failed: {e}")

        try:
            if openface_processor is not None:
                # Clear priorbox cache to prevent memory accumulation
                openface_processor.clear_cache()
                del openface_processor
        except Exception as e:
            print(f"  Warning: OpenFace processor cleanup failed: {e}")

        # Force garbage collection (always runs regardless of above errors)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def workflow_mirror_openface():
    """
    Full pipeline: Process videos through complete workflow
    (rotate, split, mirror + AU extraction)
    """
    # Select input videos
    root = tk.Tk()
    root.withdraw()

    # Make file dialog appear on top
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))

    input_paths = filedialog.askopenfilenames(
        title="Select Video Files for Processing",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    if not input_paths:
        root.destroy()
        return

    # Setup output directories using config_paths
    output_dir = config_paths.get_mirror_output_dir()
    openface_output_dir = config_paths.get_combined_data_dir()

    # Check for existing outputs and filter
    input_paths = filter_existing_outputs(
        input_paths,
        output_dir,
        openface_output_dir
    )

    if not input_paths:
        print("No videos to process (user cancelled or all files skipped)")
        root.destroy()
        return

    debug_mode = True

    # Auto-detect GPU for entire pipeline
    device = auto_detect_device()

    print(f"\nProcessing {len(input_paths)} video(s)...")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Using device: {device.upper()}")

    # Create progress window with OpenFace stage enabled
    progress_window = ProcessingProgressWindow(total_videos=len(input_paths), include_openface=True)

    # Start timing
    start_time = time.time()

    # Process videos sequentially with full pipeline
    results = []
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024**2

    for video_num, input_path in enumerate(input_paths, 1):
        print(f"\n{'='*60}")
        print(f"VIDEO {video_num} of {len(input_paths)}: {Path(input_path).name}")
        print(f"{'='*60}")

        video_name = Path(input_path).name
        result = process_single_video((
            input_path, output_dir, debug_mode,
            video_name, video_num, len(input_paths), progress_window,
            True,  # run_openface=True (full pipeline)
            device  # Use GPU for both mirroring and OpenFace
        ))
        results.append(result)
        print(f"\nCompleted {video_num}/{len(input_paths)} videos")

        # Aggressive memory cleanup after each video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Memory monitoring and deep cleanup
        if video_num % MEMORY_CHECKPOINT_INTERVAL == 0:
            gc.collect()
            gc.collect()  # Run twice for cyclic garbage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Memory checkpoint
            current_memory_mb = process.memory_info().rss / 1024**2
            memory_growth_mb = current_memory_mb - initial_memory_mb
            print(f"Memory checkpoint [{video_num}/{len(input_paths)}]: {current_memory_mb:.1f} MB (growth: +{memory_growth_mb:.1f} MB)")

    # End timing
    end_time = time.time()
    total_seconds = end_time - start_time

    # Close progress window
    progress_window.close()

    # Show results summary (compact format for large batches)
    successful_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - successful_count

    total_mirrored = sum(len([f for f in r['outputs'] if 'mirrored' in f and 'debug' not in f])
                         for r in results if r['success'])
    total_csvs = sum(len(r.get('openface_outputs', [])) for r in results if r['success'])

    # Format processing time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"

    summary = "Processing Complete!\n\n"
    summary += f"Successfully processed: {successful_count} video(s)\n"
    if failed_count > 0:
        summary += f"Failed: {failed_count} video(s)\n"
    summary += f"\nProcessing time: {time_str}\n"
    summary += "\nGenerated Files:\n"
    summary += f"  • {total_mirrored} mirrored videos\n"
    summary += f"  • {total_csvs} AU CSV files\n\n"
    summary += "Results saved to: Combined Data/\n\n"
    summary += "Next step: Open 'Combined Data' folder to view your Action Unit CSV files."

    native_dialogs.show_info("Processing Complete", summary)
    root.destroy()


def main():
    """Main entry point - go straight to full pipeline workflow"""
    print("\n" + "="*60)
    print(f"S1 FACE MIRROR v{config_paths.VERSION} - FULL PIPELINE")
    print("Face Mirroring + Action Unit Extraction")
    print("="*60)

    # Go straight to the full pipeline workflow
    workflow_mirror_openface()

if __name__ == "__main__":
    main()
