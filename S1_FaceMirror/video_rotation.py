import cv2

def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError in GUI subprocess contexts.

    Also handles AttributeError: in PyInstaller --windowed builds on Windows
    sys.stdout/sys.stderr are None, so builtins.print() blows up with
    'NoneType' object has no attribute 'write'. Catch and silently drop.
    """
    import builtins
    import sys as _sys
    if _sys.stdout is None and _sys.stderr is None:
        return
    try:
        builtins.print(*args, **kwargs)
    except (BrokenPipeError, IOError, AttributeError, OSError):
        pass  # Stdout disconnected, redirected to None, or otherwise unusable

import subprocess
import shutil
import os
import sys
import json
import re
import time
from pathlib import Path

# Windows: hide the console window that pops up on every subprocess call
# made from a PyInstaller --windowed app. Without CREATE_NO_WINDOW each
# subprocess.check_output / subprocess.Popen with shell=True spawns a
# visible cmd.exe window. On non-Windows the flag doesn't exist and we
# pass 0 (no special creation flags).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0


def _check_output(*args, **kwargs):
    """Wrapper around subprocess.check_output that hides the spawned cmd
    window on Windows (PyInstaller --windowed builds) by default. Pass
    creationflags=... explicitly to override."""
    kwargs.setdefault('creationflags', _NO_WINDOW)
    return subprocess.check_output(*args, **kwargs)


def _Popen(*args, **kwargs):
    """Wrapper around subprocess.Popen with the same Windows console-hide
    default as _check_output."""
    kwargs.setdefault('creationflags', _NO_WINDOW)
    return subprocess.Popen(*args, **kwargs)

import config_paths

# Cache FFmpeg path at module load
_FFMPEG_PATH = None
_FFPROBE_PATH = None


def get_ffmpeg():
    """Get cached FFmpeg path, checking bundled location first."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        _FFMPEG_PATH = config_paths.get_ffmpeg_path()
        if _FFMPEG_PATH is None:
            raise RuntimeError(
                "ERROR: FFmpeg not found!\n"
                "Please install FFmpeg: brew install ffmpeg"
            )
    return _FFMPEG_PATH


def get_ffprobe():
    """Get FFprobe path (same directory as FFmpeg)."""
    global _FFPROBE_PATH
    if _FFPROBE_PATH is None:
        ffmpeg = get_ffmpeg()
        ffmpeg_dir = os.path.dirname(ffmpeg)
        # On Windows the binary is ffprobe.exe -- without the extension
        # os.path.isfile() returns False even when the binary is right
        # there next to ffmpeg.exe in the bundled bin/.
        ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
        ffprobe = os.path.join(ffmpeg_dir, ffprobe_name)
        if os.path.isfile(ffprobe):
            _FFPROBE_PATH = ffprobe
        else:
            # Try system ffprobe (shutil.which auto-handles PATHEXT on Windows).
            _FFPROBE_PATH = shutil.which('ffprobe')
            if _FFPROBE_PATH is None:
                # Fallback to common locations
                for path in ['/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe', '/usr/bin/ffprobe']:
                    if os.path.isfile(path):
                        _FFPROBE_PATH = path
                        break
        if _FFPROBE_PATH is None:
            raise RuntimeError("ERROR: FFprobe not found!")
    return _FFPROBE_PATH


def get_video_frame_count(input_path):
    """
    Get total frame count of a video using ffprobe.

    Args:
        input_path: Path to video file

    Returns:
        int: Total number of frames, or 0 if unable to determine
    """
    ffprobe = get_ffprobe()
    try:
        cmd = f'"{ffprobe}" -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "{input_path}"'
        output = _check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        return int(output)
    except (subprocess.CalledProcessError, ValueError):
        # Fallback: try using nb_frames
        try:
            cmd = f'"{ffprobe}" -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
            output = _check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
            return int(output)
        except (subprocess.CalledProcessError, ValueError):
            return 0


def get_video_rotation(input_path):
    """Get video rotation from metadata using ffprobe with multiple detection methods"""
    ffprobe = get_ffprobe()
    # Comprehensive rotation detection commands
    commands = [
        # Try to get full metadata in JSON format for more comprehensive parsing
        f'"{ffprobe}" -v quiet -print_format json -show_streams "{input_path}"',

        # Specific commands for different metadata locations
        f'"{ffprobe}" -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 "{input_path}"',
        f'"{ffprobe}" -v error -select_streams v:0 -show_entries stream=rotate -of default=nw=1:nk=1 "{input_path}"',
        f'"{ffprobe}" -v error -select_streams v:0 -show_entries stream_side_data=rotation -of default=nw=1:nk=1 "{input_path}"'
    ]

    for command in commands:
        try:
            output = _check_output(command, shell=True, universal_newlines=True).strip()

            # For JSON metadata, parse and extract rotation
            if command.endswith('json'):
                try:
                    metadata = json.loads(output)
                    # Check different possible locations for rotation in JSON
                    for stream in metadata.get('streams', []):
                        # Try different rotation-related keys
                        rotation = stream.get('tags', {}).get('rotate')
                        if rotation is None:
                            rotation = stream.get('rotation')

                        if rotation is not None:
                            try:
                                rotation = int(rotation)
                                safe_print(f"Detected rotation from JSON: {rotation}")
                                return rotation
                            except ValueError:
                                continue
                        
                        # Check for displaymatrix in side data
                        if 'side_data_list' in stream:
                            for side_data in stream['side_data_list']:
                                if 'displaymatrix' in str(side_data).lower():
                                    if 'rotation of -90' in str(side_data).lower():
                                        safe_print(f"Detected -90 degree rotation in displaymatrix")
                                        return -90
                except (json.JSONDecodeError, TypeError):
                    pass

            # For other commands, try direct integer conversion
            try:
                rotation = int(output)
                safe_print(f"Detected rotation: {rotation}")
                return rotation
            except ValueError:
                continue

        except (subprocess.CalledProcessError, ValueError):
            continue

    # Special handling for iOS video files (common with .MOV files)
    try:
        # Use MediaInfo for additional metadata detection if available
        media_info_cmd = f'mediainfo --Inform="Video;%Rotation%" "{input_path}"'
        media_info_output = _check_output(media_info_cmd, shell=True, universal_newlines=True).strip()

        try:
            rotation = int(media_info_output)
            safe_print(f"Detected rotation via MediaInfo: {rotation}")
            return rotation
        except ValueError:
            pass
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check for portrait video dimensions as a fallback
    try:
        dim_cmd = f'"{ffprobe}" -v quiet -select_streams v:0 -show_entries stream=width,height -of json "{input_path}"'
        dim_output = _check_output(dim_cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        dim_data = json.loads(dim_output)
        
        if dim_data.get('streams'):
            width = int(dim_data['streams'][0].get('width', 0))
            height = int(dim_data['streams'][0].get('height', 0))
            
            # If height is significantly greater than width, it's likely a portrait video
            if height > width * 1.2:
                # Check if this is likely a mobile video
                filename = os.path.basename(input_path).lower()
                extension = os.path.splitext(filename)[1].lower()
                
                # Common indicators of mobile videos
                mobile_indicators = [".mov", ".mp4", "iphone", "ios", "img_", "vid_", "video", "android"]
                
                if any(indicator in filename.lower() for indicator in mobile_indicators) or extension in [".mov", ".mp4"]:
                    safe_print(f"Portrait orientation detected (H:{height} > W:{width}) for likely mobile video")
                    # Return -90 as this is the common value for portrait videos needing rotation
                    return -90
    except Exception as e:
        safe_print(f"Error checking video dimensions: {str(e)}")

    # If no rotation detected
    safe_print("No rotation detected")
    return 0


def normalize_rotation(rotation):
    """
    Normalize rotation to correct rotation for mobile video
    Specific handling for iOS video rotation metadata
    """
    # Normalize negative rotations
    if rotation < 0:
        rotation = 360 + rotation

    # Specific handling for common mobile video rotations
    # iOS typically uses -90 to indicate 90 degrees clockwise
    if rotation == 270:
        # This is just to ensure consistent handling later
        rotation = 270

    # Round to nearest 90 degrees
    rotation = round(rotation / 90) * 90 % 360

    return rotation


def auto_rotate_video(input_path, output_path, progress_callback=None):
    """
    Process video with ffmpeg's auto-rotation feature with real-time progress tracking
    This preserves original video quality and audio

    Args:
        input_path: Path to input video file
        output_path: Path to output rotated video file
        progress_callback: Optional callback function(stage, current, total, message, fps)
                          for progress updates

    Returns:
        str: Path to output file (or original if rotation failed)
    """
    rotation_start_time = time.time()
    safe_print(f"Auto-rotating video {input_path} using ffmpeg's auto-rotation")

    # Ensure output has proper extension
    output_path_obj = Path(output_path)
    input_path_obj = Path(input_path)
    
    # Ensure output has same extension as input
    if output_path_obj.suffix.lower() != input_path_obj.suffix.lower():
        output_path = str(output_path_obj.with_suffix(input_path_obj.suffix))
        safe_print(f"Changed output extension to match input: {output_path}")

    # Get original video codec details
    ffprobe = get_ffprobe()
    codec_cmd = f'"{ffprobe}" -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
    try:
        codec = _check_output(codec_cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        safe_print(f"Original video codec: {codec}")
        
        # Choose video codec based on input file
        if input_path_obj.suffix.lower() == '.mov':
            # For MOV files, use H.264 instead of ProRes since it's more reliable
            video_codec = "libx264"
            codec_options = "-pix_fmt yuv420p -preset medium -crf 23"
        else:
            # For other files
            if codec == "hevc":
                video_codec = "libx265"
                codec_options = "-pix_fmt yuv420p -preset medium -crf 23"
            else:
                video_codec = "libx264"
                codec_options = "-pix_fmt yuv420p -preset medium -crf 23"
    except subprocess.CalledProcessError:
        # Default to H.264 if codec detection fails
        video_codec = "libx264"
        codec_options = "-pix_fmt yuv420p -preset medium -crf 23"
    
    # Get total frame count for progress tracking
    total_frames = get_video_frame_count(input_path)
    if total_frames > 0:
        safe_print(f"Video has {total_frames} frames")
        if progress_callback:
            progress_callback('rotation', 0, total_frames, "Rotating video...", 0.0)
    else:
        safe_print("Unable to determine frame count, progress tracking disabled")
        if progress_callback:
            progress_callback('rotation', 0, 0, "Rotating video...", 0.0)

    # Create FFmpeg command with auto rotation enabled
    # Use -progress pipe:1 to get frame-by-frame progress on stdout
    ffmpeg = get_ffmpeg()
    cmd = (f'"{ffmpeg}" -y -i "{input_path}" -map 0 -map_metadata 0 '
           f'-c:a copy -c:v {video_codec} {codec_options} '
           f'-progress pipe:1 "{output_path}"')

    safe_print(f"Executing FFmpeg command with progress tracking...")
    try:
        # Use Popen to capture output in real-time.
        # CRITICAL: merge stderr into stdout. Reading only one of the two
        # captured pipes deadlocks once the unread one fills its buffer
        # (~64 KB on Windows). FFmpeg writes nontrivial stderr volume during
        # HEVC->H.264 transcoding of iOS clips (PTS warnings, encoder stats),
        # which used to hang this loop on long Paralysis Cohort videos
        # without ever printing a single progress update.
        process = _Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )

        current_frame = 0
        current_fps = 0.0
        last_reported_frame = -1
        ffmpeg_start_time = time.time()
        last_progress_time = ffmpeg_start_time

        # Tail of FFmpeg log lines (non-progress lines) -- kept for error
        # reporting on failure. With stderr merged into stdout above, this
        # is now the only place ffmpeg's diagnostic output survives.
        ffmpeg_log_tail: list = []
        FFMPEG_LOG_TAIL_MAX = 50

        # Prefixes emitted by `-progress pipe:1`. Lines starting with these
        # are progress data, not log noise.
        _PROGRESS_PREFIXES = (
            'frame=', 'fps=', 'progress=',
            'out_time=', 'out_time_us=', 'out_time_ms=',
            'total_size=', 'bitrate=', 'speed=',
            'stream_', 'dup_frames=', 'drop_frames=',
        )

        # Parse FFmpeg progress output
        # FFmpeg writes to stdout when using -progress pipe:1
        for line in process.stdout:
            line = line.strip()

            # Capture ffmpeg's diagnostic log lines (banner, codec warnings,
            # PTS errors) into a bounded ring buffer for error reporting.
            if line and not line.startswith(_PROGRESS_PREFIXES):
                ffmpeg_log_tail.append(line)
                if len(ffmpeg_log_tail) > FFMPEG_LOG_TAIL_MAX:
                    ffmpeg_log_tail.pop(0)

            # FFmpeg outputs "frame=N" to show progress
            if line.startswith('frame='):
                try:
                    current_frame = int(line.split('=')[1])
                except (ValueError, IndexError):
                    pass

            # FFmpeg outputs "fps=X" to show processing speed
            elif line.startswith('fps='):
                try:
                    current_fps = float(line.split('=')[1])
                except (ValueError, IndexError):
                    pass

            # FFmpeg also outputs "progress=end" when done
            elif line.startswith('progress='):
                progress_status = line.split('=')[1]
                if progress_status == 'end' and progress_callback and total_frames > 0:
                    progress_callback('rotation', total_frames, total_frames,
                                    "Video rotated", current_fps)
                elif progress_status == 'continue' and progress_callback and total_frames > 0:
                    # Report progress (throttle updates - every 10 frames)
                    if current_frame - last_reported_frame >= 10:
                        current_time = time.time()
                        time_since_last = current_time - last_progress_time
                        frames_since_last = current_frame - last_reported_frame

                        # Debug: Log performance every 50 frames
                        if current_frame % 50 == 0:
                            avg_fps = frames_since_last / time_since_last if time_since_last > 0 else 0
                            safe_print(f"  [Rotation Debug] Frame {current_frame}/{total_frames} | "
                                  f"FFmpeg FPS: {current_fps:.1f} | "
                                  f"Actual FPS: {avg_fps:.1f} | "
                                  f"Time since last: {time_since_last:.2f}s")

                        progress_callback('rotation', current_frame, total_frames,
                                        "Rotating video...", current_fps)
                        last_reported_frame = current_frame
                        last_progress_time = current_time

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            rotation_elapsed = time.time() - rotation_start_time
            safe_print(f"Auto-rotation complete in {rotation_elapsed:.2f}s. Output saved to {output_path}")
            if total_frames > 0:
                avg_fps = total_frames / rotation_elapsed
                safe_print(f"  Average FPS: {avg_fps:.1f} frames/sec")
            return output_path
        else:
            # stderr was merged into stdout (see Popen call above); the
            # tail of ffmpeg's log lines was accumulated as we drained the
            # combined stream.
            safe_print(f"Error during auto-rotation (return code {return_code})")
            if ffmpeg_log_tail:
                safe_print("FFmpeg log tail:")
                for log_line in ffmpeg_log_tail[-20:]:
                    safe_print(f"  {log_line}")
            safe_print("Unable to rotate video. Using original file.")
            return input_path

    except Exception as e:
        safe_print(f"Error during auto-rotation: {e}")
        safe_print("Unable to rotate video. Using original file.")
        return input_path


def process_video_rotation(input_path, output_path, progress_callback=None):
    """
    Main entry point for video rotation with progress tracking

    Args:
        input_path: Path to input video file
        output_path: Path to output rotated video file
        progress_callback: Optional callback function(stage, current, total, message, fps)
                          for progress updates

    Returns:
        str: Path to output file
    """
    safe_print(f"\nProcessing video rotation for {input_path}")

    # Send initial progress update
    if progress_callback:
        progress_callback('rotation', 0, 100, "Checking video orientation...", 0.0)

    # Get rotation from metadata using original detection method
    rotation = get_video_rotation(input_path)

    # Normalize rotation
    normalized_rotation = normalize_rotation(rotation)
    safe_print(f"Detected rotation: {rotation}°, Normalized to: {normalized_rotation}°")

    # Only process the video if rotation is needed
    if normalized_rotation in [90, 180, 270]:
        safe_print(f"Rotation needed: {normalized_rotation}°")
        return auto_rotate_video(input_path, output_path, progress_callback)
    else:
        safe_print("No rotation needed, using original file")

        # Send completion update (rotation not needed)
        if progress_callback:
            progress_callback('rotation', 100, 100, "Video ready (no rotation needed)", 0.0)

        # Just copy the file if it shouldn't be the same path
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
            return output_path
        return input_path
