import cv2

def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError in GUI subprocess contexts."""
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except (BrokenPipeError, IOError):
        pass  # Stdout disconnected

import subprocess
import shutil
import os
import json
import re
import time
from pathlib import Path


def get_video_frame_count(input_path):
    """
    Get total frame count of a video using ffprobe.

    Args:
        input_path: Path to video file

    Returns:
        int: Total number of frames, or 0 if unable to determine
    """
    try:
        cmd = f'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "{input_path}"'
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        return int(output)
    except (subprocess.CalledProcessError, ValueError):
        # Fallback: try using nb_frames
        try:
            cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
            output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
            return int(output)
        except (subprocess.CalledProcessError, ValueError):
            return 0


def get_video_rotation(input_path):
    """Get video rotation from metadata using ffprobe with multiple detection methods"""
    # Comprehensive rotation detection commands
    commands = [
        # Try to get full metadata in JSON format for more comprehensive parsing
        f'ffprobe -v quiet -print_format json -show_streams "{input_path}"',

        # Specific commands for different metadata locations
        f'ffprobe -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 "{input_path}"',
        f'ffprobe -v error -select_streams v:0 -show_entries stream=rotate -of default=nw=1:nk=1 "{input_path}"',
        f'ffprobe -v error -select_streams v:0 -show_entries stream_side_data=rotation -of default=nw=1:nk=1 "{input_path}"'
    ]

    for command in commands:
        try:
            output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()

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
        media_info_output = subprocess.check_output(media_info_cmd, shell=True, universal_newlines=True).strip()

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
        dim_cmd = f'ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height -of json "{input_path}"'
        dim_output = subprocess.check_output(dim_cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
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
    codec_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
    try:
        codec = subprocess.check_output(codec_cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
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
    cmd = (f'ffmpeg -y -i "{input_path}" -map 0 -map_metadata 0 '
           f'-c:a copy -c:v {video_codec} {codec_options} '
           f'-progress pipe:1 "{output_path}"')

    safe_print(f"Executing FFmpeg command with progress tracking...")
    try:
        # Use Popen to capture output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )

        current_frame = 0
        current_fps = 0.0
        last_reported_frame = -1
        ffmpeg_start_time = time.time()
        last_progress_time = ffmpeg_start_time

        # Parse FFmpeg progress output
        # FFmpeg writes to stdout when using -progress pipe:1
        for line in process.stdout:
            line = line.strip()

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
            # Get error output
            stderr_output = process.stderr.read()
            safe_print(f"Error during auto-rotation (return code {return_code})")
            safe_print(f"FFmpeg error: {stderr_output}")
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
