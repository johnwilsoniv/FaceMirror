import cv2
import subprocess
import shutil
import os
import json
from pathlib import Path


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
                                print(f"Detected rotation from JSON: {rotation}")
                                return rotation
                            except ValueError:
                                continue
                        
                        # Check for displaymatrix in side data
                        if 'side_data_list' in stream:
                            for side_data in stream['side_data_list']:
                                if 'displaymatrix' in str(side_data).lower():
                                    if 'rotation of -90' in str(side_data).lower():
                                        print(f"Detected -90 degree rotation in displaymatrix")
                                        return -90
                except (json.JSONDecodeError, TypeError):
                    pass

            # For other commands, try direct integer conversion
            try:
                rotation = int(output)
                print(f"Detected rotation: {rotation}")
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
            print(f"Detected rotation via MediaInfo: {rotation}")
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
                    print(f"Portrait orientation detected (H:{height} > W:{width}) for likely mobile video")
                    # Return -90 as this is the common value for portrait videos needing rotation
                    return -90
    except Exception as e:
        print(f"Error checking video dimensions: {str(e)}")

    # If no rotation detected
    print("No rotation detected")
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


def auto_rotate_video(input_path, output_path):
    """
    Process video with ffmpeg's auto-rotation feature
    This preserves original video quality and audio
    """
    print(f"Auto-rotating video {input_path} using ffmpeg's auto-rotation")
    
    # Ensure output has proper extension
    output_path_obj = Path(output_path)
    input_path_obj = Path(input_path)
    
    # Ensure output has same extension as input
    if output_path_obj.suffix.lower() != input_path_obj.suffix.lower():
        output_path = str(output_path_obj.with_suffix(input_path_obj.suffix))
        print(f"Changed output extension to match input: {output_path}")

    # Get original video codec details
    codec_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{input_path}"'
    try:
        codec = subprocess.check_output(codec_cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        print(f"Original video codec: {codec}")
        
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
    
    # Create FFmpeg command with auto rotation enabled
    cmd = (f'ffmpeg -y -v info -i "{input_path}" -map 0 -map_metadata 0 '
           f'-c:a copy -c:v {video_codec} {codec_options} "{output_path}"')

    print(f"Executing FFmpeg command: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"Auto-rotation complete. Output saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error during auto-rotation: {e}")
        # If rotation fails, return original path
        print("Unable to rotate video. Using original file.")
        return input_path


def process_video_rotation(input_path, output_path):
    """
    Main entry point for video rotation
    Uses original detection method but with ffmpeg auto-rotation
    """
    print(f"\nProcessing video rotation for {input_path}")
    
    # Get rotation from metadata using original detection method
    rotation = get_video_rotation(input_path)
    
    # Normalize rotation
    normalized_rotation = normalize_rotation(rotation)
    print(f"Detected rotation: {rotation}°, Normalized to: {normalized_rotation}°")
    
    # Only process the video if rotation is needed
    if normalized_rotation in [90, 180, 270]:
        print(f"Rotation needed: {normalized_rotation}°")
        return auto_rotate_video(input_path, output_path)
    else:
        print("No rotation needed, using original file")
        # Just copy the file if it shouldn't be the same path
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
            return output_path
        return input_path
