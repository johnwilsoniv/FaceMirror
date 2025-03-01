import cv2
import subprocess
import shutil
import os
import json


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
        # Use MediaInfo for additional metadata detection
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
        # Rotate 90 degrees counter-clockwise instead of 270 degrees
        return 90

    # Round to nearest 90 degrees
    rotation = round(rotation / 90) * 90 % 360

    return rotation


def rotate_video(input_path, output_path, angle):
    """Rotate video using OpenCV"""
    print(f"Rotating video {input_path} by {angle} degrees")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    if cap.isOpened():
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Original video dimensions: {frame_width}x{frame_height}")

        if angle == 90:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_height, frame_width))
        elif angle == 270:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_height, frame_width))
        elif angle == 180:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        else:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if angle == 90:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif angle == 270:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                rotated_frame = frame

            out.write(rotated_frame)

        cap.release()
        out.release()
        print(f"Rotation complete. Output saved to {output_path}")
    else:
        raise RuntimeError(f"Error opening video file: {input_path}")


def process_video_rotation(input_path, output_path):
    """Process video rotation based on metadata"""
    print(f"Processing video rotation for {input_path}")
    rotation = get_video_rotation(input_path)

    # Normalize rotation
    normalized_rotation = normalize_rotation(rotation)
    print(f"Normalized rotation: {normalized_rotation}")

    # Only create a rotated video if rotation is needed
    if normalized_rotation in [90, 180, 270]:
        rotate_video(input_path, output_path, normalized_rotation)
        return output_path
    else:
        print("No rotation needed")
        return input_path