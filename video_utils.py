import subprocess
import json
import logging
import tempfile
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def get_comprehensive_rotation(video_path, debug_mode=False):
    """
    Get comprehensive rotation information from video metadata using ffprobe.
    Returns rotation angle and whether conversion is needed.
    """
    try:
        # First check - direct rotation metadata
        cmd1 = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate',
            '-of', 'json',
            str(video_path)
        ]

        # Second check - side data rotation
        cmd2 = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_side_data_list',
            '-of', 'json',
            str(video_path)
        ]

        # Third check - container metadata
        cmd3 = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format_tags=rotate',
            '-of', 'json',
            str(video_path)
        ]

        rotation_angle = 0
        needs_conversion = False

        # Check 1: Stream Tags
        result = subprocess.run(cmd1, capture_output=True, text=True)
        if result.stdout:
            data = json.loads(result.stdout)
            if 'streams' in data and data['streams']:
                tags = data['streams'][0].get('tags', {})
                if 'rotate' in tags:
                    rotation_angle = float(tags['rotate'])
                    needs_conversion = True
                    if debug_mode:
                        logger.info(f"Found rotation in stream tags: {rotation_angle}")
                    return rotation_angle, needs_conversion

        # Check 2: Side Data
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.stdout:
            data = json.loads(result.stdout)
            if 'streams' in data and data['streams']:
                side_data_list = data['streams'][0].get('side_data_list', [])
                for side_data in side_data_list:
                    if 'rotation' in side_data:
                        rotation_angle = float(side_data['rotation'])
                        needs_conversion = True
                        if debug_mode:
                            logger.info(f"Found rotation in side data: {rotation_angle}")
                        return rotation_angle, needs_conversion

        # Check 3: Format Tags
        result = subprocess.run(cmd3, capture_output=True, text=True)
        if result.stdout:
            data = json.loads(result.stdout)
            if 'format' in data and 'tags' in data['format']:
                if 'rotate' in data['format']['tags']:
                    rotation_angle = float(data['format']['tags']['rotate'])
                    needs_conversion = True
                    if debug_mode:
                        logger.info(f"Found rotation in format tags: {rotation_angle}")
                    return rotation_angle, needs_conversion

        # If no rotation found, try to detect display matrix
        matrix_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=display_matrix',
            '-of', 'json',
            str(video_path)
        ]

        result = subprocess.run(matrix_cmd, capture_output=True, text=True)
        if result.stdout:
            data = json.loads(result.stdout)
            if 'streams' in data and data['streams']:
                display_matrix = data['streams'][0].get('display_matrix')
                if display_matrix:
                    # Display matrix values that indicate rotation
                    if display_matrix == '0 -1 1 1 0 0' or display_matrix == '[0, -1, 1, 1, 0, 0]':
                        rotation_angle = 90
                        needs_conversion = True
                    elif display_matrix == '-1 0 0 0 -1 1' or display_matrix == '[-1, 0, 0, 0, -1, 1]':
                        rotation_angle = 180
                        needs_conversion = True
                    elif display_matrix == '0 1 0 -1 0 1' or display_matrix == '[0, 1, 0, -1, 0, 1]':
                        rotation_angle = 270
                        needs_conversion = True

                    if needs_conversion and debug_mode:
                        logger.info(f"Found rotation in display matrix: {rotation_angle}")
                    return rotation_angle, needs_conversion

        if debug_mode:
            logger.info("No rotation metadata found")
        return 0, False

    except Exception as e:
        if debug_mode:
            logger.error(f"Error checking rotation metadata: {str(e)}")
            logger.error(f"Command outputs:")
            for cmd in [cmd1, cmd2, cmd3, matrix_cmd]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    logger.error(f"Command {cmd}: {result.stdout}")
                    if result.stderr:
                        logger.error(f"Error: {result.stderr}")
                except Exception as cmd_error:
                    logger.error(f"Error running command {cmd}: {str(cmd_error)}")
        return 0, False


def fix_rotation_with_progress(input_path, output_dir, rotation_angle, debug_mode=False, progress_callback=None):
    """
    Fix video rotation using ffmpeg with progress updates.
    Returns the path to the rotated video.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / f"rotated_{input_path.name}"
        final_output = output_dir / f"rotated_{input_path.name}"

        # Get video duration for progress calculation
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(input_path)
        ]

        try:
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
        except:
            duration = 0

        # Prepare ffmpeg command with transpose filter based on rotation angle
        transpose_filter = ""
        if rotation_angle == 90:
            transpose_filter = "transpose=1"  # 90 degrees clockwise
        elif rotation_angle == 180:
            transpose_filter = "transpose=2,transpose=2"  # 180 degrees
        elif rotation_angle == 270:
            transpose_filter = "transpose=2"  # 90 degrees counterclockwise

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
        ]

        if transpose_filter:
            cmd.extend(['-vf', transpose_filter])

        cmd.extend([
            '-metadata:s:v:0', 'rotate=0',
            '-pix_fmt', 'yuv420p',
            '-progress', 'pipe:1',
            '-y',
            str(temp_output)
        ])

        if debug_mode:
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")

        # Run the FFmpeg process and wait for completion
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Process progress updates while waiting for completion
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break

            if output and 'out_time_ms=' in output:
                time_ms = int(output.split('out_time_ms=')[1].strip())
                if duration > 0:
                    progress = (time_ms / 1000000) / duration * 100
                    if progress_callback:
                        progress_callback(min(progress, 100))

        # Wait for the process to complete
        process.wait()

        # Check if conversion was successful
        if process.returncode == 0:
            if debug_mode:
                logger.info("Rotation correction successful")
            # Copy the temp file to the final output location
            shutil.copy2(temp_output, final_output)
            return str(final_output)  # Return the path as string for compatibility
        else:
            error_output = process.stderr.read()
            if debug_mode:
                logger.error(f"FFmpeg error: {error_output}")
            return str(input_path)  # Return original path if rotation failed