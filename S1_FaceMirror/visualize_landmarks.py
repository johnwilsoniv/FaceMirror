#!/usr/bin/env python3
"""
Landmark Visualization: Compare SPIGA 98-point, SPIGA→68 mapped, and CLNF 68-point

Creates a 3-panel visualization:
- Panel A: SPIGA native 98-point WFLW landmarks
- Panel B: SPIGA mapped to 68-point dlib format
- Panel C: pyclnf 68-point landmarks

Usage:
    python visualize_landmarks.py --video "../S Data/Paralysis Cohort/IMG_8401.MOV"
    python visualize_landmarks.py --video "../S Data/Paralysis Cohort/IMG_8401.MOV" --output landmarks_comparison.png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# Colors for different landmark regions (BGR)
COLORS_98 = {
    'jawline': (0, 255, 255),      # Yellow (0-32)
    'right_brow': (255, 0, 0),     # Blue (33-41)
    'left_brow': (255, 0, 0),      # Blue (42-50)
    'nose': (0, 255, 0),           # Green (51-59)
    'right_eye': (0, 0, 255),      # Red (60-67)
    'left_eye': (0, 0, 255),       # Red (68-75)
    'outer_mouth': (255, 0, 255),  # Magenta (76-87)
    'inner_mouth': (255, 128, 0),  # Orange (88-95)
    'pupils': (255, 255, 255),     # White (96-97)
}

COLORS_68 = {
    'jawline': (0, 255, 255),      # Yellow (0-16)
    'right_brow': (255, 0, 0),     # Blue (17-21)
    'left_brow': (255, 0, 0),      # Blue (22-26)
    'nose': (0, 255, 0),           # Green (27-35)
    'right_eye': (0, 0, 255),      # Red (36-41)
    'left_eye': (0, 0, 255),       # Red (42-47)
    'outer_mouth': (255, 0, 255),  # Magenta (48-59)
    'inner_mouth': (255, 128, 0),  # Orange (60-67)
}


def get_region_98(idx):
    """Get region name for 98-point WFLW landmark index."""
    if idx <= 32:
        return 'jawline'
    elif idx <= 41:
        return 'right_brow'
    elif idx <= 50:
        return 'left_brow'
    elif idx <= 59:
        return 'nose'
    elif idx <= 67:
        return 'right_eye'
    elif idx <= 75:
        return 'left_eye'
    elif idx <= 87:
        return 'outer_mouth'
    elif idx <= 95:
        return 'inner_mouth'
    else:
        return 'pupils'


def get_region_68(idx):
    """Get region name for 68-point dlib landmark index."""
    if idx <= 16:
        return 'jawline'
    elif idx <= 21:
        return 'right_brow'
    elif idx <= 26:
        return 'left_brow'
    elif idx <= 35:
        return 'nose'
    elif idx <= 41:
        return 'right_eye'
    elif idx <= 47:
        return 'left_eye'
    elif idx <= 59:
        return 'outer_mouth'
    else:
        return 'inner_mouth'


def draw_landmarks_98(frame, landmarks, title="SPIGA 98-point"):
    """Draw 98-point WFLW landmarks with region colors."""
    vis = frame.copy()

    if landmarks is None:
        cv2.putText(vis, f"{title}: No landmarks", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        region = get_region_98(i)
        color = COLORS_98.get(region, (255, 255, 255))
        cv2.circle(vis, (int(x), int(y)), 2, color, -1)

    # Draw connections for eyes and mouth
    # Right eye (60-67)
    for i in range(60, 67):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_98['right_eye'], 1)
    cv2.line(vis, tuple(landmarks[67].astype(int)), tuple(landmarks[60].astype(int)), COLORS_98['right_eye'], 1)

    # Left eye (68-75)
    for i in range(68, 75):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_98['left_eye'], 1)
    cv2.line(vis, tuple(landmarks[75].astype(int)), tuple(landmarks[68].astype(int)), COLORS_98['left_eye'], 1)

    # Outer mouth (76-87)
    for i in range(76, 87):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_98['outer_mouth'], 1)
    cv2.line(vis, tuple(landmarks[87].astype(int)), tuple(landmarks[76].astype(int)), COLORS_98['outer_mouth'], 1)

    # Inner mouth (88-95)
    for i in range(88, 95):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_98['inner_mouth'], 1)
    cv2.line(vis, tuple(landmarks[95].astype(int)), tuple(landmarks[88].astype(int)), COLORS_98['inner_mouth'], 1)

    # Title
    cv2.putText(vis, f"A: {title}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"{len(landmarks)} points", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis


def draw_landmarks_68(frame, landmarks, title="68-point", panel_label="B"):
    """Draw 68-point dlib landmarks with region colors."""
    vis = frame.copy()

    if landmarks is None:
        cv2.putText(vis, f"{title}: No landmarks", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        region = get_region_68(i)
        color = COLORS_68.get(region, (255, 255, 255))
        cv2.circle(vis, (int(x), int(y)), 2, color, -1)

    # Draw connections
    # Jawline (0-16)
    for i in range(0, 16):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['jawline'], 1)

    # Right eyebrow (17-21)
    for i in range(17, 21):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['right_brow'], 1)

    # Left eyebrow (22-26)
    for i in range(22, 26):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['left_brow'], 1)

    # Nose bridge (27-30)
    for i in range(27, 30):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['nose'], 1)

    # Nose bottom (31-35)
    for i in range(31, 35):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['nose'], 1)

    # Right eye (36-41)
    for i in range(36, 41):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['right_eye'], 1)
    cv2.line(vis, tuple(landmarks[41].astype(int)), tuple(landmarks[36].astype(int)), COLORS_68['right_eye'], 1)

    # Left eye (42-47)
    for i in range(42, 47):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['left_eye'], 1)
    cv2.line(vis, tuple(landmarks[47].astype(int)), tuple(landmarks[42].astype(int)), COLORS_68['left_eye'], 1)

    # Outer mouth (48-59)
    for i in range(48, 59):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['outer_mouth'], 1)
    cv2.line(vis, tuple(landmarks[59].astype(int)), tuple(landmarks[48].astype(int)), COLORS_68['outer_mouth'], 1)

    # Inner mouth (60-67)
    for i in range(60, 67):
        pt1 = tuple(landmarks[i].astype(int))
        pt2 = tuple(landmarks[i+1].astype(int))
        cv2.line(vis, pt1, pt2, COLORS_68['inner_mouth'], 1)
    cv2.line(vis, tuple(landmarks[67].astype(int)), tuple(landmarks[60].astype(int)), COLORS_68['inner_mouth'], 1)

    # Title
    cv2.putText(vis, f"{panel_label}: {title}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"{len(landmarks)} points", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis


def get_video_rotation(video_path):
    """Get video rotation from metadata using ffprobe."""
    import subprocess
    import json

    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Check for rotation in tags
                    tags = stream.get('tags', {})
                    if 'rotate' in tags:
                        return int(tags['rotate'])
                    # Check side_data for displaymatrix rotation
                    for side_data in stream.get('side_data_list', []):
                        if 'rotation' in side_data:
                            return int(side_data['rotation'])
    except Exception as e:
        print(f"Error getting rotation: {e}")
    return 0


def apply_rotation(frame, rotation):
    """Apply rotation to frame based on video metadata.

    The displaymatrix rotation value indicates how to rotate the video to display correctly.
    - rotation -90 (or 270): video is rotated 90 CCW, so rotate 90 CW to correct
    - rotation 90 (or -270): video is rotated 90 CW, so rotate 90 CCW to correct
    - rotation 180: video is upside down, rotate 180 to correct
    """
    if rotation == -90 or rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def extract_frame(video_path, frame_number=30):
    """Extract a single frame from video with rotation correction."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")

    # Apply rotation based on metadata
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"Applying rotation correction: {rotation} degrees")
        frame = apply_rotation(frame, rotation)

    return frame


def main():
    parser = argparse.ArgumentParser(description='Compare landmark detectors visually')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--frame', type=int, default=30, help='Frame number to extract')
    parser.add_argument('--output', help='Output image path (optional, displays if not set)')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale factor for display')

    args = parser.parse_args()

    video_path = str(Path(args.video).resolve())
    print(f"Loading frame {args.frame} from: {video_path}")

    # Extract frame
    frame = extract_frame(video_path, args.frame)
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Initialize detectors
    print("\nInitializing SPIGA detector...")
    try:
        from spiga_detector import SPIGALandmarkDetector
        from landmark_mapper import wflw_to_dlib_68
        spiga_detector = SPIGALandmarkDetector(debug_mode=False)
        spiga_available = True
    except ImportError as e:
        print(f"SPIGA not available: {e}")
        print("Install with: pip install spiga facenet-pytorch")
        spiga_available = False

    print("Initializing CLNF detector...")
    try:
        from pyfaceau_detector import PyFaceAU68LandmarkDetector
        clnf_detector = PyFaceAU68LandmarkDetector(debug_mode=False)
        clnf_available = True
    except ImportError as e:
        print(f"CLNF not available: {e}")
        clnf_available = False

    # Get landmarks
    spiga_98 = None
    spiga_68 = None
    clnf_68 = None

    if spiga_available:
        print("\nRunning SPIGA detection...")
        spiga_68_result, info = spiga_detector.get_face_mesh(frame)
        if spiga_68_result is not None:
            spiga_68 = spiga_68_result
            # Get the native 98-point landmarks stored during detection
            spiga_98 = spiga_detector.last_landmarks_98
            print(f"  SPIGA 98-point: {spiga_98.shape if spiga_98 is not None else 'None'}")
            print(f"  SPIGA→68 mapped: {spiga_68.shape}")
        else:
            print(f"  SPIGA failed: {info}")

    if clnf_available:
        print("\nRunning CLNF detection...")
        clnf_68_result, info = clnf_detector.get_face_mesh(frame)
        if clnf_68_result is not None:
            clnf_68 = clnf_68_result
            print(f"  CLNF 68-point: {clnf_68.shape}")
        else:
            print(f"  CLNF failed: {info}")

    # Create visualization panels
    print("\nCreating visualization...")

    panel_a = draw_landmarks_98(frame, spiga_98, "SPIGA 98-point (native)")
    panel_b = draw_landmarks_68(frame, spiga_68, "SPIGA→68 (mapped)", "B")
    panel_c = draw_landmarks_68(frame, clnf_68, "CLNF 68-point", "C")

    # Combine panels horizontally
    combined = np.hstack([panel_a, panel_b, panel_c])

    # Add legend at bottom
    legend_height = 40
    legend = np.zeros((legend_height, combined.shape[1], 3), dtype=np.uint8)

    # Draw color legend
    x_offset = 10
    for name, color in [('Jaw', (0, 255, 255)), ('Brow', (255, 0, 0)),
                         ('Nose', (0, 255, 0)), ('Eye', (0, 0, 255)),
                         ('Mouth', (255, 0, 255))]:
        cv2.circle(legend, (x_offset, 20), 8, color, -1)
        cv2.putText(legend, name, (x_offset + 15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_offset += 80

    combined = np.vstack([combined, legend])

    # Scale for display
    if args.scale != 1.0:
        new_width = int(combined.shape[1] * args.scale)
        new_height = int(combined.shape[0] * args.scale)
        combined = cv2.resize(combined, (new_width, new_height))

    # Save or display
    if args.output:
        output_path = Path(args.output).resolve()
        cv2.imwrite(str(output_path), combined)
        print(f"\nSaved to: {output_path}")
    else:
        print("\nDisplaying (press any key to close)...")
        cv2.imshow('Landmark Comparison: A=SPIGA98, B=SPIGA→68, C=CLNF68', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
