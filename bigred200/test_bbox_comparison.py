#!/usr/bin/env python3
"""
Compare MTCNN bbox output between local and BR200.
Uses the same test frame to verify ONNX models produce identical results.
"""
import sys
import os
import json
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

def get_video_rotation(video_path):
    """Get rotation from video metadata."""
    cap = cv2.VideoCapture(video_path)
    rotation = 0
    if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    cap.release()
    return rotation

def rotate_frame(frame, rotation):
    """Apply rotation to frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number')
    parser.add_argument('--output', default='bbox_result.json', help='Output JSON file')
    args = parser.parse_args()

    # Import MTCNN
    from pymtcnn import MTCNN

    # Open video and extract frame
    cap = cv2.VideoCapture(args.video)
    rotation = get_video_rotation(args.video)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame}")
        sys.exit(1)

    # Apply rotation
    frame = rotate_frame(frame, rotation)

    # Convert BGR to RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(f"Frame shape: {frame_rgb.shape}")
    print(f"Video rotation: {rotation}°")

    # Initialize MTCNN with CPU backend (ONNX)
    detector = MTCNN(backend='cpu')

    # Detect faces
    bboxes, landmarks = detector.detect(frame_rgb)

    result = {
        'video': args.video,
        'frame': args.frame,
        'rotation': rotation,
        'frame_shape': list(frame_rgb.shape),
        'num_faces': 0,
        'bboxes': [],
        'landmarks': []
    }

    if bboxes is not None and len(bboxes) > 0:
        result['num_faces'] = len(bboxes)
        result['bboxes'] = bboxes.tolist()
        if landmarks is not None:
            result['landmarks'] = landmarks.tolist()

        for i, bbox in enumerate(bboxes):
            x, y, w, h, conf = bbox[:5] if len(bbox) >= 5 else (*bbox[:4], 1.0)
            print(f"Face {i}: bbox=[{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}], conf={conf:.6f}")
    else:
        print("No faces detected")

    # Save result
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to: {args.output}")

if __name__ == '__main__':
    main()
