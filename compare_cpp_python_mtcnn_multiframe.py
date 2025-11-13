#!/usr/bin/env python3
"""
Compare C++ and Python MTCNN outputs across multiple patient frames
to identify systematic differences in the detection pipeline.
"""

import cv2
import numpy as np
import subprocess
import re
from cpp_mtcnn_detector import CPPMTCNNDetector

def run_cpp_mtcnn(image_path):
    """Run C++ FeatureExtraction and parse bbox trace output"""
    cmd = [
        f"{os.path.expanduser('~')}/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction",
        "-f", image_path,
        "-out_dir", "/tmp/of_out",
        "-no_track"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr

    # Parse bbox traces
    traces = {}

    # After RNet regression
    match = re.search(r'\[C\+\+ BBOX TRACE\] After RNet regression.*?\n(.*?)(?=\n\[|$)', output, re.DOTALL)
    if match:
        boxes = []
        for line in match.group(1).split('\n'):
            if 'Box' in line:
                parts = re.findall(r'w=([\d.]+), h=([\d.]+)', line)
                if parts:
                    w, h = float(parts[0][0]), float(parts[0][1])
                    boxes.append({'w': w, 'h': h, 'square': abs(w-h) < 1.0})
        traces['rnet_regression'] = boxes

    # After RNet rectify
    match = re.search(r'\[C\+\+ BBOX TRACE\] After RNet rectify.*?\n(.*?)(?=\n\[|Total boxes)', output, re.DOTALL)
    if match:
        boxes = []
        for line in match.group(1).split('\n'):
            if 'Box' in line:
                parts = re.findall(r'w=([\d.]+), h=([\d.]+)', line)
                if parts:
                    w, h = float(parts[0][0]), float(parts[0][1])
                    boxes.append({'w': w, 'h': h, 'square': abs(w-h) < 1.0})
        traces['rnet_rectify'] = boxes

    # Total boxes to ONet
    match = re.search(r'Total boxes going to ONet: (\d+)', output)
    if match:
        traces['onet_count'] = int(match.group(1))

    # Final bbox
    match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    if match:
        traces['final_bbox'] = {
            'x': float(match.group(1)),
            'y': float(match.group(2)),
            'w': float(match.group(3)),
            'h': float(match.group(4))
        }

    return traces

def run_python_mtcnn(image_path):
    """Run Python MTCNN detector and capture bbox traces"""
    import sys
    from io import StringIO

    detector = CPPMTCNNDetector()
    img = cv2.imread(image_path)

    # Capture stdout to parse bbox traces
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    bboxes, landmarks = detector.detect(img)

    output = captured_output.getvalue()
    sys.stdout = old_stdout

    # Parse bbox traces
    traces = {}

    # After RNet regression
    match = re.search(r'\[BBOX TRACE\] After RNet regression.*?\n(.*?)(?=\n\[|$)', output, re.DOTALL)
    if match:
        boxes = []
        for line in match.group(1).split('\n'):
            if 'Box' in line:
                parts = re.findall(r'w=([\d.]+), h=([\d.]+)', line)
                if parts:
                    w, h = float(parts[0][0]), float(parts[0][1])
                    boxes.append({'w': w, 'h': h, 'square': abs(w-h) < 1.0})
        traces['rnet_regression'] = boxes

    # After RNet rectify
    match = re.search(r'\[BBOX TRACE\] After RNet rectify.*?\n(.*?)(?=\n\[|$)', output, re.DOTALL)
    if match:
        boxes = []
        for line in match.group(1).split('\n'):
            if 'Box' in line:
                parts = re.findall(r'w=([\d.]+), h=([\d.]+)', line)
                if parts:
                    w, h = float(parts[0][0]), float(parts[0][1])
                    boxes.append({'w': w, 'h': h, 'square': abs(w-h) < 1.0})
        traces['rnet_rectify'] = boxes

    # ONet count (from output or bbox count)
    traces['onet_count'] = len(bboxes)

    # Final bbox (largest width)
    if len(bboxes) > 0:
        widths = [bbox[2] for bbox in bboxes]
        largest_idx = np.argmax(widths)
        bbox = bboxes[largest_idx]
        traces['final_bbox'] = {
            'x': bbox[0],
            'y': bbox[1],
            'w': bbox[2],
            'h': bbox[3]
        }

    return traces, output

def compare_traces(cpp_traces, python_traces, frame_name):
    """Compare C++ and Python bbox traces"""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {frame_name}")
    print(f"{'='*80}")

    # RNet boxes to ONet
    print(f"\nBoxes sent to ONet:")
    print(f"  C++:    {cpp_traces.get('onet_count', 'N/A')}")
    print(f"  Python: {python_traces.get('onet_count', 'N/A')}")

    # RNet regression output (first 3 boxes)
    print(f"\nAfter RNet regression (first 3 boxes):")
    print(f"  C++:")
    for i, box in enumerate(cpp_traces.get('rnet_regression', [])[:3]):
        print(f"    Box {i}: w={box['w']:.2f}, h={box['h']:.2f}, square={box['square']}")
    print(f"  Python:")
    for i, box in enumerate(python_traces.get('rnet_regression', [])[:3]):
        print(f"    Box {i}: w={box['w']:.2f}, h={box['h']:.2f}, square={box['square']}")

    # RNet rectify output (first 3 boxes)
    print(f"\nAfter RNet rectify (first 3 boxes):")
    print(f"  C++:")
    for i, box in enumerate(cpp_traces.get('rnet_rectify', [])[:3]):
        print(f"    Box {i}: w={box['w']:.2f}, h={box['h']:.2f}, square={box['square']}")
    print(f"  Python:")
    for i, box in enumerate(python_traces.get('rnet_rectify', [])[:3]):
        print(f"    Box {i}: w={box['w']:.2f}, h={box['h']:.2f}, square={box['square']}")

    # Final bbox
    print(f"\nFinal selected bbox:")
    cpp_bbox = cpp_traces.get('final_bbox', {})
    py_bbox = python_traces.get('final_bbox', {})

    if cpp_bbox and py_bbox:
        print(f"  C++:    x={cpp_bbox['x']:.2f}, y={cpp_bbox['y']:.2f}, w={cpp_bbox['w']:.2f}, h={cpp_bbox['h']:.2f}")
        print(f"  Python: x={py_bbox['x']:.2f}, y={py_bbox['y']:.2f}, w={py_bbox['w']:.2f}, h={py_bbox['h']:.2f}")

        # Compute differences
        dx = abs(cpp_bbox['x'] - py_bbox['x'])
        dy = abs(cpp_bbox['y'] - py_bbox['y'])
        dw = abs(cpp_bbox['w'] - py_bbox['w'])
        dh = abs(cpp_bbox['h'] - py_bbox['h'])

        print(f"\n  Differences:")
        print(f"    dx={dx:.2f} pixels, dy={dy:.2f} pixels")
        print(f"    dw={dw:.2f} pixels, dh={dh:.2f} pixels")

        # IoU
        x1_min = min(cpp_bbox['x'], py_bbox['x'])
        y1_min = min(cpp_bbox['y'], py_bbox['y'])
        x1_max = max(cpp_bbox['x'], py_bbox['x'])
        y1_max = max(cpp_bbox['y'], py_bbox['y'])

        x2_cpp = cpp_bbox['x'] + cpp_bbox['w']
        y2_cpp = cpp_bbox['y'] + cpp_bbox['h']
        x2_py = py_bbox['x'] + py_bbox['w']
        y2_py = py_bbox['y'] + py_bbox['h']

        x2_min = min(x2_cpp, x2_py)
        y2_min = min(y2_cpp, y2_py)
        x2_max = max(x2_cpp, x2_py)
        y2_max = max(y2_cpp, y2_py)

        # Intersection
        inter_x1 = max(cpp_bbox['x'], py_bbox['x'])
        inter_y1 = max(cpp_bbox['y'], py_bbox['y'])
        inter_x2 = min(x2_cpp, x2_py)
        inter_y2 = min(y2_cpp, y2_py)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            cpp_area = cpp_bbox['w'] * cpp_bbox['h']
            py_area = py_bbox['w'] * py_bbox['h']
            union_area = cpp_area + py_area - inter_area
            iou = inter_area / union_area
            print(f"    IoU: {iou:.1%}")
        else:
            print(f"    IoU: 0.0%")

import os

if __name__ == "__main__":
    test_frames = [
        "calibration_frames/patient1_frame1.jpg",
        "calibration_frames/patient2_frame1.jpg"
    ]

    for frame_path in test_frames:
        frame_name = os.path.basename(frame_path)

        print(f"\n{'#'*80}")
        print(f"# Processing: {frame_name}")
        print(f"{'#'*80}")

        # Run C++ MTCNN
        print(f"\nRunning C++ MTCNN...")
        cpp_traces = run_cpp_mtcnn(frame_path)

        # Run Python MTCNN
        print(f"Running Python MTCNN...")
        python_traces, python_output = run_python_mtcnn(frame_path)

        # Compare
        compare_traces(cpp_traces, python_traces, frame_name)

    print(f"\n{'#'*80}")
    print(f"# ANALYSIS COMPLETE")
    print(f"{'#'*80}")
