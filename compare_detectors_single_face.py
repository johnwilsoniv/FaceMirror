#!/usr/bin/env python3
"""
Compare Face Detectors on Single Face Image

Tests C++ MTCNN, RetinaFace, and PyMTCNN on the same single-face image
to determine if there are systematic bbox differences that require correction.
"""

import subprocess
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent / "pyfaceau"))

def run_cpp_mtcnn(image_path, output_dir):
    """Run C++ MTCNN detector."""
    binary_path = Path.home() / "repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction"

    cmd = [
        str(binary_path),
        "-f", str(image_path),
        "-out_dir", str(output_dir),
        "-verbose"
    ]

    print("="*80)
    print("C++ MTCNN DETECTOR")
    print("="*80)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse bbox from debug output
    bbox_match = re.search(r'DEBUG_BBOX: ([\d.]+),([\d.]+),([\d.]+),([\d.]+)', output)
    init_params_match = re.search(r'DEBUG_INIT_PARAMS: scale=([\d.]+)', output)

    data = {'detector': 'C++ MTCNN'}

    if bbox_match:
        x, y, w, h = [float(bbox_match.group(i)) for i in range(1, 5)]
        data['bbox'] = (x, y, w, h)
        data['center'] = (x + w/2, y + h/2)
        data['aspect_ratio'] = w / h if h > 0 else 0
        print(f"✓ BBox: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
        print(f"  Center: ({data['center'][0]:.1f}, {data['center'][1]:.1f})")
        print(f"  Aspect: {data['aspect_ratio']:.3f}")

    if init_params_match:
        data['init_scale'] = float(init_params_match.group(1))
        print(f"✓ Init scale: {data['init_scale']:.6f}")

    return data

def run_retinaface(image_path):
    """Run RetinaFace detector."""
    from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector

    print("\n" + "="*80)
    print("RETINAFACE DETECTOR")
    print("="*80)

    img = cv2.imread(str(image_path))

    retinaface_model = Path("pyfaceau/weights/retinaface_mobilenet025_coreml.onnx")
    detector = ONNXRetinaFaceDetector(
        str(retinaface_model),
        use_coreml=False,
        confidence_threshold=0.5,
        nms_threshold=0.4
    )

    detections, _ = detector.detect_faces(img, resize=1.0)

    if len(detections) == 0:
        print("✗ No faces detected")
        return None

    # Get first (highest confidence) detection
    detection = detections[0]
    x1, y1, x2, y2 = detection[:4]
    x, y, w, h = x1, y1, x2 - x1, y2 - y1

    data = {
        'detector': 'RetinaFace',
        'bbox': (x, y, w, h),
        'center': (x + w/2, y + h/2),
        'aspect_ratio': w / h if h > 0 else 0,
        'confidence': float(detection[4]) if len(detection) > 4 else None
    }

    print(f"✓ BBox: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    print(f"  Center: ({data['center'][0]:.1f}, {data['center'][1]:.1f})")
    print(f"  Aspect: {data['aspect_ratio']:.3f}")
    if data['confidence']:
        print(f"  Confidence: {data['confidence']:.3f}")

    # Get init scale using pyCLNF PDM
    from pyclnf.core import PDM
    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))
    init_params = pdm.init_params(data['bbox'])
    data['init_scale'] = init_params[0]
    print(f"✓ Init scale: {data['init_scale']:.6f}")

    return data

def run_pymtcnn(image_path):
    """Run PyMTCNN detector."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "openface_mtcnn",
        "pyfaceau/pyfaceau/detectors/openface_mtcnn.py"
    )
    openface_mtcnn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openface_mtcnn)
    OpenFaceMTCNN = openface_mtcnn.OpenFaceMTCNN

    print("\n" + "="*80)
    print("PYMTCNN DETECTOR")
    print("="*80)

    img = cv2.imread(str(image_path))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detector = OpenFaceMTCNN()
    bboxes, _ = detector.detect(rgb_img)

    if len(bboxes) == 0:
        print("✗ No faces detected")
        return None

    # Get first detection (x1, y1, x2, y2)
    x1, y1, x2, y2 = bboxes[0]
    x, y, w, h = x1, y1, x2 - x1, y2 - y1

    data = {
        'detector': 'PyMTCNN',
        'bbox': (x, y, w, h),
        'center': (x + w/2, y + h/2),
        'aspect_ratio': w / h if h > 0 else 0
    }

    print(f"✓ BBox: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
    print(f"  Center: ({data['center'][0]:.1f}, {data['center'][1]:.1f})")
    print(f"  Aspect: {data['aspect_ratio']:.3f}")

    # Get init scale using pyCLNF PDM
    from pyclnf.core import PDM
    model_dir = Path("pyclnf/models/exported_pdm")
    pdm = PDM(str(model_dir))
    init_params = pdm.init_params(data['bbox'])
    data['init_scale'] = init_params[0]
    print(f"✓ Init scale: {data['init_scale']:.6f}")

    return data

def visualize_comparison(image_path, cpp_data, retina_data, pymtcnn_data, output_path):
    """Create detailed comparison visualization."""
    img = cv2.imread(str(image_path))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Row 1: Individual detections
    detectors = [
        (cpp_data, 'red', 'C++'),
        (retina_data, 'blue', 'RetinaFace'),
        (pymtcnn_data, 'green', 'PyMTCNN')
    ]

    for idx, (data, color, name) in enumerate(detectors):
        if data is None:
            continue

        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(rgb_img)

        if 'bbox' in data:
            x, y, w, h = data['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=3)
            ax.add_patch(rect)

            # Draw center point
            cx, cy = data['center']
            ax.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)

            # Draw bbox dimensions
            ax.text(x, y - 10, f"{w:.0f}×{h:.0f}", color=color, fontsize=10,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        title = f"{name}\nBBox: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})\n"
        title += f"Aspect: {data.get('aspect_ratio', 0):.3f}"
        if 'init_scale' in data:
            title += f" | Init Scale: {data['init_scale']:.4f}"

        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')

    # Row 2: Overlays and analysis
    # C++ vs RetinaFace
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(rgb_img)
    if cpp_data and 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2, label='C++ MTCNN')
        ax.add_patch(rect)
        ax.plot(*cpp_data['center'], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
    if retina_data and 'bbox' in retina_data:
        x, y, w, h = retina_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='blue', linewidth=2, label='RetinaFace')
        ax.add_patch(rect)
        ax.plot(*retina_data['center'], 'bo', markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.set_title("C++ MTCNN vs RetinaFace", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # C++ vs PyMTCNN
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(rgb_img)
    if cpp_data and 'bbox' in cpp_data:
        x, y, w, h = cpp_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2, label='C++ MTCNN')
        ax.add_patch(rect)
        ax.plot(*cpp_data['center'], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1)
    if pymtcnn_data and 'bbox' in pymtcnn_data:
        x, y, w, h = pymtcnn_data['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, color='green', linewidth=2, label='PyMTCNN')
        ax.add_patch(rect)
        ax.plot(*pymtcnn_data['center'], 'go', markersize=8, markeredgecolor='white', markeredgewidth=1)
    ax.set_title("C++ MTCNN vs PyMTCNN", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('off')

    # Detailed comparison table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    text = "DETECTOR COMPARISON ANALYSIS\n"
    text += "="*60 + "\n\n"

    # Reference: C++ MTCNN
    if cpp_data and 'bbox' in cpp_data:
        cx, cy, cw, ch = cpp_data['bbox']
        text += "REFERENCE: C++ MTCNN\n"
        text += f"  BBox: ({cx:.1f}, {cy:.1f}, {cw:.1f}, {ch:.1f})\n"
        text += f"  Center: ({cpp_data['center'][0]:.1f}, {cpp_data['center'][1]:.1f})\n"
        text += f"  Aspect: {cpp_data['aspect_ratio']:.4f}\n"
        text += f"  Init Scale: {cpp_data.get('init_scale', 0):.6f}\n\n"

        # RetinaFace comparison
        if retina_data and 'bbox' in retina_data:
            rx, ry, rw, rh = retina_data['bbox']
            text += "RETINAFACE vs C++ MTCNN:\n"
            text += f"  ΔX: {rx - cx:+.1f}px  ({((rx-cx)/cw)*100:+.1f}% of width)\n"
            text += f"  ΔY: {ry - cy:+.1f}px  ({((ry-cy)/ch)*100:+.1f}% of height)\n"
            text += f"  ΔWidth: {rw - cw:+.1f}px  ({((rw-cw)/cw)*100:+.1f}%)\n"
            text += f"  ΔHeight: {rh - ch:+.1f}px  ({((rh-ch)/ch)*100:+.1f}%)\n"
            text += f"  ΔCenter: {np.linalg.norm(np.array(retina_data['center']) - np.array(cpp_data['center'])):.1f}px\n"
            text += f"  ΔAspect: {retina_data['aspect_ratio'] - cpp_data['aspect_ratio']:+.4f}\n"
            text += f"  ΔInit Scale: {retina_data.get('init_scale', 0) - cpp_data.get('init_scale', 0):+.6f}\n\n"

        # PyMTCNN comparison
        if pymtcnn_data and 'bbox' in pymtcnn_data:
            px, py, pw, ph = pymtcnn_data['bbox']
            text += "PYMTCNN vs C++ MTCNN:\n"
            text += f"  ΔX: {px - cx:+.1f}px  ({((px-cx)/cw)*100:+.1f}% of width)\n"
            text += f"  ΔY: {py - cy:+.1f}px  ({((py-cy)/ch)*100:+.1f}% of height)\n"
            text += f"  ΔWidth: {pw - cw:+.1f}px  ({((pw-cw)/cw)*100:+.1f}%)\n"
            text += f"  ΔHeight: {ph - ch:+.1f}px  ({((ph-ch)/ch)*100:+.1f}%)\n"
            text += f"  ΔCenter: {np.linalg.norm(np.array(pymtcnn_data['center']) - np.array(cpp_data['center'])):.1f}px\n"
            text += f"  ΔAspect: {pymtcnn_data['aspect_ratio'] - cpp_data['aspect_ratio']:+.4f}\n"
            text += f"  ΔInit Scale: {pymtcnn_data.get('init_scale', 0) - cpp_data.get('init_scale', 0):+.6f}\n\n"

    text += "="*60 + "\n"
    text += "CORRECTION FACTOR ANALYSIS:\n"
    text += "="*60 + "\n"

    if cpp_data and retina_data and 'bbox' in cpp_data and 'bbox' in retina_data:
        cx, cy, cw, ch = cpp_data['bbox']
        rx, ry, rw, rh = retina_data['bbox']

        scale_init_diff = abs(retina_data.get('init_scale', 0) - cpp_data.get('init_scale', 0))

        if scale_init_diff < 0.01:
            text += "✓ RetinaFace: EXCELLENT agreement\n"
            text += "  No correction needed!\n"
        elif scale_init_diff < 0.1:
            text += "⚠ RetinaFace: MODERATE difference\n"
            text += "  Correction factor may improve init:\n"
            text += f"    x_offset = {(rx - cx) / rw:.3f} * width\n"
            text += f"    y_offset = {(ry - cy) / rh:.3f} * height\n"
            text += f"    width_scale = {cw / rw:.3f}\n"
            text += f"    height_scale = {ch / rh:.3f}\n"
        else:
            text += "✗ RetinaFace: LARGE difference\n"
            text += "  May need systematic correction\n"

    if cpp_data and pymtcnn_data and 'bbox' in cpp_data and 'bbox' in pymtcnn_data:
        scale_init_diff = abs(pymtcnn_data.get('init_scale', 0) - cpp_data.get('init_scale', 0))

        if scale_init_diff < 0.01:
            text += "\n✓ PyMTCNN: EXCELLENT agreement\n"
            text += "  No correction needed!\n"
        else:
            text += f"\n⚠ PyMTCNN: {scale_init_diff:.6f} scale difference\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Visualization saved: {output_path}")
    print(f"{'='*80}")
    plt.close()

def main():
    """Main comparison function."""
    image_path = Path("test_frames_single/normal_face.jpg")
    output_dir = Path("test_output/detector_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("DETECTOR COMPARISON ON SINGLE FACE")
    print("="*80)
    print(f"Image: {image_path}")
    print()
    print("Testing three detectors on the SAME face:")
    print("1. C++ MTCNN (OpenFace)")
    print("2. RetinaFace (ONNX)")
    print("3. PyMTCNN")
    print()

    # Run all detectors
    cpp_data = run_cpp_mtcnn(image_path, output_dir)
    retina_data = run_retinaface(image_path)
    pymtcnn_data = run_pymtcnn(image_path)

    # Create visualization
    output_path = output_dir / "detector_comparison_single_face.jpg"
    visualize_comparison(image_path, cpp_data, retina_data, pymtcnn_data, output_path)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if cpp_data and retina_data and 'init_scale' in cpp_data and 'init_scale' in retina_data:
        scale_diff = abs(retina_data['init_scale'] - cpp_data['init_scale'])
        print(f"\nRetinaFace vs C++ MTCNN:")
        print(f"  Init scale difference: {scale_diff:.6f}")
        if scale_diff < 0.01:
            print(f"  ✓ EXCELLENT - No correction needed")
        elif scale_diff < 0.1:
            print(f"  ⚠ MODERATE - Correction factor may help")
        else:
            print(f"  ✗ LARGE - Systematic correction recommended")

    if cpp_data and pymtcnn_data and 'init_scale' in cpp_data and 'init_scale' in pymtcnn_data:
        scale_diff = abs(pymtcnn_data['init_scale'] - cpp_data['init_scale'])
        print(f"\nPyMTCNN vs C++ MTCNN:")
        print(f"  Init scale difference: {scale_diff:.6f}")
        if scale_diff < 0.01:
            print(f"  ✓ EXCELLENT - No correction needed")

    print(f"\nVisualization: {output_path}")
    print()

if __name__ == "__main__":
    main()
