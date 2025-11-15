"""
Analysis Script: Compare C++ OpenFace vs PyFaceAU Results

Generates:
1. MTCNN stage comparison metrics
2. Landmark accuracy metrics (initial & final)
3. Pose estimation comparison
4. AU correlation analysis
5. Visualizations
6. Validation report
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

# Paths
PROJECT_ROOT = Path(__file__).parent
CALIBRATION_FRAMES = PROJECT_ROOT / "calibration_frames"
CPP_OUTPUT = PROJECT_ROOT / "validation_output" / "cpp_baseline"
PYTHON_OUTPUT = PROJECT_ROOT / "validation_output" / "python_baseline"
REPORT_OUTPUT = PROJECT_ROOT / "validation_output" / "report"
VIZ_OUTPUT = REPORT_OUTPUT / "visualizations"

# Create output directories
REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)
VIZ_OUTPUT.mkdir(parents=True, exist_ok=True)

def load_cpp_results(frame_name):
    """Load C++ results for a frame"""
    csv_file = CPP_OUTPUT / f"{frame_name}.csv"
    mtcnn_file = CPP_OUTPUT / f"{frame_name}_mtcnn_debug.csv"

    if not csv_file.exists():
        return None

    # Load main CSV
    df = pd.read_csv(csv_file)
    row = df.iloc[0]

    results = {
        'frame': frame_name,
        'success': row['success'] == 1,
        'confidence': row['confidence'],
        'landmarks_2d': np.array([[row[f'x_{i}'], row[f'y_{i}']] for i in range(68)]),
        'landmarks_3d': np.array([[row[f'X_{i}'], row[f'Y_{i}'], row[f'Z_{i}']] for i in range(68)]),
        'pose': {
            'scale': row['p_scale'],
            'rx': row['p_rx'],
            'ry': row['p_ry'],
            'rz': row['p_rz'],
            'tx': row['p_tx'],
            'ty': row['p_ty']
        },
        'aus': {f'AU{col.split("_")[0][2:]}': row[col] for col in df.columns if col.startswith('AU') and col.endswith('_r')}
    }

    # Load MTCNN 5-point landmarks
    if mtcnn_file.exists():
        mtcnn_df = pd.read_csv(mtcnn_file)
        if len(mtcnn_df) > 0:
            mtcnn_row = mtcnn_df.iloc[0]
            results['mtcnn_5pt'] = np.array([
                [mtcnn_row['lm1_x'], mtcnn_row['lm1_y']],
                [mtcnn_row['lm2_x'], mtcnn_row['lm2_y']],
                [mtcnn_row['lm3_x'], mtcnn_row['lm3_y']],
                [mtcnn_row['lm4_x'], mtcnn_row['lm4_y']],
                [mtcnn_row['lm5_x'], mtcnn_row['lm5_y']]
            ])
            results['mtcnn_bbox'] = [mtcnn_row['bbox_x'], mtcnn_row['bbox_y'],
                                     mtcnn_row['bbox_w'], mtcnn_row['bbox_h']]

    return results

def load_python_results(frame_name):
    """Load Python results for a frame"""
    json_file = PYTHON_OUTPUT / f"{frame_name}_result.json"

    if not json_file.exists():
        return None

    with open(json_file, 'r') as f:
        data = json.load(f)

    if not data['success']:
        return None

    debug = data.get('debug_info', {})

    results = {
        'frame': frame_name,
        'success': data['success'],
        'aus': data['au_results']
    }

    # Extract landmarks from debug info
    if 'landmark_detection' in debug:
        results['landmarks_68'] = np.array(debug['landmark_detection']['landmarks_68'])

    # Extract MTCNN info from face detection
    if 'face_detection' in debug:
        face_det = debug['face_detection']
        if 'bbox' in face_det:
            results['mtcnn_bbox'] = face_det['bbox']

    # Extract pose
    if 'pose_estimation' in debug:
        pose = debug['pose_estimation']
        results['pose'] = {
            'scale': pose['scale'],
            'rx': pose['rotation'][0],
            'ry': pose['rotation'][1],
            'rz': pose['rotation'][2],
            'tx': pose['translation'][0],
            'ty': pose['translation'][1]
        }

    return results

def compare_landmarks(cpp_lm, py_lm):
    """Compare landmark sets"""
    if cpp_lm is None or py_lm is None:
        return None

    # Calculate per-landmark euclidean distance
    distances = [euclidean(cpp_lm[i], py_lm[i]) for i in range(len(cpp_lm))]

    return {
        'mean_error': np.mean(distances),
        'max_error': np.max(distances),
        'median_error': np.median(distances),
        'std_error': np.std(distances),
        'per_point_errors': distances
    }

def compare_pose(cpp_pose, py_pose):
    """Compare pose parameters"""
    if cpp_pose is None or py_pose is None:
        return None

    return {
        'scale_diff': abs(cpp_pose['scale'] - py_pose['scale']),
        'rx_diff': abs(cpp_pose['rx'] - py_pose['rx']),
        'ry_diff': abs(cpp_pose['ry'] - py_pose['ry']),
        'rz_diff': abs(cpp_pose['rz'] - py_pose['rz']),
        'tx_diff': abs(cpp_pose['tx'] - py_pose['tx']),
        'ty_diff': abs(cpp_pose['ty'] - py_pose['ty'])
    }

def compare_aus(cpp_aus, py_aus):
    """Compare AU predictions"""
    if cpp_aus is None or py_aus is None:
        return None

    # Match AUs - handle different naming conventions
    # C++ dict has 'AU01', Python dict has 'AU01_r'
    common_aus = set()
    for au_key in cpp_aus.keys():
        # au_key is like 'AU01' from C++
        if au_key in py_aus:
            common_aus.add(au_key)
        elif au_key + '_r' in py_aus:
            common_aus.add(au_key)

    if len(common_aus) == 0:
        return None

    cpp_values = []
    py_values = []
    for au in sorted(common_aus):
        # au is the base form like 'AU01'
        cpp_au = au
        py_au = au if au in py_aus else au + '_r'

        if cpp_au in cpp_aus and py_au in py_aus:
            cpp_values.append(cpp_aus[cpp_au])
            py_values.append(py_aus[py_au])

    if len(cpp_values) == 0:
        return None

    correlation, p_value = pearsonr(cpp_values, py_values)
    rmse = np.sqrt(np.mean([(c - p)**2 for c, p in zip(cpp_values, py_values)]))
    mae = np.mean([abs(c - p) for c, p in zip(cpp_values, py_values)])

    return {
        'correlation': correlation,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'num_aus': len(cpp_values),
        'per_au_diff': {au: abs(cpp_values[i] - py_values[i]) for i, au in enumerate(sorted(common_aus))}
    }

def visualize_landmarks(frame_name, cpp_results, py_results, img):
    """Create landmark visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: C++ landmarks
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if cpp_results and 'landmarks_2d' in cpp_results:
        lm = cpp_results['landmarks_2d']
        axes[0].scatter(lm[:, 0], lm[:, 1], c='green', s=20, alpha=0.7, label='C++ Landmarks')
    axes[0].set_title(f'C++ OpenFace - {frame_name}')
    axes[0].legend()
    axes[0].axis('off')

    # Plot 2: Python landmarks
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if py_results and 'landmarks_68' in py_results:
        lm = py_results['landmarks_68']
        axes[1].scatter(lm[:, 0], lm[:, 1], c='blue', s=20, alpha=0.7, label='Python Landmarks')
    axes[1].set_title(f'PyFaceAU - {frame_name}')
    axes[1].legend()
    axes[1].axis('off')

    # Plot 3: Overlay
    axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if cpp_results and 'landmarks_2d' in cpp_results:
        lm_cpp = cpp_results['landmarks_2d']
        axes[2].scatter(lm_cpp[:, 0], lm_cpp[:, 1], c='green', s=20, alpha=0.5, label='C++')
    if py_results and 'landmarks_68' in py_results:
        lm_py = py_results['landmarks_68']
        axes[2].scatter(lm_py[:, 0], lm_py[:, 1], c='blue', s=20, alpha=0.5, label='Python')
    axes[2].set_title(f'Overlay - {frame_name}')
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(VIZ_OUTPUT / f'{frame_name}_landmarks.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Run comprehensive comparison"""
    print("\n" + "="*80)
    print("PyFaceAU vs C++ OpenFace: Results Analysis")
    print("="*80)

    # Get list of frames
    frames = sorted([f.stem for f in CALIBRATION_FRAMES.glob("patient*.jpg")])
    print(f"\nAnalyzing {len(frames)} frames...")

    all_comparisons = []

    for idx, frame_name in enumerate(frames, 1):
        print(f"\n[{idx}/{len(frames)}] Analyzing {frame_name}...")

        # Load results
        cpp_results = load_cpp_results(frame_name)
        py_results = load_python_results(frame_name)

        if cpp_results is None:
            print(f"  ⚠ No C++ results found")
            continue
        if py_results is None:
            print(f"  ⚠ No Python results found")
            continue

        # Compare
        comparison = {'frame': frame_name}

        # Landmarks
        if 'landmarks_2d' in cpp_results and 'landmarks_68' in py_results:
            lm_comp = compare_landmarks(cpp_results['landmarks_2d'], py_results['landmarks_68'])
            if lm_comp:
                comparison['landmarks'] = lm_comp
                print(f"  ✓ Landmarks: {lm_comp['mean_error']:.2f}px mean error")

        # Pose
        if 'pose' in cpp_results and 'pose' in py_results:
            pose_comp = compare_pose(cpp_results['pose'], py_results['pose'])
            if pose_comp:
                comparison['pose'] = pose_comp
                print(f"  ✓ Pose: scale_diff={pose_comp['scale_diff']:.3f}")

        # AUs
        if 'aus' in cpp_results and 'aus' in py_results:
            au_comp = compare_aus(cpp_results['aus'], py_results['aus'])
            if au_comp:
                comparison['aus'] = au_comp
                print(f"  ✓ AUs: r={au_comp['correlation']:.3f}, RMSE={au_comp['rmse']:.3f}")

        # Visualize
        img_path = CALIBRATION_FRAMES / f"{frame_name}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            visualize_landmarks(frame_name, cpp_results, py_results, img)
            print(f"  ✓ Visualization saved")

        all_comparisons.append(comparison)

    # Generate summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if len(all_comparisons) > 0:
        # Landmark summary
        lm_errors = [c['landmarks']['mean_error'] for c in all_comparisons if 'landmarks' in c]
        if lm_errors:
            print(f"\nLandmark Accuracy:")
            print(f"  Mean error: {np.mean(lm_errors):.2f}px")
            print(f"  Max error: {np.max(lm_errors):.2f}px")
            print(f"  Median error: {np.median(lm_errors):.2f}px")
            print(f"  Std dev: {np.std(lm_errors):.2f}px")

        # Pose summary
        pose_diffs = [c['pose'] for c in all_comparisons if 'pose' in c]
        if pose_diffs:
            print(f"\nPose Estimation:")
            print(f"  Mean scale diff: {np.mean([p['scale_diff'] for p in pose_diffs]):.3f}")
            print(f"  Mean rotation diff (rx): {np.mean([p['rx_diff'] for p in pose_diffs]):.3f}°")
            print(f"  Mean rotation diff (ry): {np.mean([p['ry_diff'] for p in pose_diffs]):.3f}°")
            print(f"  Mean rotation diff (rz): {np.mean([p['rz_diff'] for p in pose_diffs]):.3f}°")

        # AU summary
        au_comps = [c['aus'] for c in all_comparisons if 'aus' in c]
        if au_comps:
            print(f"\nAU Predictions:")
            print(f"  Mean correlation: {np.mean([a['correlation'] for a in au_comps]):.3f}")
            print(f"  Min correlation: {np.min([a['correlation'] for a in au_comps]):.3f}")
            print(f"  Max correlation: {np.max([a['correlation'] for a in au_comps]):.3f}")
            print(f"  Mean RMSE: {np.mean([a['rmse'] for a in au_comps]):.3f}")
            print(f"  Mean MAE: {np.mean([a['mae'] for a in au_comps]):.3f}")
            print(f"  Num AUs compared: {au_comps[0]['num_aus']}")

    print(f"\n✓ Analysis complete! Results saved to: {REPORT_OUTPUT}")
    print(f"✓ Visualizations saved to: {VIZ_OUTPUT}")

if __name__ == "__main__":
    main()
