#!/usr/bin/env python3
"""
Debug mean-shift transformation between C++ and Python.
Output exact values for comparison.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'pyclnf'))
sys.path.insert(0, str(Path(__file__).parent / 'pymtcnn'))

from pyclnf import CLNF
from pyclnf.core.pdm import PDM
from pyclnf.core.utils import align_shapes_with_scale

def main():
    print("=" * 80)
    print("MEAN-SHIFT TRANSFORMATION DEBUG")
    print("=" * 80)
    
    # Setup
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Normal Cohort/Shorty.mov"
    frame_idx = 160
    
    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load PDM and get initial shape
    MODEL_DIR = Path("pyclnf/models")
    pdm = PDM(str(MODEL_DIR / "exported_pdm"))
    
    # Use C++ bbox
    bbox = np.array([356.071, 733.145, 401.071, 425.58])
    
    # Initialize parameters
    init_params = pdm.calc_params_from_bbox(bbox)
    landmarks_2d = pdm.params_to_landmarks_2d(init_params)
    
    print(f"\nInitial landmarks shape: {landmarks_2d.shape}")
    print(f"Landmark 36 initial: ({landmarks_2d[36, 0]:.4f}, {landmarks_2d[36, 1]:.4f})")
    
    # Get reference shape for similarity transform
    ref_shape = pdm.mean_shape.copy()[:, :2]  # 2D mean shape
    
    # Compute similarity transform (image to reference)
    print("\n" + "=" * 80)
    print("SIMILARITY TRANSFORM COMPUTATION")
    print("=" * 80)
    
    # OpenFace computes this with align_shapes_with_scale
    sim_img_to_ref, a_sim, S = align_shapes_with_scale(landmarks_2d, ref_shape)
    
    print(f"\nsim_img_to_ref matrix:")
    print(sim_img_to_ref)
    print(f"\na (scale*cos): {sim_img_to_ref[0, 0]:.6f}")
    print(f"b (scale*sin): {sim_img_to_ref[1, 0]:.6f}")
    print(f"tx: {sim_img_to_ref[0, 2]:.6f}")
    print(f"ty: {sim_img_to_ref[1, 2]:.6f}")
    
    # Compute inverse transform
    a = sim_img_to_ref[0, 0]
    b = sim_img_to_ref[1, 0]
    det = a*a + b*b
    
    sim_ref_to_img = np.array([
        [a/det, b/det, 0],
        [-b/det, a/det, 0]
    ], dtype=np.float32)
    
    # Compute translation for inverse
    tx = sim_img_to_ref[0, 2]
    ty = sim_img_to_ref[1, 2]
    inv_tx = -(a*tx + b*ty) / det
    inv_ty = -(-b*tx + a*ty) / det
    sim_ref_to_img[0, 2] = inv_tx
    sim_ref_to_img[1, 2] = inv_ty
    
    print(f"\nsim_ref_to_img matrix:")
    print(sim_ref_to_img)
    
    # Test: transform landmark 36 to reference and back
    lm36 = landmarks_2d[36]
    lm36_ref = np.array([
        a * lm36[0] - b * lm36[1] + tx,
        b * lm36[0] + a * lm36[1] + ty
    ])
    
    a_inv = sim_ref_to_img[0, 0]
    b_inv = sim_ref_to_img[1, 0]
    lm36_back = np.array([
        a_inv * lm36_ref[0] - b_inv * lm36_ref[1] + sim_ref_to_img[0, 2],
        b_inv * lm36_ref[0] + a_inv * lm36_ref[1] + sim_ref_to_img[1, 2]
    ])
    
    print(f"\nLandmark 36 transform test:")
    print(f"  Original: ({lm36[0]:.4f}, {lm36[1]:.4f})")
    print(f"  To ref:   ({lm36_ref[0]:.4f}, {lm36_ref[1]:.4f})")
    print(f"  Back:     ({lm36_back[0]:.4f}, {lm36_back[1]:.4f})")
    
    # Now test mean-shift transformation
    print("\n" + "=" * 80)
    print("MEAN-SHIFT TRANSFORMATION TEST")
    print("=" * 80)
    
    # Assume we have a mean-shift in reference coordinates
    # Let's use typical values
    ms_ref_x = -5.0
    ms_ref_y = 3.0
    
    # Python method (current implementation)
    a_mat = sim_ref_to_img[0, 0]
    b_mat = sim_ref_to_img[1, 0]
    ms_py_x = a_mat * ms_ref_x - b_mat * ms_ref_y
    ms_py_y = b_mat * ms_ref_x + a_mat * ms_ref_y
    
    print(f"\nMean-shift in ref coords: ({ms_ref_x}, {ms_ref_y})")
    print(f"Python transform: ms_x = a*mx - b*my, ms_y = b*mx + a*my")
    print(f"  a_mat: {a_mat:.6f}, b_mat: {b_mat:.6f}")
    print(f"  Result: ({ms_py_x:.4f}, {ms_py_y:.4f})")
    
    # C++ method: row_vector * matrix^T
    # [mx, my] * [[a, b], [-b, a]] = [a*mx - b*my, b*mx + a*my]
    # This is the SAME as Python!
    
    # But C++ uses * .t() which is:
    # [mx, my] * [[a, -b], [b, a]]^T = [mx, my] * [[a, b], [-b, a]]
    # = [a*mx + (-b)*my, b*mx + a*my]
    # = [a*mx - b*my, b*mx + a*my]
    
    # Still the same!
    
    print(f"C++ uses: row_vec * matrix^T")
    print(f"  Both should give same result.")
    
    # Let's verify what the actual matrices look like by checking C++ trace
    print("\n" + "=" * 80)
    print("CHECK C++ DEBUG OUTPUT")
    print("=" * 80)
    
    # Read C++ trace to see actual mean-shift values
    trace_file = Path("/tmp/clnf_iteration_traces/cpp_trace.txt")
    if trace_file.exists():
        with open(trace_file) as f:
            lines = f.readlines()
            for line in lines[:5]:
                if not line.startswith('#'):
                    print(f"C++ trace: {line.strip()}")

if __name__ == "__main__":
    main()
